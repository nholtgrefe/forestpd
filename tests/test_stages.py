"""
Unit tests for the cluster reduction and the MAF stage, plus the max-plus kernel.

These are the seams the architecture rests on, so they are tested on their own contracts rather
than only through the optimizer: a reducer must produce a genuine tree of clusters whose cut-edges
induce the same split in every input tree, and a MAF computer must produce a genuine agreement
forest.
"""

from __future__ import annotations

import numpy as np
import pytest

from forestpd.algo.convolve import fold_max_plus, max_plus
from forestpd.algo import local_solver
from forestpd.algo.solver import solve
from forestpd.cluster import cluster_reduction, common, single_cluster
from forestpd.maf.partition import is_agreement_forest
from .fixtures.reference_pd import pd
from forestpd.core.tree import PhyloTree, leaf_masks_below
from forestpd.exceptions import ForestPDInputError, ForestPDValueError
from forestpd.maf import HeuristicMAF

from .fixtures.baselines import TrivialMAF
from .fixtures.random_instances import (
    balanced,
    caterpillar,
    clade_localized_instance,
    perturbed_tree,
    random_tree,
    star,
)


def common_splits(trees):
    """The non-trivial common splits as leaf bitmasks, each the side without leaf 0."""
    index = common.leaf_numbering(trees[0])
    if trees[0].n_leaves < 4:
        return []
    intervals = common._common_cuts(common._split_intervals(tree, index) for tree in trees)
    leaf_at = {int(position): leaf for leaf, position in enumerate(index)}
    return sorted(sum(1 << leaf_at[p] for p in range(lo, hi + 1)) for lo, hi in intervals)


def four_taxon_tree():
    """``((0,1),(2,3))`` with unit branches."""
    return PhyloTree.from_edges(
        6, [(0, 4), (1, 4), (4, 5), (2, 5), (3, 5)], [1.0] * 5, {i: i for i in range(4)}
    )


class TestCommonSplits:
    def test_identical_trees_share_every_non_trivial_split(self, rng):
        tree = random_tree(10, rng)
        rooted = tree.root_at(0)
        n = tree.n_leaves
        expected = {
            mask for mask in leaf_masks_below(rooted) if 2 <= int(mask).bit_count() <= n - 2
        }
        assert set(common_splits([tree, tree])) == expected

    def test_conflicting_trees_share_nothing(self):
        first = four_taxon_tree()
        second = PhyloTree.from_edges(
            6, [(0, 4), (2, 4), (4, 5), (1, 5), (3, 5)], [1.0] * 5, {i: i for i in range(4)}
        )
        assert common_splits([first, second]) == []

    def test_tiny_trees_have_no_non_trivial_splits(self, rng):
        assert common_splits([random_tree(3, rng)] * 2) == []

    def test_a_star_has_no_non_trivial_splits(self):
        assert common_splits([star(6), star(6)]) == []


class TestClusterTree:
    def test_single_cluster_holds_the_whole_instance(self, rng):
        trees = [random_tree(6, rng) for _ in range(2)]
        ktree = single_cluster(trees)
        assert ktree.n_clusters == 1
        assert ktree.pieces[0].star_leaf == -1
        assert ktree.pieces[0].trees[0].n_leaves == 6

    def test_a_shared_clade_splits_the_instance(self):
        ktree = cluster_reduction([four_taxon_tree(), four_taxon_tree()], weights=[1.0, 1.0])
        assert ktree.n_clusters == 2
        # Exactly one cluster is the root; the other hangs below it with a star leaf.
        assert sum(piece.star_leaf < 0 for piece in ktree.pieces) == 1

    def test_pieces_partition_the_taxa(self, rng):
        """Every taxon belongs to exactly one cluster, and only pseudo-leaves are unlabelled."""
        base = random_tree(12, rng)
        trees = [base, perturbed_tree(base, rng, moves=1)]
        ktree = cluster_reduction(trees, weights=[1.0] * len(trees))
        owned: list[int] = []
        for piece in ktree.pieces:
            owned.extend(t for t in piece.taxon_of_leaf if t >= 0)
        assert sorted(owned) == list(range(12))

    def test_every_tree_yields_the_same_local_leaf_set(self, rng):
        """The local trees of one cluster must agree on their leaf numbering, or nothing lines up."""
        base = random_tree(14, rng)
        trees = [base] + [perturbed_tree(base, rng, moves=1) for _ in range(2)]
        ktree = cluster_reduction(trees, weights=[1.0] * len(trees))
        for piece in ktree.pieces:
            sizes = {tree.n_leaves for tree in piece.trees}
            assert sizes == {piece.n_leaves}

    def test_cluster_tree_is_a_tree(self, rng):
        base = random_tree(16, rng)
        trees = [base] + [perturbed_tree(base, rng, moves=2) for _ in range(2)]
        ktree = cluster_reduction(trees, weights=[1.0] * len(trees))
        assert sum(1 for p in ktree.parent if p == -1) == 1
        assert len(ktree.postorder) == ktree.n_clusters
        seen: set[int] = set()
        for cluster in ktree.postorder:  # children strictly before parents
            for child in ktree.children[cluster]:
                assert child in seen
            seen.add(cluster)

    def test_star_leaf_has_zero_weight(self, rng):
        """
        The upward cut-edge is charged in the parent's local instance, so ``x_C^*`` must be free.

        Charging it here would double-count that branch.
        """
        base = random_tree(12, rng)
        ktree = cluster_reduction([base, perturbed_tree(base, rng, 1)], weights=[1.0, 1.0])
        for piece in ktree.pieces:
            if piece.star_leaf < 0:
                continue
            for tree in piece.trees:
                pendant = next(
                    e
                    for e in range(tree.n_edges)
                    if piece.star_leaf in (int(tree.edges[e, 0]), int(tree.edges[e, 1]))
                )
                assert tree.weight[pendant] == 0.0

    def test_local_weight_is_conserved(self, rng):
        """
        Every branch of the input is charged exactly once across the decomposition.

        Child cut-edges live in the parent's piece, star edges are free, so summing all (merged)
        local trees must reproduce the weighted total of the input.
        """
        base = random_tree(14, rng)
        trees = [base, perturbed_tree(base, rng, 2)]
        weights = [0.5, 2.0]
        ktree = cluster_reduction(trees, weights=weights)
        total = sum(tree.total_weight() for piece in ktree.pieces for tree in piece.trees)
        assert total == pytest.approx(sum(w * t.total_weight() for w, t in zip(weights, trees)))


def reweighted(tree, rng):
    """The same topology with fresh random branch lengths."""
    weight = np.array([rng.uniform(0.0, 5.0) for _ in range(tree.n_edges)])
    return PhyloTree(tree.n_nodes, tree.n_leaves, tree.edges.copy(), weight)


def contract_random_edges(tree, rng, n):
    """``tree`` with up to ``n`` random internal edges contracted: a multifurcating tree."""
    edges = [tuple(map(int, e)) for e in tree.edges]
    weight = [float(x) for x in tree.weight]
    for _ in range(n):
        internal = [
            i for i, (u, v) in enumerate(edges) if u >= tree.n_leaves and v >= tree.n_leaves
        ]
        if not internal:
            break
        i = rng.choice(internal)
        keep, drop = edges[i]
        del edges[i], weight[i]
        edges = [(keep if u == drop else u, keep if v == drop else v) for u, v in edges]
    nodes = sorted({x for e in edges for x in e})
    relabel = {v: i for i, v in enumerate(nodes)}
    return PhyloTree.from_edges(
        len(nodes),
        [(relabel[u], relabel[v]) for u, v in edges],
        weight,
        {relabel[v]: v for v in range(tree.n_leaves)},
    )


def assert_same_cluster_tree(a, b):
    """Two cluster trees are identical: shape, local trees (in order), edges and weights."""
    assert (a.parent, a.children, a.postorder) == (b.parent, b.children, b.postorder)
    for pa, pb in zip(a.pieces, b.pieces):
        assert (pa.taxon_of_leaf, pa.child_of_leaf, pa.star_leaf) == (
            pb.taxon_of_leaf,
            pb.child_of_leaf,
            pb.star_leaf,
        )
        assert pa.tree_weights == pb.tree_weights
        assert len(pa.trees) == len(pb.trees)
        for ta, tb in zip(pa.trees, pb.trees):
            assert (ta.n_nodes, ta.n_leaves) == (tb.n_nodes, tb.n_leaves)
            np.testing.assert_array_equal(ta.edges, tb.edges)
            np.testing.assert_allclose(ta.weight, tb.weight, rtol=1e-12)


class TestCompiledReduction:
    """The compiled merged reduction builds exactly the cluster tree of the Python path."""

    @pytest.fixture
    def python_path(self, monkeypatch):
        def run(trees, weights):
            monkeypatch.setattr(common, "USE_COMPILED", False)
            try:
                return cluster_reduction(trees, weights=weights)
            finally:
                monkeypatch.setattr(common, "USE_COMPILED", True)

        return run

    @pytest.mark.skipif(not common._kernels.HAVE_NUMBA, reason="Numba not installed")
    @pytest.mark.parametrize("trial", range(25))
    def test_same_cluster_tree_as_python(self, rng, python_path, trial):
        for _ in range(trial):
            rng.random()
        base = random_tree(rng.randint(4, 16), rng)
        shapes = [perturbed_tree(base, rng, rng.randint(0, 3)) for _ in range(3)]
        if trial % 3 == 0:  # multifurcating inputs: local trees of unequal size
            shapes = [contract_random_edges(s, rng, rng.randint(1, 3)) for s in shapes]
        trees = [reweighted(rng.choice(shapes), rng) for _ in range(rng.randint(1, 9))]
        weights = [rng.uniform(0.1, 2.0) for _ in trees]
        assert_same_cluster_tree(
            cluster_reduction(trees, weights=weights), python_path(trees, weights)
        )

    @pytest.mark.skipif(not common._kernels.HAVE_NUMBA, reason="Numba not installed")
    def test_stars_and_tiny_trees(self, rng, python_path):
        for trees in ([star(6), star(6)], [random_tree(3, rng)] * 2, [caterpillar(5), balanced(4)]):
            if len({t.n_leaves for t in trees}) == 1:
                weights = [1.0] * len(trees)
                assert_same_cluster_tree(
                    cluster_reduction(trees, weights=weights), python_path(trees, weights)
                )

    def test_degree_two_vertex_is_rejected(self):
        """Two edges would share a split; such trees must be suppressed before the reduction."""
        from forestpd.core.tree import suppress_degree_two
        from forestpd.exceptions import ForestPDInputError

        # A path 0 - 4 - 5 - (1, 2, 3): node 4 has degree 2.
        tree = PhyloTree.from_edges(
            6, [(0, 4), (4, 5), (1, 5), (2, 5), (3, 5)], [1.0] * 5, {i: i for i in range(4)}
        )
        with pytest.raises(ForestPDInputError, match="degree 2"):
            cluster_reduction([tree, tree], weights=[1.0, 1.0])
        fixed = suppress_degree_two(tree)
        ktree = cluster_reduction([fixed, fixed], weights=[1.0, 1.0])
        assert sum(t.total_weight() for p in ktree.pieces for t in p.trees) == pytest.approx(10.0)


class TestMergedPieces:
    """``cluster_reduction(trees, weights)`` merges equal local topologies without changing PD."""

    def test_equal_topologies_collapse_to_one_local_tree(self, rng):
        base = random_tree(12, rng)
        trees = [reweighted(base, rng) for _ in range(5)]
        weights = [rng.uniform(0.1, 2.0) for _ in trees]
        ktree = cluster_reduction(trees, weights=weights)
        assert all(len(piece.trees) == 1 for piece in ktree.pieces)
        merged_total = sum(piece.trees[0].total_weight() for piece in ktree.pieces)
        expected = sum(w * tree.total_weight() for w, tree in zip(weights, trees))
        assert merged_total == pytest.approx(expected)

    @pytest.mark.parametrize("trial", range(15))
    def test_merged_frontier_equals_one_unmerged_cluster(self, rng, trial):
        """Against the whole instance as one cluster, whose local trees are the unmerged input."""
        for _ in range(trial):
            rng.random()
        base = random_tree(rng.randint(6, 14), rng)
        # Repeated perturbations of a few shapes, each with its own lengths: many local
        # topologies recur across trees with different branch lengths.
        shapes = [perturbed_tree(base, rng, rng.randint(0, 2)) for _ in range(3)]
        trees = [reweighted(rng.choice(shapes), rng) for _ in range(rng.randint(2, 8))]
        weights = np.array([rng.uniform(0.1, 2.0) for _ in trees])
        costs = np.array([rng.randint(1, 3) for _ in range(base.n_leaves)], dtype=np.int64)
        budget = rng.randint(2, base.n_leaves)
        plain = solve(single_cluster(trees), HeuristicMAF(), costs, weights, budget)
        merged = solve(
            cluster_reduction(trees, weights=weights), HeuristicMAF(), costs, weights, budget
        )
        np.testing.assert_allclose(merged.meta["frontier"], plain.meta["frontier"], rtol=1e-12)


class TestMAF:
    @pytest.mark.parametrize("computer", [HeuristicMAF(), TrivialMAF()])
    @pytest.mark.parametrize("trial", range(12))
    def test_output_is_a_valid_agreement_forest(self, rng, computer, trial):
        for _ in range(trial):
            rng.random()
        trees = [random_tree(rng.randint(4, 10), rng) for _ in range(rng.randint(2, 4))]
        n = trees[0].n_leaves
        trees = [random_tree(n, rng) for _ in trees]
        forest = computer.compute(trees)
        assert sorted(leaf for block in forest for leaf in block) == list(range(n))
        comp_of_leaf = np.empty(n, dtype=np.int64)
        for index, block in enumerate(forest):
            for leaf in block:
                comp_of_leaf[leaf] = index
        assert is_agreement_forest(trees, comp_of_leaf, len(forest))

    def test_identical_trees_give_one_component(self, rng):
        tree = random_tree(9, rng)
        assert HeuristicMAF().compute([tree, tree, tree]) == (tuple(range(9)),)

    def test_a_single_tree_gives_one_component(self, rng):
        tree = random_tree(7, rng)
        assert HeuristicMAF().compute([tree]) == (tuple(range(7)),)

    def test_polytomies_are_handled(self):
        forest = HeuristicMAF().compute([star(6), caterpillar(6)])
        n = 6
        comp_of_leaf = np.empty(n, dtype=np.int64)
        for index, block in enumerate(forest):
            for leaf in block:
                comp_of_leaf[leaf] = index
        assert is_agreement_forest([star(6), caterpillar(6)], comp_of_leaf, len(forest))

    def test_one_displaced_taxon_costs_few_components(self, rng):
        """
        A single SPR move should leave a small forest — that is the whole point of the parameter.

        Not a minimality claim (the heuristic makes none), just a guard against it degenerating to
        singletons on near-identical input.
        """
        base = random_tree(14, rng)
        moved = perturbed_tree(base, rng, moves=1)
        assert len(HeuristicMAF().compute([base, moved])) <= 5

    def test_maximally_incongruent_trees_still_validate(self, rng):
        trees = [random_tree(8, rng) for _ in range(4)]
        forest = HeuristicMAF().compute(trees)
        comp_of_leaf = np.empty(8, dtype=np.int64)
        for index, block in enumerate(forest):
            for leaf in block:
                comp_of_leaf[leaf] = index
        assert is_agreement_forest(trees, comp_of_leaf, len(forest))

    def test_two_leaves(self, rng):
        trees = [random_tree(2, rng), random_tree(2, rng)]
        assert HeuristicMAF().compute(trees) == ((0, 1),)


class TestLookahead:
    @pytest.mark.parametrize("lookahead", [1, 4, 32, None])
    def test_every_setting_gives_an_agreement_forest(self, rng, lookahead):
        for _ in range(10):
            base = random_tree(15, rng)
            trees = [base] + [perturbed_tree(base, rng, 3) for _ in range(3)]
            forest = HeuristicMAF(lookahead=lookahead).compute(trees)
            comp = np.empty(15, dtype=np.int32)
            for i, block in enumerate(forest):
                comp[list(block)] = i
            assert is_agreement_forest(trees, comp, len(forest))

    def test_trying_every_block_is_not_worse(self, rng):
        """On these instances, full lookahead never gives a larger forest than lookahead 1."""
        for _ in range(10):
            base = random_tree(20, rng)
            trees = [base] + [perturbed_tree(base, rng, 3) for _ in range(3)]
            full = len(HeuristicMAF(lookahead=None).compute(trees))
            assert full <= len(HeuristicMAF(lookahead=1).compute(trees))

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True])
    def test_invalid_lookahead_is_rejected(self, bad):
        with pytest.raises(ForestPDValueError, match="lookahead"):
            HeuristicMAF(lookahead=bad)


class TestMaxPlus:
    def test_matches_a_direct_maximum(self, rng):
        a = np.array([rng.choice([-np.inf, rng.uniform(0, 5)]) for _ in range(9)])
        b = np.array([rng.choice([-np.inf, rng.uniform(0, 5)]) for _ in range(9)])
        got = max_plus(a, b, 12)
        for i in range(12):
            candidates = [a[p] + b[i - p] for p in range(len(a)) if 0 <= i - p < len(b)]
            want = max(candidates) if candidates else -np.inf
            assert got[i] == pytest.approx(want) or (got[i] == -np.inf and want == -np.inf)

    def test_infeasible_stays_infeasible(self):
        a = np.full(4, -np.inf)
        assert np.all(max_plus(a, np.zeros(4), 5) == -np.inf)

    def test_fold_requires_every_table_to_be_funded(self):
        """Each table is ``-inf`` at zero, so a fold of ``m`` tables needs at least ``m`` budget."""
        table = np.array([-np.inf, 1.0, 2.0, 3.0])
        folded = fold_max_plus([table, table], 3)
        assert folded[0] == -np.inf
        assert folded[1] == -np.inf
        assert folded[2] == pytest.approx(2.0)
        assert folded[3] == pytest.approx(3.0)

    def test_empty_fold_is_the_zero_allocation(self):
        folded = fold_max_plus([], 3)
        assert folded[0] == 0.0
        assert np.all(folded[1:] == -np.inf)


def test_balanced_tree_pd_sanity():
    """A quick anchor on the fixtures themselves."""
    tree = balanced(4)
    assert pd(tree, 0b1111) == pytest.approx(tree.total_weight())


class TestCladeLocalizedInstances:
    """The instance family where the cluster reduction genuinely pays."""

    @pytest.mark.parametrize("n_clades", [2, 3, 4])
    def test_fixture_builds_valid_trees(self, rng, n_clades):
        """Every clade count gives proper trees (no extra edge, no degree-2 vertex)."""
        instance, _ = clade_localized_instance(rng, n_clades=n_clades, clade_size=3, n_trees=3)
        for tree in instance.trees:
            assert tree.n_edges == tree.n_nodes - 1
            assert all(int(d) != 2 for d in tree.degrees()[tree.n_leaves :])

    def test_backbone_splits_are_common(self, rng):
        """Trees differing inside one clade must still share every backbone split."""

        instance, _ = clade_localized_instance(rng, n_clades=4, clade_size=5, n_trees=4, budget=8)
        ktree = cluster_reduction(instance.trees, weights=instance.weights)
        assert ktree.n_clusters >= 4  # at least one cluster per clade

    def test_reduction_shrinks_the_parameter(self, rng):
        """
        Local MAF sizes stay far below the whole instance's.

        Without the reduction this instance needs a hopeless number of signatures; with it, each
        cluster only has to resolve one clade's disagreement.
        """

        instance, _ = clade_localized_instance(rng, n_clades=5, clade_size=8, n_trees=5, budget=10)
        maf = HeuristicMAF()
        local_k = max(
            len(maf.compute(piece.trees))
            for piece in cluster_reduction(instance.trees, instance.weights).pieces
        )
        global_k = len(maf.compute(single_cluster(instance.trees).pieces[0].trees))
        assert local_k < global_k


class TestCompiledSignatureLoop:
    """The compiled signature loop fills the same cluster tables as the Python loop."""

    @pytest.mark.skipif(not local_solver._signature_kernel.HAVE_NUMBA, reason="Numba not installed")
    @pytest.mark.parametrize("maf", [HeuristicMAF(), TrivialMAF()], ids=["heuristic", "singleton"])
    @pytest.mark.parametrize("trial", range(15))
    def test_same_tables_as_python(self, rng, monkeypatch, maf, trial):
        for _ in range(trial):
            rng.random()
        n = rng.randint(4, 9)
        base = random_tree(n, rng)
        trees = [perturbed_tree(base, rng, rng.randint(0, 3)) for _ in range(rng.randint(1, 4))]
        weights = np.array([rng.uniform(0.25, 2.0) for _ in trees])
        costs = np.array([rng.randint(1, 3) for _ in range(n)], dtype=np.int64)
        budget = rng.randint(1, int(costs.sum()))
        ktree = cluster_reduction(trees, weights=weights)

        compiled = solve(ktree, maf, costs, weights, budget, witness_budgets=[budget])
        monkeypatch.setattr(local_solver, "USE_COMPILED", False)
        python = solve(ktree, maf, costs, weights, budget, witness_budgets=[budget])

        np.testing.assert_allclose(compiled.meta["frontier"], python.meta["frontier"], rtol=1e-12)
        assert compiled.value == pytest.approx(python.value, rel=1e-12)
        assert compiled.meta["n_signatures"] == python.meta["n_signatures"]


class TestPlainProfileFromPlus:
    """The plain profile derived from the plus profile equals the one built from the trees."""

    @staticmethod
    def aggregated(columns, weight):
        out = {}
        for key, w in zip(zip(*[c.tolist() for c in columns]), weight.tolist()):
            out[key] = out.get(key, 0.0) + w
        return out

    @pytest.mark.parametrize("trial", range(15))
    def test_matches_direct_build(self, rng, trial):
        from forestpd.algo.local import LocalInstance
        from forestpd.algo.profile import build_profile, plain_profile_from_plus
        from forestpd.core.tree import drop_leaf

        for _ in range(trial):
            rng.random()
        base = random_tree(rng.randint(6, 14), rng)
        shapes = [perturbed_tree(base, rng, rng.randint(0, 2)) for _ in range(3)]
        if trial % 3 == 0:
            shapes = [contract_random_edges(s, rng, rng.randint(1, 2)) for s in shapes]
        trees = [reweighted(rng.choice(shapes), rng) for _ in range(rng.randint(1, 6))]
        weights = np.array([rng.uniform(0.2, 2.0) for _ in trees])
        checked = 0
        for piece in cluster_reduction(trees, weights=weights).pieces:
            if piece.star_leaf < 0:
                continue
            n = piece.n_leaves
            tw = np.asarray(piece.tree_weights)
            plus = LocalInstance(piece.trees, tw, np.zeros((n, 4)), np.ones(n, np.int64), 3)
            plain_trees = tuple(drop_leaf(t, piece.star_leaf) for t in piece.trees)
            plain = LocalInstance(
                plain_trees, tw, np.zeros((n - 1, 4)), np.ones(n - 1, np.int64), 3
            )
            forest = HeuristicMAF().compute(plain_trees)
            direct = build_profile(plain, forest)
            derived = plain_profile_from_plus(
                build_profile(plus, forest + ((piece.star_leaf,),)), len(forest)
            )
            assert derived.n_components == direct.n_components
            np.testing.assert_array_equal(derived.node_offset, direct.node_offset)
            for name in ("edge", "ce"):
                cols = (
                    ("edge_only_below", "edge_only_above")
                    if name == "edge"
                    else ("ce_slot", "ce_own_bit", "ce_only_down", "ce_only_up")
                )
                a = self.aggregated(
                    [getattr(derived, c) for c in cols], getattr(derived, f"{name}_weight")
                )
                b = self.aggregated(
                    [getattr(direct, c) for c in cols], getattr(direct, f"{name}_weight")
                )
                assert a.keys() == b.keys()
                for key in a:
                    assert a[key] == pytest.approx(b[key], rel=1e-12, abs=1e-12)
            checked += 1
        assert checked or len(cluster_reduction(trees, [1.0] * len(trees)).pieces) == 1


class TestProfileFromSplits:
    """Profiles built from the clusters' split tables equal those built from their trees."""

    @staticmethod
    def rows(profile):
        """The profile's rows aggregated by key, edge pairs unordered, zero rows dropped."""
        edges: dict = {}
        for lo, hi, w in zip(
            profile.edge_only_below.tolist(),
            profile.edge_only_above.tolist(),
            profile.edge_weight.tolist(),
        ):
            key = (min(lo, hi), max(lo, hi))
            edges[key] = edges.get(key, 0.0) + w
        comp: dict = {}
        columns = zip(
            profile.ce_slot.tolist(),
            profile.ce_own_bit.tolist(),
            profile.ce_only_down.tolist(),
            profile.ce_only_up.tolist(),
            profile.ce_weight.tolist(),
        )
        for *key, w in columns:
            comp[tuple(key)] = comp.get(tuple(key), 0.0) + w
        drop = lambda d: {k: v for k, v in d.items() if abs(v) > 1e-12}  # noqa: E731
        return drop(edges), drop(comp)

    @staticmethod
    def assert_same_rows(a, b):
        assert a.keys() == b.keys()
        for key in a:
            assert a[key] == pytest.approx(b[key], rel=1e-12, abs=1e-12)

    @pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "python"])
    @pytest.mark.parametrize("trial", range(15))
    def test_matches_tree_profile(self, rng, monkeypatch, compiled, trial):
        from forestpd.algo.local import LocalInstance
        from forestpd.algo.profile import build_profile, build_profile_from_splits
        from forestpd.core.tree import drop_leaf

        monkeypatch.setattr(common, "USE_COMPILED", compiled)
        for _ in range(trial):
            rng.random()
        base = random_tree(rng.randint(6, 16), rng)
        shapes = [perturbed_tree(base, rng, rng.randint(0, 3)) for _ in range(3)]
        if trial % 3 == 0:  # multifurcating inputs
            shapes = [contract_random_edges(s, rng, rng.randint(1, 2)) for s in shapes]
        trees = [reweighted(rng.choice(shapes), rng) for _ in range(rng.randint(1, 7))]
        weights = np.array([rng.uniform(0.2, 2.0) for _ in trees])
        for piece in cluster_reduction(trees, weights=weights).pieces:
            assert piece.split_masks is not None
            n = piece.n_leaves
            tw = np.asarray(piece.tree_weights)
            star = piece.star_leaf
            plain_trees = (
                piece.trees if star < 0 else tuple(drop_leaf(t, star) for t in piece.trees)
            )
            forest = HeuristicMAF().compute(plain_trees)
            if star >= 0:
                forest = forest + ((star,),)
            local = LocalInstance(
                piece.trees,
                tw,
                np.zeros((n, 4)),
                np.ones(n, np.int64),
                3,
            )
            direct = build_profile(local, forest)
            split = build_profile_from_splits(n, piece.split_masks, piece.split_weights, forest)
            np.testing.assert_array_equal(split.node_offset, direct.node_offset)
            for a, b in zip(split.comp_trees, direct.comp_trees):
                for name in ("left", "right", "leaf", "postorder"):
                    np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
            for a, b in zip(self.rows(split), self.rows(direct)):
                self.assert_same_rows(a, b)

    @pytest.mark.skipif(not common._kernels.HAVE_NUMBA, reason="Numba not installed")
    @pytest.mark.parametrize("trial", range(10))
    def test_same_split_tables_on_both_reduction_paths(self, rng, monkeypatch, trial):
        for _ in range(trial):
            rng.random()
        base = random_tree(rng.randint(4, 16), rng)
        shapes = [perturbed_tree(base, rng, rng.randint(0, 3)) for _ in range(3)]
        if trial % 3 == 0:
            shapes = [contract_random_edges(s, rng, rng.randint(1, 3)) for s in shapes]
        trees = [reweighted(rng.choice(shapes), rng) for _ in range(rng.randint(1, 9))]
        weights = [rng.uniform(0.1, 2.0) for _ in trees]
        compiled = cluster_reduction(trees, weights=weights)
        monkeypatch.setattr(common, "USE_COMPILED", False)
        python = cluster_reduction(trees, weights=weights)
        for a, b in zip(compiled.pieces, python.pieces):
            np.testing.assert_array_equal(a.split_masks, b.split_masks)
            np.testing.assert_allclose(a.split_weights, b.split_weights, rtol=1e-12)

    def test_split_weight_is_conserved(self, rng):
        base = random_tree(14, rng)
        trees = [reweighted(perturbed_tree(base, rng, 2), rng) for _ in range(6)]
        weights = [rng.uniform(0.1, 2.0) for _ in trees]
        ktree = cluster_reduction(trees, weights=weights)
        total = sum(float(piece.split_weights.sum()) for piece in ktree.pieces)
        assert total == pytest.approx(sum(w * t.total_weight() for w, t in zip(weights, trees)))

    def test_unmerged_single_cluster_has_no_split_table(self, rng):
        trees = [random_tree(8, rng) for _ in range(2)]
        assert single_cluster(trees).pieces[0].split_masks is None

    def test_crossing_forest_is_rejected(self):
        from forestpd.algo.profile import build_profile_from_splits
        from forestpd.exceptions import ForestPDInputError

        # ((0,1),(2,3)): the central split {0,1}|{2,3} cuts both {0,2} and {1,3}.
        masks = np.array([0b0011, 0b0001, 0b0010, 0b0100, 0b1000], dtype=np.uint64)
        with pytest.raises(ForestPDInputError):
            build_profile_from_splits(4, masks, np.ones(5), ((0, 2), (1, 3)))

    @pytest.mark.parametrize("trial", range(10))
    def test_trusted_and_untrusted_stages_agree(self, rng, trial):
        class Untrusted:
            def compute(self, trees):
                return HeuristicMAF().compute(trees)

        for _ in range(trial):
            rng.random()
        n = rng.randint(5, 12)
        base = random_tree(n, rng)
        trees = [reweighted(perturbed_tree(base, rng, rng.randint(0, 3)), rng) for _ in range(4)]
        weights = np.array([rng.uniform(0.2, 2.0) for _ in trees])
        costs = np.array([rng.randint(1, 3) for _ in range(n)], dtype=np.int64)
        budget = rng.randint(1, int(costs.sum()))
        ktree = cluster_reduction(trees, weights=weights)
        trusted = solve(ktree, HeuristicMAF(), costs, weights, budget, witness_budgets=[budget])
        untrusted = solve(ktree, Untrusted(), costs, weights, budget, witness_budgets=[budget])
        np.testing.assert_allclose(trusted.meta["frontier"], untrusted.meta["frontier"], rtol=1e-12)


class TestDropLeaf:
    """The vectorized ``drop_leaf`` gives exactly the tree of the general path."""

    @pytest.mark.parametrize("trial", range(20))
    def test_identical_to_general_path(self, rng, monkeypatch, trial):
        from forestpd.core import tree as tree_module

        for _ in range(trial):
            rng.random()
        base = random_tree(rng.randint(4, 16), rng)
        shapes = [perturbed_tree(base, rng, rng.randint(0, 3)) for _ in range(3)]
        if trial % 2 == 0:  # polytomies: the neighbour may keep degree >= 3
            shapes = [contract_random_edges(s, rng, rng.randint(1, 4)) for s in shapes]
        trees = [reweighted(rng.choice(shapes), rng) for _ in range(rng.randint(1, 6))]
        local = [
            t
            for piece in cluster_reduction(trees, weights=[1.0] * len(trees)).pieces
            if piece.star_leaf >= 0
            for t in piece.trees
        ] + [star(5), random_tree(2, rng), random_tree(3, rng)]
        fast = [tree_module.drop_leaf(t, t.n_leaves - 1) for t in local]
        monkeypatch.setattr(tree_module, "_drop_leaf_fast", lambda tree, leaf: None)
        general = [tree_module.drop_leaf(t, t.n_leaves - 1) for t in local]
        for a, b in zip(fast, general):
            assert (a.n_nodes, a.n_leaves) == (b.n_nodes, b.n_leaves)
            assert a.edges.dtype == b.edges.dtype and a.weight.dtype == b.weight.dtype
            np.testing.assert_array_equal(a.edges, b.edges)
            np.testing.assert_array_equal(a.weight, b.weight)


class TestCompiledProfiles:
    """The compiled split-based profiles equal the NumPy ones, plus and plain."""

    @pytest.mark.skipif(
        not __import__("forestpd.algo._profile_kernel").algo._profile_kernel.HAVE_NUMBA,
        reason="Numba not installed",
    )
    @pytest.mark.parametrize("trial", range(15))
    def test_same_profiles_as_numpy(self, rng, monkeypatch, trial):
        from forestpd.algo import profile as profile_module
        from forestpd.core.tree import drop_leaf

        for _ in range(trial):
            rng.random()
        base = random_tree(rng.randint(6, 16), rng)
        shapes = [perturbed_tree(base, rng, rng.randint(0, 3)) for _ in range(3)]
        if trial % 3 == 0:
            shapes = [contract_random_edges(s, rng, rng.randint(1, 2)) for s in shapes]
        trees = [reweighted(rng.choice(shapes), rng) for _ in range(rng.randint(1, 7))]
        weights = np.array([rng.uniform(0.2, 2.0) for _ in trees])
        rows = TestProfileFromSplits.rows
        for piece in cluster_reduction(trees, weights=weights).pieces:
            star = piece.star_leaf
            plain_trees = (
                piece.trees if star < 0 else tuple(drop_leaf(t, star) for t in piece.trees)
            )
            forest = HeuristicMAF().compute(plain_trees)
            star_component = None
            if star >= 0:
                star_component = len(forest)
                forest = forest + ((star,),)
            args = (piece.n_leaves, piece.split_masks, piece.split_weights, forest, star_component)
            monkeypatch.setattr(profile_module, "USE_COMPILED", True)
            compiled = profile_module.build_profiles_from_splits(*args)
            monkeypatch.setattr(profile_module, "USE_COMPILED", False)
            numpy = profile_module.build_profiles_from_splits(*args)
            for a, b in zip(compiled, numpy):
                if b is None:
                    assert a is None
                    continue
                assert a.n_components == b.n_components
                np.testing.assert_array_equal(a.node_offset, b.node_offset)
                for ta, tb in zip(a.comp_trees, b.comp_trees):
                    np.testing.assert_array_equal(ta.left, tb.left)
                    np.testing.assert_array_equal(ta.leaf, tb.leaf)
                for ra, rb in zip(rows(a), rows(b)):
                    TestProfileFromSplits.assert_same_rows(ra, rb)


class TestGroupingIsExact:
    """Grouping local trees by topology never relies on the hash alone."""

    @pytest.mark.skipif(not common._kernels.HAVE_NUMBA, reason="Numba not installed")
    @pytest.mark.parametrize("trial", range(8))
    def test_colliding_hashes_give_the_same_cluster_tree(self, rng, monkeypatch, trial):
        for _ in range(trial):
            rng.random()
        base = random_tree(rng.randint(6, 16), rng)
        shapes = [perturbed_tree(base, rng, rng.randint(0, 3)) for _ in range(4)]
        if trial % 2 == 0:
            shapes = [contract_random_edges(s, rng, rng.randint(1, 3)) for s in shapes]
        trees = [reweighted(rng.choice(shapes), rng) for _ in range(rng.randint(2, 9))]
        weights = [rng.uniform(0.1, 2.0) for _ in trees]
        expected = cluster_reduction(trees, weights=weights)
        real = common._kernels.segment_hashes
        monkeypatch.setattr(
            common._kernels,
            "segment_hashes",
            lambda masks, starts, lengths: np.zeros_like(real(masks, starts, lengths)),
        )
        assert_same_cluster_tree(cluster_reduction(trees, weights=weights), expected)


class TestMaxPlusConstantRuns:
    """The compiled max-plus with the constant-run shortcut is bit-identical to the double loop."""

    @staticmethod
    def table(rng, n):
        """A table with -inf entries, a long constant run, and possibly a different last entry."""
        values = [rng.choice([-np.inf, rng.uniform(-5, 5)]) for _ in range(n)]
        start = rng.randint(0, n - 1)
        constant = rng.choice([-np.inf, rng.uniform(-5, 5), 0.0])
        values[start:] = [constant] * (n - start)
        if rng.random() < 0.5:
            values[-1] = rng.choice([-np.inf, rng.uniform(-5, 5)])
        return np.array(values)

    @pytest.mark.skipif(not local_solver._signature_kernel.HAVE_NUMBA, reason="Numba not installed")
    @pytest.mark.parametrize("trial", range(300))
    def test_bit_identical(self, rng, trial):
        from forestpd.algo._signature_kernel import _max_plus_into

        for _ in range(trial):
            rng.random()
        n_a, n_b = rng.randint(1, 40), rng.randint(1, 40)
        a, b = self.table(rng, n_a), self.table(rng, n_b)
        n_out = rng.randint(1, n_a + n_b)
        start = np.array([rng.choice([-np.inf, rng.uniform(-5, 5)]) for _ in range(n_out)])
        expected = start.copy()
        for p in range(n_a):
            for q in range(n_b):
                if p + q < n_out and a[p] + b[q] > expected[p + q]:
                    expected[p + q] = a[p] + b[q]
        out = start.copy()
        _max_plus_into(a, b, out)
        np.testing.assert_array_equal(out, expected)


class TestProfileGuards:
    """Forests the profile builders must refuse, on every path."""

    @staticmethod
    def local(trees):
        from forestpd.algo.local import LocalInstance

        n = trees[0].n_leaves
        return LocalInstance(
            tuple(trees), np.ones(len(trees)), np.zeros((n, 2)), np.ones(n, np.int64), 1
        )

    def test_more_than_63_components_is_refused(self, rng):
        from forestpd.algo.profile import build_profile, build_profile_from_splits
        from forestpd.algo import profile as profile_module

        tree = random_tree(64, rng)
        singletons = tuple((leaf,) for leaf in range(64))
        with pytest.raises(ForestPDInputError, match="at most 63"):
            build_profile(self.local([tree]), singletons)
        masks = np.array([1, 2], dtype=np.uint64)
        with pytest.raises(ForestPDInputError, match="at most 63"):
            build_profile_from_splits(64, masks, np.ones(2), singletons)
        if profile_module._profile_kernel.HAVE_NUMBA:
            with pytest.raises(ForestPDInputError, match="at most 63"):
                profile_module.build_profiles_from_splits(64, masks, np.ones(2), singletons)

    def test_crossing_forest_is_refused_by_the_compiled_path(self):
        from forestpd.algo import profile as profile_module

        if not profile_module._profile_kernel.HAVE_NUMBA:
            pytest.skip("Numba not installed")
        masks = np.array([0b0011, 0b0101, 0b0001, 0b0010, 0b0100], dtype=np.uint64)
        with pytest.raises(ForestPDInputError, match="two MAF components"):
            profile_module.build_profiles_from_splits(4, masks, np.ones(5), ((0, 2), (1, 3)))

    def test_component_with_two_topologies_is_refused(self):
        """A component inducing different topologies in two trees is no agreement forest."""
        from forestpd.algo.profile import build_profile, build_profile_from_splits

        first = PhyloTree.from_edges(
            6, [(0, 4), (1, 4), (4, 5), (2, 5), (3, 5)], [1.0] * 5, {i: i for i in range(4)}
        )
        second = PhyloTree.from_edges(
            6, [(0, 4), (2, 4), (4, 5), (1, 5), (3, 5)], [1.0] * 5, {i: i for i in range(4)}
        )
        whole = ((0, 1, 2, 3),)
        with pytest.raises(ForestPDInputError):
            build_profile(self.local([first, second]), whole)
        masks = np.array([0b0011, 0b0101], dtype=np.uint64)  # the splits 01|23 and 02|13
        with pytest.raises(ForestPDInputError, match="different topologies"):
            build_profile_from_splits(4, masks, np.ones(2), whole)


class TestBudgetSplitting:
    def test_split_budget_edges(self):
        from forestpd.algo.local_solver import split_budget

        assert split_budget([], 0) == []
        assert split_budget([], 2) is None
        infeasible = np.array([-np.inf, -np.inf, -np.inf])
        assert split_budget([infeasible], 2) is None
        assert split_budget([np.array([-np.inf, 1.0, 3.0]), np.array([-np.inf, 2.0, 2.5])], 2) == [
            1,
            1,
        ]

    def test_max_plus_numpy_fallback(self, rng, monkeypatch):
        from forestpd.algo import convolve

        a = np.array([0.0, -np.inf, 2.0, 5.0])
        b = np.array([1.0, 4.0, -np.inf])
        compiled = convolve.max_plus(a, b, 6)
        monkeypatch.setattr(convolve, "HAVE_NUMBA", False)
        np.testing.assert_array_equal(convolve.max_plus(a, b, 6), compiled)
        assert convolve.max_plus(a, np.empty(0), 3).tolist() == [-np.inf] * 3
        assert convolve.max_plus(a, b, 0).size == 0


class TestHeuristicPaths:
    def test_one_tree_is_one_component(self, rng):
        assert HeuristicMAF().compute([random_tree(7, rng)]) == (tuple(range(7)),)

    def test_polytomy_with_extra_branches_merges_into_one(self):
        """Two leaves hanging off a vertex that keeps other branches (a polytomy): the merged
        block needs a fresh leaf, and identical trees still give a single component."""
        # 0 and 1 hang off node 6, which also carries node 7 (with 2, 3) and node 8 (with 4, 5).
        edges = [(0, 6), (1, 6), (6, 7), (2, 7), (3, 7), (6, 8), (4, 8), (5, 8)]
        tree = PhyloTree.from_edges(9, edges, [1.0] * len(edges), {i: i for i in range(6)})
        assert HeuristicMAF().compute([tree, tree]) == (tuple(range(6)),)
        other = PhyloTree.from_edges(
            9,
            [(0, 6), (1, 6), (6, 7), (2, 7), (4, 7), (6, 8), (3, 8), (5, 8)],
            [1.0] * 8,
            {i: i for i in range(6)},
        )
        forest = HeuristicMAF().compute([tree, other])
        comp_of_leaf = np.empty(6, dtype=np.int32)
        for index, block in enumerate(forest):
            comp_of_leaf[list(block)] = index
        assert is_agreement_forest([tree, other], comp_of_leaf, len(forest))
        assert (0, 1) in forest or any({0, 1} <= set(block) for block in forest)
