"""Unit tests for the core tree representation, PD, embeddings and partition masks."""

from __future__ import annotations

import numpy as np
import pytest

from forestpd.maf.embedding import component_edges, extract_embedding
from forestpd.maf.partition import (
    component_edge_masks,
    is_agreement_forest,
    split_component_of_edge,
)
from .fixtures.reference_pd import pd
from forestpd.core.taxa import TaxonRegistry
from forestpd.core.tree import PhyloTree, drop_leaf, leaf_masks_below, suppress_degree_two
from forestpd.exceptions import ForestPDInputError

from .fixtures.random_instances import balanced, caterpillar, random_tree, star


def cherry_plus_one():
    """``((0,1),2)`` with unit branches; leaf 0/1 cherry at node 3, node 4 joins leaf 2."""
    return PhyloTree.from_edges(
        5, [(0, 3), (1, 3), (3, 4), (2, 4)], [1.0, 1.0, 1.0, 1.0], {0: 0, 1: 1, 2: 2}
    )


class TestPhyloTree:
    def test_leaves_occupy_the_lowest_node_ids(self):
        """The package's central invariant: leaf id == node id for every leaf."""
        tree = PhyloTree.from_edges(
            5, [(4, 0), (4, 1), (4, 3), (3, 2)], [1.0] * 4, {0: 1, 1: 2, 2: 0}
        )
        assert tree.n_leaves == 3
        assert set(np.nonzero(tree.degrees() == 1)[0]) == {0, 1, 2}

    def test_rejects_a_disconnected_edge_list(self):
        # A doubled edge between 0 and 2 leaves {1, 3} stranded on its own component.
        with pytest.raises(ForestPDInputError, match="not connected"):
            PhyloTree.from_edges(4, [(0, 2), (2, 0), (1, 3)], [1.0] * 3, {0: 0, 1: 1})

    def test_rejects_the_wrong_edge_count(self):
        with pytest.raises(ForestPDInputError, match="needs"):
            PhyloTree.from_edges(4, [(0, 2), (1, 2)], [1.0] * 2, {0: 0, 1: 1})

    def test_rejects_non_contiguous_leaf_ids(self):
        with pytest.raises(ForestPDInputError, match="0..n_leaves-1"):
            PhyloTree.from_edges(3, [(0, 2), (1, 2)], [1.0] * 2, {0: 0, 1: 5})

    def test_rooting_gives_a_consistent_postorder(self, rng):
        tree = random_tree(12, rng)
        rooted = tree.root_at(0)
        seen = set()
        for v in reversed(rooted.order.tolist()):
            parent = int(rooted.parent[v])
            assert parent == -1 or parent not in seen
            seen.add(v)
        assert len(seen) == tree.n_nodes

    def test_entry_exit_stamps_give_ancestry(self, rng):
        tree = random_tree(10, rng)
        rooted = tree.root_at(0)
        for v in range(tree.n_nodes):
            ancestors, u = {v}, v
            while int(rooted.parent[u]) >= 0:
                u = int(rooted.parent[u])
                ancestors.add(u)
            for u in range(tree.n_nodes):
                stamped = rooted.tin[u] <= rooted.tin[v] and rooted.tout[v] <= rooted.tout[u]
                assert bool(stamped) == (u in ancestors)


class TestPD:
    def test_small_cases(self):
        tree = PhyloTree.from_edges(
            5, [(0, 4), (1, 4), (4, 3), (2, 3)], [1.0, 1.0, 2.0, 1.0], {0: 0, 1: 1, 2: 2}
        )
        assert pd(tree, 0b000) == 0.0
        assert pd(tree, 0b001) == 0.0  # a single taxon spans nothing
        assert pd(tree, 0b011) == 2.0  # the cherry
        assert pd(tree, 0b111) == 5.0  # everything

    def test_all_taxa_gives_the_total_weight(self, rng):
        tree = random_tree(9, rng)
        assert pd(tree, (1 << tree.n_leaves) - 1) == pytest.approx(tree.total_weight())

    def test_monotone_under_adding_taxa(self, rng):
        tree = random_tree(8, rng)
        previous = 0.0
        mask = 0
        for leaf in range(tree.n_leaves):
            mask |= 1 << leaf
            current = pd(tree, mask)
            assert current >= previous - 1e-12
            previous = current


class TestSuppressAndDrop:
    def test_suppress_merges_weights(self):
        # 0 -- 3 -- 4 -- 1 with a pendant 2 on node 4; node 3 has degree 2.
        tree = PhyloTree(
            5,
            3,
            np.array([[0, 3], [3, 4], [4, 1], [4, 2]], dtype=np.int32),
            np.array([1.0, 2.0, 3.0, 4.0]),
        )
        reduced = suppress_degree_two(tree)
        assert reduced.n_nodes == 4
        assert reduced.total_weight() == pytest.approx(tree.total_weight())

    def test_drop_leaf_preserves_pd_of_the_rest(self, rng):
        tree = random_tree(7, rng)
        reduced = drop_leaf(tree, tree.n_leaves - 1)
        assert reduced.n_leaves == tree.n_leaves - 1
        for _ in range(20):
            picked = {rng.randrange(reduced.n_leaves) for _ in range(3)}
            mask = sum(1 << leaf for leaf in picked)
            assert pd(reduced, mask) == pytest.approx(pd(tree, mask))

    def test_drop_leaf_refuses_other_leaves(self, rng):
        tree = random_tree(5, rng)
        with pytest.raises(ForestPDInputError, match="highest-numbered"):
            drop_leaf(tree, 0)


class TestEmbedding:
    def test_single_leaf_component(self):
        tree = cherry_plus_one()
        embedding = extract_embedding(tree, [2])
        assert embedding.n_nodes == 1
        assert embedding.edge_fnode == {}

    def test_subdivided_path_maps_to_one_edge(self):
        """Both pendant edges of the cherry subdivide the single edge of the two-leaf ``F``."""
        tree = cherry_plus_one()
        embedding = extract_embedding(tree, [0, 1])
        assert embedding.n_nodes == 2
        assert set(embedding.edge_fnode.values()) == {1}
        assert len(embedding.edge_fnode) == 2

    def test_shape_is_stable_across_equivalent_trees(self, rng):
        """The same topology in two different host trees must produce the same canonical shape."""
        base = random_tree(9, rng)
        leaves = [0, 1, 2, 3]
        first = extract_embedding(base, leaves).shape
        # Re-root by relabelling internal nodes: the same tree, a different node numbering.
        shifted = PhyloTree(
            base.n_nodes, base.n_leaves, base.edges[::-1].copy(), base.weight[::-1].copy()
        )
        assert extract_embedding(shifted, leaves).shape == first

    def test_component_edges_match_the_masks(self, rng):
        """The standalone edge finder agrees with the mask-derived owner array."""
        # A contiguous block plus singletons: an agreement forest of the tree with itself, which
        # is what makes the `below & above` decoding single-bit.
        tree = random_tree(10, rng)
        comp_of_leaf = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        rooted = tree.root_at(0)
        below, above = component_edge_masks(rooted, comp_of_leaf, 9)
        owner = split_component_of_edge(below, above)
        for comp in range(9):
            leaves = [int(x) for x in np.nonzero(comp_of_leaf == comp)[0]]
            expected = sorted(component_edges(tree, leaves))
            assert sorted(int(e) for e in np.nonzero(owner == comp)[0]) == expected


class TestPartition:
    def test_masks_are_complementary_on_leaf_partitions(self, rng):
        tree = random_tree(11, rng)
        comp_of_leaf = np.array([leaf % 4 for leaf in range(tree.n_leaves)])
        rooted = tree.root_at(0)
        below, above = component_edge_masks(rooted, comp_of_leaf, 4)
        masks = leaf_masks_below(rooted)
        full = (1 << tree.n_leaves) - 1
        for e in range(tree.n_edges):
            want_below = {int(comp_of_leaf[i]) for i in range(tree.n_leaves) if masks[e] >> i & 1}
            want_above = {
                int(comp_of_leaf[i]) for i in range(tree.n_leaves) if (full & ~masks[e]) >> i & 1
            }
            got_below = {b for b in range(4) if int(below[e]) >> b & 1}
            got_above = {b for b in range(4) if int(above[e]) >> b & 1}
            assert got_below == want_below
            assert got_above == want_above

    def test_singletons_are_always_an_agreement_forest(self, rng):
        trees = [random_tree(7, rng) for _ in range(3)]
        comp_of_leaf = np.arange(7)
        assert is_agreement_forest(trees, comp_of_leaf, 7)

    def test_identical_trees_agree_on_everything(self, rng):
        tree = random_tree(7, rng)
        assert is_agreement_forest([tree, tree], np.zeros(7, dtype=np.int64), 1)

    def test_conflicting_topologies_are_rejected(self):
        """``((0,1),(2,3))`` against ``((0,2),(1,3))``: one block of all four cannot agree."""
        first = balanced(4)
        second = PhyloTree.from_edges(
            6, [(0, 4), (2, 4), (4, 5), (1, 5), (3, 5)], [1.0] * 5, {i: i for i in range(4)}
        )
        assert not is_agreement_forest([first, second], np.zeros(4, dtype=np.int64), 1)

    def test_overlapping_components_are_rejected(self):
        """
        A caterpillar's ``{0,2}`` and ``{1,3}`` both run through the spine, so they share vertices.
        """
        tree = caterpillar(5)
        comp_of_leaf = np.array([0, 1, 0, 1, 2])
        assert not is_agreement_forest([tree, tree], comp_of_leaf, 3)

    def test_star_allows_one_multi_leaf_component(self):
        """
        Every embedded subtree of a star runs through the centre, so at most one may be non-trivial.

        ``{0,1}`` plus singletons is fine; a second pair would share the centre vertex.
        """
        tree = star(6)
        assert is_agreement_forest([tree, tree], np.array([0, 0, 1, 2, 3, 4]), 5)
        assert not is_agreement_forest([tree, tree], np.array([0, 0, 1, 1, 2, 2]), 3)

    def test_shared_vertex_is_rejected_even_with_spare_vertices(self):
        """
        ``(a1,a2,b1,b2,(c,d))`` with ``{a1,a2}``, ``{b1,b2}``, ``{c}``, ``{d}``.

        Both pairs pass through the centre. The vertex above ``c`` and ``d`` lies in no component,
        so a check that only counts edges and vertices would not notice the shared centre.
        """
        tree = PhyloTree.from_edges(
            8,
            [(0, 6), (1, 6), (2, 6), (3, 6), (6, 7), (4, 7), (5, 7)],
            [1.0] * 7,
            {i: i for i in range(6)},
        )
        assert not is_agreement_forest([tree, tree], np.array([0, 0, 1, 1, 2, 3]), 4)
        assert is_agreement_forest([tree, tree], np.array([0, 0, 1, 2, 3, 4]), 5)

    def test_solver_refuses_a_forest_with_a_shared_vertex(self):
        """A user-supplied ``maf=`` returning such a partition must be rejected, not solved."""
        from forestpd.algo.solver import solve
        from forestpd.cluster import single_cluster

        tree = PhyloTree.from_edges(
            8,
            [(0, 6), (1, 6), (2, 6), (3, 6), (6, 7), (4, 7), (5, 7)],
            [1.0] * 7,
            {i: i for i in range(6)},
        )

        class SharedVertexForest:
            def compute(self, trees):
                return ((0, 1), (2, 3), (4,), (5,))

        costs = np.ones(6, dtype=np.int64)
        weights = np.ones(2)
        with pytest.raises(ForestPDInputError, match="share a vertex"):
            solve(single_cluster([tree, tree]), SharedVertexForest(), costs, weights, 3)


class TestValidationAndSmallCases:
    """Constructor checks and degenerate trees of the core types."""

    def test_from_edges_rejects_mismatched_weights(self):
        with pytest.raises(ForestPDInputError, match="equal length"):
            PhyloTree.from_edges(3, [(0, 2), (1, 2)], [1.0], {0: 0, 1: 1})

    def test_from_leaf_arrays_checks(self):
        edges = np.array([[0, 2], [1, 2]], dtype=np.int32)
        ones = np.ones(2)
        with pytest.raises(ForestPDInputError, match="equal length"):
            PhyloTree.from_leaf_arrays(3, edges, np.ones(1), np.array([0, 1]), np.array([0, 1]))
        with pytest.raises(ForestPDInputError, match="0..n_leaves-1"):
            PhyloTree.from_leaf_arrays(3, edges, ones, np.array([0, 1]), np.array([0, 2]))
        with pytest.raises(ForestPDInputError, match="edges"):
            PhyloTree.from_leaf_arrays(4, edges, ones, np.array([0, 1]), np.array([0, 1]))

    def test_single_leaf_tree(self):
        tree = PhyloTree.from_edges(1, [], [], {0: 0})
        assert (tree.n_leaves, tree.n_edges, tree.total_weight()) == (1, 0, 0.0)
        assert pd(tree, 0b1) == 0.0

    def test_neighbours_and_repr(self):
        tree = PhyloTree.from_edges(3, [(0, 2), (1, 2)], [1.0, 2.0], {0: 0, 1: 1})
        assert sorted(tree.neighbours(2)) == [(0, 0), (1, 1)]
        assert repr(tree) == "PhyloTree(n_leaves=2, n_nodes=3)"

    def test_registry(self):
        registry = TaxonRegistry(["B", "A", "C"])
        assert len(registry) == 3 and repr(registry) == "TaxonRegistry(n=3)"
        assert registry.id_array([]).tolist() == []
        assert registry.id_array(["B"]).tolist() == [1]
        with pytest.raises(ForestPDInputError):
            registry.id_array(["A", "Z"])
        with pytest.raises(ForestPDInputError):
            registry.ids_of({"Z"})
