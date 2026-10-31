"""
Survival probabilities: the optimizer maximizes the expected PD of the surviving taxa.

Every unprotected taxon ``x`` survives with probability ``a(x)``, protected ones surely, and an edge
counts when both of its sides keep a survivor. The oracle is the brute force over every affordable
set, evaluated with :func:`fixtures.reference_pd.expected_pd`, which is itself checked here against the
per-edge definition. Beyond the end-to-end comparison, the pieces the extension adds are tested on
their own contracts: the per-signature decomposition of the expected PD into ``d_S`` and the
two-sided weights, the cluster sums (empty set, singletons, ``c_down``/``c_up``), the equality of
the split and tree paths, and every compiled kernel against its NumPy/Python twin.

Without survival probabilities nothing may change; the rest of the suite checks that, since it runs
the same code with every probability zero.
"""

from __future__ import annotations

import math
import random
import re

import numpy as np
import pytest

import forestpd as fpd
from forestpd import MaxMPDInstance, optimize_maxmpd
from forestpd.algo import local_solver
from forestpd.algo import profile as profile_module
from forestpd.algo import survival as survival_module
from forestpd.algo.local import LocalInstance
from forestpd.algo.local_solver import signature_weights, solve_local
from forestpd.algo.profile import build_profile, build_profiles_from_splits
from forestpd.algo.solver import solve
from forestpd.algo.survival import cluster_sides, leaf_logs, subtree_logs
from forestpd.cluster import cluster_reduction, common, single_cluster
from .fixtures.reference_pd import expected_pd, pd
from forestpd.core.tree import leaf_masks_below
from forestpd.exceptions import ForestPDInputError
from forestpd.maf import HeuristicMAF
from forestpd.maf.maffe import rooted_newick

from .fixtures.baselines import optimize
from .fixtures.brute_force import brute_force_optimize_pd
from .fixtures.random_instances import (
    Case,
    caterpillar,
    clade_localized_instance,
    make_instance,
    perturbed_tree,
    random_instance,
    random_survival,
    random_tree,
    related_instance,
    star,
    with_survival,
)
from .fixtures.reference_dp import two_sided_pd
from .test_rooted import random_rooted, rooted_pd

CONFIGS = [
    (reduction, maf)
    for reduction in ("common", "trivial")
    for maf in ("heuristic", "singleton", "untrusted")
]
CONFIG_IDS = [f"{r}+{m}" for r, m in CONFIGS]


def close(a: float, b: float) -> bool:
    """Equal up to a relative tolerance of 1e-9 (absolute near zero)."""
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def assert_matches_oracle(case: Case, reduction: str = "common", maf: str = "heuristic") -> None:
    """Value equals brute force, and the witness is affordable and achieves it."""
    instance, budget = case
    expected, _ = brute_force_optimize_pd(instance, budget)
    result = optimize(case, reduction, maf, return_taxa=True)
    assert close(result.value, expected), f"{reduction}/{maf}: {result.value} != {expected}"
    cost = sum(int(instance.costs[instance.registry.id_of(x)]) for x in result.taxa)
    assert cost <= budget
    assert close(instance.evaluate_pd(result.taxa), expected)


def surviving(case: Case, rng: random.Random) -> Case:
    """``case`` with random survival probabilities."""
    return with_survival(case, random_survival(rng, case.instance.registry.n))


def log_loss(instance: MaxMPDInstance) -> np.ndarray:
    return np.log1p(-instance.survival)


# ---------------------------------------------------------------------------------------------
# The reference evaluation


class TestExpectedPD:
    @staticmethod
    def by_definition(tree, mask: int, survival: list[float]) -> float:
        """Per edge: weight times, for each side, 1 if protected else P(some taxon survives)."""
        below = leaf_masks_below(tree.root_at(0))
        full = (1 << tree.n_leaves) - 1

        def keeps(side: int) -> float:
            if side & mask:
                return 1.0
            lost = 1.0
            for leaf in range(tree.n_leaves):
                if (side >> leaf) & 1:
                    lost *= 1.0 - survival[leaf]
            return 1.0 - lost

        return sum(
            float(tree.weight[e]) * keeps(below[e]) * keeps(full & ~below[e])
            for e in range(tree.n_edges)
        )

    @pytest.mark.parametrize("trial", range(20))
    def test_matches_definition(self, rng, trial):
        for _ in range(trial):
            rng.random()
        tree = random_tree(rng.randint(2, 9), rng)
        survival = random_survival(rng, tree.n_leaves)
        mask = rng.randrange(1 << tree.n_leaves)
        value = expected_pd(tree, mask, np.log1p(-np.array(survival)))
        assert close(value, self.by_definition(tree, mask, survival))

    @pytest.mark.parametrize("trial", range(10))
    def test_without_survival_is_pd(self, rng, trial):
        for _ in range(trial):
            rng.random()
        tree = random_tree(rng.randint(2, 9), rng)
        mask = rng.randrange(1 << tree.n_leaves)
        assert expected_pd(tree, mask, np.zeros(tree.n_leaves)) == pytest.approx(pd(tree, mask))

    def test_protecting_more_never_hurts(self, rng):
        tree = random_tree(8, rng)
        loss = np.log1p(-np.array(random_survival(rng, 8)))
        for _ in range(30):
            mask = rng.randrange(1 << 8)
            bigger = mask | (1 << rng.randrange(8))
            assert expected_pd(tree, bigger, loss) >= expected_pd(tree, mask, loss) - 1e-12


# ---------------------------------------------------------------------------------------------
# End to end against brute force


@pytest.mark.parametrize("reduction,maf", CONFIGS, ids=CONFIG_IDS)
@pytest.mark.parametrize("trial", range(12))
def test_random_instances(rng, trial, reduction, maf):
    """Unstructured trees, uneven costs and weights, random survival probabilities."""
    for _ in range(trial):
        rng.random()
    case = random_instance(
        rng,
        n_leaves=rng.randint(3, 8),
        n_trees=rng.randint(1, 3),
        heterogeneous_costs=trial % 2 == 0,
        varied_weights=trial % 3 == 0,
    )
    assert_matches_oracle(surviving(case, rng), reduction, maf)


@pytest.mark.parametrize("reduction,maf", CONFIGS, ids=CONFIG_IDS)
@pytest.mark.parametrize("trial", range(8))
def test_related_instances(rng, trial, reduction, maf):
    """Near-identical trees: many clusters, so c_down and c_up carry most of the value."""
    for _ in range(trial):
        rng.random()
    case = related_instance(rng, n_leaves=rng.randint(5, 9), n_trees=rng.randint(2, 4))
    assert_matches_oracle(surviving(case, rng), reduction, maf)


@pytest.mark.parametrize("maf", ["heuristic", "untrusted"])
@pytest.mark.parametrize("trial", range(8))
def test_every_budget_on_many_clusters(rng, trial, maf):
    """Clade-localized instances, read out at every budget from one solve, with witnesses."""
    for _ in range(trial):
        rng.random()
    case = surviving(
        clade_localized_instance(
            rng, n_clades=rng.randint(3, 4), clade_size=3, n_trees=rng.randint(2, 3), budget=12
        ),
        rng,
    )
    instance = case.instance
    budgets = list(range(instance.n_taxa + 1))
    from .fixtures.baselines import MAFS

    results = instance.optimize_many(budgets, return_taxa=True, maf=MAFS[maf]())
    for budget, result in zip(budgets, results):
        expected, _ = brute_force_optimize_pd(instance, budget)
        assert close(result.value, expected), f"budget {budget}"
        assert sum(int(instance.costs[instance.registry.id_of(x)]) for x in result.taxa) <= budget
        assert close(instance.evaluate_pd(result.taxa), expected)


@pytest.mark.parametrize("reduction,maf", CONFIGS, ids=CONFIG_IDS)
def test_edge_cases(rng, reduction, maf):
    """Budgets and shapes at the boundaries: here the empty set and singletons often win."""
    tree = random_tree(5, rng)
    other = perturbed_tree(tree, rng, 1)
    some = [0.0, 0.5, 0.9, 0.3, 0.0]
    cases = {
        "zero budget": make_instance([tree, other], 0, survival=some),
        "nothing affordable": make_instance([tree, other], 2, costs=[3] * 5, survival=some),
        "only singletons affordable": make_instance(
            [tree, other], 2, costs=[2, 3, 3, 2, 3], survival=some
        ),
        "budget beyond total cost": make_instance([tree, other], 999, survival=some),
        "single tree": make_instance([tree], 3, survival=some),
        "identical trees": make_instance([tree, tree], 3, survival=some),
        "survival near one": make_instance([tree, other], 2, survival=[0.999999] * 5),
        "everyone else certain to die": make_instance([tree, other], 2, survival=[0.0] * 5),
        "a zero-weight tree": make_instance([tree, other], 3, weights=[0.0, 2.0], survival=some),
        "zero-length edges": make_instance(
            [random_tree(5, rng, max_weight=0), random_tree(5, rng, max_weight=0)], 2, survival=some
        ),
        "two taxa": make_instance(
            [random_tree(2, rng), random_tree(2, rng)], 1, survival=[0.4, 0.7]
        ),
        "three taxa": make_instance(
            [random_tree(3, rng), random_tree(3, rng)], 1, survival=[0.4, 0.7, 0.1]
        ),
        "star and caterpillar": make_instance(
            [star(6), caterpillar(6)], 2, survival=[0.1, 0.9, 0.5, 0.0, 0.3, 0.6]
        ),
    }
    for name, case in cases.items():
        try:
            assert_matches_oracle(case, reduction, maf)
        except AssertionError as exc:  # pragma: no cover - only on failure
            raise AssertionError(f"{name}: {exc}") from exc


@pytest.mark.parametrize(
    "switch",
    [
        (local_solver, "USE_COMPILED"),
        (profile_module, "USE_COMPILED"),
        (survival_module, "USE_COMPILED"),
        (common, "USE_COMPILED"),
    ],
    ids=["python-signature-loop", "numpy-profiles", "numpy-survival-sums", "python-reduction"],
)
@pytest.mark.parametrize("trial", range(6))
def test_every_code_path(rng, monkeypatch, switch, trial):
    """Each compiled kernel switched off in turn: the NumPy/Python twins give the same optimum."""
    monkeypatch.setattr(*switch, False)
    for _ in range(trial):
        rng.random()
    case = surviving(
        (
            clade_localized_instance(rng, n_clades=3, clade_size=3, n_trees=3, budget=5)
            if trial % 2
            else related_instance(rng, n_leaves=7, n_trees=3)
        ),
        rng,
    )
    assert_matches_oracle(case)


# ---------------------------------------------------------------------------------------------
# Properties


class TestProperties:
    @pytest.mark.parametrize("trial", range(8))
    def test_more_survival_never_lowers_the_optimum(self, rng, trial):
        """Raising one taxon's survival probability can only add expected diversity."""
        for _ in range(trial):
            rng.random()
        case = random_instance(rng, n_leaves=7, n_trees=2, heterogeneous_costs=True)
        survival = random_survival(rng, 7)
        before = with_survival(case, survival).instance.optimize(case.budget).value
        taxon = rng.randrange(7)
        survival[taxon] = survival[taxon] + (0.999 - survival[taxon]) * rng.random()
        after = with_survival(case, survival).instance.optimize(case.budget).value
        assert after >= before - 1e-9

    def test_monotone_in_budget(self, rng):
        case = surviving(related_instance(rng, n_leaves=9, n_trees=3), rng)
        values = [r.value for r in case.instance.optimize_many(range(10))]
        assert all(b >= a for a, b in zip(values, values[1:]))

    def test_zero_budget_is_the_expected_pd_of_nothing(self, rng):
        case = surviving(related_instance(rng, n_leaves=8, n_trees=3), rng)
        result = case.instance.optimize(0, return_taxa=True)
        assert result.taxa == set()
        assert close(result.value, case.instance.evaluate_pd(()))
        assert result.value > 0

    def test_scaling_weights_scales_the_optimum(self, rng):
        case = surviving(random_instance(rng, n_leaves=7, n_trees=3, varied_weights=True), rng)
        instance = case.instance
        scaled = MaxMPDInstance.from_phylotrees(
            instance.trees,
            instance.registry,
            instance.costs.tolist(),
            instance.weights * 3.5,
            instance.survival.tolist(),
        )
        assert close(scaled.optimize(case.budget).value, 3.5 * instance.optimize(case.budget).value)

    def test_explicit_zeros_equal_the_default(self, rng):
        """Survival zero for everyone is the default: same frontier, bit for bit."""
        case = related_instance(rng, n_leaves=9, n_trees=3)
        zeros = with_survival(case, [0.0] * 9)
        a = case.instance.optimize(case.budget, return_taxa=True)
        b = zeros.instance.optimize(case.budget, return_taxa=True)
        np.testing.assert_array_equal(a.meta["frontier"], b.meta["frontier"])
        assert a.taxa == b.taxa


# ---------------------------------------------------------------------------------------------
# The pieces


def single_local(instance: MaxMPDInstance) -> LocalInstance:
    """The whole instance as one local instance, with its leaf constants."""
    piece = single_cluster(instance.trees).pieces[0]
    n = piece.n_leaves
    return LocalInstance(
        piece.trees,
        instance.weights,
        np.zeros((n, 2)),
        np.ones(n, np.int64),
        1,
        leaf_log=log_loss(instance),
    )


@pytest.mark.parametrize("trial", range(15))
def test_signature_decomposition(rng, trial):
    """
    For every selection ``Q``, with ``S`` the components it hits: ``d_S`` plus each active
    component's two-sided PD of ``Q`` equals the expected PD of ``Q`` (the case table of
    :mod:`~forestpd.algo.local_solver`, checked
    straight from the definition, without the DP).
    """
    for _ in range(trial):
        rng.random()
    case = surviving(random_instance(rng, n_leaves=rng.randint(4, 8), n_trees=3), rng)
    instance = case.instance
    local = single_local(instance)
    forest = HeuristicMAF().compute(local.trees)
    profile = build_profile(local, forest)
    loss = log_loss(instance)
    for _ in range(20):
        chosen = {x for x in range(local.n_leaves) if rng.random() < 0.4} or {0}
        signature = sum(1 << i for i, block in enumerate(forest) if chosen & set(block))
        d_s, lam_down, lam_up, lam_both = signature_weights(profile, signature)
        value = d_s
        for comp in range(profile.n_components):
            if (signature >> comp) & 1:
                lo, hi = int(profile.node_offset[comp]), int(profile.node_offset[comp + 1])
                value += two_sided_pd(
                    profile.comp_trees[comp],
                    lam_down[lo:hi],
                    lam_up[lo:hi],
                    lam_both[lo:hi],
                    chosen,
                    profile.slot_survive_down[lo:hi],
                    profile.slot_survive_up[lo:hi],
                )
        mask = sum(1 << x for x in chosen)
        direct = sum(
            float(w) * expected_pd(tree, mask, loss)
            for w, tree in zip(instance.weights, instance.trees)
        )
        assert close(value, direct)


def aggregated_rows(profile):
    """A profile's rows as a signature reads them, aggregated by key; edge pairs unordered."""
    edges: dict = {}
    for lo, hi, owner, w, s_lo, s_hi in zip(
        profile.edge_only_below.tolist(),
        profile.edge_only_above.tolist(),
        profile.edge_owner.tolist(),
        profile.edge_weight.tolist(),
        profile.edge_survive_below.tolist(),
        profile.edge_survive_above.tolist(),
    ):
        if lo > hi:
            lo, hi, s_lo, s_hi = hi, lo, s_hi, s_lo
        if lo == 0:
            # One side is empty, so it is never covered: only the empty side's survive column
            # can ever be paid. The other columns are not compared (the paths may drop such rows
            # before or after merging, which changes only these unread sums).
            w = s_hi = 0.0
        acc = edges.setdefault((lo, hi, owner), [0.0, 0.0, 0.0])
        for i, value in enumerate((w, s_lo, s_hi)):
            acc[i] += value
    comps: dict = {}
    for *key, w, dd, du, db in zip(
        profile.ce_slot.tolist(),
        profile.ce_own_bit.tolist(),
        profile.ce_only_down.tolist(),
        profile.ce_only_up.tolist(),
        profile.ce_weight.tolist(),
        profile.ce_dies_down.tolist(),
        profile.ce_dies_up.tolist(),
        profile.ce_dies_both.tolist(),
    ):
        acc = comps.setdefault(tuple(key), [0.0] * 4)
        for i, value in enumerate((w, dd, du, db)):
            acc[i] += value
    nonzero = lambda d: {k: v for k, v in d.items() if max(map(abs, v)) > 1e-12}  # noqa: E731
    return nonzero(edges), nonzero(comps)


def assert_same_rows(a, b):
    for x, y in zip(aggregated_rows(a), aggregated_rows(b)):
        assert x.keys() == y.keys()
        for key in x:
            np.testing.assert_allclose(x[key], y[key], rtol=1e-12, atol=1e-12)


def pieces_with_logs(rng, n_leaves=10):
    """The pieces of a reduced instance with survival, each with its leaf constants."""
    base = random_tree(n_leaves, rng)
    trees = [perturbed_tree(base, rng, rng.randint(0, 2)) for _ in range(3)]
    weights = np.array([rng.uniform(0.2, 2.0) for _ in trees])
    loss = np.log1p(-np.array(random_survival(rng, n_leaves)))
    ktree = cluster_reduction(trees, weights=weights)
    sub = subtree_logs(ktree, loss)
    total = float(loss.sum())
    for cluster, piece in enumerate(ktree.pieces):
        yield ktree, piece, leaf_logs(piece, loss, sub, cluster, total), weights


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "numpy"])
@pytest.mark.parametrize("trial", range(10))
def test_split_profiles_match_tree_profiles(rng, monkeypatch, compiled, trial):
    """The side constants come out the same from the split table and from the trees, plus and
    plain, on the compiled and the NumPy split path."""
    monkeypatch.setattr(profile_module, "USE_COMPILED", compiled)
    for _ in range(trial):
        rng.random()
    for _, piece, leaf_log, _ in pieces_with_logs(rng):
        if piece.split_masks is None:
            continue
        n = piece.n_leaves
        tw = np.asarray(piece.tree_weights)
        local = LocalInstance(
            piece.trees, tw, np.zeros((n, 2)), np.ones(n, np.int64), 1, leaf_log=leaf_log
        )
        plain_trees = local.trees
        if piece.star_leaf >= 0:
            from forestpd.core.tree import drop_leaf

            plain_trees = tuple(drop_leaf(t, piece.star_leaf) for t in piece.trees)
        forest = HeuristicMAF().compute(plain_trees)
        star = None
        if piece.star_leaf >= 0:
            star = len(forest)
            forest = forest + ((piece.star_leaf,),)
        direct = build_profile(local, forest)
        split, plain = build_profiles_from_splits(
            n, piece.split_masks, piece.split_weights, forest, star, leaf_log
        )
        assert_same_rows(direct, split)
        if star is not None:
            assert plain is not None
            assert_same_rows(profile_module.plain_profile_from_plus(direct, star), plain)


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "numpy"])
@pytest.mark.parametrize("trial", range(10))
def test_cluster_sides_split_and_tree_paths_agree(rng, monkeypatch, compiled, trial):
    monkeypatch.setattr(survival_module, "USE_COMPILED", compiled)
    for _ in range(trial):
        rng.random()
    for _, piece, leaf_log, weights in pieces_with_logs(rng):
        if piece.split_masks is None:
            continue
        from_splits = cluster_sides(piece, leaf_log, weights)
        unsplit = type(piece)(
            trees=piece.trees,
            taxon_of_leaf=piece.taxon_of_leaf,
            child_of_leaf=piece.child_of_leaf,
            star_leaf=piece.star_leaf,
            tree_weights=piece.tree_weights,
        )
        from_trees = cluster_sides(unsplit, leaf_log, weights)
        np.testing.assert_allclose(from_splits.at_leaf, from_trees.at_leaf, rtol=1e-12, atol=1e-12)
        assert from_splits.empty == pytest.approx(from_trees.empty, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("trial", range(10))
def test_empty_and_singletons_match_direct_evaluation(rng, trial):
    """The solver's value of the empty set and of every single taxon (built from c_down, c_up and
    the cluster sums) equals the direct expected PD."""
    for _ in range(trial):
        rng.random()
    case = surviving(
        (
            clade_localized_instance(rng, n_clades=3, clade_size=4, n_trees=3, budget=3)
            if trial % 2
            else related_instance(rng, n_leaves=9, n_trees=3)
        ),
        rng,
    )
    instance = case.instance
    solution = solve(
        cluster_reduction(instance.trees, weights=instance.weights),
        HeuristicMAF(),
        instance.costs,
        instance.weights,
        1,
        log_loss=log_loss(instance),
    )
    # Costs are at least 1, so at budget 0 only the empty selection is feasible.
    assert close(float(solution.meta["frontier"][0]), instance.evaluate_pd(()))
    for label in instance.taxa:
        taxon = instance.registry.id_of(label)
        assert close(float(solution.single_values[taxon]), instance.evaluate_pd({label})), label


@pytest.mark.skipif(
    not local_solver._signature_kernel.HAVE_NUMBA, reason="the compiled loop needs Numba"
)
@pytest.mark.parametrize("trial", range(10))
def test_compiled_and_python_signature_loops_agree(rng, monkeypatch, trial):
    for _ in range(trial):
        rng.random()
    case = surviving(random_instance(rng, n_leaves=7, n_trees=3, heterogeneous_costs=True), rng)
    local = single_local(case.instance)
    local = LocalInstance(
        local.trees,
        local.tree_weight,
        np.zeros((local.n_leaves, 6)),
        case.instance.costs,
        5,
        leaf_log=local.leaf_log,
    )
    profile = build_profile(local, HeuristicMAF().compute(local.trees))
    monkeypatch.setattr(local_solver, "USE_COMPILED", True)
    compiled = solve_local(local, profile)
    monkeypatch.setattr(local_solver, "USE_COMPILED", False)
    python = solve_local(local, profile)
    np.testing.assert_allclose(compiled.table, python.table, rtol=1e-12)


# ---------------------------------------------------------------------------------------------
# Rooted PD


class TestRooted:
    @staticmethod
    def instance(rng, n_taxa, n_trees):
        labels = [f"x{i}" for i in range(n_taxa)]
        trees = [random_rooted(rng, labels) for _ in range(n_trees)]
        survival = dict(zip(labels, random_survival(rng, n_taxa)))
        costs = {x: rng.randint(1, 2) for x in labels}
        return (
            trees,
            survival,
            costs,
            fpd.MaxMPDInstance(trees, costs, survival=survival, rooted=True),
        )

    @pytest.mark.parametrize("trial", range(10))
    def test_pd_matches_direct_computation(self, trial):
        rng = random.Random(trial)
        trees, survival, _, instance = self.instance(rng, rng.randint(2, 6), rng.randint(1, 3))
        for _ in range(10):
            chosen = {x for x in instance.taxa if rng.random() < 0.4}
            direct = sum(rooted_pd(tree, chosen, survival) for tree in trees)
            assert close(instance.evaluate_pd(chosen), direct)

    @pytest.mark.parametrize("trial", range(10))
    def test_optimum_at_every_budget(self, trial):
        rng = random.Random(100 + trial)
        trees, survival, costs, instance = self.instance(rng, rng.randint(2, 6), rng.randint(1, 3))
        budgets = list(range(sum(costs.values()) + 1))
        for budget, result in zip(budgets, instance.optimize_many(budgets, return_taxa=True)):
            best = max(
                sum(rooted_pd(tree, set(chosen), survival) for tree in trees)
                for size in range(len(costs) + 1)
                for chosen in __import__("itertools").combinations(sorted(costs), size)
                if sum(costs[x] for x in chosen) <= budget
            )
            assert close(result.value, best), f"budget {budget}"
            assert sum(costs[x] for x in result.taxa) <= budget
            assert close(instance.evaluate_pd(result.taxa), best)

    @pytest.mark.parametrize("trial", range(6))
    def test_tree_weights_and_budgets_past_the_total(self, trial):
        rng = random.Random(200 + trial)
        labels = [f"x{i}" for i in range(rng.randint(2, 6))]
        trees = [random_rooted(rng, labels) for _ in range(rng.randint(1, 3))]
        weights = [round(rng.uniform(0.25, 3.0), 2) for _ in trees]
        survival = dict(zip(labels, random_survival(rng, len(labels))))
        instance = fpd.MaxMPDInstance(trees, weights=weights, survival=survival, rooted=True)
        everything = sum(w * rooted_pd(t, set(labels), survival) for w, t in zip(weights, trees))
        for budget in (len(labels), len(labels) + 1, 3 * len(labels)):
            result = instance.optimize(budget, return_taxa=True)
            assert close(result.value, everything), f"budget {budget}"
            assert len(result.meta["frontier"]) == len(labels) + 1  # unit costs: total n
            assert close(float(result.meta["frontier"][-1]), everything)
            assert close(instance.evaluate_pd(result.taxa), everything)

    def test_root_leaf_is_invisible(self):
        instance = fpd.MaxMPDInstance(
            ["((A:1,B:1):1,(C:1,D:1):1);"], survival={"A": 0.5}, rooted=True
        )
        assert instance.survival[instance.registry.id_of("A")] == 0.5
        assert float(instance.survival.sum()) == 0.5
        assert "\x00root" not in instance.taxa
        # Nothing protected: A survives with probability 1/2, giving its root path 1 + 1.
        assert instance.evaluate_pd(()) == pytest.approx(1.0)
        assert instance.optimize(0, return_taxa=True).taxa == set()


def test_rooted_newick_round_trip_through_api(rng):
    """Rooted input written from random trees solves like the brute force (another root shape)."""
    n = 6
    labels = [f"t{i}" for i in range(n)]
    newicks = []
    for _ in range(3):
        text = rooted_newick(random_tree(n, rng), rng.randrange(n))
        newicks.append(re.sub(r"\d+", lambda m: labels[int(m.group()) - 1], text))
    survival = dict(zip(labels, random_survival(rng, n)))
    instance = MaxMPDInstance(newicks, survival=survival, rooted=True, missing_branch_length=1.0)
    for budget in range(4):
        expected, _ = brute_force_optimize_pd(instance, budget)
        assert close(instance.optimize(budget).value, expected)


# ---------------------------------------------------------------------------------------------
# API


class TestAPI:
    TREES = ["(A:1,B:1,(C:1,D:1):2);", "(A:1,C:1,(B:1,D:1):2);"]

    def test_optimize_pd_takes_survival(self):
        plain = optimize_maxmpd(self.TREES, 2)
        risky = optimize_maxmpd(self.TREES, 2, survival={"A": 0.5, "C": 0.9}, return_taxa=True)
        assert risky.value > plain.value
        instance = MaxMPDInstance(self.TREES, survival={"A": 0.5, "C": 0.9})
        assert risky.value == pytest.approx(instance.optimize(2).value)
        assert instance.evaluate_pd(risky.taxa) == pytest.approx(risky.value)

    def test_mapping_and_sequence_agree(self):
        by_label = MaxMPDInstance(self.TREES, survival={"B": 0.25, "D": 0.75})
        by_id = MaxMPDInstance.from_phylotrees(
            by_label.trees, by_label.registry, survival=[0.0, 0.25, 0.0, 0.75]
        )
        np.testing.assert_array_equal(by_label.survival, by_id.survival)
        assert by_label.optimize(1).value == pytest.approx(by_id.optimize(1).value)

    def test_missing_labels_default_to_zero(self):
        instance = MaxMPDInstance(self.TREES, survival={"B": 0.25})
        assert instance.survival.tolist() == [0.0, 0.25, 0.0, 0.0]

    @pytest.mark.parametrize(
        "survival,message",
        [
            ({"A": 1.0}, r"\[0, 1\)"),
            ({"A": -0.1}, r"\[0, 1\)"),
            ({"A": float("nan")}, r"\[0, 1\)"),
            ({"A": float("inf")}, r"\[0, 1\)"),
            ({"A": True}, "must be a number"),
            ({"A": "0.5"}, "must be a number"),
            ({"Z": 0.5}, "unknown taxon"),
        ],
    )
    def test_invalid_survival_is_rejected(self, survival, message):
        with pytest.raises(ForestPDInputError, match=message):
            MaxMPDInstance(self.TREES, survival=survival)

    def test_wrong_sequence_length_is_rejected(self):
        instance = MaxMPDInstance(self.TREES)
        with pytest.raises(ForestPDInputError, match="Expected 4 survival probabilities"):
            MaxMPDInstance.from_phylotrees(instance.trees, instance.registry, survival=[0.1, 0.2])


# ---------------------------------------------------------------------------------------------
# Pinned values

#: name -> (survival probabilities, expected optimum), on the instances of test_regression. Each
#: value is also re-derived by brute force.
PINNED: dict[str, tuple[list[float], float]] = {
    "one SPR apart": ([0.0, 0.2, 0.4, 0.6, 0.8, 0.0, 0.2, 0.4, 0.6, 0.8, 0.0, 0.2], 147.90464),
    "three SPR apart": ([0.9, 0.0, 0.5, 0.0, 0.9, 0.0, 0.5, 0.0, 0.9, 0.0, 0.5, 0.0], 151.04),
    "uneven costs": ([0.3] * 12, 135.95),
}


def pinned_case(name: str) -> Case:
    from .test_regression import curated

    return with_survival(curated()[name], PINNED[name][0])


@pytest.mark.parametrize("name", sorted(PINNED))
def test_pinned_values(name):
    case = pinned_case(name)
    value = case.instance.optimize(case.budget).value
    assert value == pytest.approx(PINNED[name][1], rel=1e-12)
    assert value == pytest.approx(brute_force_optimize_pd(case.instance, case.budget)[0], rel=1e-9)


# ---------------------------------------------------------------------------------------------
# Fast evaluation: the split table against the per-tree reference


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "numpy"])
@pytest.mark.parametrize("trial", range(10))
def test_evaluate_pd_matches_per_tree_reference(rng, monkeypatch, compiled, trial):
    """evaluate_pd groups edges by split; it must equal the tree-by-tree expected PD, with and
    without survival probabilities, for every kind of set (empty, single, large)."""
    import forestpd.io.splits as splits_module

    from .fixtures.brute_force import reference_value

    monkeypatch.setattr(splits_module, "USE_COMPILED", compiled)
    for _ in range(trial):
        rng.random()
    case = random_instance(
        rng, n_leaves=rng.randint(2, 12), n_trees=rng.randint(1, 4), varied_weights=True
    )
    for instance in (case.instance, surviving(case, rng).instance):
        labels = list(instance.taxa)
        sets = [
            (),
            labels[:1],
            labels,
            *(rng.sample(labels, rng.randint(1, len(labels))) for _ in range(10)),
        ]
        for chosen in sets:
            assert close(instance.evaluate_pd(chosen), reference_value(instance, chosen))


@pytest.mark.parametrize("trial", range(6))
def test_evaluate_pd_rooted_matches_reference(trial):
    rng = random.Random(500 + trial)
    labels = [f"x{i}" for i in range(rng.randint(2, 7))]
    trees = [random_rooted(rng, labels) for _ in range(rng.randint(1, 3))]
    survival = dict(zip(labels, random_survival(rng, len(labels))))
    instance = fpd.MaxMPDInstance(trees, survival=survival, rooted=True)
    from .fixtures.brute_force import reference_value

    for _ in range(10):
        chosen = [x for x in labels if rng.random() < 0.5]
        assert close(instance.evaluate_pd(chosen), reference_value(instance, chosen))
