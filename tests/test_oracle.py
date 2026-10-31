"""
The headline correctness gate: the optimizer equals brute force.

Every stage combination is checked, because the pipeline's whole design claim is that the stages
are interchangeable — a cluster reduction that splits differently, or a coarser agreement forest,
must not move the answer. Witnesses are checked too: a value is only trustworthy if some feasible
taxon set actually realizes it.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from forestpd import MaxMPDInstance

from .fixtures.baselines import optimize
from .fixtures.brute_force import brute_force_optimize_pd, reference_value
from .fixtures.random_instances import (
    balanced,
    caterpillar,
    clade_localized_instance,
    make_instance,
    random_instance,
    random_survival,
    random_tree,
    related_instance,
    star,
)

CONFIGS = [
    (reduction, maf)
    for reduction in ("common", "trivial")
    for maf in ("heuristic", "singleton", "untrusted")
]
CONFIG_IDS = [f"{r}+{m}" for r, m in CONFIGS]


def assert_matches_oracle(case, reduction, maf):
    """Value equals brute force, and the returned witness is feasible and achieves it."""
    instance, budget = case
    expected, _ = brute_force_optimize_pd(instance, budget)
    result = optimize(case, reduction, maf, return_taxa=True)
    assert result.value == pytest.approx(expected), f"{reduction}/{maf}"

    ids = instance.registry.ids_of(result.taxa)
    cost = sum(int(instance.costs[i]) for i in ids)
    assert cost <= budget, f"witness costs {cost} > budget {budget}"
    assert instance.evaluate_pd(result.taxa) == pytest.approx(expected)


@pytest.mark.parametrize("reduction,maf", CONFIGS, ids=CONFIG_IDS)
@pytest.mark.parametrize("trial", range(25))
def test_random_instances(rng, trial, reduction, maf):
    """Unstructured random trees: the hardest case for the MAF, the easiest to get wrong."""
    for _ in range(trial):
        rng.random()  # decorrelate the trials
    case = random_instance(
        rng,
        n_leaves=rng.randint(4, 7),
        n_trees=rng.randint(1, 3),
        heterogeneous_costs=trial % 3 == 0,
        varied_weights=trial % 4 == 0,
    )
    assert_matches_oracle(case, reduction, maf)


@pytest.mark.parametrize("reduction,maf", CONFIGS, ids=CONFIG_IDS)
@pytest.mark.parametrize("trial", range(15))
def test_related_instances(rng, trial, reduction, maf):
    """Near-identical trees — the regime the FPT parameter is small in, and the intended use."""
    for _ in range(trial):
        rng.random()
    case = related_instance(
        rng, n_leaves=rng.randint(5, 8), n_trees=rng.randint(2, 4), moves=rng.randint(1, 3)
    )
    assert_matches_oracle(case, reduction, maf)


@pytest.mark.parametrize("reduction,maf", CONFIGS, ids=CONFIG_IDS)
@pytest.mark.parametrize(
    "name,trees,budget",
    [
        ("star+caterpillar", [star(6), caterpillar(6)], 3),
        ("two stars", [star(5), star(5)], 4),
        ("two caterpillars", [caterpillar(6), caterpillar(6)], 2),
        ("balanced pair", [balanced(8), balanced(8)], 4),
        ("star vs balanced", [star(8), balanced(8)], 3),
    ],
)
def test_degenerate_topologies(name, trees, budget, reduction, maf):
    """Polytomies and maximally unbalanced shapes, which stress the binarization and the MAF."""
    assert_matches_oracle(make_instance(trees, budget), reduction, maf)


@pytest.mark.parametrize("reduction,maf", CONFIGS, ids=CONFIG_IDS)
def test_edge_cases(rng, reduction, maf):
    """Budgets and instance shapes at the boundaries of what the tables can express."""
    tree = random_tree(5, rng)
    cases = {
        "single tree": make_instance([tree], 3),
        "identical trees": make_instance([tree, tree], 3),
        "zero budget": make_instance([tree, tree], 0),
        "budget one": make_instance([tree, tree], 1),
        "budget beyond total cost": make_instance([tree, tree], 999),
        "one expensive taxon": make_instance([tree, tree], 4, costs=[1, 5, 1, 1, 1]),
        "nothing affordable": make_instance([tree, tree], 2, costs=[3] * 5),
        "a zero-weight tree": make_instance([tree, tree], 3, weights=[0.0, 2.0]),
        "two taxa": make_instance([random_tree(2, rng), random_tree(2, rng)], 2),
        "three taxa": make_instance([random_tree(3, rng), random_tree(3, rng)], 2),
    }
    for name, case in cases.items():
        try:
            assert_matches_oracle(case, reduction, maf)
        except AssertionError as exc:  # pragma: no cover - only on failure
            raise AssertionError(f"{name}: {exc}") from exc


@pytest.mark.parametrize("reduction,maf", CONFIGS, ids=CONFIG_IDS)
def test_zero_weight_edges(rng, reduction, maf):
    """Zero-length branches must not be confused with absent ones."""
    trees = [random_tree(6, rng, max_weight=0) for _ in range(2)]
    assert_matches_oracle(make_instance(trees, 3), reduction, maf)


def test_all_configurations_agree(rng):
    """
    Every stage combination returns the same value on the same instance.

    A direct test of the modularity claim, independent of whether the oracle is consulted.
    """
    for _ in range(15):
        case = related_instance(rng, n_leaves=9, n_trees=3, moves=2)
        values = {
            f"{reduction}+{maf}": optimize(case, reduction, maf).value for reduction, maf in CONFIGS
        }
        assert len(set(round(v, 9) for v in values.values())) == 1, values


@pytest.mark.parametrize("trial", range(20))
def test_optimize_many_matches_oracle_at_every_budget(rng, trial):
    """One solve at the largest budget answers every smaller budget, with valid witnesses."""
    for _ in range(trial):
        rng.random()
    case = random_instance(
        rng,
        n_leaves=rng.randint(4, 7),
        n_trees=rng.randint(1, 3),
        heterogeneous_costs=trial % 2 == 0,
        varied_weights=trial % 3 == 0,
    )
    instance = case.instance
    budgets = list(range(int(instance.costs.sum()) + 2))  # including one past the total cost
    rng.shuffle(budgets)
    results = instance.optimize_many(budgets, return_taxa=True)
    for budget, result in zip(budgets, results):
        expected, _ = brute_force_optimize_pd(instance, budget)
        assert result.value == pytest.approx(expected), f"budget {budget}"
        assert result.value == pytest.approx(instance.optimize(budget).value)
        ids = instance.registry.ids_of(result.taxa)
        assert sum(int(instance.costs[i]) for i in ids) <= budget
        assert instance.evaluate_pd(result.taxa) == pytest.approx(expected)


@pytest.mark.parametrize("survive", [False, True], ids=["plain", "survival"])
@pytest.mark.parametrize("trial", range(10))
def test_optimize_many_sets_equal_separate_solves(rng, trial, survive):
    """The sets of all budgets are reconstructed together, sharing tables between budgets; each
    must be the set a solve at that budget alone returns. No brute force, so the instances can have
    many clusters and many budgets sharing them."""
    for _ in range(trial):
        rng.random()
    if trial % 2 == 0:
        base = clade_localized_instance(
            rng, n_clades=rng.randint(4, 6), clade_size=rng.randint(3, 5), n_trees=rng.randint(2, 4)
        ).instance
    else:
        base = related_instance(
            rng, n_leaves=rng.randint(15, 25), n_trees=rng.randint(2, 5)
        ).instance
    costs = [rng.randint(1, 3) for _ in range(base.n_taxa)] if trial % 3 == 0 else None
    survival = random_survival(rng, base.n_taxa) if survive else None
    instance = MaxMPDInstance.from_phylotrees(
        base.trees, base.registry, costs, base.weights, survival
    )
    budgets = list(range(int(instance.costs.sum()) + 2))  # including one past the total cost
    rng.shuffle(budgets)
    budgets += budgets[:3]  # repeated budgets
    together = instance.optimize_many(budgets, return_taxa=True)
    for budget, result in zip(budgets, together):
        alone = instance.optimize(budget, return_taxa=True)
        assert result.taxa == alone.taxa, f"budget {budget}"
        assert result.value == alone.value, f"budget {budget}"


@pytest.mark.parametrize("trial", range(12))
def test_costs_with_a_common_divisor(rng, trial):
    """Costs sharing a factor g are solved in units of g. Every budget, a multiple of g or not,
    must still match brute force, with a frontier over the original budgets and a feasible witness.
    Unit costs times g (all costs equal) are included."""
    for _ in range(trial):
        rng.random()
    unit = (2, 3, 5)[trial % 3]
    base = random_instance(
        rng, n_leaves=rng.randint(4, 6), n_trees=rng.randint(1, 3), heterogeneous_costs=trial % 2
    ).instance
    survival = random_survival(rng, base.n_taxa) if trial % 4 == 0 else None
    costs = [unit * int(c) for c in base.costs]
    instance = MaxMPDInstance.from_phylotrees(
        base.trees, base.registry, costs, base.weights, survival
    )
    budgets = list(range(sum(costs) + unit + 1))
    for budget, result in zip(budgets, instance.optimize_many(budgets, return_taxa=True)):
        expected, _ = brute_force_optimize_pd(instance, budget)
        assert result.value == pytest.approx(expected), f"budget {budget}"
        assert len(result.meta["frontier"]) == min(budget, sum(costs)) + 1
        ids = instance.registry.ids_of(result.taxa)
        assert sum(costs[i] for i in ids) <= budget
        assert instance.evaluate_pd(result.taxa) == pytest.approx(expected)
    # A single large budget is solved with DP' (budget saved), which covers the budgets from it
    # on only; those below are NaN. The whole range covers every budget (DP and DP' both).
    for result in (instance.optimize(budgets[-1]), instance.optimize_many(budgets)[-1]):
        frontier = result.meta["frontier"]
        assert not np.isnan(frontier[-1])
        if result.meta["tables"] != "DP'":
            assert not np.isnan(frontier).any()
        for budget, value in enumerate(frontier):
            if not np.isnan(value):
                assert value == pytest.approx(brute_force_optimize_pd(instance, budget)[0])


@pytest.mark.parametrize("trial", range(12))
def test_cluster_budget_cap_matches_oracle(rng, trial):
    """Many small clusters under a budget far above each cluster's subtree cost: the per-cluster
    budget cap (and the padding of its tables) must not move the optimum at any budget."""
    for _ in range(trial):
        rng.random()
    case = clade_localized_instance(
        rng, n_clades=rng.randint(3, 4), clade_size=3, n_trees=rng.randint(2, 4), budget=12
    )
    instance = case.instance
    budgets = list(range(instance.n_taxa + 1))
    for budget, result in zip(budgets, instance.optimize_many(budgets, return_taxa=True)):
        expected, _ = brute_force_optimize_pd(instance, budget)
        assert result.value == pytest.approx(expected), f"budget {budget}"
        cost = sum(int(instance.costs[i]) for i in instance.registry.ids_of(result.taxa))
        assert cost <= budget
        assert instance.evaluate_pd(result.taxa) == pytest.approx(expected)


@pytest.mark.parametrize("survive", [False, True], ids=["plain", "survival"])
@pytest.mark.parametrize("trial", range(8))
def test_many_clusters_with_costs_and_weights(rng, trial, survive):
    """Several clusters with uneven costs and tree weights, at every budget: the per-cluster budget
    split and the tree weights inside merged local trees must match brute force."""
    for _ in range(trial):
        rng.random()
    base = clade_localized_instance(
        rng, n_clades=rng.randint(3, 4), clade_size=3, n_trees=rng.randint(2, 4)
    ).instance
    n_taxa, n_trees = base.n_taxa, len(base.trees)
    costs = [rng.randint(1, 3) for _ in range(n_taxa)]
    weights = [round(rng.uniform(0.25, 3.0), 2) for _ in range(n_trees)]
    survival = random_survival(rng, n_taxa) if survive else None
    instance = MaxMPDInstance.from_phylotrees(base.trees, base.registry, costs, weights, survival)
    assert instance.structure().n_clusters > 1
    # Brute force once for all budgets: the best value of each subset cost, then a running max.
    best_at_cost = [reference_value(instance, ())] + [float("-inf")] * (sum(costs) + 1)
    labels = instance.taxa
    for size in range(1, n_taxa + 1):
        for subset in itertools.combinations(range(n_taxa), size):
            cost = sum(costs[i] for i in subset)
            value = reference_value(instance, [labels[i] for i in subset])
            best_at_cost[cost] = max(best_at_cost[cost], value)
    oracle = list(itertools.accumulate(best_at_cost, max))
    budgets = list(range(len(oracle)))
    for budget, result in zip(budgets, instance.optimize_many(budgets, return_taxa=True)):
        expected = oracle[budget]
        assert result.value == pytest.approx(expected), f"budget {budget}"
        cost = sum(costs[i] for i in instance.registry.ids_of(result.taxa))
        assert cost <= budget, f"budget {budget}"
        assert instance.evaluate_pd(result.taxa) == pytest.approx(expected)


@pytest.mark.parametrize("trial", range(10))
def test_witness_with_large_values(rng, trial):
    """Witness reconstruction must not depend on the scale of the objective: with values around
    1e9 (e.g. branch lengths in generations) an absolute tolerance fails, a relative one does not.
    """
    for _ in range(trial):
        rng.random()
    case = related_instance(rng, n_leaves=rng.randint(6, 9), n_trees=rng.randint(2, 4))
    instance = case.instance
    scaled = type(instance).from_phylotrees(
        instance.trees, instance.registry, instance.costs.tolist(), instance.weights * 1e9
    )
    budget = case.budget
    expected, _ = brute_force_optimize_pd(scaled, budget)
    result = scaled.optimize(budget, return_taxa=True)
    assert result.value == pytest.approx(expected, rel=1e-12)
    assert scaled.evaluate_pd(result.taxa) == pytest.approx(expected, rel=1e-12)
