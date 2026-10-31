"""
Structural properties the optimum must satisfy, whatever the instance.

These do not consult the oracle, so they scale to instances far larger than brute force can reach
and they catch a different class of bug: not "wrong by a little" but "wrong in a way that breaks
the objective's shape".
"""

from __future__ import annotations

import pytest

from .fixtures.baselines import optimize
from .fixtures.random_instances import make_instance, random_tree, related_instance

CONFIGS = [("common", "heuristic"), ("trivial", "singleton")]


def optimum(case, reduction="common", maf="heuristic"):
    """Objective value under one stage configuration."""
    return optimize(case, reduction, maf).value


@pytest.mark.parametrize("reduction,maf", CONFIGS)
def test_monotone_in_the_budget(rng, reduction, maf):
    """More budget can never buy less diversity."""
    trees = [random_tree(9, rng) for _ in range(3)]
    previous = -1.0
    for budget in range(0, 10):
        value = optimum(make_instance(trees, budget), reduction, maf)
        assert value >= previous - 1e-9, f"budget {budget} dropped below {budget - 1}"
        previous = value


@pytest.mark.parametrize("reduction,maf", CONFIGS)
def test_bounded_by_the_total_weight(rng, reduction, maf):
    """The objective cannot exceed every tree's whole length, and is never negative."""
    for _ in range(10):
        trees = [random_tree(8, rng) for _ in range(3)]
        weights = [1.0, 2.0, 0.5]
        ceiling = sum(w * t.total_weight() for w, t in zip(weights, trees))
        value = optimum(make_instance(trees, 5, weights=weights), reduction, maf)
        assert 0.0 <= value <= ceiling + 1e-9


@pytest.mark.parametrize("reduction,maf", CONFIGS)
def test_full_budget_selects_everything(rng, reduction, maf):
    """With budget for every taxon, the optimum is the full weighted length of all trees."""
    trees = [random_tree(8, rng) for _ in range(3)]
    weights = [1.0, 2.0, 3.0]
    case = make_instance(trees, 8, weights=weights)
    expected = sum(w * t.total_weight() for w, t in zip(weights, trees))
    assert optimum(case, reduction, maf) == pytest.approx(expected)


@pytest.mark.parametrize("reduction,maf", CONFIGS)
def test_zero_budget_yields_nothing(rng, reduction, maf):
    trees = [random_tree(7, rng) for _ in range(2)]
    assert optimum(make_instance(trees, 0), reduction, maf) == 0.0


@pytest.mark.parametrize("reduction,maf", CONFIGS)
def test_a_zero_weight_tree_contributes_nothing(rng, reduction, maf):
    """Setting ``w_j = 0`` must be identical to deleting tree ``j``."""
    trees = [random_tree(8, rng) for _ in range(3)]
    with_zero = make_instance(trees, 4, weights=[1.0, 0.0, 1.0])
    without = make_instance([trees[0], trees[2]], 4, weights=[1.0, 1.0])
    assert optimum(with_zero, reduction, maf) == pytest.approx(optimum(without, reduction, maf))


@pytest.mark.parametrize("reduction,maf", CONFIGS)
def test_duplicating_a_tree_doubles_its_contribution(rng, reduction, maf):
    """Two copies of a tree at weight 1 must equal one copy at weight 2."""
    trees = [random_tree(8, rng) for _ in range(2)]
    doubled = make_instance([trees[0], trees[0], trees[1]], 4, weights=[1.0, 1.0, 1.0])
    weighted = make_instance([trees[0], trees[1]], 4, weights=[2.0, 1.0])
    assert optimum(doubled, reduction, maf) == pytest.approx(optimum(weighted, reduction, maf))


@pytest.mark.parametrize("reduction,maf", CONFIGS)
def test_scaling_all_weights_scales_the_optimum(rng, reduction, maf):
    """The objective is linear in the tree weights, so the argmax cannot move."""
    trees = [random_tree(8, rng) for _ in range(3)]
    base = make_instance(trees, 4, weights=[1.0, 1.0, 1.0])
    scaled = make_instance(trees, 4, weights=[3.0, 3.0, 3.0])
    assert optimum(scaled, reduction, maf) == pytest.approx(3.0 * optimum(base, reduction, maf))


@pytest.mark.parametrize("reduction,maf", CONFIGS)
def test_single_tree_reduces_to_classical_pd(rng, reduction, maf):
    """
    On one tree the problem is ordinary budgeted PD, where greedy is optimal at unit costs.

    The greedy has to be *seeded with the best pair*: on an unrooted tree every single taxon has
    diversity zero, so a greedy started from the empty set picks its first taxon blind and can end
    up strictly worse than the optimum. Seeded, it matches.
    """
    from itertools import combinations

    from .fixtures.reference_pd import pd

    tree = random_tree(10, rng)
    budget = 4
    score = lambda picked: pd(tree, sum(1 << leaf for leaf in picked))  # noqa: E731
    chosen = set(max(combinations(range(tree.n_leaves), 2), key=score))
    while len(chosen) < budget:
        chosen.add(
            max(
                (leaf for leaf in range(tree.n_leaves) if leaf not in chosen),
                key=lambda leaf: score(chosen | {leaf}),
            )
        )
    assert optimum(make_instance([tree], budget), reduction, maf) == pytest.approx(score(chosen))


@pytest.mark.parametrize("reduction,maf", CONFIGS)
def test_costly_taxa_are_never_over_spent(rng, reduction, maf):
    """The witness always fits the budget, whatever the cost structure."""
    for _ in range(10):
        trees = [random_tree(9, rng) for _ in range(2)]
        costs = [rng.randint(1, 4) for _ in range(9)]
        budget = rng.randint(1, 10)
        case = make_instance(trees, budget, costs=costs)
        result = optimize(case, reduction, maf, return_taxa=True)
        ids = case.instance.registry.ids_of(result.taxa)
        assert sum(costs[i] for i in ids) <= budget
        assert case.instance.evaluate_pd(result.taxa) == pytest.approx(result.value)


def test_reductions_agree_on_larger_instances(rng):
    """
    The two cluster reductions must agree well beyond brute-force range.

    ``trivial`` gives one big cluster and a large ``k``; ``common`` shreds the same instance into
    many small ones. Identical answers is the strongest available evidence that the ``PD_C`` /
    ``PD_C^+`` bookkeeping between clusters is right.
    """
    for _ in range(6):
        case = related_instance(rng, n_leaves=14, n_trees=3, moves=1, budget=6)
        assert optimum(case, "common", "heuristic") == pytest.approx(
            optimum(case, "trivial", "heuristic")
        )
