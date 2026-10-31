"""
The complementary tables DP' (budget saved rather than allocated) and the choice between DP, DP' and
both.

Every choice must give the brute-force optimum at every budget it covers, with a feasible optimal
set, on instances with one and with several clusters, with and without survival probabilities,
costs with a common divisor and rooted PD. The automatic choice must follow the cost model, and the
budgets no computed table covers must be ``NaN`` in ``meta["frontier"]``.
"""

from __future__ import annotations

import math
import random
from itertools import combinations

import numpy as np
import pytest

import forestpd as fpd
from forestpd import MaxMPDInstance
from forestpd.algo import local_solver, solver
from forestpd.algo.solver import choose_tables, solve
from forestpd.maf import HeuristicMAF

from .fixtures.brute_force import reference_value
from .fixtures.random_instances import (
    clade_localized_instance,
    random_instance,
    random_survival,
    with_survival,
)

TABLES = ("DP", "DP'", "both")


@pytest.fixture
def force(monkeypatch):
    """Force the choice of tables for the rest of a test."""

    def set_tables(kind: str | None) -> None:
        monkeypatch.setattr(solver, "FORCE_TABLES", kind)

    return set_tables


def best_by_budget(instance: MaxMPDInstance) -> list[float]:
    """The brute-force optimum at every budget ``0 .. total cost``, from one enumeration."""
    labels = instance.taxa
    cost = {label: int(instance.costs[instance.registry.id_of(label)]) for label in labels}
    total = sum(cost.values())
    best = [-math.inf] * (total + 1)
    for size in range(len(labels) + 1):
        for subset in combinations(labels, size):
            at = sum(cost[label] for label in subset)
            best[at] = max(best[at], reference_value(instance, subset))
    for budget in range(1, total + 1):
        best[budget] = max(best[budget], best[budget - 1])
    return best


def check(instance: MaxMPDInstance, budgets: list[int]) -> list[fpd.MaxMPDResult]:
    """Solve ``budgets`` at once; every value, set and covered frontier entry is optimal."""
    best = best_by_budget(instance)
    results = instance.optimize_many(budgets, return_taxa=True)
    for budget, result in zip(budgets, results):
        want = best[min(budget, len(best) - 1)]
        assert result.value == pytest.approx(want, rel=1e-9, abs=1e-12), f"budget {budget}"
        assert result.taxa is not None
        spent = sum(int(instance.costs[instance.registry.id_of(x)]) for x in result.taxa)
        assert spent <= budget
        assert instance.evaluate_pd(result.taxa) == pytest.approx(want, rel=1e-9, abs=1e-12)
        for at, value in enumerate(result.meta["frontier"]):
            if not np.isnan(value):
                assert value == pytest.approx(best[at], rel=1e-9, abs=1e-12), f"frontier {at}"
    return results


@pytest.mark.parametrize("kind", TABLES)
@pytest.mark.parametrize("trial", range(10))
def test_every_choice_matches_brute_force(rng, force, kind, trial):
    """One cluster, random costs; survival probabilities on half of the trials."""
    for _ in range(trial):
        rng.random()
    case = random_instance(
        rng, n_leaves=rng.randint(4, 7), n_trees=rng.randint(1, 3), heterogeneous_costs=True
    )
    if trial % 2:
        case = with_survival(case, random_survival(rng, case.instance.n_taxa))
    instance = case.instance
    force(kind)
    results = check(instance, list(range(int(instance.costs.sum()) + 2)))
    assert results[0].meta["tables"] == kind


@pytest.mark.parametrize("kind", TABLES)
@pytest.mark.parametrize("trial", range(6))
def test_several_clusters(rng, force, kind, trial):
    """
    Clades that disagree inside but agree on the backbone: several clusters, so pseudo-leaves
    carry their child's DP' tables (``f_2(x_D, beta) = PD'_D^+(beta)``) and unselected children
    their ``Omega_D^0``.
    """
    for _ in range(trial):
        rng.random()
    case = clade_localized_instance(rng, n_clades=3, clade_size=4, n_trees=3)
    if trial % 2:
        case = with_survival(case, random_survival(rng, case.instance.n_taxa))
    instance = case.instance
    assert instance.structure().n_clusters > 1
    force(kind)
    check(instance, list(range(0, instance.n_taxa + 1)))


@pytest.mark.parametrize("trial", range(8))
def test_single_large_budgets_use_the_complement(rng, trial):
    """A single budget above half the total cost is solved by DP' alone, and is optimal."""
    for _ in range(trial):
        rng.random()
    case = random_instance(rng, n_leaves=6, n_trees=2, heterogeneous_costs=True)
    if trial % 2:
        case = with_survival(case, random_survival(rng, case.instance.n_taxa))
    instance = case.instance
    best = best_by_budget(instance)
    total = len(best) - 1
    for budget in range(total + 2):
        result = instance.optimize(budget, return_taxa=True)
        assert result.meta["tables"] == ("DP'" if budget > total - budget else "DP")
        assert result.value == pytest.approx(best[min(budget, total)], rel=1e-9, abs=1e-12)
        assert instance.evaluate_pd(result.taxa) == pytest.approx(result.value, rel=1e-9)


@pytest.mark.parametrize("kind", TABLES)
@pytest.mark.parametrize("trial", range(6))
def test_costs_with_a_common_divisor(rng, force, kind, trial):
    """Costs in units of their common divisor, on every table choice."""
    for _ in range(trial):
        rng.random()
    unit = (2, 3)[trial % 2]
    base = random_instance(rng, n_leaves=5, n_trees=2, heterogeneous_costs=True).instance
    costs = [unit * int(c) for c in base.costs]
    instance = MaxMPDInstance.from_phylotrees(base.trees, base.registry, costs, base.weights)
    force(kind)
    check(instance, list(range(sum(costs) + unit + 1)))


@pytest.mark.parametrize("kind", TABLES)
@pytest.mark.parametrize("trial", range(6))
def test_rooted(force, kind, trial):
    """Rooted PD: the root leaf costs one extra unit and is forced in, on DP' as on DP."""
    rng = random.Random(trial)
    labels = [f"t{i}" for i in range(rng.randint(3, 6))]
    trees = [_random_rooted(rng, labels) for _ in range(rng.randint(1, 3))]
    costs = {x: rng.randint(1, 3) for x in labels}
    survival = dict(zip(labels, random_survival(rng, len(labels)))) if trial % 2 else None
    instance = fpd.MaxMPDInstance(trees, costs, survival=survival, rooted=True)
    force(kind)
    check(instance, list(range(sum(costs.values()) + 2)))


def _random_rooted(rng: random.Random, labels: list[str]) -> str:
    """A random rooted binary Newick tree on ``labels``."""
    nodes = [f"{x}:{rng.randint(1, 9)}" for x in labels]
    while len(nodes) > 1:
        a = nodes.pop(rng.randrange(len(nodes)))
        b = nodes.pop(rng.randrange(len(nodes)))
        nodes.append(f"({a},{b}):{rng.randint(1, 9)}")
    return nodes[0] + ";"


def test_budget_at_and_above_the_total():
    """``B >= C`` leaves nothing to save: DP' has a single entry, and everything is protected."""
    instance = fpd.MaxMPDInstance(["(A:1,B:2,(C:3,D:4):5);", "(A:1,C:3,(B:2,D:4):5);"])
    everything = instance.evaluate_pd(instance.taxa)
    for budget in (4, 5, 100):
        result = instance.optimize(budget, return_taxa=True)
        assert result.meta["tables"] == "DP'"
        assert result.value == pytest.approx(everything)
        assert result.taxa == set(instance.taxa)


def test_frontier_is_nan_where_no_table_covers_it():
    """With both tables and a gap between the budgets, only the gap is NaN."""
    instance = random_instance(random.Random(3), n_leaves=10, n_trees=2).instance
    results = instance.optimize_many([1, 9])
    assert results[-1].meta["tables"] == "both"
    frontier = results[-1].meta["frontier"]
    assert not np.isnan(frontier[:2]).any() and not np.isnan(frontier[9])
    assert np.isnan(frontier[2:9]).all()


class TestChooseTables:
    def test_single_budget_takes_dp_prime_above_half(self):
        assert choose_tables([3], 10) == (3, None)
        assert choose_tables([5], 10) == (5, None)  # a tie goes to DP
        assert choose_tables([6], 10) == (-1, 6)
        assert choose_tables([12], 10) == (-1, 10)  # beyond the total: everything

    def test_a_small_curve_takes_dp(self):
        assert choose_tables(list(range(0, 21)), 100) == (20, None)

    def test_a_large_curve_takes_dp_prime(self):
        assert choose_tables(list(range(80, 101)), 100) == (-1, 80)

    def test_the_whole_range_takes_both(self):
        dp, start = choose_tables(list(range(0, 101)), 100)
        assert start == dp + 1 and dp in (49, 50)

    def test_a_range_up_to_sixty_percent_takes_dp(self):
        """``0.36 C^2`` for DP against ``0.5 C^2`` for both."""
        assert choose_tables(list(range(0, 61)), 100) == (60, None)

    def test_forcing_both_needs_two_budgets(self, force):
        force("both")
        with pytest.raises(ValueError, match="two distinct"):
            choose_tables([5], 10)


@pytest.mark.skipif(
    not local_solver._signature_kernel.HAVE_NUMBA, reason="the compiled loop needs Numba"
)
@pytest.mark.parametrize("trial", range(6))
def test_compiled_and_python_agree_on_the_complement(rng, monkeypatch, trial):
    """The compiled signature loop and the Python one give the same DP' tables."""
    for _ in range(trial):
        rng.random()
    case = clade_localized_instance(rng, n_clades=3, clade_size=4, n_trees=3)
    if trial % 2:
        case = with_survival(case, random_survival(rng, case.instance.n_taxa))
    instance = case.instance
    costs = instance.costs
    tree, _ = instance._reduction()
    frontiers = []
    for compiled in (True, False):
        monkeypatch.setattr(local_solver, "USE_COMPILED", compiled)
        solution = solve(
            tree,
            HeuristicMAF(),
            costs,
            instance.weights,
            -1,
            log_loss=instance._log_loss,
            complement_from=int(costs.sum()) // 3,
        )
        frontiers.append(solution.meta["frontier"])
    np.testing.assert_allclose(frontiers[0], frontiers[1], rtol=1e-12)
