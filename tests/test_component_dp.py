"""
Unit tests for the per-component DP (Lemma "compute-PD_i") and the signature weights.

The DP is checked against an independent exhaustive evaluation of the definitions in
:mod:`tests.fixtures.reference_dp` — enumerate leaf subsets and budget allocations, score each one
straight from Definition "two-sided-PD", keep the best. Agreement there is a much sharper statement
than end-to-end agreement, because it isolates this lemma from the rest of the pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from forestpd.algo.component_dp import component_table
from forestpd.algo.local import LocalInstance, ordinary_f2
from forestpd.algo.local_solver import signature_weights
from forestpd.algo.profile import build_profile
from forestpd.maf import HeuristicMAF

from .fixtures.random_instances import random_tree
from .fixtures.reference_dp import best_component_value, definitely_chosen_weight


def build(trees, budget, costs=None, complement=False, leaf_log=None):
    """Local instance + profile + heuristic forest for a set of trees."""
    n = trees[0].n_leaves
    cost = np.asarray(costs if costs else [1] * n, dtype=np.int64)
    local = LocalInstance(
        tuple(trees),
        np.ones(len(trees)),
        ordinary_f2(cost, budget, complement),
        cost,
        budget,
        leaf_log=leaf_log,
        complement=complement,
    )
    forest = HeuristicMAF().compute(local.trees)
    return local, build_profile(local, forest), forest


def table_args(profile, comp, weights, local, min_selected):
    """The arguments of ``component_table`` for one component under one signature."""
    _, lam_down, lam_up, lam_both = weights
    span = slice(int(profile.node_offset[comp]), int(profile.node_offset[comp + 1]))
    return (
        profile.comp_trees[comp],
        lam_down[span],
        lam_up[span],
        lam_both[span],
        profile.slot_survive_down[span],
        profile.slot_survive_up[span],
        local,
        min_selected,
    )


@pytest.mark.parametrize("complement", [False, True])
@pytest.mark.parametrize("trial", range(12))
def test_matches_exhaustive_enumeration(rng, trial, complement):
    """
    ``PD_i^{(S)}`` equals the best over all leaf subsets and table entries, for both ``l``, on
    both budget axes, with survival probabilities (so ``pi_{X_i}`` enters ``gamma``).
    """
    for _ in range(trial):
        rng.random()
    budget = rng.randint(2, 5)
    trees = [random_tree(6, rng) for _ in range(rng.randint(2, 3))]
    costs = [rng.randint(1, 2) for _ in range(6)]
    leaf_log = np.log([1 - 0.9 * rng.random() for _ in range(6)]) if trial % 2 else None
    local, profile, forest = build(trees, budget, costs, complement, leaf_log)

    for signature in range(1, 1 << profile.n_components):
        active = [c for c in range(profile.n_components) if (signature >> c) & 1]
        weights = signature_weights(profile, signature)
        for min_selected in (1, 2):
            for comp in active:
                args = table_args(profile, comp, weights, local, min_selected)
                got = component_table(*args)
                want = best_component_value(*args)
                np.testing.assert_allclose(got, want, err_msg=f"S={signature:b} comp={comp}")


def test_singleton_component_is_selectable_alone(rng):
    """
    A one-leaf component must be usable when other components are active.

    Singletons dominate real agreement forests; a DP that forbids ``|Q_i| = 1`` makes every
    signature containing one infeasible, which silently destroys the answer.
    """
    trees = [random_tree(5, rng) for _ in range(2)]
    budget = 3
    local, profile, _ = build(trees, budget)
    singles = [c for c in range(profile.n_components) if profile.comp_trees[c].n_nodes == 1]
    assert singles, "expected the heuristic forest to contain a singleton"
    signature = (1 << profile.n_components) - 1
    weights = signature_weights(profile, signature)
    table = component_table(*table_args(profile, singles[0], weights, local, 1))
    assert np.isfinite(table[1])


def test_singleton_component_alone_is_infeasible(rng):
    """The same singleton with ``|S| = 1`` cannot supply two leaves, so every entry is infeasible."""
    trees = [random_tree(5, rng) for _ in range(2)]
    budget = 3
    local, profile, _ = build(trees, budget)
    comp = next(c for c in range(profile.n_components) if profile.comp_trees[c].n_nodes == 1)
    weights = signature_weights(profile, 1 << comp)
    table = component_table(*table_args(profile, comp, weights, local, 2))
    assert not np.isfinite(table).any()


@pytest.mark.parametrize("trial", range(10))
def test_definitely_chosen_matches_the_characterization(rng, trial):
    """The vectorized ``d_S`` equals a direct scan over the characterization."""
    for _ in range(trial):
        rng.random()
    trees = [random_tree(7, rng) for _ in range(rng.randint(2, 3))]
    _, profile, _ = build(trees, 4)
    for signature in range(1, 1 << profile.n_components):
        got, _, _, _ = signature_weights(profile, signature)
        assert got == pytest.approx(definitely_chosen_weight(profile, signature))


def test_bilateral_weights_vanish_for_multi_component_signatures(rng):
    """
    With two or more active components every edge of ``H_{j,i}`` has one side covered for free.

    The components other than ``i`` are vertex-disjoint from ``H_{j,i}``, so each lies wholly on
    one side of every one of its edges — which is exactly why the bilateral term only survives at
    ``|S| = 1``.
    """
    trees = [random_tree(8, rng) for _ in range(3)]
    _, profile, _ = build(trees, 4)
    for signature in range(1, 1 << profile.n_components):
        active = bin(signature).count("1")
        _, lam_down, lam_up, lam_both = signature_weights(profile, signature)
        if active > 1:
            assert not lam_both.any(), f"S={signature:b}"
        else:
            assert not lam_down.any() and not lam_up.any(), f"S={signature:b}"


def test_cost_is_respected(rng):
    """A leaf costing more than the budget can never be funded."""
    trees = [random_tree(4, rng) for _ in range(2)]
    budget = 3
    local, profile, _ = build(trees, budget, costs=[9, 1, 1, 1])
    signature = (1 << profile.n_components) - 1
    weights = signature_weights(profile, signature)
    comp = next(
        c for c in range(profile.n_components) if 0 in [int(x) for x in profile.comp_leaves[c]]
    )
    table = component_table(*table_args(profile, comp, weights, local, 1))
    if len(profile.comp_leaves[comp]) == 1:
        assert not np.isfinite(table).any()
