"""
The manuscript's worked example, checked number for number.

The figure fixes an agreement forest, a signature, a definitely-chosen edge set and a full table of
two-sided weights, then evaluates one taxon set by hand. Reproducing those exact values pins the
implementation to the paper rather than to itself, so this file is the first thing to run after
touching :mod:`forestpd.algo.profile` or :mod:`forestpd.algo.component_dp`.
"""

from __future__ import annotations

import numpy as np
import pytest

from forestpd.algo.component_dp import component_table
from forestpd.algo.local import LocalInstance, ordinary_f2
from forestpd.algo.local_solver import signature_weights
from forestpd.algo.profile import build_profile
from .fixtures.reference_pd import pd

from .fixtures.brute_force import brute_force_optimize_pd
from .fixtures.random_instances import make_instance
from .fixtures.reference_dp import two_sided_pd

#: The forest of the figure: X_1 = {a,b,d}, X_2 = {c}, X_3 = {e,f}.
FOREST = ((0, 1, 3), (2,), (4, 5))
BUDGET = 3


@pytest.fixture
def setup(paper_trees):
    """Build the local instance, profile and all-active signature weights of the figure."""
    local = LocalInstance(
        paper_trees,
        np.ones(2),
        ordinary_f2(np.ones(6, dtype=np.int64), BUDGET),
        np.ones(6, dtype=np.int64),
        BUDGET,
    )
    profile = build_profile(local, FOREST)
    signature = 0b111
    d_s, lam_down, lam_up, lam_both = signature_weights(profile, signature)
    return local, profile, d_s, lam_down, lam_up, lam_both


def test_agreement_forest_of_the_figure_is_valid(paper_trees):
    """The figure asserts these three blocks form an agreement forest; verify it independently."""
    from forestpd.maf.partition import is_agreement_forest

    comp_of_leaf = np.array([0, 0, 1, 0, 2, 2])
    assert is_agreement_forest(list(paper_trees), comp_of_leaf, 3)


def test_definitely_chosen_weight(setup):
    """The red edges of the figure: 1+5+4 in the first tree, 2+3+1 in the second."""
    _, _, d_s, _, _, _ = setup
    assert d_s == pytest.approx(16.0)


def test_two_sided_weight_table(setup):
    """
    The figure's table of directional weights for the component ``F_1``.

    ``R_1`` is rooted by subdividing the pendant edge of the smallest leaf, so node 2 carries the
    edge ``{a,z}``, node 3 the edge ``{z,b}`` and node 4 the edge ``{z,d}``. "Down" points away
    from ``a``, which is why ``{a,z}`` reads (2, 7) rather than (7, 2).
    """
    _, profile, _, lam_down, lam_up, lam_both = setup
    lo, hi = int(profile.node_offset[0]), int(profile.node_offset[1])
    assert list(lam_down[lo:hi]) == [0.0, 0.0, 2.0, 6.0, 6.0]
    assert list(lam_up[lo:hi]) == [0.0, 0.0, 7.0, 0.0, 3.0]
    # |S| > 1, so every bilateral weight vanishes: some other active component always covers a side.
    assert not lam_both[lo:hi].any()


def test_pd_of_single_taxon_matches_the_paper(setup):
    """``PD^(2)`` of ``{a}`` in ``F_1`` is 7 + 0 + 3 = 10, as computed in the manuscript."""
    _, profile, _, lam_down, lam_up, lam_both = setup
    lo, hi = int(profile.node_offset[0]), int(profile.node_offset[1])
    value = two_sided_pd(
        profile.comp_trees[0], lam_down[lo:hi], lam_up[lo:hi], lam_both[lo:hi], {0}
    )
    assert value == pytest.approx(10.0)


def test_component_dp_admits_a_single_taxon(setup):
    """
    With ``|S| >= 2`` a component may contribute exactly one taxon.

    This is the case the manuscript's own example needs and the one an ``o``-guard tied to the
    component instead of the local instance would wrongly forbid: at budget 1 only one leaf is
    affordable, so a ``-inf`` here would mean the whole example is unreachable.
    """
    local, profile, _, lam_down, lam_up, lam_both = setup
    lo, hi = int(profile.node_offset[0]), int(profile.node_offset[1])
    table = component_table(
        profile.comp_trees[0],
        lam_down[lo:hi],
        lam_up[lo:hi],
        lam_both[lo:hi],
        profile.slot_survive_down[lo:hi],
        profile.slot_survive_up[lo:hi],
        local,
        min_selected=1,
    )
    assert np.isfinite(table[1])
    # {b} scores 2 + 6 + 3 = 11, beating {a}'s 10 and {d}'s 6 + 2 = 8.
    assert table[1] == pytest.approx(11.0)


def test_component_dp_refuses_a_single_taxon_when_alone(setup):
    """With ``|S| = 1`` the component must supply two leaves itself, so budget 1 is infeasible."""
    local, profile, _, lam_down, lam_up, lam_both = setup
    lo, hi = int(profile.node_offset[0]), int(profile.node_offset[1])
    table = component_table(
        profile.comp_trees[0],
        lam_down[lo:hi],
        lam_up[lo:hi],
        lam_both[lo:hi],
        profile.slot_survive_down[lo:hi],
        profile.slot_survive_up[lo:hi],
        local,
        min_selected=2,
    )
    assert table[1] == -np.inf


def test_decomposition_reproduces_the_direct_score(setup, paper_trees):
    """
    The whole point of the construction: ``d_S`` plus the per-component scores is the real PD.

    For ``{a, c, e}``: 16 + 10 + 0 + 7 = 33, and summing Steiner-tree weights in both input trees
    independently gives 18 + 15 = 33.
    """
    _, profile, d_s, lam_down, lam_up, lam_both = setup
    selected = {0, 2, 4}  # a, c, e
    total = d_s
    for comp, block in enumerate(FOREST):
        lo, hi = int(profile.node_offset[comp]), int(profile.node_offset[comp + 1])
        total += two_sided_pd(
            profile.comp_trees[comp],
            lam_down[lo:hi],
            lam_up[lo:hi],
            lam_both[lo:hi],
            selected & set(block),
        )
    mask = sum(1 << t for t in selected)
    direct = sum(pd(tree, mask) for tree in paper_trees)
    assert total == pytest.approx(33.0)
    assert direct == pytest.approx(33.0)


def test_full_optimum_of_the_figure(paper_trees):
    """End to end on the figure's instance, against the brute-force oracle."""
    instance, _ = make_instance(list(paper_trees), budget=BUDGET)
    expected, _ = brute_force_optimize_pd(instance, BUDGET)
    result = instance.optimize(BUDGET, return_taxa=True)
    assert result.value == pytest.approx(expected)
    assert instance.evaluate_pd(result.taxa) == pytest.approx(expected)
