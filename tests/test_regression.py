"""
Pinned baselines on a curated instance library.

Every value here was verified against :func:`brute_force_optimize_pd` when it was pinned, so a
change to any of them means the algorithm's answer moved — which is either a bug or a deliberate
change that must be re-justified against the oracle. This is the cheapest possible early warning:
the oracle tests are randomized, these are not.

``k`` and the cluster count are pinned too. They do not affect correctness, but a jump in either
means the cluster reduction or the MAF heuristic changed behaviour, which is worth noticing.
"""

from __future__ import annotations

import random

import pytest

from .fixtures.brute_force import brute_force_optimize_pd
from .fixtures.random_instances import (
    balanced,
    caterpillar,
    make_instance,
    perturbed_tree,
    random_tree,
    star,
)


def curated():
    """The instance library, built deterministically from fixed seeds."""
    base = random_tree(12, random.Random(4242))
    return {
        # A maximal polytomy against a maximally unbalanced tree.
        "caterpillar vs star": make_instance([caterpillar(6), star(6)], 3),
        # Perfectly congruent: the cluster reduction should shred it into congruent pieces.
        "balanced pair": make_instance([balanced(8), balanced(8)], 4),
        # One displaced taxon — the regime the FPT parameter is designed for.
        "one SPR apart": make_instance([base, perturbed_tree(base, random.Random(1), 1)], 5),
        # Enough disagreement that a real signature loop runs.
        "three SPR apart": make_instance([base, perturbed_tree(base, random.Random(2), 3)], 6),
        # Heterogeneous costs, so the budget convolution is genuinely exercised.
        "uneven costs": make_instance(
            [base, perturbed_tree(base, random.Random(3), 2)],
            7,
            costs=[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        ),
        # Three trees at different weights.
        "weighted trees": make_instance(
            [
                base,
                perturbed_tree(base, random.Random(4), 1),
                perturbed_tree(base, random.Random(5), 1),
            ],
            5,
            weights=[0.5, 1.5, 2.0],
        ),
    }


#: name -> (value, MAF size, cluster count). Verified against brute force when pinned.
BASELINE = {
    # k was 6 (all singletons) until the contraction learned to merge polytomy groups.
    "caterpillar vs star": (9.0, 4, 1),
    "balanced pair": (18.0, 1, 6),
    "one SPR apart": (130.0, 1, 10),
    "three SPR apart": (144.0, 4, 2),
    "uneven costs": (126.5, 2, 8),
    "weighted trees": (252.0, 3, 1),
}


@pytest.mark.parametrize("name", sorted(BASELINE))
def test_value_matches_the_baseline(name):
    instance, budget = curated()[name]
    result = instance.optimize(budget, return_taxa=True)
    value, _, _ = BASELINE[name]
    assert result.value == pytest.approx(value)
    # The pinned witness must still be feasible and still achieve the pinned value.
    ids = instance.registry.ids_of(result.taxa)
    assert sum(int(instance.costs[i]) for i in ids) <= budget
    assert instance.evaluate_pd(result.taxa) == pytest.approx(value)


@pytest.mark.parametrize("name", sorted(BASELINE))
def test_structure_matches_the_baseline(name):
    """``k`` and the cluster count: not correctness, but a tripwire on the two stages."""
    instance, budget = curated()[name]
    meta = instance.optimize(budget).meta
    _, maf_size, clusters = BASELINE[name]
    assert meta["max_maf_size"] == maf_size
    assert meta["n_clusters"] == clusters


@pytest.mark.parametrize("name", sorted(BASELINE))
def test_baseline_still_equals_brute_force(name):
    """Re-derive each pinned value from the oracle, so the baseline can never drift silently."""
    instance, budget = curated()[name]
    expected, _ = brute_force_optimize_pd(instance, budget)
    assert expected == pytest.approx(BASELINE[name][0])
