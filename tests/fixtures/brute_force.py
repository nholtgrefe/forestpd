"""
Brute-force MaxMPD solver: the correctness oracle of the test suite.

Enumerates every taxon subset within budget and evaluates the objective directly from the
definition. Exponential in the number of taxa, and that is the point: it shares no code with the
optimizer beyond the per-tree reference evaluation :func:`.reference_pd.expected_pd` (see
:func:`reference_value`), so agreement between the two is real evidence.

With survival probabilities the empty set and single taxa can have positive value, and in rooted
mode a set is evaluated with its root path, so every subset is enumerated.
"""

from __future__ import annotations

from itertools import combinations

import math

from .reference_pd import expected_pd
from forestpd.instance import MaxMPDInstance


def reference_value(instance: MaxMPDInstance, taxa) -> float:
    """
    The objective of protecting ``taxa``, tree by tree with :func:`expected_pd`.

    Independent of :meth:`MaxMPDInstance.evaluate_pd` (which groups the edges by split). In rooted
    mode the root leaf is protected and the root-leaf edge and stem corrected as documented there.
    """
    ids = instance.registry.ids_of(taxa)
    selected = bool(ids)
    loss = instance._log_loss
    if instance.rooted:
        ids.add(instance._root_leaf)
    mask = sum(1 << taxon for taxon in ids)
    value = sum(
        float(w) * expected_pd(tree, mask, loss)
        for w, tree in zip(instance.weights, instance.trees)
    )
    if not instance.rooted:
        return value
    survives = 1.0 if selected else -math.expm1(float(loss.sum()))
    return value + (instance._stem - instance._root_offset) * survives


def brute_force_optimize_pd(
    instance: MaxMPDInstance,
    budget: int,
    return_taxa: bool = False,
) -> tuple[float, set[str] | None]:
    """
    Exactly solve MaxMPD by enumeration.

    Parameters
    ----------
    instance : MaxMPDInstance
        The normalized instance.
    budget : int
        Total budget ``B``.
    return_taxa : bool, default False
        Also return an achieving taxon-label set.

    Returns
    -------
    tuple[float, set[str] | None]
        The optimal value and (optionally) a witness.
    """
    labels = instance.taxa
    cost = {label: int(instance.costs[instance.registry.id_of(label)]) for label in labels}
    best_value = reference_value(instance, ())
    best: tuple[str, ...] = ()
    for size in range(1, len(labels) + 1):
        for subset in combinations(labels, size):
            if sum(cost[label] for label in subset) > budget:
                continue
            value = reference_value(instance, subset)
            if value > best_value:
                best_value = value
                best = subset

    taxa = set(best) if return_taxa else None
    return best_value, taxa


__all__ = ["brute_force_optimize_pd", "reference_value"]
