"""
Validation of the instance data: per-tree weights, taxon costs, survival probabilities and budgets.

Shared by :class:`~forestpd.instance.MaxMPDInstance`, :mod:`forestpd.utils` and
:mod:`forestpd.io.splits`; every function raises :class:`~forestpd.exceptions.ForestPDInputError`
on invalid input.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from .core.taxa import TaxonRegistry
from .core.tree import PhyloTree, has_degree_two
from .exceptions import ForestPDInputError
from .io.newick import ROOT_LABEL


def weights_to_array(
    weights: Sequence[float] | Mapping[int, float] | None, n_trees: int
) -> np.ndarray:
    """
    Validate per-tree weights ``w_j`` and return them as an array (no rescaling).

    Parameters
    ----------
    weights : Sequence[float] | Mapping[int, float] | None
        Per-tree weights, either aligned with the trees or keyed by tree index (missing indices
        weigh ``1.0``). ``None`` means all ``1.0``.
    n_trees : int
        Number of trees ``t``.

    Returns
    -------
    numpy.ndarray
        ``float64[t]`` array of weights.

    Raises
    ------
    ForestPDInputError
        If a weight is negative or non-finite, a mapping key is not a tree index, or the sequence
        length is wrong.
    """
    if weights is None:
        return np.ones(n_trees, dtype=np.float64)
    if isinstance(weights, Mapping):
        for key in weights:
            if (
                isinstance(key, (bool, np.bool_))
                or not isinstance(key, (int, np.integer))
                or not 0 <= key < n_trees
            ):
                raise ForestPDInputError(
                    f"Weight given for tree {key!r}, but trees are indexed 0..{n_trees - 1}"
                )
        values = [float(weights.get(j, 1.0)) for j in range(n_trees)]
    else:
        values = [float(w) for w in weights]
        if len(values) != n_trees:
            raise ForestPDInputError(f"Expected {n_trees} weights, got {len(values)}")
    for j, w in enumerate(values):
        if not math.isfinite(w) or w < 0:
            raise ForestPDInputError(
                f"Weight for tree {j} must be finite and non-negative, got {w}"
            )
    return np.asarray(values, dtype=np.float64)


def costs_to_array(
    registry: TaxonRegistry, costs: Mapping[str, int] | Sequence[int] | None
) -> np.ndarray:
    """
    Validate taxon costs and return them as an array indexed by taxon id.

    Parameters
    ----------
    registry : TaxonRegistry
        The taxon registry fixing the id of each label.
    costs : Mapping[str, int] | Sequence[int] | None
        Integer cost per taxon label (missing labels cost 1), or a sequence of costs indexed by
        taxon id. ``None`` means unit costs.

    Returns
    -------
    numpy.ndarray
        ``int64[n]`` array of costs, indexed by taxon id.

    Raises
    ------
    ForestPDInputError
        If a cost is given for an unknown label, is not a positive integer, or a sequence
        has the wrong length.
    """
    result = np.ones(registry.n, dtype=np.int64)
    if costs is None:
        return result
    if isinstance(costs, Mapping):
        items = [
            (label, registry.id_of(label) if _user_label(label, registry) else -1, cost)
            for label, cost in costs.items()
        ]
    else:
        values = list(costs)
        if len(values) != registry.n:
            raise ForestPDInputError(f"Expected {registry.n} costs, got {len(values)}")
        items = [(registry.label_of(i), i, cost) for i, cost in enumerate(values)]
    for label, taxon, cost in items:
        if taxon < 0:
            raise ForestPDInputError(f"Cost given for unknown taxon: {label!r}")
        if isinstance(cost, (bool, np.bool_)) or not isinstance(cost, (int, np.integer)):
            raise ForestPDInputError(
                f"Cost for taxon {label!r} must be an integer, got {type(cost).__name__}"
            )
        if cost < 1:
            raise ForestPDInputError(f"Cost for taxon {label!r} must be at least 1, got {cost}")
        result[taxon] = cost
    return result


def survival_to_array(
    registry: TaxonRegistry, survival: Mapping[str, float] | Sequence[float] | None
) -> np.ndarray:
    """
    Validate survival probabilities and return them as an array indexed by taxon id.

    Parameters
    ----------
    registry : TaxonRegistry
        The taxon registry fixing the id of each label.
    survival : Mapping[str, float] | Sequence[float] | None
        Probability per taxon label (missing labels have ``0``), or a sequence indexed by taxon
        id. ``None`` means ``0`` for every taxon.

    Returns
    -------
    numpy.ndarray
        ``float64[n]`` array of probabilities, indexed by taxon id.

    Raises
    ------
    ForestPDInputError
        If a probability is given for an unknown label, is not a number in ``[0, 1)``, or a
        sequence has the wrong length.
    """
    result = np.zeros(registry.n, dtype=np.float64)
    if survival is None:
        return result
    if isinstance(survival, Mapping):
        items = [
            (label, registry.id_of(label) if _user_label(label, registry) else -1, value)
            for label, value in survival.items()
        ]
    else:
        values = list(survival)
        if len(values) != registry.n:
            raise ForestPDInputError(
                f"Expected {registry.n} survival probabilities, got {len(values)}"
            )
        items = [(registry.label_of(i), i, value) for i, value in enumerate(values)]
    for label, taxon, value in items:
        if taxon < 0:
            raise ForestPDInputError(f"Survival probability given for unknown taxon: {label!r}")
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ForestPDInputError(
                f"Survival probability for taxon {label!r} must be a number, "
                f"got {type(value).__name__}"
            )
        if not (0.0 <= float(value) < 1.0):
            raise ForestPDInputError(
                f"Survival probability for taxon {label!r} must lie in [0, 1), got {value}"
            )
        result[taxon] = float(value)
    return result


def validate_budget(budget: int) -> None:
    """
    Raise :class:`ForestPDInputError` unless ``budget`` is a non-negative integer.

    Parameters
    ----------
    budget : int
        The total budget ``B``.

    Raises
    ------
    ForestPDInputError
        If ``budget`` is not an integer (Python or NumPy; ``bool`` is rejected) or is negative.
    """
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)):
        raise ForestPDInputError(f"budget must be an integer, got {type(budget).__name__}")
    if budget < 0:
        raise ForestPDInputError(f"budget must be non-negative, got {budget}")


def validate_uniform(trees: Sequence[PhyloTree], registry: TaxonRegistry) -> None:
    """
    Check that every tree has the same leaf set, namely all taxa in ``registry``, and no
    internal vertex of degree 2 (trees read from Newick or PhyloZoo never have one).

    Parameters
    ----------
    trees : Sequence[PhyloTree]
        The input trees.
    registry : TaxonRegistry
        The registry of all taxa in the instance.

    Raises
    ------
    ForestPDInputError
        If the trees do not all span the same taxon set, or one has an internal vertex of
        degree 2.
    """
    for j, tree in enumerate(trees):
        if tree.n_leaves != registry.n:
            raise ForestPDInputError(
                f"Tree {j} has {tree.n_leaves} taxa but the instance has {registry.n}. "
                "All trees must carry the same taxon set."
            )
        if has_degree_two(tree):
            raise ForestPDInputError(
                f"Tree {j} has an internal vertex of degree 2; suppress it first with "
                "forestpd.core.tree.suppress_degree_two (this does not change any PD)."
            )


def taxon_ids(registry: TaxonRegistry, taxa: Iterable[str]) -> set[int]:
    """
    The ids of a set of taxon labels given by a user.

    Parameters
    ----------
    registry : TaxonRegistry
        The taxon registry.
    taxa : Iterable[str]
        Taxon labels; a single string is rejected rather than read as its characters.

    Returns
    -------
    set[int]
        Their ids.

    Raises
    ------
    ForestPDInputError
        If ``taxa`` is a string, or a label is unknown (the reserved root label included).
    """
    if isinstance(taxa, str):
        raise ForestPDInputError(
            f"Expected a collection of taxon labels, got the string {taxa!r}; "
            f"wrap a single label as [{taxa!r}]."
        )
    ids = set()
    for label in taxa:
        if not _user_label(label, registry):
            raise ForestPDInputError(f"Unknown taxon: {label!r}")
        ids.add(registry.id_of(label))
    return ids


def _user_label(label: object, registry: TaxonRegistry) -> bool:
    """Whether ``label`` names a taxon a user may refer to (not the reserved root leaf)."""
    return label != ROOT_LABEL and label in registry


__all__ = [
    "taxon_ids",
    "weights_to_array",
    "costs_to_array",
    "survival_to_array",
    "validate_budget",
    "validate_uniform",
]
