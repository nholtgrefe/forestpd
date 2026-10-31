"""
One-call entry points.

Each function builds a :class:`~forestpd.instance.MaxMPDInstance` and calls one of its methods:

=============================  ===============================================================
:func:`optimize_maxmpd`        :meth:`~forestpd.instance.MaxMPDInstance.optimize`
:func:`optimize_many_maxmpd`   :meth:`~forestpd.instance.MaxMPDInstance.optimize_many`
:func:`evaluate_maxmpd`        :meth:`~forestpd.instance.MaxMPDInstance.evaluate_pd`
=============================  ===============================================================

Building an instance parses and validates the trees, so for repeated analyses of the same data
(several budgets, several taxon sets) build the instance once and call its methods instead.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from .io.convert import MissingBranchLength, Trees
from .instance import MaxMPDInstance, MaxMPDResult
from .maf import MAFComputer


def optimize_maxmpd(
    trees: Trees,
    budget: int,
    costs: Mapping[str, int] | None = None,
    weights: Sequence[float] | Mapping[int, float] | None = None,
    survival: Mapping[str, float] | None = None,
    *,
    return_taxa: bool = False,
    maf: MAFComputer | None = None,
    missing_branch_length: MissingBranchLength = "error",
    rooted: bool = False,
) -> MaxMPDResult:
    """
    Maximize phylogenetic diversity across several phylogenetic trees under a budget.

    Equivalent to ``MaxMPDInstance(trees, costs, weights, survival).optimize(budget)``. For repeated
    analyses on the same data (several budgets, or evaluating taxon sets), build the
    :class:`~forestpd.instance.MaxMPDInstance` once instead, so the trees are parsed only once.

    Parameters
    ----------
    trees : str | os.PathLike | Sequence
        The path of a file holding one Newick tree per line, or a list of Newick strings and/or
        PhyloZoo ``SemiDirectedPhyNetwork`` / ``DirectedPhyNetwork`` trees (with the ``phylozoo``
        extra). Every Newick tree ends with ``;``. All trees must carry the same taxon set.
    budget : int
        Total integer budget ``B`` (``>= 0``).
    costs : Mapping[str, int] | None, optional
        Integer cost per taxon label, at least 1. ``None`` means unit costs; missing labels cost 1.
    weights : Sequence[float] | Mapping[int, float] | None, optional
        Per-tree weight ``w_j``, in the order of the trees (the list, or the file), or keyed by
        tree index (missing indices weigh ``1.0``). ``None`` means all ``1.0``.
    survival : Mapping[str, float] | None, optional
        Probability ``a(x)`` in ``[0, 1)`` that taxon ``x`` survives when not protected; the
        objective is then the expected PD of the survivors. ``None`` means ``0`` for every taxon.
    return_taxa : bool, default False
        Also reconstruct an optimal taxon set.
    maf : MAFComputer | None, optional
        Agreement-forest stage. ``None`` (default) uses :class:`~forestpd.maf.HeuristicMAF`.
    missing_branch_length : str | float, default "error"
        Policy for edges without a branch length: ``"error"``, ``"zero"``, or a numeric default.
    rooted : bool, default False
        Use rooted PD (see :class:`~forestpd.instance.MaxMPDInstance`).

    Returns
    -------
    MaxMPDResult
        The optimal value, optionally an optimal taxon set, and diagnostics, including
        ``meta["frontier"]``: the optimal value for every budget ``0..budget`` (up to the total
        cost of all taxa; beyond it the value stays the same). A budget above half the total cost
        is solved with the complementary tables, which cover only the budgets from it on; the
        rest of the frontier is then ``NaN`` (see :class:`~forestpd.instance.MaxMPDResult`).

    Raises
    ------
    ForestPDInputError
        If any input is malformed, the trees do not share one taxon set, or the MAF stage does
        not return an agreement forest.

    Warns
    -----
    ForestPDWarning
        If the trees look rooted but ``rooted`` is False (see
        :class:`~forestpd.instance.MaxMPDInstance`).

    Examples
    --------
    >>> import forestpd as fpd
    >>> r = fpd.optimize_maxmpd(["(A,B,(C,D));", "(A,C,(B,D));"], budget=2,
    ...                         return_taxa=True, missing_branch_length=1.0)
    >>> r.value
    6.0
    >>> r.taxa in ({"A", "D"}, {"B", "C"})
    True
    """
    instance = MaxMPDInstance(
        trees,
        costs,
        weights,
        survival,
        missing_branch_length=missing_branch_length,
        rooted=rooted,
    )
    return instance.optimize(budget, return_taxa=return_taxa, maf=maf)


def optimize_many_maxmpd(
    trees: Trees,
    budgets: Iterable[int],
    costs: Mapping[str, int] | None = None,
    weights: Sequence[float] | Mapping[int, float] | None = None,
    survival: Mapping[str, float] | None = None,
    *,
    return_taxa: bool = False,
    maf: MAFComputer | None = None,
    missing_branch_length: MissingBranchLength = "error",
    rooted: bool = False,
) -> list[MaxMPDResult]:
    """
    :func:`optimize_maxmpd` for several budgets at once, from a single solve.

    Equivalent to ``MaxMPDInstance(trees, costs, weights, survival).optimize_many(budgets)``. For
    repeated analyses on the same data, build the :class:`~forestpd.instance.MaxMPDInstance` once
    instead.

    Parameters
    ----------
    trees : str | os.PathLike | Sequence
        As for :func:`optimize_maxmpd`.
    budgets : Iterable[int]
        The budgets (each ``>= 0``).
    costs, weights, survival, maf, missing_branch_length, rooted
        As for :func:`optimize_maxmpd`.
    return_taxa : bool, default False
        Also reconstruct an optimal taxon set for each budget.

    Returns
    -------
    list[MaxMPDResult]
        One result per budget, in the given order.

    Raises
    ------
    ForestPDInputError
        If any input is malformed, the trees do not share one taxon set, or the MAF stage does
        not return an agreement forest.

    Warns
    -----
    ForestPDWarning
        If the trees look rooted but ``rooted`` is False (see
        :class:`~forestpd.instance.MaxMPDInstance`).

    Examples
    --------
    >>> import forestpd as fpd
    >>> results = fpd.optimize_many_maxmpd(["(A:1,B:1,(C:1,D:1):2);", "(A:1,C:1,(B:1,D:1):2);"],
    ...                                    [1, 2, 3])
    >>> [r.value for r in results]
    [0.0, 8.0, 10.0]
    """
    instance = MaxMPDInstance(
        trees,
        costs,
        weights,
        survival,
        missing_branch_length=missing_branch_length,
        rooted=rooted,
    )
    return instance.optimize_many(budgets, return_taxa=return_taxa, maf=maf)


def evaluate_maxmpd(
    trees: Trees,
    taxa: Iterable[str],
    weights: Sequence[float] | Mapping[int, float] | None = None,
    survival: Mapping[str, float] | None = None,
    *,
    missing_branch_length: MissingBranchLength = "error",
    rooted: bool = False,
) -> float:
    """
    The objective value of protecting a given taxon set.

    Equivalent to ``MaxMPDInstance(trees, weights=weights, survival=survival).evaluate_pd(taxa)``.
    For repeated analyses on the same data (several taxon sets), build the
    :class:`~forestpd.instance.MaxMPDInstance` once instead.

    Parameters
    ----------
    trees : str | os.PathLike | Sequence
        As for :func:`optimize_maxmpd`.
    taxa : Iterable[str]
        The protected taxon labels.
    weights, survival, missing_branch_length, rooted
        As for :func:`optimize_maxmpd`.

    Returns
    -------
    float
        The weighted sum of phylogenetic diversities over all trees (its expected value over the
        surviving taxa, with survival probabilities).

    Raises
    ------
    ForestPDInputError
        If any input is malformed, a label is unknown, or the trees do not share one taxon set.

    Warns
    -----
    ForestPDWarning
        If the trees look rooted but ``rooted`` is False (see
        :class:`~forestpd.instance.MaxMPDInstance`).

    Examples
    --------
    >>> import forestpd as fpd
    >>> fpd.evaluate_maxmpd(["(A:1,B:1,(C:1,D:1):2);", "(A:1,C:1,(B:1,D:1):2);"], {"A", "B"})
    6.0
    """
    instance = MaxMPDInstance(
        trees,
        weights=weights,
        survival=survival,
        missing_branch_length=missing_branch_length,
        rooted=rooted,
    )
    return instance.evaluate_pd(taxa)


__all__ = ["optimize_maxmpd", "optimize_many_maxmpd", "evaluate_maxmpd"]
