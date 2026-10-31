"""
The local instance solved at one cluster.

A :class:`LocalInstance` is the input of Algorithm 3: the trees restricted to one cluster, on a
shared local leaf numbering, together with the tables ``f_2`` that give each leaf a value per
budget (Lemma "compute-PD_i" and Section "Cluster Reduction" of the manuscript).

Clusters exchange information only through ``f_2``. Each leaf has a cost ``c(x)`` and an upper
bound cost ``c'(x)``: ``c = c'`` for a taxon; a pseudo-leaf ``x_D`` has ``c(x_D) = 0`` and
``c'(x_D) = c(X_D)``, the cost of the taxa below the child cluster ``D``; the artificial leaf
``x_C^*`` (the manuscript's ``star_C``) costs nothing. The DP runs on one of two budget axes:

``DP`` (``complement=False``)
    ``beta`` is the budget allocated, and a table entry is the best value with *at most*
    ``beta`` allocated.
``DP'`` (``complement=True``)
    ``beta`` is the budget *not* allocated (saved), and an entry is the best value with *at least*
    ``beta`` saved. The tables run up to ``B' = c'(X) - B`` instead of ``B``, which is shorter
    when ``B`` is more than half the total cost.

The paper's ``f_2(x, 0)`` (the leaf not selected) is kept apart as :attr:`LocalInstance.idle`,
and :attr:`LocalInstance.f2` holds the value of the leaf *selected*, on the run's axis:

=================  ============================================  =================================
leaf               ``f2[x, beta]`` (DP)                          ``f2[x, beta]`` (DP')
=================  ============================================  =================================
taxon              ``0`` for ``beta >= c(x)``                    ``0`` at ``beta = 0``
``x_D``            ``PD_D^+(beta)``, ``1 <= beta <= B - 1``      ``PD'_D^+(beta)``, ``beta < c'(x_D)``
``x_C^*``          ``0`` at ``beta = 0``                         ``0`` at ``beta = 0``
=================  ============================================  =================================

and ``-inf`` elsewhere. ``idle`` is ``0`` for a taxon and ``Omega_D^0 = PD_D^+(0)`` for ``x_D`` (the
expected weight below ``D``'s cut when nothing below it is selected; zero without survival
probabilities). ``x_C^*`` is always selected, since its singleton component is forced into every
signature, so its ``idle`` (``0``) is never read. On DP' an unselected leaf saves exactly
``c'(x)``.

``f_2(x_D, B) = -inf`` as in the paper: if the whole budget is spent below ``D``, nothing is left
for the second leaf that every local solution must contain.

Each local leaf also carries ``leaf_log``, the summed ``log(1 - a(x))`` of the taxa it stands for:
one taxon, a child cluster's subtree, or (``x_C^*``) everything outside the cluster. The side
constants of every edge are sums of these.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.tree import PhyloTree

#: Value of an infeasible table entry.
NEG_INF = -np.inf


@dataclass(frozen=True)
class LocalInstance:
    """
    One cluster's problem: trees, per-leaf value tables, costs and budget.

    Attributes
    ----------
    trees : tuple[PhyloTree, ...]
        Local trees, all on leaves ``0..n_leaves-1``: one per input tree, or one per distinct
        local topology when merged (see :attr:`~forestpd.cluster.base.LocalPiece.tree_weights`).
    tree_weight : numpy.ndarray
        The weight of each local tree: the per-tree weights ``w_j``, or ``1.0`` for merged trees
        (whose branch lengths already include ``w_j``).
    f2 : numpy.ndarray
        ``float64[n_leaves, budget+1]``; the value of each leaf when selected, per ``beta`` on the
        run's axis (see the module docstring), ``-inf`` where it cannot be.
    cost : numpy.ndarray
        ``int64[n_leaves]``; ``c(x)``.
    budget : int
        The last index of the tables: on DP the budget ``B``, on DP' the saved budget
        ``B' = c'(X) - B``; both capped at the cost of the taxa at or below the cluster.
    leaf_log : numpy.ndarray | None
        ``float64[n_leaves]``; the summed ``log(1 - a(x))`` of the taxa each leaf stands for.
        ``None`` means no survival probabilities (all zero). The leaves must stand for *all* taxa
        for the side constants to be right, so a non-root cluster's ``I_C`` (without ``x_C^*``)
        is profiled through ``I_C^+`` (see :func:`~forestpd.algo.profile.plain_profile_from_plus`).
    idle : numpy.ndarray | None
        ``float64[n_leaves]``; the paper's ``f_2(x, 0)``, the value of each leaf when not selected.
        ``None`` means all zero.
    cost_max : numpy.ndarray | None
        ``int64[n_leaves]``; the upper bound cost ``c'(x)``, what an unselected leaf saves on DP'.
        ``None`` means ``cost``.
    complement : bool
        Whether the tables are on the DP' axis (budget saved, at least ``beta``) rather than the DP
        axis (budget allocated, at most ``beta``).
    """

    trees: tuple[PhyloTree, ...]
    tree_weight: np.ndarray
    f2: np.ndarray
    cost: np.ndarray
    budget: int
    leaf_log: np.ndarray | None = None
    idle: np.ndarray | None = None
    cost_max: np.ndarray | None = None
    complement: bool = False

    @property
    def n_leaves(self) -> int:
        """Number of local leaves."""
        return self.trees[0].n_leaves

    @property
    def leaf_log_or_zero(self) -> np.ndarray:
        """``leaf_log``, with ``None`` read as all zero."""
        if self.leaf_log is None:
            return np.zeros(self.n_leaves)
        return self.leaf_log

    @property
    def idle_or_zero(self) -> np.ndarray:
        """``idle``, with ``None`` read as all zero."""
        if self.idle is None:
            return np.zeros(self.n_leaves)
        return self.idle

    @property
    def cost_max_or_cost(self) -> np.ndarray:
        """``cost_max``, with ``None`` read as ``cost``."""
        if self.cost_max is None:
            return self.cost
        return self.cost_max


def ordinary_f2(costs: np.ndarray, budget: int, complement: bool = False) -> np.ndarray:
    """
    The ``f2`` rows of plain taxa: their value when selected, which is ``0`` where affordable.

    Parameters
    ----------
    costs : numpy.ndarray
        ``int64[n]``; cost ``c(x)`` per leaf.
    budget : int
        The last table index (``B`` on DP, ``B'`` on DP').
    complement : bool, default False
        Whether the rows are on the DP' axis.

    Returns
    -------
    numpy.ndarray
        ``float64[n, budget+1]``. On DP ``0`` at ``beta >= c(x)``; on DP' ``0`` at ``beta = 0``
        (a selected taxon saves nothing of its own cost); ``-inf`` elsewhere.

    Examples
    --------
    >>> ordinary_f2(np.array([2]), 3)
    array([[-inf, -inf,   0.,   0.]])
    >>> ordinary_f2(np.array([2]), 3, complement=True)
    array([[  0., -inf, -inf, -inf]])
    """
    table = np.full((len(costs), budget + 1), NEG_INF)
    if complement:
        table[:, 0] = 0.0
        return table
    for leaf, cost in enumerate(costs):
        if cost <= budget:
            table[leaf, int(cost) :] = 0.0
    return table


def pseudo_leaf_f2(
    child_table: np.ndarray, width: int, budget: int, complement: bool = False
) -> np.ndarray:
    """
    Turn a child cluster's ``PD_D^+`` table into the pseudo-leaf's ``f2`` row.

    Parameters
    ----------
    child_table : numpy.ndarray
        ``PD_D^+`` on the run's axis, as computed for the child cluster.
    width : int
        Length of the row: the parent cluster's last table index plus one.
    budget : int
        The DP's budget ``B``; ignored on DP'.
    complement : bool, default False
        Whether the tables are on the DP' axis.

    Returns
    -------
    numpy.ndarray
        ``float64[width]`` ready to be dropped into :attr:`LocalInstance.f2`. On DP, ``x_D`` is
        selected only with positive budget and never with all of ``B`` (the paper's
        ``f_2(x_D, B) = -inf``); on DP' every entry of the child's table carries over.
    """
    row = np.full(width, NEG_INF)
    span = min(width, child_table.size)
    if complement:
        row[:span] = child_table[:span]
        return row
    top = min(span, budget)  # entries 1 .. B - 1
    if top > 1:
        row[1:top] = child_table[1:top]
    return row


__all__ = ["LocalInstance", "NEG_INF", "ordinary_f2", "pseudo_leaf_f2"]
