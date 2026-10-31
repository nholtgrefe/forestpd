"""
The budgeted DP within one MAF component (Lemma "compute-PD_i").

Computes ``PD_i^{(S)}(beta)``, the best two-sided PD within one component, on the local instance's
budget axis (see :mod:`~forestpd.algo.local`): with at most ``beta`` allocated to the component
(the manuscript's ``DP``), or with at least ``beta`` of its cost saved (``DP'``). The manuscript's
table is ``DP[v, l, s_up, s_down, beta]``:

``l``
    how many leaves the component must contribute: more than ``l``, with ``l = [|S| = 1]``.
``s_up``
    whether a leaf of this component is selected outside the subtree of ``v``. It decides whether
    the edge's weight towards the outside counts.
``s_down``
    whether a leaf in the subtree of ``v`` is selected.

Every local solution must select at least two leaves of the local instance; otherwise a
pseudo-leaf value ``PD_D^+``, which assumes something outside ``D`` is selected, would be counted
without that being the case. If ``|S| >= 2`` this holds automatically, since every active component
contributes a leaf. If ``|S| = 1``, the single component must contribute two leaves. (Requiring two
leaves from every component instead would exclude singleton components entirely, and most
components of a forest are singletons.) ``l`` is constant over the recursion, so it is a parameter
(``min_selected = l + 1``) that selects the leaf initialization rather than a DP state.

Each edge of ``R_i`` (named by its lower endpoint ``v``) contributes ``gamma_v(q, s)``, where
``q = s_down(v)`` and ``s`` is ``v``'s ``s_up``:

    ``gamma_v(q, s) = lambda_down alpha_down(q) + lambda_up alpha_up(s)
    + lambda_both alpha_down(q) alpha_up(s)``,

with ``alpha(1) = 1`` and ``alpha(0) = pi_{X_i}``, the probability that an unselected taxon of the
component on that side survives (zero without survival probabilities).

Shortcuts, all leaving the table unchanged:

* The ``s_down = 0`` entries describe one configuration (nothing selected below ``v``), so their
  value does not depend on ``beta`` over the range where they are feasible: every ``beta`` on DP,
  ``beta <= c'(R_v)`` on DP', where the unselected subtree saves exactly the upper bound cost of its
  leaves. They are stored as one number per node and ``s_up`` (``empty``), with that offset.
* When both children are occupied, both child states are ``s_up = 1`` and the ``gamma`` terms are
  constants, so this case does not depend on ``s_up``: one convolution per node serves both values.

The running time is ``O(n_i * B^2)`` on DP and ``O(n_i * B'^2)`` on DP'.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .convolve import max_plus, saved_shift
from .local import NEG_INF, LocalInstance
from .profile import ComponentTree


class ComponentDP(NamedTuple):
    """
    A filled component table.

    Attributes
    ----------
    occ : numpy.ndarray
        ``float64[n_nodes, 2, width]``; ``DP[v, l, s_up, 1, beta]``, indexed ``[v, s_up, beta]``.
    empty : numpy.ndarray
        ``float64[n_nodes, 2]``; ``DP[v, l, s_up, 0, beta]``, the same for every feasible ``beta``.
    offset : numpy.ndarray
        ``int64[n_nodes]``; what an unselected subtree adds on the budget axis: ``0`` on DP,
        ``c'(R_v)`` on DP'.
    gamma : list[tuple[tuple[float, float], tuple[float, float]]]
        ``gamma_v(q, s)`` of the edge above each node, as ``gamma[v][q][s]`` (see :func:`gammas`).
    """

    occ: np.ndarray
    empty: np.ndarray
    offset: np.ndarray
    gamma: list[tuple[tuple[float, float], tuple[float, float]]]


def component_table(
    comp_tree: ComponentTree,
    lam_down: np.ndarray,
    lam_up: np.ndarray,
    lam_both: np.ndarray,
    survive_down: np.ndarray,
    survive_up: np.ndarray,
    local: LocalInstance,
    min_selected: int,
) -> np.ndarray:
    """
    Compute ``PD_i^{(S)}(0..width-1)`` for one component.

    Parameters
    ----------
    comp_tree : ComponentTree
        The rooted binary ``R_i``.
    lam_down, lam_up, lam_both : numpy.ndarray
        The two-sided weights of this component's edges, indexed by ``R_i`` node (an edge is named
        by its lower endpoint). ``lam_down`` pays when the subtree below keeps a survivor,
        ``lam_up`` when the other side does, ``lam_both`` only when both do.
    survive_down, survive_up : numpy.ndarray
        ``pi_{X_i}`` below and above each edge, indexed by ``R_i`` node
        (:attr:`~forestpd.algo.profile.ClusterProfile.slot_survive_down`).
    local : LocalInstance
        The local instance: leaf tables, upper bound costs and budget axis.
    min_selected : int
        Minimum number of leaves the component must contribute (1 or 2).

    Returns
    -------
    numpy.ndarray
        ``float64[local.budget + 1]``; ``-inf`` where infeasible.
    """
    dp = component_dp_table(
        comp_tree, lam_down, lam_up, lam_both, survive_down, survive_up, local, min_selected
    )
    return root_table(dp, comp_tree)


def root_table(dp: ComponentDP, comp_tree: ComponentTree) -> np.ndarray:
    """
    ``PD_i^{(S)}`` read from a filled :func:`component_dp_table`, as :func:`component_table`.

    The component is active, so its root is occupied, with nothing of it outside (``s_up = 0``).

    Parameters
    ----------
    dp : ComponentDP
        The filled table.
    comp_tree : ComponentTree
        The rooted binary ``R_i`` it was filled for.

    Returns
    -------
    numpy.ndarray
        ``float64[width]``.
    """
    table: np.ndarray = dp.occ[comp_tree.root, 0].copy()
    return table


def node_offsets(comp_tree: ComponentTree, local: LocalInstance) -> np.ndarray:
    """
    ``int64[n_nodes]``; what an unselected subtree adds on the budget axis.

    Zero on DP; on DP' the upper bound cost ``c'`` of the leaves below each node.
    """
    offset = np.zeros(comp_tree.n_nodes, dtype=np.int64)
    if not local.complement:
        return offset
    cost_max = local.cost_max_or_cost
    for node in comp_tree.postorder.tolist():
        leaf = int(comp_tree.leaf[node])
        if leaf >= 0:
            offset[node] = int(cost_max[leaf])
        else:
            offset[node] = offset[int(comp_tree.left[node])] + offset[int(comp_tree.right[node])]
    return offset


def component_dp_table(
    comp_tree: ComponentTree,
    lam_down: np.ndarray,
    lam_up: np.ndarray,
    lam_both: np.ndarray,
    survive_down: np.ndarray,
    survive_up: np.ndarray,
    local: LocalInstance,
    min_selected: int,
) -> ComponentDP:
    """
    Fill the whole table in postorder.

    Parameters
    ----------
    comp_tree : ComponentTree
        The rooted binary ``R_i``.
    lam_down, lam_up, lam_both : numpy.ndarray
        The two-sided weights of this component's edges, indexed by ``R_i`` node.
    survive_down, survive_up : numpy.ndarray
        ``pi_{X_i}`` below and above each edge, indexed by ``R_i`` node.
    local : LocalInstance
        The local instance.
    min_selected : int
        Minimum number of leaves the component must contribute (1 or 2).

    Returns
    -------
    ComponentDP
        The filled table, ``-inf`` where infeasible.
    """
    width = local.budget + 1
    dp = ComponentDP(
        occ=np.full((comp_tree.n_nodes, 2, width), NEG_INF),
        empty=np.full((comp_tree.n_nodes, 2), NEG_INF),
        offset=node_offsets(comp_tree, local),
        gamma=gammas(lam_down, lam_up, lam_both, survive_down, survive_up),
    )
    idle = local.idle_or_zero
    for node in comp_tree.postorder:
        node = int(node)
        leaf = int(comp_tree.leaf[node])
        if leaf >= 0:
            _init_leaf(dp, node, leaf, local.f2, idle, min_selected)
        else:
            _combine(dp, node, comp_tree, width)
    return dp


def _init_leaf(
    dp: ComponentDP,
    node: int,
    leaf: int,
    f2: np.ndarray,
    idle: np.ndarray,
    min_selected: int,
) -> None:
    """
    Initialize the DP rows of a leaf node.

    ============  ========================================  ==========================
    ``s_up``      selected (``occ``)                        not selected (``empty``)
    ============  ========================================  ==========================
    ``1``         ``f2[x]``                                 ``idle[x]``
    ``0``         ``f2[x]`` if ``min_selected == 1``        ``-inf``
    ============  ========================================  ==========================

    ``empty[x, 0]`` is ``-inf``: nothing selected below and nothing of the component outside means
    an empty component, which an active component is not.

    Parameters
    ----------
    dp : ComponentDP
        The table, written in place at ``node``.
    node : int
        The ``R_i`` node holding the leaf.
    leaf : int
        The local leaf id at ``node``.
    f2 : numpy.ndarray
        ``float64[n_leaves, width]``; the value of each leaf when selected.
    idle : numpy.ndarray
        ``float64[n_leaves]``; the value of each leaf when not selected.
    min_selected : int
        Minimum number of leaves the component must contribute (1 or 2).
    """
    dp.occ[node, 1] = f2[leaf]
    if min_selected == 1:
        dp.occ[node, 0] = f2[leaf]
    dp.empty[node, 1] = idle[leaf]


def _combine(dp: ComponentDP, node: int, comp_tree: ComponentTree, width: int) -> None:
    """
    Fill an internal node's DP rows from its two children.

    Takes the best over the four regimes: both children occupied, only the left, only the right, or
    neither. A child left unselected adds its offset on the budget axis (nothing on DP, its whole
    upper bound cost on DP').

    Parameters
    ----------
    dp : ComponentDP
        The table; the children's rows are read, ``node``'s rows written.
    node : int
        The internal ``R_i`` node.
    comp_tree : ComponentTree
        The rooted binary ``R_i``.
    width : int
        Length of the tables.
    """
    occ, empty, offset, gamma = dp
    w1 = int(comp_tree.left[node])
    w2 = int(comp_tree.right[node])
    g1 = gamma[w1]
    g2 = gamma[w2]

    shared = max_plus(occ[w1, 1], occ[w2, 1], width) + (g1[1][1] + g2[1][1])
    left = saved_shift(occ[w1], int(offset[w2]))  # both s_up rows, shifted by w2's offset
    right = saved_shift(occ[w2], int(offset[w1]))

    for s in (0, 1):
        out = shared.copy()
        left_only = left[s] + (g1[1][s] + empty[w2, 1] + g2[0][1])
        np.maximum(out, left_only, out=out)
        right_only = right[s] + (g2[1][s] + empty[w1, 1] + g1[0][1])
        np.maximum(out, right_only, out=out)
        occ[node, s] = out
        empty[node, s] = empty[w1, s] + g1[0][s] + empty[w2, s] + g2[0][s]


def gammas(
    lam_down: np.ndarray,
    lam_up: np.ndarray,
    lam_both: np.ndarray,
    survive_down: np.ndarray,
    survive_up: np.ndarray,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """
    ``gamma_v(q, s)`` for the edge above every node, as ``gamma[v][q][s]``.

    ``q`` says a selected leaf lies below the edge, ``s`` that one lies above it; a side without a
    selected leaf keeps a survivor of the component with probability ``pi``. Without survival
    probabilities ``pi = 0`` and this is ``((0, up), (down, down + up + both))``.

    Parameters
    ----------
    lam_down, lam_up, lam_both : numpy.ndarray
        The two-sided weights, indexed by ``R_i`` node.
    survive_down, survive_up : numpy.ndarray
        ``pi_{X_i}`` below and above each edge, indexed by ``R_i`` node.

    Returns
    -------
    list[tuple[tuple[float, float], tuple[float, float]]]
        One ``gamma[q][s]`` table per node.
    """
    # Components are small, so plain floats beat vectorized NumPy here.
    columns = (lam_down.tolist(), lam_up.tolist(), lam_both.tolist())
    pis = (survive_down.tolist(), survive_up.tolist())
    return [
        (
            (down * p_down + up * p_up + both * p_down * p_up, down * p_down + up + both * p_down),
            (down + up * p_up + both * p_up, down + up + both),
        )
        for down, up, both, p_down, p_up in zip(*columns, *pis)
    ]


__all__ = [
    "ComponentDP",
    "component_table",
    "component_dp_table",
    "gammas",
    "root_table",
    "node_offsets",
]
