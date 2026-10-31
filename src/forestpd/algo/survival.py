"""
Survival probabilities: side constants per local leaf, and the per-cluster sums around the DP.

Every unprotected taxon ``x`` survives independently with probability ``a(x)``, and protected ones
surely. An edge counts when both of its sides keep a survivor. A side holding no selected taxon
keeps one with probability ``1 - q``, where its *side constant* ``q = prod (1 - a(x))`` runs over
every taxon on that side of the original tree, selected or not, so it does not depend on the
signature. All constants are kept in log space, as sums of ``log(1 - a(x))``, which is finite
because ``a(x) < 1``.

A local leaf stands for a fixed set of taxa: one taxon, the subtree of a child cluster ``D``
(pseudo-leaf ``x_D``), or everything outside the cluster (``x_C^*``). So a side's ``log q`` is the
sum of :func:`leaf_logs` over the local leaves on it. The profile rows use this per edge (see
:mod:`~forestpd.algo.profile`); this module adds what the DP itself does not see:

``c_down(D)``
    The manuscript's ``Omega_D^0 = PD_D^+(0)``: the expected weight of the edges below the cut
    above ``D`` when nothing below it is selected (their upper side is then covered by the rest of
    the solution). An unselected pseudo-leaf ``x_D`` is worth this much (its ``idle`` value).
``c_up(C)``
    The manuscript's ``Omega_C^1``: the expected weight of the edges outside the region below
    ``C`` when the whole solution lies below ``C`` (their side towards ``C`` is covered). Part of
    ``PD_C``, added when reading out the answer.
The empty set and singletons
    The DP only counts solutions with at least two selected leaves in their lowest cluster. With
    survival probabilities an empty or one-taxon selection is still worth something, so these
    values are computed directly.

All of them are sums over a cluster's edges of ``w (1 - q)`` of "the side without some leaf", which
:func:`cluster_sides` gives for every local leaf at once, in linear time. Without survival
probabilities every ``q`` is ``1`` and all of these are zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..cluster.base import ClusterTree, LocalPiece
from ..core.tree import RootedTree
from . import _profile_kernel

#: Compute the split-table sums with the compiled kernel when Numba is installed. The NumPy path
#: gives the same values; set ``False`` to force it (tests).
USE_COMPILED = True


@dataclass(frozen=True)
class ClusterSides:
    """
    The signature-independent survival sums over one cluster's own edges.

    The cluster's own edges are those of its local trees ``I_C^+``: the edges inside the cluster,
    including the cut edges to its children (as the pendant edges of their pseudo-leaves).

    Attributes
    ----------
    at_leaf : numpy.ndarray
        ``float64[n_leaves]``; per local leaf ``l``, ``sum w (1 - q)`` of the side of each edge that
        does not contain ``l``: what the edges pay when ``l`` is the only covered leaf.
    empty : float
        ``sum w (1 - q_a)(1 - q_b)``: what the edges pay when nothing is selected.
    """

    at_leaf: np.ndarray
    empty: float


def subtree_logs(cluster_tree: ClusterTree, log_loss: np.ndarray) -> list[float]:
    """
    The summed ``log(1 - a(x))`` of the taxa at or below each cluster.

    Parameters
    ----------
    cluster_tree : ClusterTree
        The tree of clusters ``K``.
    log_loss : numpy.ndarray
        ``float64[n_taxa]``; ``log(1 - a(x))`` per taxon id.

    Returns
    -------
    list[float]
        One sum per cluster.
    """
    total = [0.0] * cluster_tree.n_clusters
    for cluster in cluster_tree.postorder:
        piece = cluster_tree.pieces[cluster]
        own = sum(float(log_loss[t]) for t in piece.taxon_of_leaf if t >= 0)
        total[cluster] = own + sum(total[d] for d in cluster_tree.children[cluster])
    return total


def leaf_logs(
    piece: LocalPiece, log_loss: np.ndarray, subtree: Sequence[float], cluster: int, total: float
) -> np.ndarray:
    """
    The summed ``log(1 - a(x))`` of the taxa each local leaf of a cluster stands for.

    Parameters
    ----------
    piece : LocalPiece
        The cluster's geometry.
    log_loss : numpy.ndarray
        ``float64[n_taxa]``; ``log(1 - a(x))`` per taxon id.
    subtree : Sequence[float]
        :func:`subtree_logs`.
    cluster : int
        The cluster's id.
    total : float
        The sum over all taxa.

    Returns
    -------
    numpy.ndarray
        ``float64[n_leaves]``: a taxon's own value, a child's subtree sum, and for ``x_C^*`` the
        sum over all taxa outside the cluster's subtree. Together the leaves cover every taxon once.
    """
    out = np.empty(piece.n_leaves)
    for leaf in range(piece.n_leaves):
        taxon = piece.taxon_of_leaf[leaf]
        child = piece.child_of_leaf[leaf]
        if taxon >= 0:
            out[leaf] = log_loss[taxon]
        elif child >= 0:
            out[leaf] = subtree[child]
        else:  # x_C^*: everything outside this cluster's subtree
            out[leaf] = total - subtree[cluster]
    return out


def cluster_sides(
    piece: LocalPiece, leaf_log: np.ndarray, tree_weights: np.ndarray
) -> ClusterSides:
    """
    The survival sums of :class:`ClusterSides` for one cluster.

    From the cluster's split table when it has one, else from its local trees; the two agree.

    Parameters
    ----------
    piece : LocalPiece
        The cluster's geometry.
    leaf_log : numpy.ndarray
        ``float64[n_leaves]``; :func:`leaf_logs`.
    tree_weights : numpy.ndarray
        ``float64[t]``; the per-tree weights ``w_j``, used only when the piece's local trees are not
        merged (merged trees carry their own weights).

    Returns
    -------
    ClusterSides
        The per-leaf sums and the empty-selection sum.

    Examples
    --------
    One edge between two leaves, the second surviving with probability ``1/2``: covering leaf 0
    pays ``w / 2``, covering leaf 1 pays nothing, and an empty selection pays nothing either.

    >>> from forestpd.cluster.base import single_cluster
    >>> from forestpd.core.tree import PhyloTree
    >>> tree = PhyloTree.from_edges(2, [(0, 1)], [4.0], {0: 0, 1: 1})
    >>> piece = single_cluster([tree]).pieces[0]
    >>> sides = cluster_sides(piece, np.log([1.0, 0.5]), np.ones(1))
    >>> sides.at_leaf.tolist(), sides.empty
    ([2.0, 0.0], 0.0)
    """
    leaf_log = np.ascontiguousarray(leaf_log, dtype=np.float64)
    total = float(leaf_log.sum())
    if piece.split_masks is not None and piece.split_weights is not None:
        masks = piece.split_masks.astype(np.uint64, copy=False)
        weights = piece.split_weights.astype(np.float64, copy=False)
        if USE_COMPILED and _profile_kernel.HAVE_NUMBA:
            at_leaf, empty = _profile_kernel.split_sides(masks, weights, leaf_log, total)
            return ClusterSides(at_leaf=at_leaf, empty=float(empty))
        return _split_sides(masks, weights, leaf_log, total)
    if piece.tree_weights is not None:  # merged local trees carry their own weights
        tree_weights = np.asarray(piece.tree_weights, dtype=np.float64)
    root = piece.star_leaf if piece.star_leaf >= 0 else 0
    at_leaf = np.zeros(piece.n_leaves)
    empty = 0.0
    for tree, tree_weight in zip(piece.trees, tree_weights.tolist()):
        rooted = tree.root_at(root)
        log_below = below_logs(rooted, leaf_log)
        weight = tree_weight * tree.weight
        keep_below = 0.0 - np.expm1(log_below)
        keep_above = 0.0 - np.expm1(total - log_below)
        survive_below = weight * keep_below
        survive_above = weight * keep_above
        empty += float((weight * keep_below * keep_above).sum())
        # Leaf l pays survive_below on every edge, except on the edges from the root to l, where
        # the side without l is the one above: accumulate that difference along each root path.
        change = (survive_above - survive_below).tolist()
        acc = [0.0] * tree.n_nodes
        parent = rooted.parent.tolist()
        parent_edge = rooted.parent_edge.tolist()
        for v in rooted.order.tolist():
            e = parent_edge[v]
            if e >= 0:
                acc[v] = acc[parent[v]] + change[e]
        at_leaf += float(survive_below.sum()) + np.asarray(acc[: piece.n_leaves])
    return ClusterSides(at_leaf=at_leaf, empty=empty)


def _split_sides(
    masks: np.ndarray, weights: np.ndarray, leaf_log: np.ndarray, total: float
) -> ClusterSides:
    """:func:`cluster_sides` from a split table, in NumPy (masks: the sides away from the root)."""
    n_leaves = leaf_log.size
    shifts = np.arange(n_leaves, dtype=np.uint64)
    bits = ((masks[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.float64)
    log_a = bits @ leaf_log
    keep_a = 0.0 - np.expm1(log_a)
    keep_b = 0.0 - np.expm1(total - log_a)
    at_leaf = float((weights * keep_a).sum()) + bits.T @ (weights * keep_b - weights * keep_a)
    return ClusterSides(at_leaf=at_leaf, empty=float((weights * keep_a * keep_b).sum()))


def below_logs(rooted: RootedTree, leaf_log: np.ndarray) -> np.ndarray:
    """
    Per edge, the summed ``leaf_log`` of the leaves on its far-from-root side.

    Parameters
    ----------
    rooted : RootedTree
        A rooting of the tree.
    leaf_log : numpy.ndarray
        ``float64[n_leaves]``.

    Returns
    -------
    numpy.ndarray
        ``float64[n_edges]``.
    """
    tree = rooted.tree
    sub = np.asarray(leaf_log, dtype=np.float64).tolist() + [0.0] * (tree.n_nodes - tree.n_leaves)
    out = [0.0] * tree.n_edges
    parent = rooted.parent.tolist()
    parent_edge = rooted.parent_edge.tolist()
    for v in reversed(rooted.order.tolist()):
        e = parent_edge[v]
        if e >= 0:
            out[e] = sub[v]
            sub[parent[v]] += sub[v]
    return np.asarray(out, dtype=np.float64)


def split_logs(
    masks: np.ndarray, n_leaves: int, leaf_log: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """
    The log side constants of each split: of the masked side, and of the other side.

    Parameters
    ----------
    masks : numpy.ndarray
        ``uint64``; one side of each split.
    n_leaves : int
        Number of local leaves (at most 64).
    leaf_log : numpy.ndarray | None
        ``float64[n_leaves]``, or ``None`` for all zero.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``float64`` per split, for the masked side and for its complement.
    """
    if leaf_log is None:
        zero = np.zeros(masks.size)
        return zero, zero.copy()
    shifts = np.arange(n_leaves, dtype=np.uint64)
    bits = ((masks[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.float64)
    log_a = bits @ np.asarray(leaf_log, dtype=np.float64)
    return log_a, float(np.sum(leaf_log)) - log_a


__all__ = [
    "ClusterSides",
    "cluster_sides",
    "subtree_logs",
    "leaf_logs",
    "below_logs",
    "split_logs",
]
