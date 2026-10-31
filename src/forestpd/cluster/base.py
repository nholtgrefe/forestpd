"""
Cluster-reduction data types.

The paper's *tree of clusters* ``K`` is the object this layer produces. Deleting every common
cut-edge from any input tree leaves the same collection of connected pieces, and contracting each
piece gives a tree whose edges are exactly those cut-edges. Each node of ``K`` is one cluster; its
local instance is that piece with every incident cut-edge turned into a pendant pseudo-leaf.

This is the first of the three stages (cluster -> MAF -> PD). All downstream machinery consumes
:class:`ClusterTree` and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..core.tree import PhyloTree


@dataclass(frozen=True)
class LocalPiece:
    """
    One cluster's geometry: the input trees restricted to it, with pseudo-leaves for its neighbours.

    All trees share one local leaf numbering, laid out in three contiguous blocks:

    1. the cluster's *own* taxa ``X_C^o``, in increasing global taxon id;
    2. one pseudo-leaf ``x_D`` per child cluster ``D``, in increasing child cluster index;
    3. (non-root clusters only) the parent-direction leaf ``x_C^*``, always **last**.

    Attributes
    ----------
    trees : tuple[PhyloTree, ...]
        The local trees, one per input tree, all on leaves ``0..n_leaves-1``.
    taxon_of_leaf : tuple[int, ...]
        Global taxon id of each local leaf; ``-1`` for a pseudo-leaf.
    child_of_leaf : tuple[int, ...]
        Child cluster index of each local leaf; ``-1`` for anything that is not a child pseudo-leaf.
    star_leaf : int
        Local id of the parent-direction leaf ``x_C^*``, or ``-1`` at the root cluster.
    tree_weights : tuple[float, ...] | None
        Weight of each local tree, when the trees are not one per input tree: local trees of equal
        topology merged into one, with edge weights ``sum_j w_j * len_j(e)`` and tree weight ``1``.
        ``None`` means one local tree per input tree, weighted by the input tree weights.
    split_masks : numpy.ndarray | None
        ``uint64``; the cluster's distinct local splits, over the edges of all input trees. Each
        is the mask of the local leaves on the side away from the cluster's root leaf (leaf ``0``
        at the root cluster, ``x_C^*`` elsewhere), in increasing order. The zero-weight pendant
        edge of ``x_C^*`` is not included. ``None`` when not computed (no tree weights given, or
        more than 64 local leaves).
    split_weights : numpy.ndarray | None
        ``float64``; per split, ``sum_j w_j * len_j(e)`` over the edges ``e`` that induce it.
    """

    trees: tuple[PhyloTree, ...]
    taxon_of_leaf: tuple[int, ...]
    child_of_leaf: tuple[int, ...]
    star_leaf: int
    tree_weights: tuple[float, ...] | None = None
    split_masks: np.ndarray | None = field(default=None, compare=False, repr=False)
    split_weights: np.ndarray | None = field(default=None, compare=False, repr=False)

    @property
    def n_leaves(self) -> int:
        """Number of local leaves, including pseudo-leaves."""
        return len(self.taxon_of_leaf)


@dataclass(frozen=True)
class ClusterTree:
    """
    The rooted tree of clusters ``K``.

    Attributes
    ----------
    pieces : tuple[LocalPiece, ...]
        One local instance per cluster, indexed by cluster id.
    parent : tuple[int, ...]
        Parent cluster id, ``-1`` at the root.
    children : tuple[tuple[int, ...], ...]
        Child cluster ids per cluster.
    postorder : tuple[int, ...]
        Cluster ids, children before parents.
    root : int
        Root cluster id.
    """

    pieces: tuple[LocalPiece, ...]
    parent: tuple[int, ...]
    children: tuple[tuple[int, ...], ...]
    postorder: tuple[int, ...]
    root: int

    @property
    def n_clusters(self) -> int:
        """Number of clusters."""
        return len(self.pieces)


def single_cluster(trees: Sequence[PhyloTree]) -> ClusterTree:
    """
    The tree of clusters with one cluster holding the whole instance.

    Parameters
    ----------
    trees : Sequence[PhyloTree]
        The input trees, all on leaves ``0..n_leaves-1``.

    Returns
    -------
    ClusterTree
        A one-node cluster tree whose piece is ``trees`` unchanged.
    """
    n_leaves = trees[0].n_leaves
    piece = LocalPiece(
        trees=tuple(trees),
        taxon_of_leaf=tuple(range(n_leaves)),
        child_of_leaf=tuple(-1 for _ in range(n_leaves)),
        star_leaf=-1,
    )
    return ClusterTree(pieces=(piece,), parent=(-1,), children=((),), postorder=(0,), root=0)


__all__ = ["LocalPiece", "ClusterTree", "single_cluster"]
