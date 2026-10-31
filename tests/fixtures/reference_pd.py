"""
Reference phylogenetic diversity, straight from the definition, for the tests.

An edge ``e`` lies in the minimal subtree (Steiner tree) spanning a leaf set ``A`` exactly when
*both* sides of the split induced by ``e`` meet ``A``. Phylogenetic diversity is the total branch
length of that subtree::

    PD_T(A) = sum of len(e) over edges whose split has both sides meeting A.

The package never calls this: the optimizer works with ``k``-bit component masks instead of these
``n``-bit taxon masks, and :meth:`~forestpd.MaxMPDInstance.evaluate_pd` with a split table. Both
are checked against it.

With survival probabilities, an unprotected taxon ``x`` survives independently with probability
``a(x)`` and a protected one surely, and the objective is the *expected* PD of the survivors. An
edge counts when both of its sides keep a survivor, so it contributes::

    len(e) * P(side L keeps a survivor) * P(side R keeps a survivor),

where a side holding a protected taxon keeps one surely, and otherwise with probability
``1 - prod (1 - a(x))`` over its taxa. :func:`expected_pd` evaluates this; with ``a = 0`` it equals
:func:`pd`.
"""

from __future__ import annotations

import math

import numpy as np

from forestpd.core.tree import PhyloTree, leaf_masks_below


def pd(tree: PhyloTree, leaf_mask: int) -> float:
    """
    Phylogenetic diversity of a leaf set in a tree.

    Parameters
    ----------
    tree : PhyloTree
        The tree.
    leaf_mask : int
        Bitmask of selected leaves (bit ``i`` set means leaf ``i`` is selected).

    Returns
    -------
    float
        ``PD_T(leaf_mask)``; zero for the empty set and for any single leaf.

    Examples
    --------
    >>> t = PhyloTree.from_edges(5, [(0, 4), (1, 4), (4, 3), (2, 3)],
    ...                          [1.0, 1.0, 2.0, 1.0], {0: 0, 1: 1, 2: 2})
    >>> pd(t, 0b011)      # leaves 0 and 1: the cherry
    2.0
    >>> pd(t, 0b111)      # all three leaves: the whole tree
    5.0
    >>> pd(t, 0b001)
    0.0
    """
    if leaf_mask == 0 or leaf_mask & (leaf_mask - 1) == 0:
        return 0.0  # empty or single leaf spans nothing
    rooted = tree.root_at(0)
    below = leaf_masks_below(rooted)
    full = (1 << tree.n_leaves) - 1
    total = 0.0
    for e in range(tree.n_edges):
        low = below[e]
        if (low & leaf_mask) and ((full & ~low) & leaf_mask):
            total += float(tree.weight[e])
    return total


def expected_pd(tree: PhyloTree, leaf_mask: int, log_loss: np.ndarray) -> float:
    """
    Expected phylogenetic diversity of the survivors, given a protected leaf set.

    Parameters
    ----------
    tree : PhyloTree
        The tree.
    leaf_mask : int
        Bitmask of protected leaves; they survive surely.
    log_loss : numpy.ndarray
        ``float64[n_leaves]``; ``log(1 - a(x))`` per leaf, the log-probability that the leaf is lost
        when unprotected (``0`` for ``a(x) = 0``).

    Returns
    -------
    float
        ``sum_e len(e) * P(one side keeps a survivor) * P(the other does)``. With ``log_loss`` all
        zero this equals ``pd(tree, leaf_mask)`` (up to the order of summation).

    Examples
    --------
    >>> t = PhyloTree.from_edges(5, [(0, 4), (1, 4), (4, 3), (2, 3)],
    ...                          [1.0, 1.0, 2.0, 1.0], {0: 0, 1: 1, 2: 2})
    >>> expected_pd(t, 0b011, np.zeros(3))      # nobody else survives: plain PD
    2.0
    >>> expected_pd(t, 0b011, np.log([1.0, 1.0, 0.5]))   # leaf 2 survives with probability 1/2
    3.5
    """
    rooted = tree.root_at(0)
    n_leaves = tree.n_leaves
    loss = log_loss.tolist()
    # Per node, below it: the summed log-loss of its unprotected leaves, and its protected leaves.
    lost = [0.0] * tree.n_nodes
    kept = [0] * tree.n_nodes
    for leaf in range(n_leaves):
        if (leaf_mask >> leaf) & 1:
            kept[leaf] = 1
        else:
            lost[leaf] = loss[leaf]
    total_lost = sum(lost[:n_leaves])
    total_kept = sum(kept[:n_leaves])
    parent = rooted.parent.tolist()
    parent_edge = rooted.parent_edge.tolist()
    weight = tree.weight.tolist()
    total = 0.0
    for v in reversed(rooted.order.tolist()):
        e = parent_edge[v]
        if e < 0:
            continue
        below = 1.0 if kept[v] else 0.0 - math.expm1(lost[v])
        above = 1.0 if total_kept - kept[v] else 0.0 - math.expm1(total_lost - lost[v])
        total += weight[e] * below * above
        lost[parent[v]] += lost[v]
        kept[parent[v]] += kept[v]
    return total


__all__ = ["pd", "expected_pd"]
