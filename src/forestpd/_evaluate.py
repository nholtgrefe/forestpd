"""
Fast evaluation of the objective of a taxon set: the trees as one weighted split table.

The objective of a protected set ``S`` is a sum over all edges of all trees of
``w_j * len_j(e) * P(one side keeps a survivor) * P(the other side does)``, where a side holding a
taxon of ``S`` keeps one surely and otherwise with probability ``1 - prod (1 - a(x))`` over its
taxa. Both probabilities depend on the edge only through its split, so the edges are grouped by
split once (:func:`forestpd.io.splits.split_system`, compiled) and each distinct split carries its
summed weight ``W_s``. Each split is stored by its smaller side, as a sparse row of taxon ids.

Scoring a set is then two sparse sums per split, independent of the number of trees: the protected
taxa on the stored side (the other side has the rest), and the summed ``log(1 - a(x))`` of its
unprotected taxa (the other side has the rest of the total). This regroups the per-edge sum of
the per-edge definition exactly; without survival probabilities it is the split
diversity of ``S``, i.e. its weighted PD.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .core.tree import PhyloTree
from .io import _splits_kernel
from .io import splits as _splits


class SplitTable:
    """
    The weighted splits of a set of trees, ready to score taxon sets.

    Parameters
    ----------
    trees : Sequence[PhyloTree]
        The trees, all on leaves ``0..n_taxa-1``.
    weights : numpy.ndarray
        ``float64[t]``; the per-tree weights.
    log_loss : numpy.ndarray
        ``float64[n_taxa]``; ``log(1 - a(x))`` per taxon.

    Examples
    --------
    >>> from forestpd.core.tree import PhyloTree
    >>> t = PhyloTree.from_edges(6, [(0, 4), (1, 4), (4, 5), (2, 5), (3, 5)], [1.0] * 5,
    ...                          {i: i for i in range(4)})
    >>> table = SplitTable([t, t], np.array([1.0, 0.5]), np.zeros(4))
    >>> table.value([0, 2])          # the path 0 - 4 - 5 - 2, in both trees: 3 + 1.5
    4.5
    """

    def __init__(self, trees: Sequence[PhyloTree], weights: np.ndarray, log_loss: np.ndarray):
        n_taxa = trees[0].n_leaves
        members, offsets, split_weights = _splits._flat_split_system(trees, weights)
        if _splits.USE_COMPILED and _splits_kernel.HAVE_NUMBA:
            members, offsets = _splits_kernel.smaller_sides(members, offsets, n_taxa)
        else:
            members, offsets = _smaller_sides(members, offsets, n_taxa)
        self.n_taxa = n_taxa
        self.members = members
        self.weights = split_weights
        self.log_loss = np.asarray(log_loss, dtype=np.float64)
        # Rows are nonempty (every side holds a taxon), so reduceat over these starts is exact.
        self.starts = offsets[:-1]

    def value(self, protected: Sequence[int]) -> float:
        """
        The objective of protecting the taxa ``protected`` (taxon ids).

        Parameters
        ----------
        protected : Sequence[int]
            Distinct taxon ids.

        Returns
        -------
        float
            ``sum_s W_s P(side A keeps a survivor) P(side B does)``.
        """
        if self.weights.size == 0:
            return 0.0
        is_protected = np.zeros(self.n_taxa, dtype=bool)
        is_protected[np.asarray(list(protected), dtype=np.int64)] = True
        n_protected = int(is_protected.sum())
        unprotected_loss = np.where(is_protected, 0.0, self.log_loss)
        total_loss = float(unprotected_loss.sum())

        on_side = is_protected[self.members]
        protected_a = np.add.reduceat(on_side.astype(np.int64), self.starts)
        loss_a = np.add.reduceat(unprotected_loss[self.members], self.starts)
        protected_b = n_protected - protected_a
        loss_b = total_loss - loss_a
        keeps_a = np.where(protected_a > 0, 1.0, 0.0 - np.expm1(loss_a))
        keeps_b = np.where(protected_b > 0, 1.0, 0.0 - np.expm1(loss_b))
        return float(np.dot(self.weights, keeps_a * keeps_b))


def _smaller_sides(
    members: np.ndarray, offsets: np.ndarray, n_taxa: int
) -> tuple[np.ndarray, np.ndarray]:
    """The NumPy version of :func:`forestpd.io._splits_kernel.smaller_sides`."""
    sides = []
    everyone = np.arange(n_taxa)
    for r in range(offsets.size - 1):
        side = members[offsets[r] : offsets[r + 1]]
        sides.append(side if 2 * side.size <= n_taxa else np.setdiff1d(everyone, side))
    out_offsets = np.zeros(len(sides) + 1, dtype=np.int64)
    np.cumsum([side.size for side in sides], out=out_offsets[1:])
    flat = np.concatenate(sides) if sides else np.empty(0, dtype=np.int64)
    return flat.astype(np.int64), out_offsets


__all__ = ["SplitTable"]
