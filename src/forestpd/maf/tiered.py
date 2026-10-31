"""
The heuristic forest, improved by rooted maffe where it is large.

The DP's work per cluster grows with ``2^k``, so a smaller forest only matters on clusters where
``k`` is large. :class:`TieredMAF` runs the fast heuristic everywhere and calls rooted maffe
(:class:`~forestpd.maf.MaffeRMAF`, a minimum rooted agreement forest) only on the clusters where
the heuristic leaves ``k >= k_threshold``, keeping maffe's forest when it has fewer components.
Like :class:`~forestpd.maf.MaffeRMAF` itself, maffe's forests are trusted, not checked.

The default threshold of 14 comes from full solves on simulated gene trees: below it the DP of a
cluster takes milliseconds and no maffe call pays off; above it, rooted maffe costs well under a
second per cluster and removed one or two components from the hardest clusters, never making a
solve slower and often making it about twice as fast.
"""

from __future__ import annotations

from typing import Sequence

from ..core.tree import PhyloTree
from ..exceptions import ForestPDError, ForestPDValueError
from .base import Forest, Rooting
from .heuristic import HeuristicMAF
from .maffe import MaffeRMAF


class TieredMAF:
    """
    The heuristic forest, replaced by rooted maffe's on clusters where it has ``k >= k_threshold``.

    Needs maffe installed (see :class:`~forestpd.maf.MaffeRMAF`). Maffe's forest is used if it has
    fewer components; like :class:`~forestpd.maf.MaffeRMAF`, it is trusted, not checked. If maffe
    cannot handle a cluster (it raises a :class:`~forestpd.exceptions.ForestPDError`, e.g. for a
    polytomy), the heuristic forest is kept. Other exceptions from maffe itself are not caught.

    Parameters
    ----------
    k_threshold : int, default 14
        Smallest heuristic forest size at which rooted maffe is tried.

    Attributes
    ----------
    heuristic : HeuristicMAF
        The first stage.
    maffe : MaffeRMAF
        The second stage.
    trusted : bool
        ``True``: both stages return agreement forests by construction.
    uses_rooting : bool
        ``True``: the solver's :class:`~forestpd.maf.base.Rooting` is passed on to maffe.

    Raises
    ------
    ForestPDValueError
        If ``k_threshold`` is not an integer of at least 2.
    ForestPDImportError
        If maffe is not installed.
    """

    trusted = True
    uses_rooting = True

    def __init__(self, k_threshold: int = 14) -> None:
        if isinstance(k_threshold, bool) or not isinstance(k_threshold, int) or k_threshold < 2:
            raise ForestPDValueError(f"k_threshold must be an integer >= 2, got {k_threshold!r}")
        self.k_threshold = k_threshold
        self.heuristic = HeuristicMAF()
        self.maffe = MaffeRMAF()

    def compute(self, trees: Sequence[PhyloTree], rooting: Rooting | None = None) -> Forest:
        """
        The heuristic forest of ``trees``, or rooted maffe's where that is smaller.

        Parameters
        ----------
        trees : Sequence[PhyloTree]
            The local trees of one cluster, all on leaves ``0..n_leaves-1``.
        rooting : Rooting | None, optional
            Where maffe roots the trees (see :class:`~forestpd.maf.MaffeRMAF`).

        Returns
        -------
        Forest
            The forest used.
        """
        forest = self.heuristic.compute(trees)
        if len(forest) < self.k_threshold:
            return forest
        try:
            candidate = self.maffe.compute(trees, rooting=rooting)
        except ForestPDError:  # maffe cannot handle this cluster (e.g. a polytomy)
            return forest
        return candidate if len(candidate) < len(forest) else forest


__all__ = ["TieredMAF"]
