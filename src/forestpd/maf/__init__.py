"""
MAF module.

Stage 2 of the pipeline: an agreement forest per cluster. :class:`HeuristicMAF`
(common-neighbourhood contraction) is the default; it returns a valid, usually small, forest.
:class:`MaffeRMAF` and :class:`MaffeUMAF` give a minimum rooted or unrooted forest through the
optional maffe package, and :class:`TieredMAF` runs the heuristic and improves it with rooted maffe
on clusters where its ``k`` is large. Any object implementing :class:`MAFComputer` can be passed as
``maf=`` instead.
"""

from __future__ import annotations

from .base import Forest, MAFComputer, Rooting, singleton_forest
from .heuristic import HeuristicMAF
from .maffe import MaffeRMAF, MaffeUMAF
from .tiered import TieredMAF

__all__ = [
    "MAFComputer",
    "Forest",
    "HeuristicMAF",
    "TieredMAF",
    "MaffeRMAF",
    "MaffeUMAF",
    "Rooting",
    "singleton_forest",
]
