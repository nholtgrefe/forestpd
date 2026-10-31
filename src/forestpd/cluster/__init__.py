"""
Cluster module.

Stage 1 of the pipeline: the cluster reduction. A reducer splits the input
trees into the tree of clusters ``K``; :func:`cluster_reduction` cuts every non-trivial split
common to all trees.
"""

from __future__ import annotations

from .base import ClusterTree, LocalPiece, single_cluster
from .common import cluster_reduction

__all__ = [
    "ClusterTree",
    "LocalPiece",
    "single_cluster",
    "cluster_reduction",
]
