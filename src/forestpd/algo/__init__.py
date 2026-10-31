"""
Stage 3 of the pipeline: the phylogenetic-diversity dynamic program.

Modules, in the order the algorithm uses them:

===========================  =============================================================
:mod:`.local`                ``LocalInstance`` and the ``f_2`` tables that link clusters
:mod:`.profile`              per-cluster precomputation, done once before the ``2^k`` loop
:mod:`.survival`             survival probabilities: side constants and the per-cluster sums
:mod:`.convolve`             max-plus convolution (compiled with Numba)
:mod:`.component_dp`         ``PD_i^{(S)}``, Lemma "compute-PD_i"
:mod:`.local_solver`         Algorithm 3: one cluster, all signatures
:mod:`.solver`               Algorithm 1: bottom-up pass over the tree of clusters
:mod:`.witness`              reconstruction of an optimal taxon set
===========================  =============================================================
"""

from __future__ import annotations

from .component_dp import component_dp_table, component_table
from .convolve import HAVE_NUMBA, fold_max_plus, max_plus
from .local import LocalInstance, NEG_INF, ordinary_f2, pseudo_leaf_f2
from .local_solver import LocalSolution, signature_weights, solve_local
from .profile import ClusterProfile, ComponentTree, build_profile
from .solver import Solution, solve

__all__ = [
    "LocalInstance",
    "NEG_INF",
    "ordinary_f2",
    "pseudo_leaf_f2",
    "ClusterProfile",
    "ComponentTree",
    "build_profile",
    "max_plus",
    "fold_max_plus",
    "HAVE_NUMBA",
    "component_table",
    "component_dp_table",
    "LocalSolution",
    "solve_local",
    "signature_weights",
    "solve",
    "Solution",
]
