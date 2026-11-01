"""
forestpd — maximizing phylogenetic diversity under phylogenetic and extinction uncertainty.

Solves MaxMPD: given unrooted weighted trees on a shared taxon set, per-tree weights ``w_j``,
integer taxon costs and a budget ``B``, choose ``S`` with ``c(S) <= B`` maximizing
``sum_j w_j PD_{T_j}(S)`` — or, with per-taxon survival probabilities, its expected value over the
taxa that survive. Given a maximum agreement forest (MAF) of size ``k``, it solves MaxMPD in
``O(2^k * n * B^2)`` time, where ``n`` is the total size of the trees. ``k`` is taken per common
cluster, so it measures how much the trees disagree locally.

The public entry points are :class:`MaxMPDInstance` (build it once, then call
:meth:`~MaxMPDInstance.optimize`, :meth:`~MaxMPDInstance.optimize_many` and
:meth:`~MaxMPDInstance.evaluate_pd`) and the one-call functions :func:`optimize_maxmpd`,
:func:`optimize_many_maxmpd` and :func:`evaluate_maxmpd`. The pipeline has three stages —
:mod:`forestpd.cluster`, :mod:`forestpd.maf`, :mod:`forestpd.algo` — described in ``README.md``.
"""

from __future__ import annotations

__version__ = "0.1.1"

from .api import evaluate_maxmpd, optimize_many_maxmpd, optimize_maxmpd
from .core import PhyloTree, TaxonRegistry
from .exceptions import (
    ForestPDError,
    ForestPDImportError,
    ForestPDInputError,
    ForestPDValueError,
    ForestPDWarning,
)
from .instance import MaxMPDInstance, MaxMPDResult, MaxMPDStructure
from . import io, utils
from .algo.convolve import HAVE_NUMBA as _HAVE_NUMBA

if not _HAVE_NUMBA:  # pragma: no cover - depends on the environment
    import warnings as _warnings

    _warnings.warn(
        "Numba could not be imported, so forestpd runs its NumPy/Python fallbacks: the same "
        "results, but several times slower. Install numba (a dependency of forestpd) for full speed.",
        ForestPDWarning,
        stacklevel=2,
    )

__all__ = [
    "__version__",
    # Public API
    "MaxMPDInstance",
    "optimize_maxmpd",
    "optimize_many_maxmpd",
    "evaluate_maxmpd",
    "MaxMPDResult",
    "MaxMPDStructure",
    "io",
    "utils",
    # Core types
    "PhyloTree",
    "TaxonRegistry",
    # Exceptions
    "ForestPDError",
    "ForestPDValueError",
    "ForestPDInputError",
    "ForestPDImportError",
    "ForestPDWarning",
]
