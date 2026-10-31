"""
Max-plus convolution, the ``B^2`` inner kernel of the algorithm.

It is used in two places: splitting a budget between the two children of a node in the component
DP, and splitting a budget across the components of a signature (Lemma "compute-D_Sb").

Infeasible entries are ``-inf``, which stays ``-inf`` under addition, so no masking is needed. A
Numba kernel is used when Numba is installed; otherwise a NumPy version performs one vector
operation per left-hand index.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: The compiled kernel, or ``None`` when Numba is unavailable.
_max_plus_kernel: Any

try:  # pragma: no cover - depends on the environment
    # Older Numba releases ship no type information; newer ones do, hence both codes.
    from numba import njit  # type: ignore[import-untyped,unused-ignore]

    @njit(cache=True, fastmath=False)
    def _max_plus_kernel(a, b, out):  # pragma: no cover - compiled
        """
        Compiled in-place max-plus convolution: ``out[p + q] = max(out[p + q], a[p] + b[q])``.

        Parameters
        ----------
        a, b : numpy.ndarray
            Contiguous ``float64`` value tables; ``-inf`` marks an infeasible entry.
        out : numpy.ndarray
            ``float64`` result buffer, pre-filled with ``-inf`` and updated in place.
        """
        n_a = a.shape[0]
        n_b = b.shape[0]
        n_out = out.shape[0]
        for p in range(n_a):
            av = a[p]
            if av == -np.inf:
                continue
            hi = n_b if p + n_b <= n_out else n_out - p
            for q in range(hi):
                value = av + b[q]
                if value > out[p + q]:
                    out[p + q] = value

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - only if Numba cannot be imported
    _max_plus_kernel = None
    HAVE_NUMBA = False

#: The signature loop's kernel, which skips a table's constant runs (a plain taxon from its cost
#: on, a child cluster past its budget cap) with a running maximum, bit-identical to the plain
#: loop; used for long tables. ``None`` without Numba.
_max_plus_runs: Any
try:  # pragma: no cover - depends on the environment
    from ._signature_kernel import MIN_SHORTCUT
    from ._signature_kernel import _max_plus_into as _max_plus_runs
except ImportError:  # pragma: no cover - only if Numba cannot be imported
    _max_plus_runs = None
    MIN_SHORTCUT = 0


def max_plus(a: np.ndarray, b: np.ndarray, n_out: int) -> np.ndarray:
    """
    ``out[i] = max over p + q == i of a[p] + b[q]``, truncated to ``n_out`` entries.

    Parameters
    ----------
    a, b : numpy.ndarray
        ``float64`` value tables; ``-inf`` marks an infeasible entry.
    n_out : int
        Length of the result.

    Returns
    -------
    numpy.ndarray
        ``float64[n_out]``, filled with ``-inf`` where no split is feasible.

    Examples
    --------
    >>> a = np.array([0.0, 1.0, 2.0])
    >>> b = np.array([0.0, 5.0, 6.0])
    >>> max_plus(a, b, 3)
    array([0., 5., 6.])
    """
    out = np.full(max(n_out, 0), -np.inf)
    if n_out <= 0 or a.size == 0 or b.size == 0:
        return out
    if HAVE_NUMBA:
        a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
        if _max_plus_runs is not None and n_out >= MIN_SHORTCUT:
            _max_plus_runs(a, b, out)
        else:
            _max_plus_kernel(a, b, out)
        return out
    for p in range(min(a.size, n_out)):
        av = a[p]
        if av == -np.inf:
            continue
        span = min(b.size, n_out - p)
        np.maximum(out[p : p + span], av + b[:span], out=out[p : p + span])
    return out


def saved_shift(table: np.ndarray, offset: int) -> np.ndarray:
    """
    ``table`` moved ``offset`` entries up along its last axis, for "at least ``beta``" tables.

    Entry ``beta`` of the result is entry ``max(0, beta - offset)`` of ``table``: something else
    saving exactly ``offset`` leaves ``beta - offset`` to be saved by ``table``'s part (DP', see
    :mod:`~forestpd.algo.local`). With ``offset = 0`` (always on DP) ``table`` itself is returned.

    Examples
    --------
    >>> saved_shift(np.array([3.0, 2.0, 1.0, 0.0]), 2)
    array([3., 3., 3., 2.])
    """
    if offset <= 0:
        return table
    width = table.shape[-1]
    index = np.maximum(np.arange(width) - offset, 0)
    shifted: np.ndarray = table[..., index]
    return shifted


def fold_max_plus(tables: list[np.ndarray], budget: int) -> np.ndarray:
    """
    Combine per-component tables into one budget table by repeated max-plus convolution.

    Parameters
    ----------
    tables : list[numpy.ndarray]
        One ``float64[budget+1]`` table per active component.
    budget : int
        The working budget ``B``.

    Returns
    -------
    numpy.ndarray
        ``float64[budget+1]``; entry ``beta`` is the best total over all ways to split ``beta``
        among the components. With no tables, ``0`` at budget 0 and ``-inf`` elsewhere.
    """
    if not tables:
        result = np.full(budget + 1, -np.inf)
        result[0] = 0.0
        return result
    acc = tables[0]
    for table in tables[1:]:
        acc = max_plus(acc, table, budget + 1)
    return acc


__all__ = ["max_plus", "fold_max_plus", "saved_shift", "HAVE_NUMBA"]
