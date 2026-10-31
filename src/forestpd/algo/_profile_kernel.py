"""
The compiled row computation of :func:`~forestpd.algo.profile.build_profiles_from_splits`.

Building a cluster's profile from its split table is a handful of small array operations per
cluster. The work is tiny, but with thousands of clusters the per-call overhead of NumPy dominates.
These kernels do the whole row computation of one cluster in compiled loops: the masks and side
constants of every split, the merging of equal rows, the plain profile's rows (the plus profile's
with the ``x_C^*`` bit cleared, merged again), and the cluster's survival sums
(:func:`~forestpd.algo.survival.cluster_sides`).

Component-edge rows are keyed by ``(component, away, down, up)``, where ``away`` is the
component's leaves on the side away from its smallest leaf. The ``R_i`` slot is a function of
``(component, away)``, so merging on these keys merges exactly the rows that share a slot; the
slots themselves are assigned afterwards in Python, where ``F_i`` is built.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: The compiled kernels, or ``None`` when Numba is unavailable.
split_rows: Any
merge_rows: Any
plain_rows: Any
split_sides: Any

try:  # pragma: no cover - depends on the environment
    from numba import njit  # type: ignore[import-untyped,unused-ignore]

    ONE = np.uint64(1)
    ZERO = np.uint64(0)

    @njit(cache=True)
    def split_rows(masks, weights, comp_mask, smallest, full, leaf_log, total):  # pragma: no cover
        """
        The unmerged edge and component-edge rows of a cluster's splits.

        Returns ``(ok, edge_keys, edge_cols, comp_keys, comp_cols)``, all column-major (one row
        per column, so each column is contiguous): ``ok`` is ``False`` when some split cuts two
        components; ``edge_keys`` is ``uint64[3, E]`` (the two "wholly on one side" masks, smaller
        first, and the owner bit; rows that can never pay are dropped) with ``edge_cols``
        ``float64[3, E]`` (``w``, then ``w (1 - q)`` of the two sides in the same order);
        ``comp_keys`` is ``uint64[4, E_c]`` (component, away, down, up) with ``comp_cols``
        ``float64[4, E_c]`` (``w``, ``w q_down``, ``w q_up``, ``w q_down q_up``, with ``q`` the
        side constant of the components in the side's mask, as
        :func:`~forestpd.algo.profile._dies_columns`).
        """
        n = masks.shape[0]
        k = comp_mask.shape[0]
        n_leaves = leaf_log.shape[0]
        edge_keys = np.empty((3, n), dtype=np.uint64)
        edge_cols = np.empty((3, n), dtype=np.float64)
        comp_keys = np.empty((4, n), dtype=np.uint64)
        comp_cols = np.empty((4, n), dtype=np.float64)
        comp_log = np.zeros(k)
        for c in range(k):
            for leaf in range(n_leaves):
                if (comp_mask[c] >> np.uint64(leaf)) & ONE:
                    comp_log[c] += leaf_log[leaf]
        n_edge = 0
        n_comp = 0
        for s in range(n):
            side_a = masks[s]
            side_b = ~side_a & full
            only_a = ZERO
            only_b = ZERO
            owner = -1
            n_inside = 0
            log_only_a = 0.0
            log_only_b = 0.0
            for c in range(k):
                in_a = (side_a & comp_mask[c]) != ZERO
                in_b = (side_b & comp_mask[c]) != ZERO
                if in_a and in_b:
                    n_inside += 1
                    owner = c
                elif in_a:
                    only_a |= ONE << np.uint64(c)
                    log_only_a += comp_log[c]
                elif in_b:
                    only_b |= ONE << np.uint64(c)
                    log_only_b += comp_log[c]
            if n_inside > 1:
                return False, edge_keys[:, :0], edge_cols[:, :0], comp_keys[:, :0], comp_cols[:, :0]
            log_a = 0.0
            for leaf in range(n_leaves):
                if (side_a >> np.uint64(leaf)) & ONE:
                    log_a += leaf_log[leaf]
            log_b = total - log_a
            w = weights[s]
            survive_a = w * (0.0 - np.expm1(log_a))
            survive_b = w * (0.0 - np.expm1(log_b))
            if only_a <= only_b:
                low, high, survive_low, survive_high = only_a, only_b, survive_a, survive_b
            else:
                low, high, survive_low, survive_high = only_b, only_a, survive_b, survive_a
            if high != ZERO and (low != ZERO or survive_low != 0.0):
                edge_keys[0, n_edge] = low
                edge_keys[1, n_edge] = high
                edge_keys[2, n_edge] = ONE << np.uint64(owner) if owner >= 0 else ZERO
                edge_cols[0, n_edge] = w
                edge_cols[1, n_edge] = survive_low
                edge_cols[2, n_edge] = survive_high
                n_edge += 1
            if owner >= 0:
                cm = comp_mask[owner]
                if (side_a & (ONE << np.uint64(smallest[owner]))) == ZERO:
                    away, down, up = side_a & cm, only_a, only_b
                    q_down, q_up = np.exp(log_only_a), np.exp(log_only_b)
                else:
                    away, down, up = side_b & cm, only_b, only_a
                    q_down, q_up = np.exp(log_only_b), np.exp(log_only_a)
                comp_keys[0, n_comp] = np.uint64(owner)
                comp_keys[1, n_comp] = away
                comp_keys[2, n_comp] = down
                comp_keys[3, n_comp] = up
                comp_cols[0, n_comp] = w
                comp_cols[1, n_comp] = w * q_down
                comp_cols[2, n_comp] = w * q_up
                comp_cols[3, n_comp] = w * q_down * q_up
                n_comp += 1
        return (
            True,
            edge_keys[:, :n_edge],
            edge_cols[:, :n_edge],
            comp_keys[:, :n_comp],
            comp_cols[:, :n_comp],
        )

    @njit(cache=True)
    def merge_rows(keys, cols):  # pragma: no cover - compiled
        """
        The distinct key rows in lexicographic order, with their summed weight columns.

        Column-major: ``keys`` is ``uint64[m, n]`` and ``cols`` ``float64[c, n]``.
        """
        m, n = keys.shape
        n_cols = cols.shape[0]
        order = np.arange(n)
        for col in range(m - 1, -1, -1):  # stable sorts, last key first
            column = np.empty(n, dtype=np.uint64)
            for i in range(n):
                column[i] = keys[col, order[i]]
            order = order[np.argsort(column, kind="mergesort")]
        out_keys = np.empty_like(keys)
        out_cols = np.empty((n_cols, n), dtype=np.float64)
        count = 0
        for i in range(n):
            r = order[i]
            same = count > 0
            if same:
                for col in range(m):
                    if keys[col, r] != out_keys[col, count - 1]:
                        same = False
                        break
            if same:
                for col in range(n_cols):
                    out_cols[col, count - 1] += cols[col, r]
            else:
                for col in range(m):
                    out_keys[col, count] = keys[col, r]
                for col in range(n_cols):
                    out_cols[col, count] = cols[col, r]
                count += 1
        return out_keys[:, :count], out_cols[:, :count]

    @njit(cache=True)
    def plain_rows(edge_keys, edge_cols, comp_keys, comp_cols, star_bit):  # pragma: no cover
        """
        The plain profile's rows from the plus profile's (column-major, as :func:`split_rows`):
        ``x_C^*``'s bit cleared, rows that can never pay dropped, equal rows merged. The side
        constants are unchanged.
        """
        keep = ~star_bit
        n = edge_keys.shape[1]
        e_keys = np.empty((3, n), dtype=np.uint64)
        e_cols = np.empty((3, n), dtype=np.float64)
        count = 0
        for i in range(n):
            a = edge_keys[0, i] & keep
            b = edge_keys[1, i] & keep
            survive_a = edge_cols[1, i]
            survive_b = edge_cols[2, i]
            if a <= b:
                low, high, survive_low, survive_high = a, b, survive_a, survive_b
            else:
                low, high, survive_low, survive_high = b, a, survive_b, survive_a
            if high != ZERO and (low != ZERO or survive_low != 0.0):
                e_keys[0, count] = low
                e_keys[1, count] = high
                e_keys[2, count] = edge_keys[2, i]
                e_cols[0, count] = edge_cols[0, i]
                e_cols[1, count] = survive_low
                e_cols[2, count] = survive_high
                count += 1
        c_keys = comp_keys.copy()
        for i in range(c_keys.shape[1]):
            c_keys[2, i] &= keep
            c_keys[3, i] &= keep
        edges = merge_rows(e_keys[:, :count], e_cols[:, :count])
        comps = merge_rows(c_keys, comp_cols)
        return edges[0], edges[1], comps[0], comps[1]

    @njit(cache=True)
    def split_sides(masks, weights, leaf_log, total):  # pragma: no cover - compiled
        """
        :func:`~forestpd.algo.survival.cluster_sides` from a split table: per local leaf ``l``,
        the summed ``w (1 - q)`` of the side without ``l``, and the summed ``w (1 - q_a)(1 - q_b)``.
        """
        n_leaves = leaf_log.shape[0]
        at_leaf = np.zeros(n_leaves)
        base = 0.0
        empty = 0.0
        for s in range(masks.shape[0]):
            side = masks[s]
            log_a = 0.0
            for leaf in range(n_leaves):
                if (side >> np.uint64(leaf)) & ONE:
                    log_a += leaf_log[leaf]
            survive_a = 0.0 - np.expm1(log_a)
            survive_b = 0.0 - np.expm1(total - log_a)
            w = weights[s]
            base += w * survive_a
            empty += w * survive_a * survive_b
            change = w * survive_b - w * survive_a
            for leaf in range(n_leaves):
                if (side >> np.uint64(leaf)) & ONE:
                    at_leaf[leaf] += change
        for leaf in range(n_leaves):
            at_leaf[leaf] += base
        return at_leaf, empty

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - depends on the environment
    split_rows = None
    merge_rows = None
    plain_rows = None
    split_sides = None
    HAVE_NUMBA = False


__all__ = ["HAVE_NUMBA", "split_rows", "merge_rows", "plain_rows", "split_sides"]
