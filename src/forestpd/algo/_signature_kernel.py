"""
The compiled signature loop of :func:`~forestpd.algo.local_solver.solve_local`.

Per cluster the solver enumerates up to ``2^k`` signatures, and for each it computes the
definitely-chosen weight, the two-sided weights, one component DP per active component, and a
max-plus fold. In Python each signature costs tens of microseconds of interpreter overhead, which
dominates once ``k`` is large. This kernel runs the whole loop compiled, when Numba is installed.

It mirrors the Python code step by step (:func:`~forestpd.algo.local_solver.signature_weights`,
:mod:`~forestpd.algo.component_dp`, :func:`~forestpd.algo.convolve.fold_max_plus`) and
enumerates signatures in the same order, so ties resolve to the same signature. Sums may differ
from NumPy's pairwise summation in the last bits.

Component trees are packed into the shared ``R_i`` address space of the profile: global node
``node_offset[i] + v`` is node ``v`` of ``R_i``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: The compiled kernel, or ``None`` when Numba is unavailable.
signature_loop: Any

try:  # pragma: no cover - depends on the environment
    from numba import njit  # type: ignore[import-untyped,unused-ignore]

    NEG = -np.inf

    #: Constant runs shorter than this are not worth the special case.
    MIN_RUN = 8
    #: Below this table length the plain double loop is faster than looking for runs.
    MIN_SHORTCUT = 128

    @njit(cache=True, inline="always")
    def _constant_run(b):  # pragma: no cover - compiled
        """
        ``(m, e)`` with ``b[m:e]`` constant, ``e`` the end of ``b`` or one before it (a table
        may differ in its last entry only), and ``m`` as small as possible; ``m == e`` if there
        is no long run.
        """
        n = b.shape[0]
        best_m, best_e = n, n
        for e in (n, n - 1):
            if e < 1:
                continue
            m = e - 1
            while m > 0 and b[m - 1] == b[e - 1]:
                m -= 1
            if e - m >= MIN_RUN and m < best_m:
                best_m, best_e = m, e
        return best_m, best_e

    @njit(cache=True, inline="always")
    def _max_plus_plain(a, b, out):  # pragma: no cover - compiled
        """``out[p + q] = max(out[p + q], a[p] + b[q])``: the plain double loop."""
        n_a, n_b, n_out = a.shape[0], b.shape[0], out.shape[0]
        for p in range(n_a):
            av = a[p]
            if av == NEG:
                continue
            hi = n_b if p + n_b <= n_out else n_out - p
            for q in range(hi):
                value = av + b[q]
                if value > out[p + q]:
                    out[p + q] = value

    @njit(cache=True, inline="always")
    def _max_plus_run(a, b, m, e, out):  # pragma: no cover - compiled
        """:func:`_max_plus_into` where ``b`` is constant on ``[m, e)``."""
        n_a, n_b, n_out = a.shape[0], b.shape[0], out.shape[0]
        for p in range(n_a):
            av = a[p]
            if av == NEG:
                continue
            for q in range(min(m, n_out - p)):
                value = av + b[q]
                if value > out[p + q]:
                    out[p + q] = value
            for q in range(e, min(n_b, n_out - p)):
                value = av + b[q]
                if value > out[p + q]:
                    out[p + q] = value
        constant = b[m]
        if constant == NEG:
            return
        running = NEG  # max of a[0 .. r - m]
        for r in range(m, n_out):
            if r - m < n_a and a[r - m] > running:
                running = a[r - m]
            low = r - e + 1
            if low <= 0:
                if running == NEG:
                    continue
                value = running + constant
            else:  # the run no longer reaches back to p = 0: take the window directly
                window = NEG
                for p in range(low, min(r - m, n_a - 1) + 1):
                    if a[p] > window:
                        window = a[p]
                if window == NEG:
                    continue
                value = window + constant
            if value > out[r]:
                out[r] = value

    @njit(cache=True)
    def _max_plus_into(a, b, out):  # pragma: no cover - compiled
        """
        ``out[p + q] = max(out[p + q], a[p] + b[q])``; ``out`` pre-filled by the caller.

        Where one table is constant on a long run ``[m, e)`` (a plain taxon from its cost on, a
        child cluster's table past its own budget cap), the run's contribution to ``out[r]`` is
        that constant plus the maximum of the other table over ``r - e < p <= r - m``: a running
        maximum instead of a loop, so the cost drops from ``O(|a| |b|)`` to ``O(|a| m)``. Adding
        the same constant preserves the order of floats, so the result is bit-identical to the
        plain double loop. Callers use the plain loop for short tables, where it is faster.
        """
        m_a, e_a = _constant_run(a)
        m_b, e_b = _constant_run(b)
        if m_b < e_b and m_b <= m_a:
            _max_plus_run(a, b, m_b, e_b, out)
        elif m_a < e_a:
            _max_plus_run(b, a, m_a, e_a, out)
        else:
            _max_plus_plain(a, b, out)

    @njit(cache=True)
    def _component_table(
        comp, node_offset, left, right, leaf, axis, post, post_offset, roots, lam_down, lam_up,
        lam_both, survive_down, survive_up, f2, idle, budget, min_selected, occ, empty,
        table, scratch,
    ):  # fmt: skip  # pragma: no cover - compiled
        """
        :func:`~forestpd.algo.component_dp.component_table` for component ``comp``, into
        ``table``; ``scratch`` is a work buffer of length ``B + 1``.
        """
        width = budget + 1
        for i in range(post_offset[comp], post_offset[comp + 1]):
            node = post[i]
            x = leaf[node]
            if x >= 0:  # _init_leaf
                for beta in range(width):
                    occ[node, 1, beta] = f2[x, beta]
                    occ[node, 0, beta] = f2[x, beta] if min_selected == 1 else NEG
                empty[node, 1] = idle[x]
                empty[node, 0] = NEG
                continue
            # _combine, with gammas' gamma[q][s] per child edge.
            w1, w2 = left[node], right[node]
            d1, u1, b1 = lam_down[w1], lam_up[w1], lam_both[w1]
            p1, r1 = survive_down[w1], survive_up[w1]
            d2, u2, b2 = lam_down[w2], lam_up[w2], lam_both[w2]
            p2, r2 = survive_down[w2], survive_up[w2]
            g1_00, g1_01 = d1 * p1 + u1 * r1 + b1 * p1 * r1, d1 * p1 + u1 + b1 * p1
            g1_10, g1_11 = d1 + u1 * r1 + b1 * r1, d1 + u1 + b1
            g2_00, g2_01 = d2 * p2 + u2 * r2 + b2 * p2 * r2, d2 * p2 + u2 + b2 * p2
            g2_10, g2_11 = d2 + u2 * r2 + b2 * r2, d2 + u2 + b2
            shared = scratch
            shared[:] = NEG
            if width >= MIN_SHORTCUT:
                _max_plus_into(occ[w1, 1], occ[w2, 1], shared)
            else:
                _max_plus_plain(occ[w1, 1], occ[w2, 1], shared)
            bonus = g1_11 + g2_11
            for beta in range(width):
                shared[beta] = shared[beta] + bonus
            off1, off2 = axis[w1], axis[w2]
            for s in range(2):
                g1_1s = g1_11 if s == 1 else g1_10
                g2_1s = g2_11 if s == 1 else g2_10
                g1_0s = g1_01 if s == 1 else g1_00
                g2_0s = g2_01 if s == 1 else g2_00
                for beta in range(width):
                    occ[node, s, beta] = shared[beta]
                add_left = (g1_1s + empty[w2, 1]) + g2_01
                add_right = (g2_1s + empty[w1, 1]) + g1_01
                if off1 == 0 and off2 == 0:  # always on DP: no shift
                    for beta in range(width):
                        value = occ[w1, s, beta] + add_left
                        if value > occ[node, s, beta]:
                            occ[node, s, beta] = value
                        value = occ[w2, s, beta] + add_right
                        if value > occ[node, s, beta]:
                            occ[node, s, beta] = value
                else:
                    for beta in range(width):
                        value = occ[w1, s, max(beta - off2, 0)] + add_left
                        if value > occ[node, s, beta]:
                            occ[node, s, beta] = value
                        value = occ[w2, s, max(beta - off1, 0)] + add_right
                        if value > occ[node, s, beta]:
                            occ[node, s, beta] = value
                empty[node, s] = ((empty[w1, s] + g1_0s) + empty[w2, s]) + g2_0s
        root = roots[comp]
        for beta in range(width):
            table[beta] = occ[root, 0, beta]

    @njit(cache=True)
    def signature_loop(
        edge_below, edge_above, edge_owner, edge_weight, edge_survive_below, edge_survive_above,
        ce_slot, ce_own, ce_down, ce_up, ce_weight, ce_dies_down, ce_dies_up, ce_dies_both,
        survive_down, survive_up, node_offset, left, right, leaf, axis, post, post_offset, roots,
        f2, idle, comp_idle, comp_saved, budget, forced, complement,
    ):  # fmt: skip  # pragma: no cover - compiled
        """
        :func:`~forestpd.algo.local_solver.solve_local`'s loop over all signatures.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``best`` (``float64[B+1]``) and ``best_signature`` (``int64[B+1]``, ``-1`` where
            nothing is feasible).
        """
        k = roots.shape[0]
        width = budget + 1
        n_slots = node_offset[k]
        best = np.full(width, NEG)
        best_signature = np.full(width, -1, dtype=np.int64)
        occ = np.empty((max(n_slots, 1), 2, width))
        empty = np.empty((max(n_slots, 1), 2))
        lam_down = np.zeros(n_slots)
        lam_up = np.zeros(n_slots)
        lam_both = np.zeros(n_slots)
        tables = np.empty((k, width))
        acc = np.empty(width)
        nxt = np.empty(width)
        scratch = np.empty(width)

        optional = np.empty(k, dtype=np.int64)
        m = 0
        for c in range(k):
            if c != forced:
                optional[m] = c
                m += 1
        low_size = 1 if forced < 0 else 0
        high = m if complement else min(m, budget)
        active = np.empty(k, dtype=np.int64)
        chosen = np.empty(k, dtype=np.int64)
        for size in range(low_size, high + 1):
            for i in range(size):
                chosen[i] = i
            more = True
            while more:
                # The signature: optional[chosen[:size]] plus the forced component, sorted.
                mask = np.uint64(0) if forced < 0 else np.uint64(1) << np.uint64(forced)
                n_active = 0
                placed = forced < 0
                for i in range(size):
                    comp = optional[chosen[i]]
                    if not placed and forced < comp:
                        active[n_active] = forced
                        n_active += 1
                        placed = True
                    active[n_active] = comp
                    n_active += 1
                    mask |= np.uint64(1) << np.uint64(comp)
                if not placed:
                    active[n_active] = forced
                    n_active += 1

                # signature_weights
                d_s = 0.0
                for e in range(edge_weight.shape[0]):
                    has_below = (edge_below[e] & mask) != 0
                    has_above = (edge_above[e] & mask) != 0
                    if has_below and has_above:
                        d_s += edge_weight[e]
                    elif (edge_owner[e] & mask) == 0:
                        if has_above:
                            d_s += edge_survive_below[e]
                        elif has_below:
                            d_s += edge_survive_above[e]
                for slot in range(n_slots):
                    lam_down[slot] = 0.0
                    lam_up[slot] = 0.0
                    lam_both[slot] = 0.0
                for e in range(ce_weight.shape[0]):
                    if (ce_own[e] & mask) == 0:
                        continue
                    down_covered = (ce_down[e] & mask) != 0
                    up_covered = (ce_up[e] & mask) != 0
                    w = ce_weight[e]
                    if not down_covered and up_covered:
                        lam_down[ce_slot[e]] += ce_dies_down[e]
                        d_s += w - ce_dies_down[e]
                    elif down_covered and not up_covered:
                        lam_up[ce_slot[e]] += ce_dies_up[e]
                        d_s += w - ce_dies_up[e]
                    elif not down_covered and not up_covered:
                        dd, du, db = ce_dies_down[e], ce_dies_up[e], ce_dies_both[e]
                        lam_down[ce_slot[e]] += dd - db
                        lam_up[ce_slot[e]] += du - db
                        lam_both[ce_slot[e]] += db
                        d_s += ((w - dd) - du) + db
                # inactive_idle and inactive_saved
                idle_sum = 0.0
                saved = 0
                for c in range(k):
                    if (mask & (np.uint64(1) << np.uint64(c))) == 0:
                        idle_sum += comp_idle[c]
                        saved += comp_saved[c]
                d_s += idle_sum

                min_selected = 2 if n_active == 1 else 1
                for i in range(n_active):
                    _component_table(
                        active[i], node_offset, left, right, leaf, axis, post, post_offset, roots,
                        lam_down, lam_up, lam_both, survive_down, survive_up, f2, idle, budget,
                        min_selected, occ, empty, tables[i], scratch,
                    )  # fmt: skip
                # fold_max_plus, then (DP') saved_shift, then + d_s
                for beta in range(width):
                    acc[beta] = tables[0, beta]
                for i in range(1, n_active):
                    for beta in range(width):
                        nxt[beta] = NEG
                    if width >= MIN_SHORTCUT:
                        _max_plus_into(acc, tables[i], nxt)
                    else:
                        _max_plus_plain(acc, tables[i], nxt)
                    for beta in range(width):
                        acc[beta] = nxt[beta]
                for beta in range(width):
                    candidate = acc[max(beta - saved, 0)] + d_s
                    if candidate > best[beta]:
                        best[beta] = candidate
                        best_signature[beta] = np.int64(mask)

                # Next combination of size `size` from range(m), lexicographically.
                if size == 0:
                    more = False
                else:
                    i = size - 1
                    while i >= 0 and chosen[i] == m - size + i:
                        i -= 1
                    if i < 0:
                        more = False
                    else:
                        chosen[i] += 1
                        for j in range(i + 1, size):
                            chosen[j] = chosen[j - 1] + 1
        return best, best_signature

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - only if Numba cannot be imported
    signature_loop = None
    HAVE_NUMBA = False


__all__ = ["HAVE_NUMBA", "signature_loop"]
