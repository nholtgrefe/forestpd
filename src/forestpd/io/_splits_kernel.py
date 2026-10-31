"""
The compiled passes of :func:`~forestpd.io.splits.split_system`.

The split system of many trees on many taxa is mostly repetition: every tree induces about ``n``
splits, and most recur across trees. These kernels find every edge's split in one pass per tree
and list the taxa of each *distinct* split only once:

* :func:`split_keys` roots each tree at leaf 0 and gives every edge the key of its lower side (the
  side without taxon 0): two sums of random 64-bit hashes of the taxa below it, and their number.
  Equal splits get equal keys; two different splits share a key with probability about
  ``2^-128``.
* :func:`split_members` lists the taxa below one representative edge per distinct split.
* :func:`smaller_sides` replaces each split's side by the smaller of its two sides (for the split
  table of :mod:`forestpd._evaluate`).

Trees are packed: the edges of all trees in one array, tree ``j``'s at
``edge_offset[j]:edge_offset[j+1]``, with node ids local to the tree and ``0..n_leaves-1`` the
leaves (leaf id == taxon id).
"""

from __future__ import annotations

from typing import Any

#: The compiled kernels, or ``None`` when Numba is unavailable.
split_keys: Any
split_members: Any
smaller_sides: Any

try:  # pragma: no cover - depends on the environment
    import numpy as np
    from numba import njit  # type: ignore[import-untyped,unused-ignore]

    @njit(cache=True)
    def _rooted(edges, a, n_nodes, root):  # pragma: no cover - compiled
        """Parent, parent edge (local index) and preorder of one packed tree rooted at ``root``."""
        m = n_nodes - 1
        degree = np.zeros(n_nodes + 1, dtype=np.int64)
        for e in range(m):
            degree[edges[a + e, 0] + 1] += 1
            degree[edges[a + e, 1] + 1] += 1
        start = np.cumsum(degree)
        fill = start[:-1].copy()
        to = np.empty(2 * m, dtype=np.int64)
        via = np.empty(2 * m, dtype=np.int64)
        for e in range(m):
            u, v = edges[a + e, 0], edges[a + e, 1]
            to[fill[u]] = v
            via[fill[u]] = e
            fill[u] += 1
            to[fill[v]] = u
            via[fill[v]] = e
            fill[v] += 1
        parent = np.full(n_nodes, -1, dtype=np.int64)
        parent_edge = np.full(n_nodes, -1, dtype=np.int64)
        order = np.empty(n_nodes, dtype=np.int64)
        stack = np.empty(n_nodes, dtype=np.int64)
        seen = np.zeros(n_nodes, dtype=np.bool_)
        stack[0] = root
        seen[root] = True
        top, pos = 1, 0
        while top > 0:
            top -= 1
            v = stack[top]
            order[pos] = v
            pos += 1
            for i in range(start[v], start[v + 1]):
                w = to[i]
                if not seen[w]:
                    seen[w] = True
                    parent[w] = v
                    parent_edge[w] = via[i]
                    stack[top] = w
                    top += 1
        return parent, parent_edge, order

    @njit(cache=True)
    def split_keys(
        edges, edge_offset, n_nodes, n_leaves, lengths, tree_weight, h1, h2
    ):  # pragma: no cover - compiled
        """
        Per packed edge: the key ``(k1, k2, count)`` of its side without taxon 0, its weight
        ``w_j * len_j(e)``, and its tree and lower node.
        """
        total = edges.shape[0]
        k1 = np.empty(total, dtype=np.uint64)
        k2 = np.empty(total, dtype=np.uint64)
        count = np.empty(total, dtype=np.int64)
        weight = np.empty(total, dtype=np.float64)
        tree = np.empty(total, dtype=np.int64)
        lower = np.empty(total, dtype=np.int64)
        for j in range(n_nodes.shape[0]):
            a = edge_offset[j]
            nn = n_nodes[j]
            parent, parent_edge, order = _rooted(edges, a, nn, 0)
            s1 = np.zeros(nn, dtype=np.uint64)
            s2 = np.zeros(nn, dtype=np.uint64)
            c = np.zeros(nn, dtype=np.int64)
            for x in range(1, n_leaves):  # leaf 0 is the root: never below an edge
                s1[x] = h1[x]
                s2[x] = h2[x]
                c[x] = 1
            for i in range(nn - 1, 0, -1):
                v = order[i]
                p = parent[v]
                e = a + parent_edge[v]
                k1[e] = s1[v]
                k2[e] = s2[v]
                count[e] = c[v]
                weight[e] = tree_weight[j] * lengths[e]
                tree[e] = j
                lower[e] = v
                s1[p] += s1[v]
                s2[p] += s2[v]
                c[p] += c[v]
        return k1, k2, count, weight, tree, lower

    @njit(cache=True)
    def split_members(
        edges, edge_offset, n_nodes, n_leaves, rep_tree, rep_node, rep_count
    ):  # pragma: no cover - compiled
        """
        The taxa below each representative edge, as one flat array with offsets (the
        representatives sorted by tree; each side's taxa in increasing order).
        """
        n_reps = rep_tree.shape[0]
        out_offset = np.zeros(n_reps + 1, dtype=np.int64)
        for r in range(n_reps):
            out_offset[r + 1] = out_offset[r] + rep_count[r]
        out = np.empty(out_offset[-1], dtype=np.int64)
        r = 0
        while r < n_reps:
            j = rep_tree[r]
            nn = n_nodes[j]
            parent, parent_edge, order = _rooted(edges, edge_offset[j], nn, 0)
            tin = np.empty(nn, dtype=np.int64)
            for i in range(nn):
                tin[order[i]] = i
            size = np.ones(nn, dtype=np.int64)  # subtree sizes in nodes: preorder intervals
            for i in range(nn - 1, 0, -1):
                size[parent[order[i]]] += size[order[i]]
            while r < n_reps and rep_tree[r] == j:
                v = rep_node[r]
                k = out_offset[r]
                for i in range(tin[v], tin[v] + size[v]):
                    x = order[i]
                    if x < n_leaves:
                        out[k] = x
                        k += 1
                out[out_offset[r] : out_offset[r + 1]].sort()
                r += 1
        return out, out_offset

    @njit(cache=True)
    def smaller_sides(members, offsets, n_taxa):  # pragma: no cover - compiled
        """
        Each split's smaller side, as a flat array with offsets: the stored side where it holds at
        most half of the taxa, else its complement (each side's taxa in increasing order).
        """
        n_splits = offsets.shape[0] - 1
        out_offset = np.zeros(n_splits + 1, dtype=np.int64)
        for r in range(n_splits):
            size = offsets[r + 1] - offsets[r]
            out_offset[r + 1] = out_offset[r] + min(size, n_taxa - size)
        out = np.empty(out_offset[-1], dtype=np.int64)
        mark = np.zeros(n_taxa, dtype=np.bool_)
        for r in range(n_splits):
            a, b = offsets[r], offsets[r + 1]
            k = out_offset[r]
            if 2 * (b - a) <= n_taxa:
                for i in range(a, b):
                    out[k] = members[i]
                    k += 1
                continue
            for i in range(a, b):
                mark[members[i]] = True
            for x in range(n_taxa):
                if not mark[x]:
                    out[k] = x
                    k += 1
            for i in range(a, b):
                mark[members[i]] = False
        return out, out_offset

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - depends on the environment
    split_keys = None
    split_members = None
    smaller_sides = None
    HAVE_NUMBA = False


__all__ = ["HAVE_NUMBA", "split_keys", "split_members", "smaller_sides"]
