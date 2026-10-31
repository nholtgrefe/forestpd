"""
Compiled per-tree passes of the cluster reduction, used when Numba is installed.

The reduction touches every edge of every input tree a few times: root the tree at leaf 0, find
each edge's split interval (Day's numbering), label every node with its cluster, and key every edge
by its split in its cluster's local instance. In pure Python that is a few milliseconds per tree,
which dominates for thousands of trees. Here all trees are packed into flat arrays and each pass is
one compiled loop over them.

Local split keys are exact ``uint64`` bitmasks, so this path is used only when every cluster has at
most 64 local leaves; :mod:`~forestpd.cluster.common` falls back to its Python path otherwise.

Packed layout: tree ``j`` owns edges ``edge_offset[j]:edge_offset[j + 1]`` of ``edges`` (node ids
local to the tree, leaves ``0..n_leaves-1``) and has ``n_nodes[j]`` nodes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: The compiled kernels, or ``None`` when Numba is unavailable.
split_keys: Any
local_edges: Any
tree_labels: Any
segment_hashes: Any
assign_groups: Any
cluster_order: Any

try:  # pragma: no cover - depends on the environment
    from numba import njit  # type: ignore[import-untyped,unused-ignore]

    @njit(cache=True)
    def _root_at_leaf_zero(edges, n_nodes):  # pragma: no cover - compiled
        """Parent, parent edge and a BFS order (parents first) of one tree rooted at node 0."""
        n_edges = edges.shape[0]
        degree = np.zeros(n_nodes + 1, dtype=np.int64)
        for e in range(n_edges):
            degree[edges[e, 0] + 1] += 1
            degree[edges[e, 1] + 1] += 1
        start = np.cumsum(degree)
        fill = start[:-1].copy()
        to = np.empty(2 * n_edges, dtype=np.int64)
        via = np.empty(2 * n_edges, dtype=np.int64)
        for e in range(n_edges):
            u, v = edges[e, 0], edges[e, 1]
            to[fill[u]] = v
            via[fill[u]] = e
            fill[u] += 1
            to[fill[v]] = u
            via[fill[v]] = e
            fill[v] += 1
        parent = np.full(n_nodes, -1, dtype=np.int64)
        parent_edge = np.full(n_nodes, -1, dtype=np.int64)
        order = np.empty(n_nodes, dtype=np.int64)
        order[0] = 0
        head, tail = 0, 1
        while head < tail:
            v = order[head]
            head += 1
            for i in range(start[v], start[v + 1]):
                if via[i] != parent_edge[v]:
                    w = to[i]
                    parent[w] = v
                    parent_edge[w] = via[i]
                    order[tail] = w
                    tail += 1
        return parent, parent_edge, order

    @njit(cache=True)
    def _interval_keys(parent, parent_edge, order, n_leaves, index, n_edges):  # pragma: no cover
        """
        Per edge, ``lo * n + hi`` when the leaves below it are the indices ``lo..hi`` and number
        between 2 and ``n - 2``; ``-1`` otherwise.
        """
        n_nodes = parent.shape[0]
        lo = np.full(n_nodes, n_leaves, dtype=np.int64)
        hi = np.full(n_nodes, -1, dtype=np.int64)
        count = np.zeros(n_nodes, dtype=np.int64)
        for leaf in range(n_leaves):
            lo[leaf] = index[leaf]
            hi[leaf] = index[leaf]
            count[leaf] = 1
        for i in range(n_nodes - 1, 0, -1):
            v = order[i]
            p = parent[v]
            if lo[v] < lo[p]:
                lo[p] = lo[v]
            if hi[v] > hi[p]:
                hi[p] = hi[v]
            count[p] += count[v]
        keys = np.full(n_edges, -1, dtype=np.int64)
        for v in range(n_nodes):
            e = parent_edge[v]
            if e >= 0:
                size = count[v]
                if hi[v] - lo[v] + 1 == size and 2 <= size <= n_leaves - 2:
                    keys[e] = lo[v] * n_leaves + hi[v]
        return keys

    @njit(cache=True)
    def _labels(parent, parent_edge, order, keys, common):  # pragma: no cover - compiled
        """Cluster per node: ``c + 1`` below the common split ``common[c]``, else the parent's."""
        n_nodes = parent.shape[0]
        label = np.zeros(n_nodes, dtype=np.int64)
        for i in range(1, n_nodes):
            v = order[i]
            key = keys[parent_edge[v]]
            cut = -1
            if key >= 0 and common.shape[0] > 0:
                at = np.searchsorted(common, key)
                if at < common.shape[0] and common[at] == key:
                    cut = at
            label[v] = label[parent[v]] if cut < 0 else cut + 1
        return label

    @njit(cache=True)
    def split_keys(edges, edge_offset, n_nodes, n_leaves, index):  # pragma: no cover - compiled
        """
        :func:`_interval_keys` of every packed tree.

        Parameters
        ----------
        edges : numpy.ndarray
            ``int32[E, 2]``; all trees' edges, packed.
        edge_offset : numpy.ndarray
            ``int64[t + 1]``; tree ``j``'s edges are ``edge_offset[j]:edge_offset[j + 1]``.
        n_nodes : numpy.ndarray
            ``int64[t]``; node count per tree.
        n_leaves : int
            Number of taxa.
        index : numpy.ndarray
            ``int64[n_leaves]``; Day's leaf numbering.

        Returns
        -------
        numpy.ndarray
            ``int64[E]``; the interval key of every packed edge, ``-1`` if not a candidate split.
        """
        out = np.empty(edges.shape[0], dtype=np.int64)
        for j in range(n_nodes.shape[0]):
            a, b = edge_offset[j], edge_offset[j + 1]
            tree_edges = edges[a:b]
            parent, parent_edge, order = _root_at_leaf_zero(tree_edges, n_nodes[j])
            out[a:b] = _interval_keys(parent, parent_edge, order, n_leaves, index, b - a)
        return out

    @njit(cache=True)
    def local_edges(
        edges, lengths, edge_offset, n_nodes, n_leaves, index, common, slot, child_slot, weights
    ):  # pragma: no cover - compiled
        """
        Every packed edge's owning cluster, local split mask, endpoints and weighted length.

        An edge belongs to the cluster of its endpoint nearer leaf 0. Its mask holds the local
        leaves below it: the cluster's own taxa, and for a cut-edge the pseudo-leaf ``x_D`` of the
        child cluster ``D`` it leads to.

        Parameters
        ----------
        edges, edge_offset, n_nodes, n_leaves, index
            As for :func:`split_keys`.
        lengths : numpy.ndarray
            ``float64[E]``; packed branch lengths.
        common : numpy.ndarray
            ``int64[k]``; sorted interval keys of the common splits (cut ``c`` is ``common[c]``).
        slot : numpy.ndarray
            ``int64[n_leaves]``; local leaf id of each taxon in its own cluster.
        child_slot : numpy.ndarray
            ``int64[n_clusters]``; local leaf id of each cluster's pseudo-leaf in its parent.
        weights : numpy.ndarray
            ``float64[t]``; per-tree weights.

        Returns
        -------
        tuple[numpy.ndarray, ...]
            ``owner``, ``mask`` (``uint64``), ``upper``, ``lower`` (``-1 - D`` for the pendant
            edge of ``x_D``) and ``wlen`` (weight times length), each of length ``E``; and
            ``top`` (``int32[t, n_clusters]``), each cluster's node incident to its parent cut
            (``-1`` at the root cluster).
        """
        n_edges_total = edges.shape[0]
        t = n_nodes.shape[0]
        n_clusters = child_slot.shape[0]
        owner = np.empty(n_edges_total, dtype=np.int32)
        mask = np.empty(n_edges_total, dtype=np.uint64)
        upper = np.empty(n_edges_total, dtype=np.int32)
        lower = np.empty(n_edges_total, dtype=np.int32)
        wlen = np.empty(n_edges_total, dtype=np.float64)
        top = np.full((t, n_clusters), -1, dtype=np.int32)
        one = np.uint64(1)
        for j in range(t):
            a, b = edge_offset[j], edge_offset[j + 1]
            tree_edges = edges[a:b]
            parent, parent_edge, order = _root_at_leaf_zero(tree_edges, n_nodes[j])
            keys = _interval_keys(parent, parent_edge, order, n_leaves, index, b - a)
            label = _labels(parent, parent_edge, order, keys, common)
            below = np.zeros(n_nodes[j], dtype=np.uint64)
            for leaf in range(n_leaves):
                below[leaf] = one << np.uint64(slot[leaf])
            for i in range(n_nodes[j] - 1, 0, -1):  # children before parents
                v = order[i]
                p = parent[v]
                c = label[p]
                e = a + parent_edge[v]
                if label[v] == c:
                    key = below[v]
                    lower[e] = v
                else:
                    key = one << np.uint64(child_slot[label[v]])
                    lower[e] = -1 - label[v]
                    top[j, label[v]] = v
                below[p] |= key
                owner[e] = c
                mask[e] = key
                upper[e] = p
                wlen[e] = weights[j] * lengths[e]
        return owner, mask, upper, lower, wlen, top

    @njit(cache=True)
    def tree_labels(edges, n_nodes, n_leaves, index, common):  # pragma: no cover - compiled
        """Cluster per node of one tree (see :func:`_labels`)."""
        parent, parent_edge, order = _root_at_leaf_zero(edges, n_nodes)
        keys = _interval_keys(parent, parent_edge, order, n_leaves, index, edges.shape[0])
        return _labels(parent, parent_edge, order, keys, common)

    @njit(cache=True)
    def cluster_order(owner, tree_of_edge, mask, n_clusters):  # pragma: no cover - compiled
        """
        The edge order by ``(cluster, tree, mask)``, in linear time.

        A stable counting sort by cluster keeps the packed tree order, so each ``(cluster, tree)``
        run is contiguous; each run (one local tree, a few edges) is then insertion-sorted by mask.
        Returns the order and the start of every run.
        """
        n = owner.shape[0]
        count = np.zeros(n_clusters + 1, dtype=np.int64)
        for e in range(n):
            count[owner[e] + 1] += 1
        for c in range(n_clusters):
            count[c + 1] += count[c]
        order = np.empty(n, dtype=np.int64)
        for e in range(n):
            order[count[owner[e]]] = e
            count[owner[e]] += 1
        is_start = np.zeros(n, dtype=np.bool_)
        i = 0
        while i < n:
            j = i + 1
            while (
                j < n
                and owner[order[j]] == owner[order[i]]
                and tree_of_edge[order[j]] == tree_of_edge[order[i]]
            ):
                j += 1
            is_start[i] = True
            for p in range(i + 1, j):  # insertion sort of the run by mask
                e = order[p]
                q = p - 1
                while q >= i and mask[order[q]] > mask[e]:
                    order[q + 1] = order[q]
                    q -= 1
                order[q + 1] = e
            i = j
        return order, np.flatnonzero(is_start)

    @njit(cache=True)
    def segment_hashes(masks, starts, lengths):  # pragma: no cover - compiled
        """
        A 64-bit hash of each segment ``masks[starts[s] : starts[s] + lengths[s]]``.

        Only used to bring candidate equal segments together; equality is always checked exactly
        (:func:`assign_groups`).
        """
        n = starts.shape[0]
        out = np.empty(n, dtype=np.uint64)
        for s in range(n):
            h = np.uint64(lengths[s]) * np.uint64(0x9E3779B97F4A7C15)
            for i in range(starts[s], starts[s] + lengths[s]):
                h ^= (
                    masks[i]
                    + np.uint64(0x9E3779B97F4A7C15)
                    + (h << np.uint64(6))
                    + (h >> np.uint64(2))
                )
                h ^= h >> np.uint64(31)
                h *= np.uint64(0xBF58476D1CE4E5B9)
            out[s] = h
        return out

    @njit(cache=True)
    def assign_groups(masks, starts, lengths, order, owner, hashes):  # pragma: no cover - compiled
        """
        Group segments of the same owner with equal mask rows.

        ``order`` lists the segments sorted by ``(owner, hash, segment index)``. Within a run of
        equal ``(owner, hash)`` each segment is compared exactly with the run's groups so far, so
        hash collisions cannot merge different rows. Returns each segment's group and each group's
        first (lowest-index) segment; groups are numbered in order of creation.
        """
        n = order.shape[0]
        group = np.full(n, -1, dtype=np.int64)
        first = np.empty(n, dtype=np.int64)
        n_groups = 0
        i = 0
        while i < n:
            j = i
            s0 = order[i]
            while j < n and owner[order[j]] == owner[s0] and hashes[order[j]] == hashes[s0]:
                j += 1
            run_start = n_groups
            for p in range(i, j):
                s = order[p]
                found = -1
                for g in range(run_start, n_groups):
                    r = first[g]
                    if lengths[r] != lengths[s]:
                        continue
                    same = True
                    for q in range(lengths[s]):
                        if masks[starts[r] + q] != masks[starts[s] + q]:
                            same = False
                            break
                    if same:
                        found = g
                        break
                if found < 0:
                    first[n_groups] = s
                    found = n_groups
                    n_groups += 1
                group[s] = found
            i = j
        return group, first[:n_groups]

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - only if Numba cannot be imported
    split_keys = None
    local_edges = None
    tree_labels = None
    segment_hashes = None
    assign_groups = None
    cluster_order = None
    HAVE_NUMBA = False


__all__ = [
    "HAVE_NUMBA",
    "split_keys",
    "local_edges",
    "tree_labels",
    "segment_hashes",
    "assign_groups",
]
