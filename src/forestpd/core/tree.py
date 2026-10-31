"""
Unrooted weighted phylogenetic trees and rootings of them.

:class:`PhyloTree` is the algorithmic core's own tree representation, deliberately decoupled from
PhyloZoo. It fixes one invariant that the whole package relies on:

    **Node ids ``0..n_leaves-1`` are exactly the leaves, and leaf id == node id.**

Everything downstream (agreement forests, local instances, the per-component DP) therefore indexes
leaves and nodes with the same integers, which removes a whole class of translation bugs.

Trees are *uniform*: every input tree carries the same leaf set. The paper's non-leaf-uniform
generalization (MaxMPD with non-uniform leaf sets) is W[1]-hard even when every local MAF has two
components, so it is rejected at the API boundary rather than half-supported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..exceptions import ForestPDInputError


class PhyloTree:
    """
    An unrooted, weighted phylogenetic tree.

    The constructor stores its arguments as given and performs no validation; use
    :meth:`from_edges` to build a tree from an arbitrary node numbering with checks.

    Parameters
    ----------
    n_nodes : int
        Number of nodes, including the leaves.
    n_leaves : int
        Number of leaves; nodes ``0..n_leaves-1`` must be exactly the leaves.
    edges : numpy.ndarray
        ``int32`` array of shape ``(n_edges, 2)`` holding each edge's two endpoints.
    weight : numpy.ndarray
        ``float64`` branch length per edge, aligned with ``edges``.

    Attributes
    ----------
    n_nodes : int
        Number of nodes. Nodes ``0..n_leaves-1`` are the leaves (leaf id == node id).
    n_leaves : int
        Number of leaves.
    edges : numpy.ndarray
        ``int32`` array of shape ``(n_edges, 2)`` holding each edge's two endpoints.
    weight : numpy.ndarray
        ``float64`` branch length per edge, aligned with ``edges``.

    Examples
    --------
    >>> t = PhyloTree.from_edges(3, [(0, 2), (1, 2)], [1.0, 2.0], {0: 0, 1: 1})
    >>> t.n_leaves, t.n_edges, t.total_weight()
    (2, 2, 3.0)
    """

    __slots__ = ("n_nodes", "n_leaves", "edges", "weight", "_adj", "_rooted")

    def __init__(
        self,
        n_nodes: int,
        n_leaves: int,
        edges: np.ndarray,
        weight: np.ndarray,
    ) -> None:
        self.n_nodes = int(n_nodes)
        self.n_leaves = int(n_leaves)
        self.edges = edges
        self.weight = weight
        self._adj: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._rooted: "RootedTree | None" = None

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return int(self.edges.shape[0])

    @classmethod
    def from_edges(
        cls,
        n_nodes: int,
        edges: Sequence[tuple[int, int]],
        weights: Sequence[float],
        leaf_of_node: Mapping[int, int],
    ) -> "PhyloTree":
        """
        Build a tree from an arbitrary node numbering, relabelling leaves to ``0..n_leaves-1``.

        Parameters
        ----------
        n_nodes : int
            Number of nodes in the *input* numbering (ids ``0..n_nodes-1``).
        edges : Sequence[tuple[int, int]]
            Undirected edges as input-node pairs; must form a tree.
        weights : Sequence[float]
            Branch length per edge, aligned with ``edges``.
        leaf_of_node : Mapping[int, int]
            Input node id -> leaf id. Leaf ids must be exactly ``0..n_leaves-1``.

        Returns
        -------
        PhyloTree
            The tree, with leaves renumbered to their leaf ids and internal nodes numbered
            ``n_leaves..n_nodes-1`` in input order.

        Raises
        ------
        ForestPDInputError
            If sizes mismatch, leaf ids are not contiguous, or the edges are not a tree.
        """
        if len(edges) != len(weights):
            raise ForestPDInputError("edges and weights must have equal length")
        n_leaves = len(leaf_of_node)
        if sorted(leaf_of_node.values()) != list(range(n_leaves)):
            raise ForestPDInputError("leaf ids must be exactly 0..n_leaves-1")
        if len(edges) != n_nodes - 1:
            raise ForestPDInputError(
                f"a tree on {n_nodes} nodes needs {n_nodes - 1} edges, got {len(edges)}"
            )

        # Relabel so the leaves occupy 0..n_leaves-1 and internal nodes follow.
        relabel = [-1] * n_nodes
        for node, leaf in leaf_of_node.items():
            relabel[node] = leaf
        nxt = n_leaves
        for node in range(n_nodes):
            if relabel[node] == -1:
                relabel[node] = nxt
                nxt += 1

        arr = np.empty((len(edges), 2), dtype=np.int32)
        for i, (u, v) in enumerate(edges):
            arr[i, 0] = relabel[u]
            arr[i, 1] = relabel[v]
        tree = cls(n_nodes, n_leaves, arr, np.asarray(weights, dtype=np.float64))
        tree._check_connected()
        return tree

    @classmethod
    def from_leaf_arrays(
        cls,
        n_nodes: int,
        edges: np.ndarray,
        weights: np.ndarray,
        leaf_nodes: np.ndarray,
        leaf_ids: np.ndarray,
    ) -> "PhyloTree":
        """
        :meth:`from_edges` for edges already known to form a tree, from arrays.

        Gives exactly the tree :meth:`from_edges` would, without its connectivity search: the
        caller guarantees that the edges form a tree (a parsed tree does, by construction).

        Parameters
        ----------
        n_nodes : int
            Number of nodes in the input numbering.
        edges : numpy.ndarray
            ``int[E, 2]`` input-node pairs forming a tree.
        weights : numpy.ndarray
            ``float64[E]`` branch lengths.
        leaf_nodes : numpy.ndarray
            ``int[L]`` input node id of each leaf.
        leaf_ids : numpy.ndarray
            ``int[L]`` leaf id of each, a permutation of ``0..L-1``.

        Returns
        -------
        PhyloTree
            The tree, numbered as by :meth:`from_edges`.

        Raises
        ------
        ForestPDInputError
            If sizes mismatch or the leaf ids are not exactly ``0..L-1``.
        """
        if len(edges) != len(weights):
            raise ForestPDInputError("edges and weights must have equal length")
        n_leaves = len(leaf_nodes)
        if not np.array_equal(np.sort(leaf_ids), np.arange(n_leaves)):
            raise ForestPDInputError("leaf ids must be exactly 0..n_leaves-1")
        if len(edges) != n_nodes - 1:
            raise ForestPDInputError(
                f"a tree on {n_nodes} nodes needs {n_nodes - 1} edges, got {len(edges)}"
            )
        relabel = np.full(n_nodes, -1, dtype=np.int64)
        relabel[leaf_nodes] = leaf_ids
        internal = np.flatnonzero(relabel < 0)
        relabel[internal] = np.arange(n_leaves, n_leaves + internal.size)
        arr = relabel[np.asarray(edges, dtype=np.int64)].astype(np.int32).reshape(-1, 2)
        return cls(n_nodes, n_leaves, arr, np.asarray(weights, dtype=np.float64))

    def _check_connected(self) -> None:
        """
        Raise unless the edges connect every node (with ``n-1`` edges, that means: a tree).

        Raises
        ------
        ForestPDInputError
            If some node is unreachable from node 0.
        """
        if self.n_nodes <= 1:
            return
        start, to, _ = self.adjacency()
        seen = np.zeros(self.n_nodes, dtype=bool)
        seen[0] = True
        stack = [0]
        count = 1
        while stack:
            v = stack.pop()
            for idx in range(start[v], start[v + 1]):
                w = int(to[idx])
                if not seen[w]:
                    seen[w] = True
                    count += 1
                    stack.append(w)
        if count != self.n_nodes:
            raise ForestPDInputError("edge list is not connected (not a tree)")

    def adjacency(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the adjacency in CSR form, computed once and cached.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            ``(start, to, via)``: the neighbours of node ``v`` are ``to[start[v]:start[v+1]]``,
            reached along edge indices ``via[start[v]:start[v+1]]``.
        """
        if self._adj is None:
            # Each node's incident edges in increasing edge index: a stable sort of the edge
            # endpoints, read as (u0, v0, u1, v1, ...).
            ends = self.edges.ravel().astype(np.int64)
            order = np.argsort(ends, kind="stable")
            to = self.edges[:, ::-1].ravel()[order].astype(np.int32)
            via = (order // 2).astype(np.int32)
            start = np.zeros(self.n_nodes + 1, dtype=np.int64)
            np.cumsum(np.bincount(ends, minlength=self.n_nodes), out=start[1:])
            self._adj = (start, to, via)
        return self._adj

    def degrees(self) -> np.ndarray:
        """
        Degree of every node.

        Returns
        -------
        numpy.ndarray
            ``int32[n_nodes]`` array of node degrees.
        """
        start, _, _ = self.adjacency()
        return np.diff(start).astype(np.int32)

    def neighbours(self, node: int) -> list[tuple[int, int]]:
        """
        The ``(neighbour, edge index)`` pairs at ``node``.

        Parameters
        ----------
        node : int
            Node id.

        Returns
        -------
        list[tuple[int, int]]
            One ``(neighbour, edge index)`` pair per edge incident to ``node``.
        """
        start, to, via = self.adjacency()
        return [(int(to[i]), int(via[i])) for i in range(start[node], start[node + 1])]

    def root_at(self, root: int) -> "RootedTree":
        """
        Root the tree at ``root`` (see :class:`RootedTree`), caching the most recent rooting.

        Parameters
        ----------
        root : int
            Node id to root at.

        Returns
        -------
        RootedTree
            The rooting of this tree at ``root``.
        """
        if self._rooted is not None and self._rooted.root == root:
            return self._rooted
        rooted = RootedTree.build(self, root)
        self._rooted = rooted
        return rooted

    def total_weight(self) -> float:
        """
        Total branch length.

        Returns
        -------
        float
            Sum of all edge weights.
        """
        return float(self.weight.sum())

    def __repr__(self) -> str:
        return f"PhyloTree(n_leaves={self.n_leaves}, n_nodes={self.n_nodes})"


@dataclass(frozen=True)
class RootedTree:
    """
    A rooting of a :class:`PhyloTree`: the traversal arrays every linear-time step needs.

    Attributes
    ----------
    tree : PhyloTree
        The underlying unrooted tree.
    root : int
        Root node id.
    parent : numpy.ndarray
        ``int32[n_nodes]``; ``-1`` at the root.
    parent_edge : numpy.ndarray
        ``int32[n_nodes]``; edge index up to the parent, ``-1`` at the root.
    order : numpy.ndarray
        ``int32[n_nodes]``; a preorder. Reversed, it is a postorder (children before parents).
    tin, tout : numpy.ndarray
        ``int32[n_nodes]``; DFS entry/exit stamps: ``u`` is an ancestor of ``v`` (or ``v``
        itself) exactly when ``tin[u] <= tin[v]`` and ``tout[v] <= tout[u]``.
    """

    tree: PhyloTree
    root: int
    parent: np.ndarray
    parent_edge: np.ndarray
    order: np.ndarray
    tin: np.ndarray
    tout: np.ndarray

    @classmethod
    def build(cls, tree: PhyloTree, root: int) -> "RootedTree":
        """
        Root ``tree`` at node ``root`` with an iterative DFS (no recursion-depth limit).

        Parameters
        ----------
        tree : PhyloTree
            The unrooted tree.
        root : int
            Node id to root at.

        Returns
        -------
        RootedTree
            The traversal arrays of the rooting.
        """
        # Plain lists in the loop (scalar NumPy indexing is several times slower); arrays at the
        # end.
        n = tree.n_nodes
        start_a, to_a, via_a = tree.adjacency()
        start, to, via = start_a.tolist(), to_a.tolist(), via_a.tolist()
        parent = [-1] * n
        parent_edge = [-1] * n
        order: list[int] = []
        tin = [0] * n
        tout = [0] * n

        cursor = start[:-1]
        root = int(root)
        clock = 1
        order.append(root)
        stack = [root]
        while stack:
            v = stack[-1]
            descended = False
            while cursor[v] < start[v + 1]:
                idx = cursor[v]
                cursor[v] += 1
                if via[idx] == parent_edge[v]:
                    continue
                w = to[idx]
                parent[w] = v
                parent_edge[w] = via[idx]
                tin[w] = clock
                clock += 1
                order.append(w)
                stack.append(w)
                descended = True
                break
            if not descended:
                tout[v] = clock
                clock += 1
                stack.pop()
        as_array = lambda values: np.array(values, dtype=np.int32)  # noqa: E731
        return cls(
            tree,
            root,
            as_array(parent),
            as_array(parent_edge),
            as_array(order),
            as_array(tin),
            as_array(tout),
        )


def leaf_masks_below(rooted: RootedTree) -> list[int]:
    """
    Per edge, the bitmask of leaves on its far-from-root side.

    Parameters
    ----------
    rooted : RootedTree
        A rooting of the tree.

    Returns
    -------
    list[int]
        Per edge index, the bitmask (bit ``i`` = leaf ``i``) of leaves below that edge.
    """
    tree = rooted.tree
    sub = [0] * tree.n_nodes
    for leaf in range(tree.n_leaves):
        sub[leaf] = 1 << leaf
    below = [0] * tree.n_edges
    for v in reversed(rooted.order.tolist()):
        pe = int(rooted.parent_edge[v])
        if pe >= 0:
            below[pe] = sub[v]
            sub[int(rooted.parent[v])] |= sub[v]
    return below


def has_degree_two(tree: PhyloTree) -> bool:
    """
    Whether some internal node of ``tree`` has degree 2.

    Such a node gives its two edges the same split, so trees with one are not accepted as input;
    :func:`suppress_degree_two` removes them without changing any PD.

    Parameters
    ----------
    tree : PhyloTree
        The tree.

    Returns
    -------
    bool
        True if some node other than a leaf has exactly two neighbours.

    Examples
    --------
    >>> path = PhyloTree.from_edges(4, [(0, 3), (3, 2), (2, 1)], [1.0] * 3, {0: 0, 1: 1})
    >>> has_degree_two(path), has_degree_two(suppress_degree_two(path))
    (True, False)
    """
    degree = np.bincount(tree.edges.ravel(), minlength=tree.n_nodes)
    return bool(np.any(degree[tree.n_leaves :] == 2))


def suppress_degree_two(tree: PhyloTree) -> PhyloTree:
    """
    Merge out every internal node of degree 2, summing the two branch lengths.

    Parameters
    ----------
    tree : PhyloTree
        The input tree.

    Returns
    -------
    PhyloTree
        ``tree`` itself if it has no internal degree-2 node, otherwise a new tree with each
        maximal degree-2 chain merged into a single edge. Leaf ids are unchanged.
    """
    deg = tree.degrees()
    keep_node = np.ones(tree.n_nodes, dtype=bool)
    for v in range(tree.n_leaves, tree.n_nodes):
        keep_node[v] = deg[v] != 2
    if keep_node.all():
        return tree

    start, to, via = tree.adjacency()
    visited = np.zeros(tree.n_edges, dtype=bool)
    chain_edges: list[tuple[int, int]] = []
    chain_weight: list[float] = []

    for e0 in range(tree.n_edges):
        if visited[e0]:
            continue
        visited[e0] = True
        total = float(tree.weight[e0])
        ends: list[int] = []
        for side in (0, 1):
            prev = int(tree.edges[e0, 1 - side])
            cur = int(tree.edges[e0, side])
            while not keep_node[cur]:
                # `cur` has degree 2, so exactly one neighbour is not `prev`.
                nxt = nxt_edge = -1
                for i in range(start[cur], start[cur + 1]):
                    if int(to[i]) != prev:
                        nxt, nxt_edge = int(to[i]), int(via[i])
                        break
                visited[nxt_edge] = True
                total += float(tree.weight[nxt_edge])
                prev, cur = cur, nxt
            ends.append(cur)
        chain_edges.append((ends[0], ends[1]))
        chain_weight.append(total)

    kept = [v for v in range(tree.n_nodes) if keep_node[v]]
    remap = {v: i for i, v in enumerate(kept)}
    new_edges = np.array([[remap[a], remap[b]] for a, b in chain_edges], dtype=np.int32)
    if new_edges.size == 0:
        new_edges = np.empty((0, 2), dtype=np.int32)
    return PhyloTree(
        len(kept), tree.n_leaves, new_edges, np.asarray(chain_weight, dtype=np.float64)
    )


def drop_leaf(tree: PhyloTree, leaf: int) -> PhyloTree:
    """
    Remove the highest-numbered leaf and its edge, then suppress the resulting degree-2 node.

    Parameters
    ----------
    tree : PhyloTree
        The input tree.
    leaf : int
        The leaf to remove; must be ``tree.n_leaves - 1``.

    Returns
    -------
    PhyloTree
        The tree without ``leaf``, with degree-2 nodes suppressed.

    Raises
    ------
    ForestPDInputError
        If ``leaf`` is not the highest-numbered leaf.
    """
    if leaf != tree.n_leaves - 1:
        raise ForestPDInputError("drop_leaf only removes the highest-numbered leaf")
    fast = _drop_leaf_fast(tree, leaf)
    if fast is not None:
        return fast
    keep = [
        e for e in range(tree.n_edges) if leaf not in (int(tree.edges[e, 0]), int(tree.edges[e, 1]))
    ]
    if not keep:
        # Degenerate: the tree was a single edge, so one isolated leaf remains.
        return PhyloTree(1, 1, np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float64))
    used = sorted({int(x) for e in keep for x in tree.edges[e]})
    remap = {v: i for i, v in enumerate(used)}
    new_edges = np.array(
        [[remap[int(tree.edges[e, 0])], remap[int(tree.edges[e, 1])]] for e in keep],
        dtype=np.int32,
    )
    reduced = PhyloTree(len(used), tree.n_leaves - 1, new_edges, tree.weight[keep].copy())
    return suppress_degree_two(reduced)


def _drop_leaf_fast(tree: PhyloTree, leaf: int) -> PhyloTree | None:
    """
    :func:`drop_leaf` for the common case, with an identical result.

    The case: the leaf hangs off an internal node ``u`` and no internal node has degree 2. Then
    the only node to suppress is ``u`` (if its degree drops to 2), and :func:`suppress_degree_two`
    would merge its two remaining edges into the first of them, in place, keeping that edge's
    orientation. Node ids shift down past the removed nodes. Local trees are small, so plain lists
    beat NumPy here.

    Parameters
    ----------
    tree : PhyloTree
        The input tree.
    leaf : int
        The highest-numbered leaf.

    Returns
    -------
    PhyloTree | None
        The tree without ``leaf``, or ``None`` outside the common case.
    """
    pairs = tree.edges.tolist()
    n_leaves = tree.n_leaves
    degree = [0] * tree.n_nodes
    row = -1
    for i, (a, b) in enumerate(pairs):
        degree[a] += 1
        degree[b] += 1
        if a == leaf or b == leaf:
            if row >= 0:
                return None
            row = i
    if row < 0:
        return None
    a, b = pairs[row]
    u = a if b == leaf else b
    if u < n_leaves or 2 in degree[n_leaves:]:
        return None

    weight = tree.weight.tolist()
    del pairs[row], weight[row]
    low, high = leaf, tree.n_nodes  # the removed node ids; high = n_nodes means "none"
    if degree[u] == 3:
        first, second = [i for i, pair in enumerate(pairs) if u in pair]
        s0, s1 = pairs[second]
        other = s0 if s1 == u else s1
        f0, f1 = pairs[first]
        pairs[first] = [other if f0 == u else f0, other if f1 == u else f1]
        weight[first] = weight[first] + weight[second]
        del pairs[second], weight[second]
        low, high = min(leaf, u), max(leaf, u)
    n_removed = 1 if high == tree.n_nodes else 2
    edges = np.array(
        [[x - (x > low) - (x > high), y - (y > low) - (y > high)] for x, y in pairs],
        dtype=np.int32,
    ).reshape(-1, 2)
    return PhyloTree(
        tree.n_nodes - n_removed, n_leaves - 1, edges, np.array(weight, dtype=np.float64)
    )


__all__ = [
    "PhyloTree",
    "RootedTree",
    "leaf_masks_below",
    "suppress_degree_two",
    "has_degree_two",
    "drop_leaf",
]
