"""
Extracting the topology of a MAF component from each tree.

For a component with leaf set ``X_i`` and an input tree ``T_j``, the algorithm uses:

* ``H_{j,i} = T_j<X_i>``, the subtree of ``T_j`` spanning ``X_i``;
* ``F_i``, the component itself: ``H_{j,i}`` with its degree-2 vertices suppressed;
* ``pi_{j,i}``, which maps each edge of ``H_{j,i}`` to the edge of ``F_i`` whose subdivision path
  contains it.

:func:`extract_embedding` computes all three in one traversal of the component. The embedded
subtrees of a forest are vertex-disjoint, so extracting all components of one tree costs ``O(n)``
in total.

``F_i`` is the same tree for every ``j``, but its node numbering must also agree across trees, so
that weights from different trees are added to the same edges. This is done with a canonical
rooting: root at the smallest leaf of ``X_i`` and order children by the smallest leaf below them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..core.tree import PhyloTree

#: A component's topology as nested tuples of leaf ids; equal shapes mean equal topologies.
Shape = object


@dataclass(frozen=True)
class Embedding:
    """
    One component's structure inside one tree.

    Attributes
    ----------
    n_nodes : int
        Number of nodes of ``F_i`` under the canonical rooting (root = smallest leaf of ``X_i``).
    parent : numpy.ndarray
        ``int32[n_nodes]``; parent in ``F_i``, ``-1`` at the root.
    leaf : numpy.ndarray
        ``int32[n_nodes]``; leaf id at this node of ``F_i``, or ``-1`` for internal nodes.
    children : tuple[tuple[int, ...], ...]
        Children per ``F_i`` node, in canonical order.
    edge_fnode : dict[int, int]
        Host-tree edge index -> the ``F_i`` node whose parent edge it subdivides (an ``F_i`` edge
        is named by its lower endpoint).
    shape : Shape
        Canonical topology, comparable across trees.
    """

    n_nodes: int
    parent: np.ndarray
    leaf: np.ndarray
    children: tuple[tuple[int, ...], ...]
    edge_fnode: dict[int, int]
    shape: Shape


def component_edges(tree: PhyloTree, leaves: Sequence[int]) -> list[int]:
    """
    The edges of ``tree<leaves>``: those with component leaves on both sides.

    Parameters
    ----------
    tree : PhyloTree
        The host tree.
    leaves : Sequence[int]
        The component's leaf ids.

    Returns
    -------
    list[int]
        Edge indices of ``H = tree<leaves>``, in increasing order of their lower endpoint; empty
        when ``leaves`` has at most one element.
    """
    marked = set(int(x) for x in leaves)
    total = len(marked)
    if total <= 1:
        return []
    rooted = tree.root_at(0)
    count = np.zeros(tree.n_nodes, dtype=np.int32)
    for leaf in marked:
        count[leaf] = 1
    for idx in range(tree.n_nodes - 1, -1, -1):
        v = int(rooted.order[idx])
        p = int(rooted.parent[v])
        if p >= 0:
            count[p] += count[v]
    out = []
    for v in range(tree.n_nodes):
        pe = int(rooted.parent_edge[v])
        if pe >= 0 and 0 < count[v] < total:
            out.append(pe)
    return out


def extract_embedding(
    tree: PhyloTree,
    leaves: Sequence[int],
    edges: Sequence[int] | None = None,
) -> Embedding:
    """
    Extract ``F_i``, its canonical numbering, and ``pi_{j,i}`` for one component inside ``tree``.

    Parameters
    ----------
    tree : PhyloTree
        The host tree.
    leaves : Sequence[int]
        The component's leaf ids (any order; at least one).
    edges : Sequence[int] | None, optional
        The edges of ``H_{j,i}``, if already known. Computed via :func:`component_edges` otherwise.

    Returns
    -------
    Embedding
        ``F_i`` under its canonical numbering, together with ``pi_{j,i}`` as ``edge_fnode``.

    Examples
    --------
    >>> from forestpd.core.tree import PhyloTree
    >>> # ((0,1),2), with the path from the cherry to 2 subdivided by node 4.
    >>> t = PhyloTree.from_edges(5, [(0, 3), (1, 3), (3, 4), (2, 4)],
    ...                          [1.0, 1.0, 1.0, 1.0], {0: 0, 1: 1, 2: 2})
    >>> emb = extract_embedding(t, [0, 1])
    >>> emb.n_nodes                 # F is the two-leaf tree 0 -- 1
    2
    >>> len(emb.edge_fnode)         # both pendant edges subdivide that single F edge
    2
    """
    component = sorted(int(x) for x in leaves)
    root = component[0]
    if len(component) == 1:
        return Embedding(
            n_nodes=1,
            parent=np.array([-1], dtype=np.int32),
            leaf=np.array([root], dtype=np.int32),
            children=((),),
            edge_fnode={},
            shape=root,
        )
    if edges is None:
        edges = component_edges(tree, component)
    is_leaf = set(component)

    # 1. Root H at the smallest component leaf. All later passes walk H only, never the whole tree.
    order, children, up_edge = _orient(tree, edges, root)

    # 2. The vertices of F are the component leaves and the vertices where H branches. Every other
    #    vertex of H has exactly one child and only subdivides an edge of F. first_f[v] is the
    #    first vertex of F at or below v.
    first_f: dict[int, int] = {}
    for v in reversed(order):
        is_f_vertex = v in is_leaf or len(children[v]) >= 2
        first_f[v] = v if is_f_vertex else first_f[children[v][0]]

    # 3. Canonical order: children by the smallest component leaf below them.
    smallest: dict[int, int] = {}
    for v in reversed(order):
        below = [smallest[c] for c in children[v]]
        smallest[v] = min(below + [v]) if v in is_leaf else min(below)

    # 4. Number the vertices of F in preorder, children in canonical order.
    f_id: dict[int, int] = {}
    parent: list[int] = []
    leaf: list[int] = []
    kids: list[list[int]] = []
    stack = [(root, -1)]  # (vertex of F, id of its parent in F)
    while stack:
        v, up = stack.pop()
        f_id[v] = len(parent)
        parent.append(up)
        leaf.append(v if v in is_leaf else -1)
        kids.append([])
        if up >= 0:
            kids[up].append(f_id[v])
        f_children = sorted((first_f[c] for c in children[v]), key=smallest.__getitem__)
        stack.extend((c, f_id[v]) for c in reversed(f_children))

    # 5. pi: every edge of H belongs to the F edge above the first F vertex below it.
    edge_fnode = {up_edge[v]: f_id[first_f[v]] for v in order if v != root}

    # 6. The shape, built bottom-up (children have larger preorder ids than their parent).
    shape: list[Shape] = [None] * len(parent)
    for i in range(len(parent) - 1, -1, -1):
        shape[i] = tuple(shape[c] for c in kids[i]) if kids[i] else leaf[i]

    return Embedding(
        n_nodes=len(parent),
        parent=np.asarray(parent, dtype=np.int32),
        leaf=np.asarray(leaf, dtype=np.int32),
        children=tuple(tuple(k) for k in kids),
        edge_fnode=edge_fnode,
        shape=shape[0],
    )


def _orient(
    tree: PhyloTree, edges: Sequence[int], root: int
) -> tuple[list[int], dict[int, list[int]], dict[int, int]]:
    """
    Root the subtree formed by ``edges`` at ``root``.

    Parameters
    ----------
    tree : PhyloTree
        The host tree.
    edges : Sequence[int]
        The edges of the subtree ``H``.
    root : int
        The vertex to root at (a leaf of ``H``).

    Returns
    -------
    tuple[list[int], dict[int, list[int]], dict[int, int]]
        A preorder of the vertices of ``H``, the children of each vertex, and the edge from each
        vertex up to its parent (absent for the root).
    """
    neighbours: dict[int, list[tuple[int, int]]] = {}
    for e in edges:
        u, v = int(tree.edges[e, 0]), int(tree.edges[e, 1])
        neighbours.setdefault(u, []).append((v, e))
        neighbours.setdefault(v, []).append((u, e))
    order = [root]
    children: dict[int, list[int]] = {root: []}
    up_edge: dict[int, int] = {}
    stack = [root]
    while stack:
        v = stack.pop()
        for w, e in neighbours[v]:
            if w not in children:
                children[w] = []
                children[v].append(w)
                up_edge[w] = e
                order.append(w)
                stack.append(w)
    return order, children, up_edge


def component_shape(
    tree: PhyloTree, leaves: Sequence[int], edges: Sequence[int] | None = None
) -> Shape:
    """
    The canonical topology of ``tree<leaves>``; equal shapes mean the same tree.

    Parameters
    ----------
    tree : PhyloTree
        The host tree.
    leaves : Sequence[int]
        The component's leaf ids.
    edges : Sequence[int] | None, optional
        The edges of ``tree<leaves>``, if already known. By default None (computed).

    Returns
    -------
    Shape
        Nested tuples of leaf ids, comparable across trees.
    """
    return extract_embedding(tree, leaves, edges).shape


__all__ = [
    "Embedding",
    "Shape",
    "extract_embedding",
    "component_edges",
    "component_shape",
]
