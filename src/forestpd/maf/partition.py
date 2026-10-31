"""
Component masks per edge, and agreement-forest validation.

For a partition of the leaves into components and a tree rooted at leaf 0, each edge ``e`` gets two
``k``-bit masks:

``below[e]``
    the components with at least one leaf on the far side of ``e`` (away from the root);
``above[e]``
    the components with at least one leaf on the near side.

From these, per edge:

* ``below & ~above`` is the set of components entirely below ``e``;
* ``above & ~below`` is the set of components entirely above ``e``;
* ``below & above`` is the component that ``e`` lies inside (at most one, by vertex-disjointness),
  i.e. ``e`` is an edge of that component's ``H_{j,i}``.

These masks are used by the definitely-chosen edge test, the two-sided weights, and the validation
of agreement forests.

Example: the tree ``((0,1),(2,3))`` with components ``{0,1}``, ``{2}`` and ``{3}`` (bits 0, 1, 2),
rooted at leaf 0. The edge from the cherry ``{0,1}`` to the cherry ``{2,3}`` has components 1 and 2
below it and component 0 above it, so it lies inside no component; the pendant edge of leaf 1 lies
inside component 0, whose leaves 0 and 1 are on either side of it.

>>> import numpy as np
>>> from forestpd.core.tree import PhyloTree
>>> t = PhyloTree.from_edges(6, [(0, 4), (1, 4), (4, 5), (2, 5), (3, 5)], [1.0] * 5,
...                          {i: i for i in range(4)})
>>> below, above = component_edge_masks(t.root_at(0), np.array([0, 0, 1, 2]), 3)
>>> [format(int(m), "03b") for m in below]      # edges (0,4) (1,4) (4,5) (2,5) (3,5)
['111', '001', '110', '010', '100']
>>> [format(int(m), "03b") for m in above]
['001', '111', '001', '101', '011']
>>> [format(int(m), "03b") for m in lies_inside(below, above)]
['001', '001', '000', '000', '000']
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..exceptions import ForestPDInputError, ForestPDValueError
from .embedding import component_shape
from ..core.tree import PhyloTree, RootedTree

#: Widest component count the ``uint64`` mask representation supports. Any ``k`` near this makes the
#: ``2^k`` signature loop hopeless, so it is a sanity bound rather than a practical limit.
MAX_COMPONENTS = 63

#: Marker for an edge lying inside two components at once (only produced with ``strict=False``).
SHARED_EDGE = -2

_ZERO = np.uint64(0)


def component_edge_masks(
    rooted: RootedTree, comp_of_leaf: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per edge, the component masks of its two sides.

    Parameters
    ----------
    rooted : RootedTree
        A rooting of the tree.
    comp_of_leaf : numpy.ndarray
        Leaf -> component index.
    n_components : int
        Number of components ``k`` (at most :data:`MAX_COMPONENTS`).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(below, above)``, ``uint64`` arrays indexed by edge: the components with a leaf on the
        far-from-root and near-to-root side of each edge, respectively.

    Raises
    ------
    ForestPDValueError
        If ``n_components`` exceeds :data:`MAX_COMPONENTS` (the masks have 64 bits).
    """
    if n_components > MAX_COMPONENTS:
        raise ForestPDValueError(
            f"{n_components} components; component masks support at most {MAX_COMPONENTS}"
        )
    tree = rooted.tree
    n = tree.n_nodes
    parent, parent_edge = rooted.parent, rooted.parent_edge

    def own(v: int) -> int:
        """The bit of ``v``'s component if ``v`` is a leaf, else no bits."""
        return 1 << int(comp_of_leaf[v]) if v < tree.n_leaves else 0

    # Bottom-up: in_subtree[v] = the components with a leaf in the subtree of v. The edge above v
    # has exactly those components below it.
    in_subtree = np.array([own(v) for v in range(n)], dtype=np.uint64)
    below = np.zeros(tree.n_edges, dtype=np.uint64)
    for v in rooted.order[::-1]:
        v = int(v)
        if parent[v] >= 0:
            below[parent_edge[v]] = in_subtree[v]
            in_subtree[parent[v]] |= in_subtree[v]

    # Top-down: outside[v] = the components with a leaf outside the subtree of v. For a child c of
    # v, that is everything outside v, plus v itself, plus the subtrees of c's siblings. The
    # siblings' union is taken from prefix and suffix ORs, so each node costs O(degree).
    children: list[list[int]] = [[] for _ in range(n)]
    for v in rooted.order[1:]:
        children[int(parent[v])].append(int(v))
    outside = np.zeros(n, dtype=np.uint64)
    above = np.zeros(tree.n_edges, dtype=np.uint64)
    for v in rooted.order:
        v = int(v)
        kids = children[v]
        if not kids:
            continue
        kid_masks = [in_subtree[c] for c in kids]
        after = [_ZERO] * (len(kids) + 1)  # after[i] = union of kid_masks[i:]
        for i in range(len(kids) - 1, -1, -1):
            after[i] = after[i + 1] | kid_masks[i]
        before = _ZERO  # union of kid_masks[:i]
        for i, c in enumerate(kids):
            outside[c] = outside[v] | own(v) | before | after[i + 1]
            above[parent_edge[c]] = outside[c]
            before |= kid_masks[i]
    return below, above


def is_agreement_forest(
    trees: Sequence[PhyloTree], comp_of_leaf: np.ndarray, n_components: int
) -> bool:
    """
    Check whether a leaf partition is an agreement forest of ``trees``.

    Checks that each component induces the same topology in every tree, and that in every tree
    the components' embedded subtrees are pairwise vertex-disjoint.

    Parameters
    ----------
    trees : Sequence[PhyloTree]
        The trees, all on the same leaf set.
    comp_of_leaf : numpy.ndarray
        Leaf -> component index.
    n_components : int
        Number of components.

    Returns
    -------
    bool
        True if both conditions hold in every tree.
    """
    members = [
        [int(x) for x in np.nonzero(comp_of_leaf == comp)[0]] for comp in range(n_components)
    ]
    shapes: list[set[object]] = [set() for _ in range(n_components)]
    for tree in trees:
        rooted = tree.root_at(0)
        below, above = component_edge_masks(rooted, comp_of_leaf, n_components)
        # Each edge lies in at most one component's embedded subtree, so slicing the edge list by
        # component costs one pass and gives every extraction its own O(|H|) budget.
        owner = split_component_of_edge(below, above, strict=False)
        if (owner == SHARED_EDGE).any():
            return False
        if not components_are_vertex_disjoint(tree, owner):
            return False
        for comp in range(n_components):
            if len(members[comp]) <= 2:
                continue  # one leaf or a pair admits only one topology
            edges = [int(e) for e in np.nonzero(owner == comp)[0]]
            shapes[comp].add(component_shape(tree, members[comp], edges))
    return all(len(s) <= 1 for s in shapes)


def split_component_of_edge(
    below: np.ndarray, above: np.ndarray, strict: bool = True
) -> np.ndarray:
    """
    Per edge, the index of the component it splits, or ``-1``.

    Parameters
    ----------
    below, above : numpy.ndarray
        The per-edge component masks.
    strict : bool, default True
        Raise when an edge lies inside two components. With ``strict=False`` such edges are marked
        :data:`SHARED_EDGE` instead.

    Returns
    -------
    numpy.ndarray
        ``int64`` array indexed by edge: the component index the edge lies inside, ``-1`` for an
        edge outside every embedded subtree, or :data:`SHARED_EDGE` (only when not ``strict``).

    Raises
    ------
    ForestPDInputError
        If ``strict`` and some edge splits two components.
    """
    inside = lies_inside(below, above)
    owner = np.full(below.shape[0], -1, dtype=np.int64)
    for e in np.nonzero(inside)[0]:
        value = int(inside[e])
        if has_several(value):
            if strict:
                raise ForestPDInputError(
                    f"edge {int(e)} lies inside two MAF components at once; the partition is not "
                    "an agreement forest"
                )
            owner[e] = SHARED_EDGE
        else:
            owner[e] = value.bit_length() - 1  # the index of the single set bit
    return owner


def entirely_below(below: np.ndarray, above: np.ndarray) -> np.ndarray:
    """
    Per edge, the components with all their leaves below it.

    Parameters
    ----------
    below, above : numpy.ndarray
        The per-edge component masks from :func:`component_edge_masks`.

    Returns
    -------
    numpy.ndarray
        ``uint64`` mask per edge.
    """
    mask: np.ndarray = below & ~above
    return mask


def entirely_above(below: np.ndarray, above: np.ndarray) -> np.ndarray:
    """
    Per edge, the components with all their leaves above it.

    Parameters
    ----------
    below, above : numpy.ndarray
        The per-edge component masks from :func:`component_edge_masks`.

    Returns
    -------
    numpy.ndarray
        ``uint64`` mask per edge.
    """
    mask: np.ndarray = above & ~below
    return mask


def lies_inside(below: np.ndarray, above: np.ndarray) -> np.ndarray:
    """
    Per edge, the components with leaves on both sides, i.e. whose subtree contains the edge.

    Parameters
    ----------
    below, above : numpy.ndarray
        The per-edge component masks from :func:`component_edge_masks`.

    Returns
    -------
    numpy.ndarray
        ``uint64`` mask per edge; in an agreement forest each has at most one bit set.
    """
    mask: np.ndarray = below & above
    return mask


def has_several(mask: int) -> bool:
    """
    Whether a mask has more than one bit set.

    Parameters
    ----------
    mask : int
        A component mask.

    Returns
    -------
    bool
        True if ``mask`` contains two or more components.

    Examples
    --------
    >>> has_several(0b0100), has_several(0b0110)
    (False, True)
    """
    return bool(mask & (mask - 1))  # clearing the lowest set bit leaves something


def components_are_vertex_disjoint(tree: PhyloTree, owner: np.ndarray) -> bool:
    """
    Whether the embedded subtrees share no vertex.

    Every endpoint of a component edge is assigned to that edge's component; the subtrees are
    vertex-disjoint exactly when no vertex is assigned to two different components.

    Parameters
    ----------
    tree : PhyloTree
        The host tree ``T_j``.
    owner : numpy.ndarray
        Per-edge owning component, as returned by :func:`split_component_of_edge` (with no
        :data:`SHARED_EDGE` entries).

    Returns
    -------
    bool
        True if the embedded subtrees are pairwise vertex-disjoint.

    Examples
    --------
    >>> import numpy as np
    >>> star = PhyloTree.from_edges(5, [(0, 4), (1, 4), (2, 4), (3, 4)], [1.0] * 4,
    ...                             {i: i for i in range(4)})
    >>> components_are_vertex_disjoint(star, np.array([0, 0, 1, 1]))  # both pass the centre
    False
    >>> components_are_vertex_disjoint(star, np.array([0, 0, -1, -1]))
    True
    """
    inside = owner >= 0
    nodes = tree.edges[inside].ravel()
    comps = np.repeat(owner[inside], 2)
    order = np.argsort(nodes, kind="stable")
    nodes, comps = nodes[order], comps[order]
    clash = (nodes[1:] == nodes[:-1]) & (comps[1:] != comps[:-1])
    return not bool(clash.any())


__all__ = [
    "component_edge_masks",
    "split_component_of_edge",
    "entirely_below",
    "entirely_above",
    "lies_inside",
    "has_several",
    "components_are_vertex_disjoint",
    "is_agreement_forest",
    "MAX_COMPONENTS",
    "SHARED_EDGE",
]
