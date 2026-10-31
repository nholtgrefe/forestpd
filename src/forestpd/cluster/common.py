"""
Exhaustive common-cluster reduction.

Finds every non-trivial split shared by all input trees, cuts them all at once, and assembles the
resulting pieces into the tree of clusters ``K``.

Trivial splits (a single leaf against the rest) are deliberately *not* cut. They are common to every
tree.

Cost is ``O(t * n)``, via Day's algorithm: number the leaves by a DFS of the first tree, and every
cluster of that tree becomes a contiguous index interval. A cluster of any other tree equals one of
them exactly when its own index range is contiguous, has the matching size, and appears in the
first tree's interval set. No leaf bitmasks are ever compared, which is what makes it linear rather
than quadratic in the number of taxa.

Pieces need no matching across trees. With every tree rooted at leaf 0, each piece other than the
one holding leaf 0 hangs below exactly one cut, so cluster ``c + 1`` is "the piece below cut ``c``"
and cluster ``0`` is the piece holding leaf 0, in every tree alike.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..core.tree import PhyloTree, has_degree_two
from ..exceptions import ForestPDInputError
from . import _kernels
from .base import ClusterTree, LocalPiece, single_cluster

#: Use the compiled passes of :mod:`._kernels` for merged reductions when Numba is installed. The
#: Python path computes the same cluster tree; tests switch this off to exercise it.
USE_COMPILED = True

#: Local split masks are ``uint64`` in the compiled path, so it handles clusters up to this size.
_MAX_COMPILED_LEAVES = 64


def leaf_numbering(tree: PhyloTree) -> np.ndarray:
    """
    Day's leaf numbering: index each leaf by its position in a DFS from leaf 0.

    Every cluster of ``tree`` occupies a contiguous index interval; leaf 0 gets index 0.

    Parameters
    ----------
    tree : PhyloTree
        The tree whose DFS fixes the numbering (the first input tree).

    Returns
    -------
    numpy.ndarray
        ``int64[n_leaves]``; DFS position of each leaf id.
    """
    rooted = tree.root_at(0)
    index = np.full(tree.n_leaves, -1, dtype=np.int64)
    position = 0
    for v in rooted.order:
        if v < tree.n_leaves:
            index[int(v)] = position
            position += 1
    return index


def _edge_intervals(tree: PhyloTree, index: list[int]) -> tuple[list[int], list[int], list[int]]:
    """
    Per edge, the ``(lo, hi, count)`` of the leaf indices on its far-from-root side (rooted at
    leaf 0): the smallest and largest index below the edge, and the number of leaves below it.

    On plain lists, which is several times faster than scalar NumPy.
    """
    rooted = tree.root_at(0)
    n, n_leaves = tree.n_nodes, tree.n_leaves
    parent = rooted.parent.tolist()
    lo = index + [n_leaves] * (n - n_leaves)  # leaf ids are 0..n_leaves-1, so leaves come first
    hi = index + [-1] * (n - n_leaves)
    count = [1] * n_leaves + [0] * (n - n_leaves)
    for v in reversed(rooted.order.tolist()):
        p = parent[v]
        if p >= 0:
            if lo[v] < lo[p]:
                lo[p] = lo[v]
            if hi[v] > hi[p]:
                hi[p] = hi[v]
            count[p] += count[v]

    e_lo = [0] * tree.n_edges
    e_hi = [0] * tree.n_edges
    e_count = [0] * tree.n_edges
    for v, pe in enumerate(rooted.parent_edge.tolist()):
        if pe >= 0:
            e_lo[pe], e_hi[pe], e_count[pe] = lo[v], hi[v], count[v]
    return e_lo, e_hi, e_count


def _split_intervals(tree: PhyloTree, index: np.ndarray) -> list[tuple[int, int] | None]:
    """
    Per edge, its non-trivial split as an interval of the first tree's numbering, if it is one.

    Parameters
    ----------
    tree : PhyloTree
        One input tree.
    index : numpy.ndarray
        The shared leaf numbering from :func:`leaf_numbering`.

    Returns
    -------
    list[tuple[int, int] | None]
        Per edge, ``(lo, hi)`` when the leaves below it are exactly the indices ``lo..hi`` and
        number between 2 and ``n - 2``; ``None`` otherwise.
    """
    n = tree.n_leaves
    lo, hi, count = _edge_intervals(tree, index.tolist())
    return [
        (a, b) if b - a + 1 == size and 2 <= size <= n - 2 else None
        for a, b, size in zip(lo, hi, count)
    ]


def _common_cuts(
    per_tree: Iterable[list[tuple[int, int] | None]],
) -> dict[tuple[int, int], int]:
    """The intervals found in every tree, each mapped to a cut id (in sorted order)."""
    common: set[tuple[int, int]] | None = None
    for intervals in per_tree:
        found = {key for key in intervals if key is not None}
        common = found if common is None else common & found
    return {key: cut for cut, key in enumerate(sorted(common or ()))}


def cluster_reduction(
    trees: Sequence[PhyloTree], weights: Sequence[float] | np.ndarray
) -> ClusterTree:
    """
    Exhaustive common-cluster reduction: cut every non-trivial common split.

    The resulting pieces become the clusters of the tree of clusters ``K``, rooted at the cluster
    holding leaf 0.

    Parameters
    ----------
    trees : Sequence[PhyloTree]
        The input trees, all on leaves ``0..n_leaves-1``.
    weights : Sequence[float] | numpy.ndarray
        Per-tree weights ``w_j``. The local trees of each cluster that have the same topology are
        merged into one, with edge weights ``sum_j w_j * len_j(e)`` (see
        :attr:`~forestpd.cluster.base.LocalPiece.tree_weights`). The objective is linear in these
        products, so this changes nothing downstream except the number of local trees, which is
        the number of distinct local topologies instead of ``t``.

    Returns
    -------
    ClusterTree
        The tree of clusters ``K``; a single cluster when no non-trivial split is common.
        Every piece has ``tree_weights`` set, except for a single taxon (no edges at all).

    Raises
    ------
    ForestPDInputError
        If a tree has an internal vertex of degree 2 (its two edges would share a split). Trees
        read from Newick or PhyloZoo never do; suppress them with
        :func:`~forestpd.core.tree.suppress_degree_two`.

    Examples
    --------
    >>> from forestpd.core.tree import PhyloTree
    >>> # ((A,B),(C,D)) twice: {A,B} is a common split, so K has two clusters.
    >>> mk = lambda: PhyloTree.from_edges(
    ...     6, [(0, 4), (1, 4), (4, 5), (2, 5), (3, 5)], [1.0] * 5, {0: 0, 1: 1, 2: 2, 3: 3})
    >>> ktree = cluster_reduction([mk(), mk()], weights=[0.5, 0.5])
    >>> ktree.n_clusters, [len(p.trees) for p in ktree.pieces]  # the two equal trees are merged
    (2, [1, 1])
    """
    trees = list(trees)
    for j, tree in enumerate(trees):
        if has_degree_two(tree):
            raise ForestPDInputError(
                f"Tree {j} has an internal vertex of degree 2; suppress it first with "
                "forestpd.core.tree.suppress_degree_two (this does not change any PD)."
            )
    if trees[0].n_leaves < 2:  # a single taxon: no edges, nothing to cut
        return single_cluster(trees)
    if USE_COMPILED and _kernels.HAVE_NUMBA:
        compiled = _compiled_reduction(trees, np.asarray(weights, dtype=np.float64))
        if compiled is not None:
            return compiled
    index = leaf_numbering(trees[0])
    if trees[0].n_leaves < 4:
        intervals: list[list[tuple[int, int] | None]] = [[None] * t.n_edges for t in trees]
    else:
        intervals = [_split_intervals(tree, index) for tree in trees]
    cut_id = _common_cuts(intervals)
    n_clusters = len(cut_id) + 1
    cluster_of_node = [
        _label_clusters(tree, found, cut_id) for tree, found in zip(trees, intervals)
    ]
    parent, children, postorder = _cluster_tree_shape(trees[0], cluster_of_node[0], n_clusters)
    pieces = _merged_pieces(trees, list(weights), cluster_of_node, parent, children)
    return ClusterTree(
        pieces=pieces,
        parent=tuple(parent),
        children=tuple(tuple(cs) for cs in children),
        postorder=tuple(postorder),
        root=0,
    )


def _label_clusters(
    tree: PhyloTree,
    intervals: list[tuple[int, int] | None],
    cut_id: dict[tuple[int, int], int],
) -> list[int]:
    """
    The cluster of every node: ``c + 1`` below cut ``c``, ``0`` for the piece holding leaf 0.

    Parameters
    ----------
    tree : PhyloTree
        One input tree.
    intervals : list[tuple[int, int] | None]
        Its split interval per edge, from :func:`_split_intervals`.
    cut_id : dict[tuple[int, int], int]
        Common interval -> cut id, from :func:`_common_cuts`.

    Returns
    -------
    list[int]
        Cluster id per node.
    """
    rooted = tree.root_at(0)
    parent = rooted.parent.tolist()
    parent_edge = rooted.parent_edge.tolist()
    cluster = [0] * tree.n_nodes
    for v in rooted.order.tolist()[1:]:  # preorder, so the parent is labelled first
        key = intervals[parent_edge[v]]
        cut = cut_id.get(key) if key is not None else None
        cluster[v] = cluster[parent[v]] if cut is None else cut + 1
    return cluster


def _cluster_tree_shape(
    tree: PhyloTree, cluster_of_node: list[int], n_clusters: int
) -> tuple[list[int], list[list[int]], list[int]]:
    """
    Read ``K``, rooted at cluster 0, off one tree's cluster labels.

    Parameters
    ----------
    tree : PhyloTree
        One input tree (any will do; the first is used).
    cluster_of_node : list[int]
        Its cluster id per node, from :func:`_label_clusters`.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    tuple[list[int], list[list[int]], list[int]]
        ``(parent, children, postorder)``: parent cluster (``-1`` at the root), sorted children per
        cluster, and the clusters with children before parents.
    """
    rooted = tree.root_at(0)
    parent = [-1] * n_clusters
    first_seen: list[int] = []
    seen = [False] * n_clusters
    for v in rooted.order:
        c = cluster_of_node[int(v)]
        if not seen[c]:
            seen[c] = True
            first_seen.append(c)
            if c != 0:
                parent[c] = cluster_of_node[int(rooted.parent[int(v)])]
    children: list[list[int]] = [[] for _ in range(n_clusters)]
    for c in range(1, n_clusters):
        children[parent[c]].append(c)
    return parent, children, first_seen[::-1]


class _LeafLayout:
    """
    A cluster's local leaf numbering: own taxa, then child pseudo-leaves, then ``x_C^*``.

    Parameters
    ----------
    own_taxa : list[int]
        The cluster's own taxa, in increasing global id.
    child_clusters : list[int]
        Its children in ``K``, in increasing order.
    parent_cluster : int
        Its parent in ``K``, or ``-1`` at the root.
    """

    def __init__(self, own_taxa: list[int], child_clusters: list[int], parent_cluster: int):
        self.slot_of_taxon = {t: i for i, t in enumerate(own_taxa)}
        self.slot_of_child = {d: len(own_taxa) + i for i, d in enumerate(child_clusters)}
        self.star_leaf = -1 if parent_cluster < 0 else len(own_taxa) + len(child_clusters)
        self.n_leaves = len(own_taxa) + len(child_clusters) + (0 if self.star_leaf < 0 else 1)

    def piece(
        self,
        trees: tuple[PhyloTree, ...],
        tree_weights: tuple[float, ...] | None = None,
        splits: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> LocalPiece:
        """The local instance with these local trees (and optionally its split table)."""
        taxon_of_leaf = [-1] * self.n_leaves
        for t, slot in self.slot_of_taxon.items():
            taxon_of_leaf[slot] = t
        child_of_leaf = [-1] * self.n_leaves
        for d, slot in self.slot_of_child.items():
            child_of_leaf[slot] = d
        return LocalPiece(
            trees=trees,
            taxon_of_leaf=tuple(taxon_of_leaf),
            child_of_leaf=tuple(child_of_leaf),
            star_leaf=self.star_leaf,
            tree_weights=tree_weights,
            split_masks=None if splits is None else splits[0],
            split_weights=None if splits is None else splits[1],
        )


def _split_table(masks: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    The distinct masks among ``masks``, in increasing order, with their summed weights.

    Parameters
    ----------
    masks : numpy.ndarray
        ``uint64`` split masks, with repeats.
    weights : numpy.ndarray
        ``float64`` weight per mask.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The distinct masks and the summed weight of each.
    """
    order = np.argsort(masks, kind="stable")
    masks, weights = masks[order], weights[order]
    if masks.size == 0:
        return masks, weights
    starts = np.flatnonzero(np.concatenate(([True], masks[1:] != masks[:-1])))
    return masks[starts], np.add.reduceat(weights, starts)


def _layouts(
    n_taxa: int, labels: list[int], parent: list[int], children: list[list[int]]
) -> tuple[list[_LeafLayout], list[int], list[int]]:
    """
    Every cluster's local leaf numbering, from one tree's cluster labels.

    Parameters
    ----------
    n_taxa : int
        Number of taxa.
    labels : list[int]
        Cluster id per node of any input tree (the taxa are the same in all).
    parent : list[int]
        Parent cluster per cluster, ``-1`` at the root.
    children : list[list[int]]
        Sorted child clusters per cluster.

    Returns
    -------
    tuple[list[_LeafLayout], list[int], list[int]]
        The layouts; the local leaf id of each taxon in its own cluster; and the local leaf id of
        each cluster's pseudo-leaf in its parent cluster (``0`` at the root).
    """
    n_clusters = len(parent)
    own_taxa: list[list[int]] = [[] for _ in range(n_clusters)]
    for v in range(n_taxa):
        own_taxa[labels[v]].append(v)
    layouts = [_LeafLayout(own_taxa[c], children[c], parent[c]) for c in range(n_clusters)]
    slot = [layouts[labels[v]].slot_of_taxon[v] for v in range(n_taxa)]
    child_slot = [0] * n_clusters
    for c in range(1, n_clusters):
        child_slot[c] = layouts[parent[c]].slot_of_child[c]
    return layouts, slot, child_slot


def _compiled_reduction(trees: list[PhyloTree], weights: np.ndarray) -> ClusterTree | None:
    """
    :func:`cluster_reduction` with merging, through the compiled passes of :mod:`._kernels`.

    Produces the same cluster tree as the Python path: the same cut numbering (common splits in
    sorted interval order), the same local trees in the same order (by first occurrence), each
    built from the same representative. Returns ``None`` where that path does not apply: a cluster
    with more than 64 local leaves. Input trees have no degree-2 vertices (checked by
    :func:`cluster_reduction`).

    Parameters
    ----------
    trees : list[PhyloTree]
        The input trees, all on leaves ``0..n_leaves-1``.
    weights : numpy.ndarray
        ``float64[t]``; per-tree weights.

    Returns
    -------
    ClusterTree | None
        The tree of clusters with merged local trees, or ``None``.
    """
    t = len(trees)
    n_taxa = trees[0].n_leaves
    n_edges = np.array([tree.n_edges for tree in trees], dtype=np.int64)
    n_nodes = np.array([tree.n_nodes for tree in trees], dtype=np.int64)
    edges = np.concatenate([tree.edges for tree in trees]).astype(np.int32, copy=False)
    lengths = np.concatenate([tree.weight for tree in trees]).astype(np.float64, copy=False)
    edge_offset = np.zeros(t + 1, dtype=np.int64)
    np.cumsum(n_edges, out=edge_offset[1:])
    tree_of_edge = np.repeat(np.arange(t, dtype=np.int32), n_edges)

    index = leaf_numbering(trees[0])
    keys = _kernels.split_keys(edges, edge_offset, n_nodes, n_taxa, index)
    # The common splits: the first tree's interval keys found in every tree. Without degree-2
    # vertices (rejected upfront) a split occurs at most once per tree, so counting occurrences
    # counts trees.
    candidates = np.unique(keys[: edge_offset[1]])
    candidates = candidates[candidates >= 0]
    common = candidates
    if candidates.size:
        position = np.minimum(np.searchsorted(candidates, keys), candidates.size - 1)
        hits = np.bincount(position[candidates[position] == keys], minlength=candidates.size)
        common = candidates[hits == t]
    del keys
    n_clusters = len(common) + 1

    labels0 = _kernels.tree_labels(
        edges[: edge_offset[1]], n_nodes[0], n_taxa, index, common
    ).tolist()
    parent, children, postorder = _cluster_tree_shape(trees[0], labels0, n_clusters)
    layouts, slot, child_slot = _layouts(n_taxa, labels0, parent, children)
    if max(layout.n_leaves for layout in layouts) > _MAX_COMPILED_LEAVES:
        return None

    owner, mask, upper, lower, wlen, top = _kernels.local_edges(
        edges,
        lengths,
        edge_offset,
        n_nodes,
        n_taxa,
        index,
        common,
        np.array(slot, dtype=np.int64),
        np.array(child_slot, dtype=np.int64),
        weights,
    )
    del edges, lengths
    pieces = _grouped_pieces(owner, mask, upper, lower, wlen, top, tree_of_edge, layouts, n_taxa)
    return ClusterTree(
        pieces=pieces,
        parent=tuple(parent),
        children=tuple(tuple(cs) for cs in children),
        postorder=tuple(postorder),
        root=0,
    )


def _grouped_pieces(
    owner: np.ndarray,
    mask: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    wlen: np.ndarray,
    top: np.ndarray,
    tree_of_edge: np.ndarray,
    layouts: list[_LeafLayout],
    n_taxa: int,
) -> tuple[LocalPiece, ...]:
    """
    Group each cluster's local trees by topology, from the packed output of the kernels.

    A cluster's local tree in tree ``j`` is the row of its edges' masks in increasing order; two
    local trees have the same topology exactly when their rows are equal.

    Parameters
    ----------
    owner, mask, upper, lower, wlen, top
        The output of :func:`._kernels.local_edges`.
    tree_of_edge : numpy.ndarray
        ``int32[E]``; the input tree of each packed edge (non-decreasing).
    layouts : list[_LeafLayout]
        Each cluster's local leaf numbering.
    n_taxa : int
        Number of taxa.

    Returns
    -------
    tuple[LocalPiece, ...]
        One local instance per cluster, with ``tree_weights`` set.
    """
    n_clusters = len(layouts)
    # One linear-time sort for all clusters: by (cluster, tree, mask). Each (cluster, tree) run is
    # then one local tree as its row of masks in increasing order.
    order, starts = _kernels.cluster_order(owner, tree_of_edge, mask, n_clusters)
    mask, wlen = mask[order], wlen[order]
    lengths = np.diff(np.append(starts, order.size))
    run_owner = owner[order[starts]].astype(np.int64)
    run_tree = tree_of_edge[order[starts]].astype(np.int64)

    # Group equal rows per cluster (exactly; the hash only brings candidates together), and number
    # the groups by first occurrence, as the Python path does.
    hashes = _kernels.segment_hashes(mask, starts, lengths)
    by_hash = np.lexsort((np.arange(starts.size), hashes, run_owner))
    group, first = _kernels.assign_groups(mask, starts, lengths, by_hash, run_owner, hashes)
    renumber = np.empty(first.size, dtype=np.int64)
    by_first = np.argsort(first, kind="stable")
    renumber[by_first] = np.arange(first.size)
    group, first = renumber[group], first[by_first]

    # Summed weights per group and position: the members' rows line up position by position.
    group_length = lengths[first]
    group_offset = np.zeros(first.size + 1, dtype=np.int64)
    np.cumsum(group_length, out=group_offset[1:])
    position = np.arange(order.size) - np.repeat(starts, lengths)
    slot = np.repeat(group_offset[group], lengths) + position
    summed = np.bincount(slot, weights=wlen, minlength=int(group_offset[-1]))

    # Split tables: every group's masks with their summed weights, merged per cluster.
    within = np.arange(int(group_offset[-1])) - np.repeat(group_offset[:-1], group_length)
    group_masks = mask[np.repeat(starts[first], group_length) + within]
    group_owner = np.repeat(run_owner[first], group_length)
    split_order = np.lexsort((group_masks, group_owner))
    so, sm, sw = group_owner[split_order], group_masks[split_order], summed[split_order]
    new_split = np.concatenate(([True], (so[1:] != so[:-1]) | (sm[1:] != sm[:-1])))
    split_starts = np.flatnonzero(new_split)
    so, sm, sw = so[split_starts], sm[split_starts], np.add.reduceat(sw, split_starts)
    split_bounds = np.searchsorted(so, np.arange(n_clusters + 1))
    group_bounds = np.searchsorted(run_owner[first], np.arange(n_clusters + 1))

    # Only the representatives' endpoints are needed, laid out like the summed weights.
    rep_edges = order[np.repeat(starts[first], group_length) + within]
    upper_list, lower_list = upper[rep_edges].tolist(), lower[rep_edges].tolist()
    summed_list, offsets = summed.tolist(), group_offset.tolist()
    rep_tree = run_tree[first].tolist()
    pieces = []
    for c in range(n_clusters):
        local_trees = []
        for g in range(int(group_bounds[c]), int(group_bounds[c + 1])):
            begin, end = offsets[g], offsets[g + 1]
            pairs = list(zip(upper_list[begin:end], lower_list[begin:end]))
            local_trees.append(
                _assemble(
                    pairs, int(top[rep_tree[g], c]), summed_list[begin:end], layouts[c], n_taxa
                )
            )
        part = slice(int(split_bounds[c]), int(split_bounds[c + 1]))
        pieces.append(
            layouts[c].piece(tuple(local_trees), (1.0,) * len(local_trees), (sm[part], sw[part]))
        )
    return tuple(pieces)


def _merged_pieces(
    trees: list[PhyloTree],
    weights: Sequence[float],
    cluster_of_node: list[list[int]],
    parent: list[int],
    children: list[list[int]],
) -> tuple[LocalPiece, ...]:
    """
    Every cluster's local instance, with the local trees of equal topology merged into one.

    Each edge of each input tree is keyed by its split in its cluster's local instance: the local
    leaves below it, with every tree rooted at leaf 0 (the cluster holding it) or at ``x_C^*``
    (any other cluster), so the key does not depend on the tree. Two local trees have the same
    topology exactly when they have the same set of keys. A merged tree's edge weight is
    ``sum_j w_j * len_j(e)`` over its members, and its tree weight is ``1``.

    Parameters
    ----------
    trees : list[PhyloTree]
        The input trees.
    weights : Sequence[float]
        Per-tree weights ``w_j``.
    cluster_of_node : list[list[int]]
        Per input tree, its cluster id per node (:func:`_label_clusters`).
    parent : list[int]
        Parent cluster per cluster, ``-1`` at the root.
    children : list[list[int]]
        Sorted child clusters per cluster.

    Returns
    -------
    tuple[LocalPiece, ...]
        One local instance per cluster, with ``tree_weights`` set.
    """
    n_clusters = len(parent)
    n_taxa = trees[0].n_leaves
    layouts, slot, child_slot = _layouts(n_taxa, cluster_of_node[0], parent, children)

    # Per cluster: split keys -> (representative's edges as (upper, lower) node pairs in key order,
    # with lower = -1 - D for the pendant edge of x_D; its top node; the summed weights).
    # The group also keeps its edges' split keys, which give the cluster's split table.
    groups: list[
        dict[tuple[int, ...], tuple[list[tuple[int, int]], int, list[float], tuple[int, ...]]]
    ]
    groups = [{} for _ in range(n_clusters)]
    for j, (tree, labels) in enumerate(zip(trees, cluster_of_node)):
        per_cluster, top, parents, lengths = _cluster_edges(
            tree, labels, slot, child_slot, n_clusters
        )
        w = float(weights[j])
        # No input tree has a degree-2 vertex, so no two edges of a local tree share a split and
        # the sorted splits identify the local topology.
        for c, edges in enumerate(per_cluster):
            edges.sort()
            signature, nodes = zip(*edges)
            keys = signature
            group = groups[c].get(signature)
            if group is None:
                pairs = [(parents[v], v if labels[v] == c else -1 - labels[v]) for v in nodes]
                groups[c][signature] = (pairs, top[c], [w * lengths[v] for v in nodes], keys)
            else:
                summed = group[2]
                for i, v in enumerate(nodes):
                    summed[i] += w * lengths[v]

    pieces = []
    for c in range(n_clusters):
        local_trees = tuple(
            _assemble(pairs, top_node, summed, layouts[c], n_taxa)
            for pairs, top_node, summed, _ in groups[c].values()
        )
        splits = None
        if layouts[c].n_leaves <= _MAX_COMPILED_LEAVES:
            all_keys = [key for group in groups[c].values() for key in group[3]]
            all_weights = [x for group in groups[c].values() for x in group[2]]
            splits = _split_table(
                np.array(all_keys, dtype=np.uint64), np.array(all_weights, dtype=np.float64)
            )
        pieces.append(layouts[c].piece(local_trees, (1.0,) * len(local_trees), splits))
    return tuple(pieces)


def _cluster_edges(
    tree: PhyloTree, labels: list[int], slot: list[int], child_slot: list[int], n_clusters: int
) -> tuple[list[list[tuple[int, int]]], list[int], list[int], list[float]]:
    """
    One tree's edges grouped by the cluster that owns them, each with its local split key.

    An edge belongs to the cluster of its endpoint nearer leaf 0, so a cut-edge is the pendant edge
    of ``x_D`` in the parent cluster (its pendant ``x_C^*`` edge in ``D`` always has weight zero).
    Its key is the bitmask of the local leaves below it. Each edge is named by its lower endpoint.

    Parameters
    ----------
    tree : PhyloTree
        One input tree.
    labels : list[int]
        Its cluster id per node.
    slot : list[int]
        Local leaf id of each taxon in its own cluster.
    child_slot : list[int]
        Local leaf id of each cluster's pseudo-leaf in its parent cluster.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    tuple[list[list[tuple[int, int]]], list[int], list[int], list[float]]
        Per cluster its ``(key, lower endpoint)`` pairs; per cluster its top node (``-1`` at the
        root cluster); the parent of each node; the length of each node's parent edge.
    """
    rooted = tree.root_at(0)
    parents = rooted.parent.tolist()
    weight = tree.weight.tolist()
    lengths = [weight[e] if e >= 0 else 0.0 for e in rooted.parent_edge.tolist()]
    n_taxa = tree.n_leaves
    below = [1 << slot[v] if v < n_taxa else 0 for v in range(tree.n_nodes)]
    per_cluster: list[list[tuple[int, int]]] = [[] for _ in range(n_clusters)]
    top = [-1] * n_clusters
    for v in reversed(rooted.order.tolist()[1:]):  # children before parents
        p = parents[v]
        c = labels[p]
        if labels[v] == c:
            key = below[v]
        else:
            key = 1 << child_slot[labels[v]]
            top[labels[v]] = v
        per_cluster[c].append((key, v))
        below[p] |= key
    return per_cluster, top, parents, lengths


def _assemble(
    pairs: list[tuple[int, int]],
    top: int,
    weights: list[float],
    layout: _LeafLayout,
    n_taxa: int,
) -> PhyloTree:
    """
    One merged local tree from its representative's edges (see :func:`_merged_pieces`).

    Parameters
    ----------
    pairs : list[tuple[int, int]]
        ``(upper, lower)`` input-tree node pairs; ``lower = -1 - D`` for the pendant edge of
        ``x_D``.
    top : int
        The cluster's node incident to its parent cut, where ``x_C^*`` hangs by a zero-weight edge;
        ``-1`` at the root cluster.
    weights : list[float]
        Edge weight per pair.
    layout : _LeafLayout
        The cluster's local leaf numbering.
    n_taxa : int
        Number of taxa of the input trees.

    Returns
    -------
    PhyloTree
        The local tree on leaves ``0..n_leaves-1``.
    """
    leaf_of_node: dict[int, int] = {}
    n_nodes = 0
    local: dict[int, int] = {}

    def local_id(v: int) -> int:
        nonlocal n_nodes
        if v not in local:
            local[v] = n_nodes
            if v < n_taxa:
                leaf_of_node[n_nodes] = layout.slot_of_taxon[v]
            n_nodes += 1
        return local[v]

    edges: list[tuple[int, int]] = []
    edge_weights = list(weights)
    for upper, lower in pairs:
        a = local_id(upper)
        if lower >= 0:
            b = local_id(lower)
        else:
            b = n_nodes
            leaf_of_node[b] = layout.slot_of_child[-1 - lower]
            n_nodes += 1
        edges.append((a, b))
    if top >= 0:
        a = local_id(top)
        leaf_of_node[n_nodes] = layout.star_leaf
        edges.append((a, n_nodes))
        edge_weights.append(0.0)
        n_nodes += 1
    # The numbering of PhyloTree.from_edges (leaves by leaf id, then internal nodes in input order),
    # without its checks: the pieces of a tree are trees by construction.
    relabel = [-1] * n_nodes
    for node, leaf_id in leaf_of_node.items():
        relabel[node] = leaf_id
    next_id = len(leaf_of_node)
    for node in range(n_nodes):
        if relabel[node] < 0:
            relabel[node] = next_id
            next_id += 1
    arr = np.array([(relabel[a], relabel[b]) for a, b in edges], dtype=np.int32).reshape(-1, 2)
    return PhyloTree(n_nodes, len(leaf_of_node), arr, np.asarray(edge_weights, dtype=np.float64))


__all__ = ["cluster_reduction"]
