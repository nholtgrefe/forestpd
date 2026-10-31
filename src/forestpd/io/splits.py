"""
Trees as one weighted split system, and writing it as NEXUS (e.g. for PDA).

For unrooted PD, ``sum_j w_j PD_{T_j}(S)`` is exactly the split diversity of the union of the trees'
splits, each weighted by the summed ``w_j * len_j(e)`` of the edges inducing it (pendant splits
included). :func:`split_system` computes that split system; :func:`to_splits_nexus` writes it in the
``Taxa``/``Splits`` NEXUS format that split-diversity tools such as PDA read. A split system holds
unrooted trees only: no costs, survival probabilities or roots.
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import numpy as np

from .._validate import weights_to_array
from ..core.tree import PhyloTree, leaf_masks_below
from ..exceptions import ForestPDInputError
from . import _splits_kernel
from .convert import MissingBranchLength, Trees, build_registry, read_trees, to_phylotree

#: Compute split systems with the compiled kernels (:mod:`._splits_kernel`) when Numba is
#: installed. The Python path gives the same splits; set ``False`` to force it (tests).
USE_COMPILED = True

#: Seed of the random taxon hashes that identify splits in the compiled path.
_HASH_SEED = 20261013


def split_system(
    trees: Trees | Sequence[PhyloTree],
    weights: Sequence[float] | Mapping[int, float] | None = None,
    *,
    labels: Sequence[str] | None = None,
    missing_branch_length: MissingBranchLength = "error",
) -> tuple[list[np.ndarray], np.ndarray, tuple[str, ...]]:
    """
    The trees as one weighted split system.

    Every distinct split of the trees appears once, as its side without taxon 0, with weight
    ``sum_j w_j * len_j(e)`` over the edges of all trees inducing it. The split diversity of a taxon
    set (the summed weight of the splits separating two of its taxa) then equals its unrooted
    ``sum_j w_j PD_{T_j}``. Trees are read as unrooted.

    Parameters
    ----------
    trees : str | os.PathLike | Sequence
        The path of a file holding one Newick tree per line, or a list of Newick strings and/or
        PhyloZoo ``SemiDirectedPhyNetwork`` / ``DirectedPhyNetwork`` trees (with the ``phylozoo``
        extra), or a list of :class:`~forestpd.core.tree.PhyloTree` (e.g. from :func:`prune`).
        All trees must carry the same taxon set.
    weights : Sequence[float] | Mapping[int, float] | None, optional
        Per-tree weight ``w_j``, in the order of the trees, or keyed by tree index. ``None`` means
        all ``1.0`` (pass ``1 / t`` each for the mean PD).
    labels : Sequence[str] | None, optional
        For :class:`~forestpd.core.tree.PhyloTree` input only: the label of each leaf id (e.g.
        ``registry.labels`` from :func:`prune`). ``None`` uses the leaf ids themselves.
    missing_branch_length : str | float, default "error"
        Policy for edges without a branch length (text and PhyloZoo input).

    Returns
    -------
    tuple[list[numpy.ndarray], numpy.ndarray, tuple[str, ...]]
        Per split, the sorted taxon ids of its side without taxon 0; the ``float64`` split
        weights; and the label of each taxon id.

    Raises
    ------
    ForestPDInputError
        If a tree or weight is invalid, the trees do not share one taxon set, or ``labels`` is
        given for text input or has the wrong length.

    Examples
    --------
    >>> import forestpd as fpd
    >>> sides, weights, labels = fpd.io.splits.split_system(
    ...     ["(A:1,B:1,(C:1,D:1):2);", "(A:1,C:1,(B:1,D:1):2);"])
    >>> sorted(("".join(labels[x] for x in side), float(w))
    ...        for side, w in zip(sides, weights))   # each pendant split twice, each cherry once
    [('B', 2.0), ('BCD', 2.0), ('BD', 2.0), ('C', 2.0), ('CD', 2.0), ('D', 2.0)]
    """
    phylo_trees, taxon_labels = _as_phylotrees(trees, labels, missing_branch_length)
    tree_weight = weights_to_array(weights, len(phylo_trees))
    members, offsets, split_weights = _flat_split_system(phylo_trees, tree_weight)
    sides = np.split(members, offsets[1:-1]) if split_weights.size else []
    return sides, split_weights, taxon_labels


def to_splits_nexus(
    trees: Trees | Sequence[PhyloTree],
    path: str | os.PathLike[str],
    weights: Sequence[float] | Mapping[int, float] | None = None,
    *,
    labels: Sequence[str] | None = None,
    missing_branch_length: MissingBranchLength = "error",
) -> int:
    """
    Write the trees' weighted split system (:func:`split_system`) as a NEXUS file, e.g. for PDA.

    The file has a ``Taxa`` block and a ``Splits`` block (``labels=no weights=yes``), each split
    given as its weight and the 1-based taxon indices of one side. A split system holds unrooted
    trees only: no costs, survival probabilities or roots.

    Parameters
    ----------
    trees : str | os.PathLike | Sequence
        As for :func:`split_system`.
    path : str | os.PathLike
        The file to write.
    weights, labels, missing_branch_length
        As for :func:`split_system`.

    Returns
    -------
    int
        The number of splits written.

    Raises
    ------
    ForestPDInputError
        As :func:`split_system`.
    OSError
        If ``path`` cannot be written.
    """
    phylo_trees, taxon_labels = _as_phylotrees(trees, labels, missing_branch_length)
    tree_weight = weights_to_array(weights, len(phylo_trees))
    members, offsets, split_weights = _flat_split_system(phylo_trees, tree_weight)
    n = len(taxon_labels)
    lines = ["#nexus", "", "BEGIN Taxa;", f"DIMENSIONS ntax={n};", "TAXLABELS"]
    lines += [f"[{i}] {_quote(label)}" for i, label in enumerate(taxon_labels, start=1)]
    lines += [";", "END; [Taxa]", "", "BEGIN Splits;"]
    lines += [f"DIMENSIONS ntax={n} nsplits={split_weights.size};"]
    lines += ["FORMAT labels=no weights=yes confidences=no intervals=no;", "MATRIX"]
    one_based = (members + 1).tolist()
    bounds = offsets.tolist()
    for r, weight in enumerate(split_weights.tolist()):
        side = one_based[bounds[r] : bounds[r + 1]]
        members_text = " ".join(map(str, side))
        lines.append(f"[{r + 1}, size={len(side)}] \t {weight!r} \t  {members_text},")
    lines += [";", "END; [Splits]", ""]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return int(split_weights.size)


def _as_phylotrees(
    trees: Trees | Sequence[PhyloTree],
    labels: Sequence[str] | None,
    missing_branch_length: MissingBranchLength,
) -> tuple[tuple[PhyloTree, ...], tuple[str, ...]]:
    """The trees as PhyloTrees on one taxon set, with the label of each taxon id."""
    items = None if isinstance(trees, (str, os.PathLike)) else list(trees)
    if items and any(isinstance(tree, PhyloTree) for tree in items):
        phylo = tuple(tree for tree in items if isinstance(tree, PhyloTree))
        if len(phylo) != len(items):
            raise ForestPDInputError("Do not mix PhyloTrees with other tree inputs.")
        n = phylo[0].n_leaves
        for j, tree in enumerate(phylo):
            if tree.n_leaves != n:
                raise ForestPDInputError(
                    f"Tree {j} has {tree.n_leaves} taxa but tree 0 has {n}. "
                    "All trees must carry the same taxon set."
                )
        names = tuple(str(x) for x in range(n)) if labels is None else tuple(labels)
        if len(names) != n:
            raise ForestPDInputError(f"Expected {n} labels, got {len(names)}")
        return phylo, names
    if labels is not None:
        raise ForestPDInputError("labels are only for PhyloTree input; text input has its own.")
    parsed = read_trees(trees)
    registry = build_registry(parsed)
    for j, read in enumerate(parsed):
        if len(read.leaf_names) != registry.n:
            raise ForestPDInputError(
                f"Tree {j} has {len(read.leaf_names)} taxa but the trees have {registry.n} in "
                "total. All trees must carry the same taxon set."
            )
    phylo = tuple(to_phylotree(read, registry, missing_branch_length) for read in parsed)
    return phylo, registry.labels


def _flat_split_system(
    trees: Sequence[PhyloTree], tree_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The distinct splits as a flat array of taxon ids with offsets, and their weights.

    Compiled: hash keys per edge, one sort to merge equal splits, the taxa of each distinct split
    listed once (see :mod:`._splits_kernel`). Python: exact leaf bitmasks per edge.
    """
    n_taxa = trees[0].n_leaves
    if USE_COMPILED and _splits_kernel.HAVE_NUMBA:
        n_nodes = np.array([tree.n_nodes for tree in trees], dtype=np.int64)
        edge_offset = np.zeros(len(trees) + 1, dtype=np.int64)
        np.cumsum([tree.n_edges for tree in trees], out=edge_offset[1:])
        edges = np.concatenate([tree.edges for tree in trees]).astype(np.int64)
        lengths = np.concatenate([tree.weight for tree in trees]).astype(np.float64)
        rng = np.random.default_rng(_HASH_SEED)
        # Two random odd 64-bit hashes per taxon; a split's key is their sums over its side.
        h1, h2 = (
            rng.integers(0, 2**63, size=n_taxa, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
            for _ in range(2)
        )
        k1, k2, count, weight, tree_of, lower = _splits_kernel.split_keys(
            edges, edge_offset, n_nodes, n_taxa, lengths, tree_weight, h1, h2
        )
        if k1.size == 0:
            return np.empty(0, dtype=np.int64), np.zeros(1, dtype=np.int64), np.empty(0)
        order = np.lexsort((count, k2, k1))
        s1, s2, sc = k1[order], k2[order], count[order]
        starts = np.ones(order.size, dtype=bool)
        starts[1:] = (s1[1:] != s1[:-1]) | (s2[1:] != s2[:-1]) | (sc[1:] != sc[:-1])
        first = np.flatnonzero(starts)
        summed = np.add.reduceat(weight[order], first)
        reps = order[first]
        by_tree = np.argsort(tree_of[reps], kind="stable")
        reps, summed = reps[by_tree], summed[by_tree]
        members, offsets = _splits_kernel.split_members(
            edges, edge_offset, n_nodes, n_taxa, tree_of[reps], lower[reps], count[reps]
        )
        return members, offsets, summed

    by_mask: dict[int, float] = {}
    for w, tree in zip(tree_weight.tolist(), trees):
        for e, mask in enumerate(leaf_masks_below(tree.root_at(0))):
            by_mask[mask] = by_mask.get(mask, 0.0) + w * float(tree.weight[e])
    sides = [[x for x in range(n_taxa) if (mask >> x) & 1] for mask in by_mask]
    offsets = np.zeros(len(sides) + 1, dtype=np.int64)
    np.cumsum([len(side) for side in sides], out=offsets[1:])
    members = np.array([x for side in sides for x in side], dtype=np.int64)
    return members, offsets, np.array(list(by_mask.values()), dtype=np.float64)


def _quote(label: str) -> str:
    """A NEXUS taxon label: single-quoted, with embedded quotes doubled."""
    return "'" + label.replace("'", "''") + "'"


__all__ = ["split_system", "to_splits_nexus"]
