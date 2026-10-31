"""
Utilities for preparing tree sets.

:func:`prune` restricts a set of trees to a chosen set of taxa, the usual first step when the trees
carry more taxa than the analysis is about (outgroups, other accessions, species outside the
conservation target). Its output goes straight into
:meth:`~forestpd.instance.MaxMPDInstance.from_phylotrees`.

:func:`split_system` gives the trees as one weighted split system (defined in
:mod:`forestpd.io.splits`, which also writes it as a NEXUS file for tools such as PDA).
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from .core.taxa import TaxonRegistry
from .core.tree import PhyloTree
from .exceptions import ForestPDInputError
from ._validate import weights_to_array
from .io.convert import MissingBranchLength, Trees, read_trees, to_phylotree
from .io.newick import ParsedTree
from .io.splits import split_system


def prune(
    trees: Trees,
    taxa: Iterable[str],
    weights: Sequence[float] | Mapping[int, float] | None = None,
    *,
    missing_branch_length: MissingBranchLength = "error",
) -> tuple[tuple[PhyloTree, ...], np.ndarray, TaxonRegistry]:
    """
    Prune every tree to ``taxa``, dropping the trees that do not contain all of them.

    Each tree is restricted to the subtree spanning ``taxa``: the other leaves are removed, and so
    is every vertex left as a leaf or with only two neighbours, whose two edges are merged into one
    (lengths summed). Unrooted PD of any set of the kept taxa is therefore unchanged. Trees are
    read as unrooted; pruning a rooted tree would move its root, so this is for unrooted PD.

    Parameters
    ----------
    trees : str | os.PathLike | Sequence
        The path of a file holding one Newick tree per line, or a list of Newick strings and/or
        PhyloZoo ``SemiDirectedPhyNetwork`` / ``DirectedPhyNetwork`` trees (with the ``phylozoo``
        extra). Every Newick tree ends with ``;``. The trees may carry different taxon sets.
    taxa : Iterable[str]
        The taxa to keep; at least two.
    weights : Sequence[float] | Mapping[int, float] | None, optional
        Per-tree weight ``w_j``, in the order of the trees (the list, or the file), or keyed by
        tree index. ``None`` means all ``1.0``. Removed trees lose their weight; the others are
        not rescaled.
    missing_branch_length : str | float, default "error"
        Policy for edges without a branch length: ``"error"``, ``"zero"``, or a numeric default.
        Applied after pruning; a merged edge has a length if any of its parts had one.

    Returns
    -------
    tuple[tuple[PhyloTree, ...], numpy.ndarray, TaxonRegistry]
        ``(trees, weights, registry)``: the pruned trees in input order, numbered against
        ``registry``; the ``float64`` weights of the kept trees; and the registry of the kept taxa.
        Use them as ``MaxMPDInstance.from_phylotrees(trees, registry, weights=weights)``.

    Raises
    ------
    ForestPDInputError
        If fewer than two taxa are given, no tree contains all of them, or a tree or weight is
        invalid.

    Examples
    --------
    >>> import forestpd as fpd
    >>> trees, weights, registry = fpd.utils.prune(
    ...     ["((A:1,B:1):1,(C:1,(D:1,E:1):1):1);", "((A:1,C:1):1,(B:1,D:1):1);"],
    ...     {"A", "B", "C"}, weights=[2.0, 3.0])
    >>> len(trees), weights.tolist(), registry.labels
    (2, [2.0, 3.0], ('A', 'B', 'C'))
    >>> instance = fpd.MaxMPDInstance.from_phylotrees(trees, registry, weights=weights)
    >>> instance.evaluate_pd({"A", "C"})    # 2 * (1 + 3) + 3 * (1 + 1)
    14.0
    """
    if isinstance(taxa, str):
        raise ForestPDInputError(
            f"Expected a collection of taxon labels, got the string {taxa!r}; "
            f"wrap a single label as [{taxa!r}]."
        )
    keep = set(taxa)
    for label in keep:
        if not isinstance(label, str):
            raise ForestPDInputError(f"Taxa must be labels (strings), got {type(label).__name__}")
    if len(keep) < 2:
        raise ForestPDInputError(f"At least two taxa are needed to prune to, got {len(keep)}")
    parsed = read_trees(trees)
    all_weights = weights_to_array(weights, len(parsed))
    registry = TaxonRegistry(keep)
    kept_trees: list[PhyloTree] = []
    kept_weights: list[float] = []
    for j, tree in enumerate(parsed):
        if keep <= set(tree.leaf_names):
            kept_trees.append(to_phylotree(_restrict(tree, keep), registry, missing_branch_length))
            kept_weights.append(float(all_weights[j]))
    if not kept_trees:
        raise ForestPDInputError(f"No tree contains all {len(keep)} taxa.")
    return tuple(kept_trees), np.asarray(kept_weights, dtype=np.float64), registry


def _restrict(tree: ParsedTree, keep: set[str]) -> ParsedTree:
    """
    The subtree of ``tree`` spanning the leaves labelled in ``keep``, degree-2 vertices suppressed.

    Parameters
    ----------
    tree : ParsedTree
        An unrooted parsed tree containing every label of ``keep``.
    keep : set[str]
        The labels to keep.

    Returns
    -------
    ParsedTree
        The restricted tree; a merged edge's length is the sum of its parts' lengths, missing only
        if all of them are.
    """
    # Adjacency with branch lengths (None where missing).
    adj: dict[int, dict[int, float | None]] = {v: {} for v in range(tree.n_nodes)}
    for (u, v), length, present in zip(
        tree.edges.tolist(), tree.lengths.tolist(), tree.has_length.tolist()
    ):
        adj[u][v] = adj[v][u] = float(length) if present else None
    labels = {v: name for v, name in tree.leaf_labels.items() if name in keep}
    # Remove unwanted leaves, and the internal vertices that become leaves.
    stack = [v for v in adj if len(adj[v]) <= 1 and v not in labels]
    while stack:
        v = stack.pop()
        for w in adj.pop(v):
            del adj[w][v]
            if len(adj[w]) <= 1 and w not in labels:
                stack.append(w)
    # Suppress the degree-2 vertices left behind, merging their two edges into one.
    for v in [v for v in adj if len(adj[v]) == 2 and v not in labels]:
        (a, la), (b, lb) = adj.pop(v).items()
        del adj[a][v], adj[b][v]
        merged = None if la is None and lb is None else (la or 0.0) + (lb or 0.0)
        adj[a][b] = adj[b][a] = merged
    index = {v: i for i, v in enumerate(adj)}
    pairs = [(u, v) for u in adj for v in adj[u] if u < v]
    return ParsedTree.from_lists(
        n_nodes=len(index),
        edges=[(index[u], index[v]) for u, v in pairs],
        lengths=[adj[u][v] for u, v in pairs],
        leaf_labels={index[v]: name for v, name in labels.items()},
    )


__all__ = ["prune", "split_system"]
