"""
Input conversion: a tree file, Newick strings or PhyloZoo trees -> internal representation.

Turns every accepted tree input into a :class:`~forestpd.io.newick.ParsedTree`, numbers the
taxa of all of them against one shared :class:`~forestpd.core.taxa.TaxonRegistry`, and builds the
:class:`~forestpd.core.tree.PhyloTree` objects the algorithm runs on.

The trees are given either as the path of one file holding one Newick tree per line, or as a list
of Newick strings and/or PhyloZoo trees. Newick is read by
:func:`forestpd.io.newick.parse_newick`, forestpd's own parser (no PhyloZoo needed). PhyloZoo
trees are handed to :mod:`forestpd.io.pz`, which is imported only when such an object actually
arrives; so this module works without the optional ``phylozoo`` extra.

Assumptions: identical taxon labels denote the same taxon. Every tree must carry the same taxon
set; that is checked once the registry is known, by ``MaxMPDInstance``.
Branch lengths are not assumed present: the ``missing_branch_length`` policy decides what to do
when one is absent.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Any, Iterator, Sequence, Union

import numpy as np

from .newick import ROOT_LABEL, ParsedTree, parse_newick
from ..core.tree import PhyloTree
from ..exceptions import ForestPDInputError, ForestPDWarning
from ..core.taxa import TaxonRegistry

#: One tree in a list of trees: a Newick string or (with the ``phylozoo`` extra) a PhyloZoo
#: ``SemiDirectedPhyNetwork`` / ``DirectedPhyNetwork`` tree.
TreeInput = Union[str, Any]

#: The trees accepted by the public API: the path of a file holding one Newick tree per line, or a
#: list of :data:`TreeInput`.
Trees = Union[str, "os.PathLike[str]", Sequence[TreeInput]]

#: Policy for edges without a branch length: ``"error"``, ``"zero"``, or a numeric default.
MissingBranchLength = Union[str, float, int]


def _resolve_weight(bl: float | None, policy: MissingBranchLength, u: object, v: object) -> float:
    """
    Resolve a (possibly missing) branch length to a non-negative float per ``policy``.

    Parameters
    ----------
    bl : float | None
        The edge's branch length, or ``None`` if absent.
    policy : MissingBranchLength
        ``"error"``, ``"zero"``, or a finite numeric default used when ``bl`` is ``None``.
    u, v : object
        The edge's endpoints, used only in error messages.

    Returns
    -------
    float
        The resolved branch length.

    Raises
    ------
    ForestPDInputError
        If ``bl`` is missing under the ``"error"`` policy, the policy is invalid or non-finite,
        or ``bl`` is negative or non-finite.
    """
    if bl is None:
        if policy == "error":
            raise ForestPDInputError(
                f"Edge ({u!r}, {v!r}) has no branch length. Provide one, or set "
                "missing_branch_length to 'zero' or a numeric default."
            )
        if policy == "zero":
            return 0.0
        if isinstance(policy, (int, float)) and not isinstance(policy, bool):
            if not math.isfinite(policy):
                raise ForestPDInputError(
                    f"missing_branch_length default must be finite, got {policy!r}"
                )
            return float(policy)
        raise ForestPDInputError(f"Invalid missing_branch_length policy: {policy!r}")
    value = float(bl)
    if not math.isfinite(value) or value < 0:
        raise ForestPDInputError(
            f"Edge ({u!r}, {v!r}) has an invalid branch length {value} "
            "(must be finite, non-negative)"
        )
    return value


def to_phylotree(
    parsed: ParsedTree,
    registry: TaxonRegistry | None = None,
    missing_branch_length: MissingBranchLength = "error",
) -> PhyloTree:
    """
    Number a parsed tree's taxa against ``registry`` and build a :class:`PhyloTree`.

    Parameters
    ----------
    parsed : ParsedTree
        The tree, as read from Newick or converted from PhyloZoo.
    registry : TaxonRegistry | None, default None
        Global taxon id mapping. Must contain every taxon of ``parsed``. If ``None``, a registry
        is built from this tree's taxa.
    missing_branch_length : str | float, default "error"
        Policy for edges lacking a branch length: ``"error"``, ``"zero"``, or a numeric value.

    Returns
    -------
    PhyloTree
        The tree, with leaf ids equal to the registry's taxon ids.

    Raises
    ------
    ForestPDInputError
        If a taxon is missing from ``registry``, or an edge has a missing/negative branch length
        under the chosen policy.

    Examples
    --------
    >>> from forestpd.io.newick import parse_newick
    >>> tree = to_phylotree(parse_newick("(A:1,B:2,C:3);"))
    >>> tree.n_leaves, tree.total_weight()
    (3, 6.0)
    """
    if registry is None:
        registry = build_registry([parsed])
    weights = _resolve_weights(parsed, missing_branch_length)
    leaf_ids = registry.id_array(parsed.leaf_names)
    return PhyloTree.from_leaf_arrays(
        parsed.n_nodes, parsed.edges, weights, parsed.leaf_nodes, leaf_ids
    )


def _resolve_weights(parsed: ParsedTree, policy: MissingBranchLength) -> np.ndarray:
    """
    Every edge's weight per ``policy``, as :func:`_resolve_weight` would give it edge by edge.

    The first edge (in edge order) that :func:`_resolve_weight` would reject raises its error.

    Parameters
    ----------
    parsed : ParsedTree
        The tree.
    policy : MissingBranchLength
        The missing-length policy.

    Returns
    -------
    numpy.ndarray
        ``float64[E]``; the resolved branch lengths.

    Raises
    ------
    ForestPDInputError
        As :func:`_resolve_weight`, for the first offending edge.
    """
    lengths, present = parsed.lengths, parsed.has_length
    with np.errstate(invalid="ignore"):
        bad = present & ~(np.isfinite(lengths) & (lengths >= 0))
    default = 0.0
    if not present.all():
        try:
            default = _resolve_weight(None, policy, None, None)
        except ForestPDInputError:
            bad |= ~present
    if bad.any():
        e = int(np.flatnonzero(bad)[0])
        labels = parsed.leaf_labels
        u, v = (int(x) for x in parsed.edges[e])
        _resolve_weight(
            float(lengths[e]) if present[e] else None, policy, labels.get(u, u), labels.get(v, v)
        )
    return np.where(present, lengths, default)


def _iter_file(path: str, rooted: bool = False) -> Iterator[ParsedTree]:
    """
    Parse a file holding one Newick tree per line (blank lines are skipped), one line at a time.

    The file is read line by line, so its text is never held in memory as a whole.

    Parameters
    ----------
    path : str
        Path to the file.
    rooted : bool, default False
        Parse in rooted mode (see :func:`~forestpd.io.newick.parse_newick`).

    Yields
    ------
    ParsedTree
        The trees, in file order.

    Raises
    ------
    ForestPDInputError
        If the file cannot be read, holds no tree, or holds an invalid tree.
    """
    found = False
    try:
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    tree = parse_newick(line.rstrip("\r\n"), rooted)
                except ForestPDInputError as exc:
                    raise ForestPDInputError(f"In file {path!r}, line {number}: {exc}") from exc
                found = True
                yield tree
    except OSError as exc:
        raise ForestPDInputError(f"Could not read tree file {path!r}: {exc}") from exc
    if not found:
        raise ForestPDInputError(f"Tree file {path!r} contains no trees.")


def _read(tree: TreeInput, rooted: bool = False) -> ParsedTree:
    """
    Read one element of a list of trees into a :class:`ParsedTree`.

    Parameters
    ----------
    tree : TreeInput
        A Newick string or a PhyloZoo network.
    rooted : bool, default False
        Parse in rooted mode.

    Returns
    -------
    ParsedTree
        The parsed tree.

    Raises
    ------
    ForestPDInputError
        If the element is a file path or has an unsupported type, or parsing fails.
    """
    if isinstance(tree, os.PathLike) or (
        isinstance(tree, str) and "(" not in tree and not tree.rstrip().endswith(";")
    ):
        raise ForestPDInputError(
            f"{os.fspath(tree)!r} is not a Newick tree. To read trees from a file, pass the "
            "file's path as trees itself, not inside a list."
        )
    if isinstance(tree, str):
        return parse_newick(tree, rooted)
    if type(tree).__module__.split(".")[0] == "phylozoo":
        from .pz import parsed_from_phylozoo

        return parsed_from_phylozoo(tree, rooted)
    raise ForestPDInputError(
        f"Unsupported tree input of type {type(tree).__name__}; expected a Newick string or a "
        "PhyloZoo tree."
    )


def build_registry(trees: Sequence[ParsedTree]) -> TaxonRegistry:
    """
    Build a :class:`TaxonRegistry` from the union of the trees' taxa.

    Parameters
    ----------
    trees : Sequence[ParsedTree]
        The parsed input trees.

    Returns
    -------
    TaxonRegistry
        A registry over the union of their taxon labels.

    Raises
    ------
    ForestPDInputError
        If the trees carry no taxa at all.
    """
    labels: set[str] = set()
    for parsed in trees:
        labels.update(parsed.leaf_names)
    if not labels:
        raise ForestPDInputError("No taxa found in the input trees.")
    return TaxonRegistry(labels)


def read_trees(trees: Trees, rooted: bool = False) -> list[ParsedTree]:
    """
    Read the accepted tree inputs into :class:`~forestpd.io.newick.ParsedTree` objects.

    Parameters
    ----------
    trees : Trees
        The path of a file holding one Newick tree per line, or a list of Newick strings and/or
        PhyloZoo ``SemiDirectedPhyNetwork`` / ``DirectedPhyNetwork`` trees.
    rooted : bool, default False
        Keep the roots (see :func:`~forestpd.io.newick.parse_newick`).

    Returns
    -------
    list[ParsedTree]
        The trees, in input (or file) order. Their taxon sets are not compared.

    Raises
    ------
    ForestPDInputError
        If no trees are given, or an input cannot be read or parsed as a tree.
    """
    return list(_iter_trees(trees, rooted))


def _iter_trees(trees: Trees, rooted: bool = False) -> Iterator[ParsedTree]:
    """:func:`read_trees`, one tree at a time (a file is read line by line)."""
    if isinstance(trees, (str, os.PathLike)):
        path = os.fspath(trees)
        if "(" in path or ";" in path:
            raise ForestPDInputError(
                "trees is a single Newick string. Pass a list of Newick strings, or the path of a "
                "file holding the trees."
            )
        if not os.path.isfile(path):
            raise ForestPDInputError(f"Tree file not found: {path!r}")
        yield from _iter_file(path, rooted)
        return
    if not trees:
        raise ForestPDInputError("At least one tree is required.")
    for tree in trees:
        yield _read(tree, rooted)


def coerce_trees(
    trees: Trees,
    missing_branch_length: MissingBranchLength = "error",
    rooted: bool = False,
) -> tuple[tuple[PhyloTree, ...], TaxonRegistry, list[float]]:
    """
    Read the trees and turn them into internal trees sharing one registry.

    Warns (:class:`~forestpd.exceptions.ForestPDWarning`) when the input looks rooted but
    ``rooted`` is False (a tree marked ``[&R]``, a PhyloZoo ``DirectedPhyNetwork``, or every tree
    with a binary root), and when ``rooted`` is True but a tree is marked ``[&U]``.

    Parameters
    ----------
    trees : Trees
        The path of a file holding one Newick tree per line, or a list of Newick strings and/or
        PhyloZoo ``SemiDirectedPhyNetwork`` / ``DirectedPhyNetwork`` trees.
    missing_branch_length : str | float, default "error"
        Branch-length policy (see :func:`to_phylotree`).
    rooted : bool, default False
        Keep the roots (see :func:`~forestpd.io.newick.parse_newick`).

    Returns
    -------
    tuple[tuple[PhyloTree, ...], TaxonRegistry, list[float]]
        The internal trees, in input (or file) order, the shared registry, and the stem length of
        every tree (all zero when not ``rooted``).

    Raises
    ------
    ForestPDInputError
        If no trees are given, an input cannot be parsed as a tree, the trees do not all carry the
        same taxon set, a branch length is invalid under the chosen policy, or unrooted input uses
        the reserved root label.

    Warns
    -----
    ForestPDWarning
        As described above.
    """
    if not isinstance(trees, (str, os.PathLike)):
        trees = list(trees)  # may be read twice (see below)
    # Each tree is converted as soon as it is read, so only the compact internal trees stay in
    # memory, not every tree's parse (with its labels as strings) nor a file's whole text. The
    # taxon ids come from the first tree: ids follow sorted labels, so with equal taxon sets this is
    # the registry of all trees. Anything else (another taxon set, an invalid branch length, the
    # reserved root label) reads all trees first instead, with the same errors and warnings.
    registry: TaxonRegistry | None = None
    internal, hints, stems = [], [], []
    for parsed_tree in _iter_trees(trees, rooted):
        if registry is None:
            registry = build_registry([parsed_tree])
        if len(parsed_tree.leaf_names) != registry.n or (
            not rooted and ROOT_LABEL in parsed_tree.leaf_names
        ):
            return _coerce_all(trees, missing_branch_length, rooted)
        try:
            internal.append(to_phylotree(parsed_tree, registry, missing_branch_length))
        except ForestPDInputError:
            return _coerce_all(trees, missing_branch_length, rooted)
        hints.append(parsed_tree.root_hint)
        stems.append(parsed_tree.stem)
    assert registry is not None  # _iter_trees yields at least one tree or raises
    _warn_rooting(hints, rooted)
    return tuple(internal), registry, stems


def _coerce_all(
    trees: Trees, missing_branch_length: MissingBranchLength, rooted: bool
) -> tuple[tuple[PhyloTree, ...], TaxonRegistry, list[float]]:
    """:func:`coerce_trees`, reading every tree before converting any."""
    parsed = read_trees(trees, rooted)
    _warn_rooting([p.root_hint for p in parsed], rooted, stacklevel=5)
    if not rooted and any(ROOT_LABEL in p.leaf_names for p in parsed):
        raise ForestPDInputError(f"The leaf label {ROOT_LABEL!r} is reserved for rooted mode.")
    registry = build_registry(parsed)
    # Checked before numbering: a tree lacking some taxa would get leaf ids outside 0..n-1.
    for j, p in enumerate(parsed):
        missing = sorted(set(registry.labels).difference(p.leaf_names))
        if missing:
            shown = ", ".join(repr(label) for label in missing[:5])
            raise ForestPDInputError(
                f"Tree {j} lacks {len(missing)} of the {registry.n} taxa of all trees ({shown}"
                f"{', ...' if len(missing) > 5 else ''}). All trees must carry the same taxon set."
            )
    internal = tuple(to_phylotree(p, registry, missing_branch_length) for p in parsed)
    return internal, registry, [p.stem for p in parsed]


def _warn_rooting(hints: Sequence[str | None], rooted: bool, stacklevel: int = 4) -> None:
    """
    The warnings of :func:`coerce_trees` about the input's apparent rooting; ``stacklevel`` points
    them at the caller of ``MaxMPDInstance``.
    """
    if not rooted and ("R" in hints or all(h == "binary" for h in hints)):
        warnings.warn(
            "The input trees appear to be rooted, but PD is computed on unrooted trees. Pass "
            "rooted=True for rooted PD.",
            ForestPDWarning,
            stacklevel=stacklevel,
        )
    if rooted and "U" in hints:
        warnings.warn(
            "A tree is marked [&U] (unrooted) but rooted=True; its top node is used as the root.",
            ForestPDWarning,
            stacklevel=stacklevel,
        )


__all__ = [
    "read_trees",
    "TreeInput",
    "Trees",
    "MissingBranchLength",
    "to_phylotree",
    "build_registry",
    "coerce_trees",
]
