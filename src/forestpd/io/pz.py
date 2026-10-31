"""
Conversion between forestpd trees and PhyloZoo networks.

This is the only module that imports PhyloZoo, and it needs the optional extra
(``pip install forestpd[phylozoo]``). Nothing else in forestpd imports it unless a PhyloZoo
object is passed in, so the core runs without PhyloZoo (or NetworkX) installed.

Examples
--------
>>> import forestpd as fpd
>>> from forestpd.io.pz import to_phylozoo
>>> instance = fpd.MaxMPDInstance(["(A:1,B:1,(C:1,D:1):2);"])
>>> network = to_phylozoo(instance.trees[0], instance.registry)
>>> sorted(network.taxa)
['A', 'B', 'C', 'D']
"""

from __future__ import annotations

from typing import Any

from .convert import MissingBranchLength, to_phylotree
from .newick import ParsedTree, from_parents
from ..core.tree import PhyloTree
from ..exceptions import ForestPDImportError, ForestPDInputError
from ..core.taxa import TaxonRegistry

try:
    from phylozoo import DirectedPhyNetwork, SemiDirectedPhyNetwork
    from phylozoo.core.network.sdnetwork.classifications import is_tree
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ForestPDImportError(
        "Converting to and from PhyloZoo needs the optional extra: "
        "pip install 'forestpd[phylozoo]'"
    ) from exc


def parsed_from_phylozoo(network: Any, rooted: bool = False) -> ParsedTree:
    """
    Read a PhyloZoo tree into a :class:`ParsedTree`.

    A ``DirectedPhyNetwork`` is rooted and can be read in either mode. A ``SemiDirectedPhyNetwork``
    is unrooted, so it cannot be read with ``rooted=True``.

    Parameters
    ----------
    network : SemiDirectedPhyNetwork | DirectedPhyNetwork
        A tree (no hybrid edges).
    rooted : bool, default False
        Keep the root (see :func:`~forestpd.io.newick.parse_newick`).

    Returns
    -------
    ParsedTree
        The tree, with leaf labels and branch lengths (``None`` where absent).

    Raises
    ------
    ForestPDInputError
        If ``network`` is not a PhyloZoo tree, or is a ``SemiDirectedPhyNetwork`` with
        ``rooted=True``.
    """
    if isinstance(network, DirectedPhyNetwork):
        return _parsed_from_directed(network, rooted)
    if not isinstance(network, SemiDirectedPhyNetwork):
        raise ForestPDInputError(
            "Expected a PhyloZoo SemiDirectedPhyNetwork or DirectedPhyNetwork, got "
            f"{type(network).__name__}"
        )
    if rooted:
        raise ForestPDInputError(
            "A SemiDirectedPhyNetwork is unrooted, so it cannot be used with rooted=True. Use a "
            "DirectedPhyNetwork or a rooted Newick string."
        )
    if not is_tree(network):
        raise ForestPDInputError("Input network is not a tree (it has hybrid edges).")

    nodes = list(network.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}
    leaf_labels: dict[int, str] = {}
    for leaf in network.leaves:
        label = network.get_label(leaf)
        leaf_labels[node_index[leaf]] = str(leaf if label is None else label)
    edges: list[tuple[int, int]] = []
    lengths: list[float | None] = []
    for u, v, key in network.tree_edges:
        edges.append((node_index[u], node_index[v]))
        lengths.append(network.get_branch_length(u, v, key))
    return ParsedTree.from_lists(len(nodes), edges, lengths, leaf_labels, check_tree=True)


def _parsed_from_directed(network: Any, rooted: bool) -> ParsedTree:
    """
    Read a PhyloZoo ``DirectedPhyNetwork`` tree into a :class:`ParsedTree`.

    Parameters
    ----------
    network : DirectedPhyNetwork
        A rooted tree (no hybrid edges).
    rooted : bool
        Keep the root.

    Returns
    -------
    ParsedTree
        The tree.

    Raises
    ------
    ForestPDInputError
        If ``network`` has hybrid edges.
    """
    if not network.is_tree:
        raise ForestPDInputError("Input network is not a tree (it has hybrid edges).")
    root = network.root_node
    index = {root: 0}
    parent: list[int] = [-1]
    length: list[float | None] = [None]
    label: list[str | None] = [None]
    children: dict[object, list[tuple[object, float | None]]] = {}
    for u, v, key in network.tree_edges:
        children.setdefault(u, []).append((v, network.get_branch_length(u, v, key)))
    stack = [root]
    while stack:
        u = stack.pop()
        for v, bl in children.get(u, []):
            index[v] = len(parent)
            parent.append(index[u])
            length.append(bl)
            label.append(None)
            stack.append(v)
    for leaf in network.leaves:
        name = network.get_label(leaf)
        label[index[leaf]] = str(leaf if name is None else name)
    return from_parents(parent, length, label, rooted, hint="R")


def from_phylozoo(
    network: Any,
    registry: TaxonRegistry | None = None,
    missing_branch_length: MissingBranchLength = "error",
) -> PhyloTree:
    """
    Convert a PhyloZoo tree into a :class:`PhyloTree`.

    Parameters
    ----------
    network : SemiDirectedPhyNetwork | DirectedPhyNetwork
        A tree (no hybrid edges); a directed one is read as unrooted.
    registry : TaxonRegistry | None, default None
        Global taxon id mapping; ``None`` builds one from this tree's taxa.
    missing_branch_length : str | float, default "error"
        Policy for edges lacking a branch length: ``"error"``, ``"zero"``, or a numeric value.

    Returns
    -------
    PhyloTree
        The tree, with leaf ids equal to the registry's taxon ids.

    Raises
    ------
    ForestPDInputError
        If ``network`` is not a PhyloZoo network or not a tree, a taxon is missing from
        ``registry``, or a branch length is invalid under the chosen policy.
    """
    return to_phylotree(parsed_from_phylozoo(network), registry, missing_branch_length)


def to_phylozoo(tree: PhyloTree, registry: TaxonRegistry) -> Any:
    """
    Convert a :class:`PhyloTree` into a PhyloZoo ``SemiDirectedPhyNetwork``.

    Parameters
    ----------
    tree : PhyloTree
        The tree, e.g. one of ``instance.trees``.
    registry : TaxonRegistry
        The registry the tree's leaf ids refer to, e.g. ``instance.registry``.

    Returns
    -------
    SemiDirectedPhyNetwork
        The tree as a PhyloZoo network: leaves labelled by taxon, edges carrying
        ``branch_length``.
    """
    return SemiDirectedPhyNetwork(
        undirected_edges=[
            {"u": int(u), "v": int(v), "branch_length": float(w)}
            for (u, v), w in zip(tree.edges, tree.weight)
        ],
        nodes=[(leaf, {"label": registry.label_of(leaf)}) for leaf in range(tree.n_leaves)],
    )


__all__ = ["from_phylozoo", "to_phylozoo", "parsed_from_phylozoo"]
