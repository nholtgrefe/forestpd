"""
Agreement forests from maffe (optional).

`maffe <https://github.com/roehrt/maffe>`_ computes maximum agreement forests. It is a separate
package, not a dependency of forestpd, and not on PyPI yet; it is built from its repository (a
C++23 compiler is needed). This module imports it only when a computer below is created, so the
rest of forestpd works without it.

Two plain adapters, with no policy of their own (deciding when maffe is worth calling, and falling
back when it cannot run, is :class:`~forestpd.maf.TieredMAF`'s job):

* :class:`MaffeRMAF` solves the *rooted* problem (``maffe.solve``). Any rooted agreement forest is
  also an unrooted one, so its forest is always valid. Where to root is not its decision: the
  solver passes the cluster's root leaf (see :class:`~forestpd.maf.base.Rooting`).
* :class:`MaffeUMAF` solves the *unrooted* problem (``maffe.solve_unrooted``, only in maffe's
  ``unrooted`` branch so far). Its forest is never larger than :class:`MaffeRMAF`'s, but it is
  slower, and its running time grows quickly with the number of leaves.

Both need binary trees and raise :class:`~forestpd.exceptions.ForestPDValueError` on a polytomy.
maffe's Python bindings take no time limit, so a hard cluster can take long.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Iterable, Sequence

import numpy as np

from ..core.tree import PhyloTree
from ..exceptions import ForestPDImportError, ForestPDInputError, ForestPDValueError
from ..io.newick import to_newick
from .base import Forest, Rooting, drop_leaf_from_forest

#: How to install maffe. PyPI has wheels for Linux and Windows on x86_64 (Python >= 3.11); on
#: other platforms maffe is built from source, as the installation guide describes.
INSTALL_HINT = (
    "pip install 'forestpd[maffe]'  (Python >= 3.11; on macOS and Linux on ARM, build maffe "
    "from source: https://nholtgrefe.github.io/forestpd/installation.html)"
)


class MaffeRMAF:
    """
    Minimum rooted agreement forest by maffe, rooted where the solver says.

    With a :class:`~forestpd.maf.base.Rooting` every tree is rooted on the pendant edge of the
    cluster's root leaf (``x_C^*``, or leaf 0 at the top cluster) and ``x_C^*`` is removed from the
    result. Without one, the trees are rooted at leaf 0.

    Marked ``trusted``: maffe's forests are agreement forests by construction, so the solver builds
    the profiles from the split tables (fast), with only its partial check. The partition itself
    is always checked here, and :class:`~forestpd.maf.TieredMAF` validates maffe's forest fully
    before using it.

    Parameters
    ----------
    approx_factor : float, default 1.0
        Passed to maffe: accept a forest within this factor of the minimum. ``1.0`` is exact.
    approx_offset : int, default 0
        Passed to maffe: accept a forest within this many components of the minimum.
    log_level : int | str, default 0
        Passed to maffe: ``0``/``"quiet"``, ``1``/``"normal"`` or ``2``/``"verbose"``.

    Attributes
    ----------
    uses_rooting : bool
        ``True``: the solver passes each cluster's :class:`~forestpd.maf.base.Rooting`.
    trusted : bool
        ``True`` (see above).

    Raises
    ------
    ForestPDImportError
        If maffe is not installed.
    """

    uses_rooting = True
    trusted = True

    def __init__(
        self, approx_factor: float = 1.0, approx_offset: int = 0, log_level: int | str = 0
    ) -> None:
        self._maffe = _import_maffe("solve")
        self.approx_factor = approx_factor
        self.approx_offset = approx_offset
        self.log_level = log_level

    def compute(self, trees: Sequence[PhyloTree], rooting: Rooting | None = None) -> Forest:
        """
        Compute a minimum rooted agreement forest of ``trees`` with maffe.

        Parameters
        ----------
        trees : Sequence[PhyloTree]
            The local trees of one cluster, all on leaves ``0..n_leaves-1``. They must be binary.
        rooting : Rooting | None, optional
            Where to root. ``None`` roots ``trees`` at leaf 0.

        Returns
        -------
        Forest
            The components on the leaves of ``trees``, each sorted, ordered by smallest leaf.

        Raises
        ------
        ForestPDValueError
            If a tree is not binary.
        ForestPDInputError
            If maffe's components are not a partition of the leaves.
        """
        if rooting is None:
            rooting = Rooting(trees, 0, extra=False)
        work = rooting.trees
        n_leaves = work[0].n_leaves
        forest = _trivial_forest(work)
        if forest is None:
            newicks = sorted({rooted_newick(tree, rooting.leaf) for tree in work})
            if len(newicks) == 1:
                forest = (tuple(range(n_leaves)),)
            else:
                components = self._maffe.solve(
                    newicks,
                    approx_factor=self.approx_factor,
                    approx_offset=self.approx_offset,
                    log_level=self.log_level,
                )
                forest = _as_forest(components, n_leaves)
        return drop_leaf_from_forest(forest, rooting.leaf) if rooting.extra else forest


class MaffeUMAF:
    """
    Minimum unrooted agreement forest by maffe.

    Uses ``maffe.solve_unrooted``, which so far exists only in maffe's ``unrooted`` branch. Marked
    ``trusted``, as :class:`MaffeRMAF`.

    Parameters
    ----------
    log_level : int | str, default 0
        Passed to maffe: ``0``/``"quiet"``, ``1``/``"normal"`` or ``2``/``"verbose"``.

    Attributes
    ----------
    trusted : bool
        ``True``, as for :class:`MaffeRMAF`.

    Raises
    ------
    ForestPDImportError
        If maffe, or its unrooted solver, is not installed.
    """

    trusted = True

    def __init__(self, log_level: int | str = 0) -> None:
        self._maffe = _import_maffe("solve_unrooted")
        self.log_level = log_level

    def compute(self, trees: Sequence[PhyloTree]) -> Forest:
        """
        Compute a minimum unrooted agreement forest of ``trees`` with maffe.

        Parameters
        ----------
        trees : Sequence[PhyloTree]
            The local trees of one cluster, all on leaves ``0..n_leaves-1``. They must be binary.

        Returns
        -------
        Forest
            The components, each a sorted tuple of leaf ids, ordered by smallest leaf.

        Raises
        ------
        ForestPDValueError
            If a tree is not binary.
        ForestPDInputError
            If maffe's components are not a partition of the leaves.
        """
        n_leaves = trees[0].n_leaves
        forest = _trivial_forest(trees)
        if forest is not None:
            return forest
        labels = [str(leaf + 1) for leaf in range(n_leaves)]
        # maffe identifies equal topologies itself, but sending each once saves the writing.
        newicks = sorted({to_newick(tree, labels, lengths=False) for tree in trees})
        if len(newicks) == 1:  # solve_unrooted needs two trees; one topology agrees with itself
            return (tuple(range(n_leaves)),)
        components = self._maffe.solve_unrooted(newicks, log_level=self.log_level)
        return _as_forest(components, n_leaves)


def _import_maffe(function: str) -> ModuleType:
    """Import maffe and check it has ``function``, or explain how to install it."""
    try:
        module = importlib.import_module("maffe")
    except ImportError as exc:
        raise ForestPDImportError(f"this needs maffe; install it with: {INSTALL_HINT}") from exc
    if not hasattr(module, function):
        raise ForestPDImportError(
            f"the installed maffe has no {function}(); install a version with it: {INSTALL_HINT}"
        )
    return module


def _trivial_forest(trees: Sequence[PhyloTree]) -> Forest | None:
    """
    The one-component forest when the trees cannot disagree, ``None`` if maffe is needed.

    Raises
    ------
    ForestPDValueError
        If maffe would be needed but a tree is not binary.
    """
    n_leaves = trees[0].n_leaves
    if n_leaves <= 3:
        # Every unrooted tree on at most three leaves is the same, so they all agree. (A rooted
        # tree on the pendant edge of one of them is then the same too.)
        return (tuple(range(n_leaves)),) if n_leaves else ()
    for tree in trees:
        if not is_binary(tree):
            raise ForestPDValueError("maffe needs binary trees; this cluster has a polytomy")
    return None


def is_binary(tree: PhyloTree) -> bool:
    """
    Whether every internal node of ``tree`` has degree three.

    Parameters
    ----------
    tree : PhyloTree
        The tree.

    Returns
    -------
    bool
        True if the tree is binary (maffe accepts only binary trees).

    Examples
    --------
    >>> t = PhyloTree.from_edges(5, [(0, 4), (1, 4), (2, 4), (3, 4)], [1.0] * 4,
    ...                          {i: i for i in range(4)})
    >>> is_binary(t)
    False
    """
    return bool(np.all(tree.degrees()[tree.n_leaves :] == 3))


def rooted_newick(tree: PhyloTree, root_leaf: int) -> str:
    """
    Topology of ``tree`` rooted on the pendant edge of ``root_leaf``, leaf ``i`` labelled ``i+1``.

    maffe numbers leaves from 1, as in the PACE format, and needs no branch lengths.

    Parameters
    ----------
    tree : PhyloTree
        The unrooted tree.
    root_leaf : int
        The leaf whose pendant edge carries the root.

    Returns
    -------
    str
        The rooted Newick string, without branch lengths.

    Examples
    --------
    >>> t = PhyloTree.from_edges(6, [(0, 4), (1, 4), (4, 5), (2, 5), (3, 5)], [1.0] * 5,
    ...                          {i: i for i in range(4)})
    >>> rooted_newick(t, 0)
    '((2,(3,4)),1);'
    >>> rooted_newick(t, 3)
    '(((1,2),3),4);'
    """
    rooted = tree.root_at(root_leaf)
    parent = rooted.parent.tolist()
    order = rooted.order.tolist()
    children: list[list[int]] = [[] for _ in range(tree.n_nodes)]
    for v in order[1:]:
        children[parent[v]].append(v)
    # Postorder, so every child's text exists before its parent's (no recursion-depth limit).
    text: list[str] = [""] * tree.n_nodes
    for v in reversed(order[1:]):
        if children[v]:
            text[v] = "(" + ",".join(text[c] for c in children[v]) + ")"
        else:
            text[v] = str(v + 1)
    (top,) = children[root_leaf]
    return f"({text[top]},{root_leaf + 1});"


def _as_forest(components: Iterable[Iterable[Any]], n_leaves: int) -> Forest:
    """
    Turn maffe's 1-based label components into a forest, checking they partition the leaves.

    Examples
    --------
    >>> _as_forest([["3", "1"], ["2"]], 3)
    ((0, 2), (1,))
    """
    forest: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for component in components:
        try:
            leaves = tuple(sorted(int(label) - 1 for label in component))
        except ValueError as exc:
            raise ForestPDInputError(f"maffe returned an unknown leaf label: {exc}") from exc
        if not leaves:
            raise ForestPDInputError("maffe returned an empty component")
        if any(not 0 <= leaf < n_leaves or leaf in seen for leaf in leaves):
            raise ForestPDInputError(f"maffe's components are not a partition: {leaves}")
        seen.update(leaves)
        forest.append(leaves)
    if len(seen) != n_leaves:
        raise ForestPDInputError(f"maffe's components miss {n_leaves - len(seen)} leaves")
    return tuple(sorted(forest))


__all__ = ["MaffeRMAF", "MaffeUMAF"]
