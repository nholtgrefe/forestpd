"""
Protocol for the MAF stage.

A MAF computer receives the local trees of one cluster and returns a partition of their common
leaf set into agreement-forest components. The component topologies ``F_i``, the embedded subtrees
``H_{j,i}`` and the maps ``pi_{j,i}`` are derived afterwards in :mod:`forestpd.maf.embedding`, so
a MAF computer only has to produce the partition.

A computer that solves a *rooted* problem (e.g. :class:`~forestpd.maf.MaffeRMAF`) needs to know
where to root each cluster. The solver can tell it: if the computer has ``uses_rooting = True``,
it is also given a :class:`Rooting`. The root is the cluster's root leaf in the tree of clusters:
``x_C^*`` (the leaf standing for the rest of the trees) in every cluster but the top one, and leaf 0
there. With ``rooted=True``, leaf 0 is the root leaf ``rho``, so every cluster is rooted towards the
input trees' true root, and a minimum rooted forest of a cluster has at most as many components as
a minimum rooted forest of the whole input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from ..core.tree import PhyloTree

#: A forest, as a tuple of components; each component is a tuple of leaf ids.
Forest = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Rooting:
    """
    Where a cluster's trees are rooted, for MAF computers with ``uses_rooting = True``.

    Attributes
    ----------
    trees : Sequence[PhyloTree]
        The cluster's trees containing the root leaf. When ``extra`` is true these are the trees
        passed to :meth:`MAFComputer.compute` plus the leaf ``x_C^*``; otherwise they are the
        same trees.
    leaf : int
        The root leaf: ``x_C^*`` (always the last leaf) or leaf 0 at the top cluster.
    extra : bool
        Whether ``leaf`` is an extra leaf, absent from the trees passed to ``compute``. A forest
        computed on ``trees`` must then have it removed (see :func:`drop_leaf_from_forest`).
    """

    trees: Sequence[PhyloTree]
    leaf: int
    extra: bool


@runtime_checkable
class MAFComputer(Protocol):
    """
    Strategy that partitions a cluster's leaf set into agreement-forest components.

    Any object with a matching :meth:`compute` method satisfies the protocol. An optional
    attribute ``trusted = True`` declares that every forest it returns is an agreement forest by
    construction; the solver then builds the clusters' profiles from their split tables, which is
    faster but checks the forest only partially. Without it, every forest is fully validated.
    An optional attribute ``uses_rooting = True`` makes the solver pass ``rooting=`` (a
    :class:`Rooting`) as well.
    """

    def compute(self, trees: Sequence[PhyloTree]) -> Forest:
        """
        Return an agreement forest of ``trees`` as a partition of leaves ``0..n_leaves-1``.

        The forest must be a valid agreement forest; it need not be minimum.

        Parameters
        ----------
        trees : Sequence[PhyloTree]
            The local trees of one cluster, all on leaves ``0..n_leaves-1``.

        Returns
        -------
        Forest
            The components, each a tuple of leaf ids; together they partition the leaf set.
        """
        ...


def singleton_forest(n_leaves: int) -> Forest:
    """
    The all-singletons forest.

    Parameters
    ----------
    n_leaves : int
        Number of leaves.

    Returns
    -------
    Forest
        ``((0,), (1,), ..., (n_leaves-1,))``.

    Examples
    --------
    >>> singleton_forest(3)
    ((0,), (1,), (2,))
    """
    return tuple((leaf,) for leaf in range(n_leaves))


def drop_leaf_from_forest(forest: Forest, leaf: int) -> Forest:
    """
    Remove the last leaf ``leaf`` from a forest; its component loses it, or vanishes if a singleton.

    Removing a leaf from an agreement forest of some trees gives an agreement forest of the trees
    with that leaf removed, so this turns a forest computed with ``x_C^*`` into one without it.

    Parameters
    ----------
    forest : Forest
        A forest on leaves ``0..leaf``.
    leaf : int
        The leaf to remove; it must be the largest leaf id, so the others keep their ids.

    Returns
    -------
    Forest
        The forest on leaves ``0..leaf-1``, components ordered by smallest leaf.

    Examples
    --------
    >>> drop_leaf_from_forest(((0, 3), (1, 2)), 3)
    ((0,), (1, 2))
    >>> drop_leaf_from_forest(((0, 1), (2,), (3,)), 3)
    ((0, 1), (2,))
    """
    reduced = (tuple(x for x in component if x != leaf) for component in forest)
    return tuple(sorted(component for component in reduced if component))


__all__ = ["MAFComputer", "Forest", "Rooting", "drop_leaf_from_forest", "singleton_forest"]
