"""
Per-cluster precomputation that does not depend on the signature.

The signature loop runs ``2^k`` times, so everything that does not depend on the signature is
computed once here. Inside the loop only a few masked reductions over flat arrays remain. The
module builds three structures:

* Edge tables: every edge of every tree, with the component masks of its two sides
  (``only_below`` / ``only_above``) and its scaled weight ``w_j * len_j(e)``. The definitely-chosen
  test of Algorithm 2 is then one vectorized comparison over this array.
* Component-edge tables: the edges that lie inside some ``H_{j,i}``, each with the ``F_i`` node it
  maps to under ``pi_{j,i}`` and its two sides oriented according to the rooting of ``F_i``. The
  two-sided weights of Definition "two-sided-MAF" are then three ``bincount`` scatters.
* Component DP trees: each ``F_i`` rooted and binarized into the ``R_i`` of Lemma "compute-PD_i",
  with the nodes of all components in one index space so that the weights fit in a single array.

``only_below`` and ``only_above`` never contain the component that an edge lies inside, since that
component has leaves on both sides. The test "some active component lies entirely on each side" is
therefore the same for edges inside and outside components, and Algorithm 2 needs no special case.

A signature sees an edge only through its masks, so in both edge tables the edges with equal masks
(from different trees, mostly) are merged into one row carrying their summed weight. ``d_S`` and the
two-sided weights are the same sums, regrouped; the per-signature work shrinks by the number of
trees sharing each mask pattern.

Survival probabilities enter through the *side constants* ``q`` of an edge: the probability that
every unprotected taxon on one side is lost, ``prod (1 - a(x))`` over all taxa on that side. A side
of a local edge stands for a fixed set of taxa (pseudo-leaves for whole child subtrees, ``x_C^*``
for everything outside the cluster), so ``q`` is a property of the edge's local split, and each row
carries a few products of its weight with its ``q`` values as extra weight columns. Merging rows
sums them like the weight. The signature loop combines them per case (see
:func:`~forestpd.algo.local_solver.signature_weights`); without survival probabilities every ``q``
is ``1`` and the extra columns change nothing. The constants are computed in log space,
``log q = sum log(1 - a(x))``, which is finite because ``a(x) < 1``.

For an edge inside a component ``F_i`` the manuscript splits a side into the taxa of ``X_i`` on it
and the rest (``Y_S`` in Definition "two-sided-MAF"). The rest consists of whole other components,
the ones in the row's side mask, so its constant ``1 - p`` is the product of their constants and is
computed from the masks after merging. The part in ``X_i`` is fixed by the ``F_i`` edge, so its
survival probability ``pi_{X_i}`` is one number per ``R_i`` slot and side
(:attr:`ClusterProfile.slot_survive_down`, :attr:`ClusterProfile.slot_survive_up`), used by the
component DP's ``gamma``.

Edge rows also keep the bit of the component the edge lies inside (``0`` for none): an edge with
active components on one side only pays ``w (1 - q)`` of the other side, unless it lies inside an
active component, whose own rows then account for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..maf.embedding import Embedding, extract_embedding
from ..core.tree import PhyloTree, RootedTree
from ..maf.partition import (
    MAX_COMPONENTS,
    component_edge_masks,
    components_are_vertex_disjoint,
    entirely_above,
    entirely_below,
    split_component_of_edge,
)
from ..exceptions import ForestPDInputError
from ..maf.base import Forest
from . import _profile_kernel
from .local import LocalInstance
from .survival import below_logs, split_logs

#: Build split-based profiles with the compiled kernels (:mod:`._profile_kernel`) when Numba is
#: installed. The NumPy path computes the same profiles; set ``False`` to force it (tests).
USE_COMPILED = True


@dataclass(frozen=True)
class ComponentTree:
    """
    The rooted binary tree ``R_i`` the per-component DP runs on.

    ``F_i`` rooted on the pendant edge of its smallest leaf, with polytomies binarized by
    zero-weight edges. A single-leaf component is its own root and has no edges.

    Attributes
    ----------
    n_nodes : int
        Nodes of ``R_i``.
    left, right : numpy.ndarray
        ``int32[n_nodes]``; children, ``-1`` at leaves.
    leaf : numpy.ndarray
        ``int32[n_nodes]``; local leaf id, ``-1`` at internal nodes.
    postorder : numpy.ndarray
        ``int32[n_nodes]``; children before parents.
    root : int
        Root node of ``R_i``.
    """

    n_nodes: int
    left: np.ndarray
    right: np.ndarray
    leaf: np.ndarray
    postorder: np.ndarray
    root: int


@dataclass(frozen=True)
class ClusterProfile:
    """
    Signature-independent tables for one cluster.

    Attributes
    ----------
    n_components : int
        ``k``.
    comp_leaves : tuple[tuple[int, ...], ...]
        Local leaf ids per component.
    comp_trees : tuple[ComponentTree, ...]
        The ``R_i``.
    node_offset : numpy.ndarray
        ``int64[k+1]``; component ``i``'s ``R_i`` nodes occupy
        ``node_offset[i] .. node_offset[i+1]`` in the shared weight address space.
    edge_only_below, edge_only_above : numpy.ndarray
        ``uint64[E]``; the components lying wholly on each side of an edge, one row per distinct
        pair and owner. A row with one empty side is kept only if it can still pay (its other side's
        survive column is nonzero).
    edge_owner : numpy.ndarray
        ``uint64[E]``; the bit of the component the edges lie inside, ``0`` for none.
    edge_weight : numpy.ndarray
        ``float64[E]``; the summed scaled weights ``w_j * len_j(e)`` of the edges with that pair.
    edge_survive_below, edge_survive_above : numpy.ndarray
        ``float64[E]``; the summed ``w (1 - q)`` of the side below / above: what the edges pay when
        only the other side is covered.
    ce_slot : numpy.ndarray
        ``int64[E_c]``; the shared-address-space node whose parent edge each one subdivides.
    ce_own_bit : numpy.ndarray
        ``uint64[E_c]``; the bit of the component each edge belongs to.
    ce_only_down, ce_only_up : numpy.ndarray
        ``uint64[E_c]``; the two sides, oriented to ``R_i``'s rooting rather than the host tree's.
    ce_weight : numpy.ndarray
        ``float64[E_c]``; summed scaled weights of the component edges with that row's slot and
        masks (one row per distinct combination).
    ce_dies_down, ce_dies_up, ce_dies_both : numpy.ndarray
        ``float64[E_c]``; the summed ``w q_down``, ``w q_up`` and ``w q_down q_up``, where ``q`` is
        the side constant of the taxa on that side *outside* the edge's own component: the
        manuscript's ``1 - p`` in Definition "two-sided-MAF".
    slot_survive_down, slot_survive_up : numpy.ndarray
        ``float64[n_slots]``; per ``R_i`` node, the probability ``pi_{X_i}`` that an unselected
        taxon of its own component survives below its parent edge / on the other side of it.
    """

    n_components: int
    comp_leaves: tuple[tuple[int, ...], ...]
    comp_trees: tuple[ComponentTree, ...]
    node_offset: np.ndarray
    edge_only_below: np.ndarray
    edge_only_above: np.ndarray
    edge_weight: np.ndarray
    ce_slot: np.ndarray
    ce_own_bit: np.ndarray
    ce_only_down: np.ndarray
    ce_only_up: np.ndarray
    ce_weight: np.ndarray
    edge_owner: np.ndarray
    edge_survive_below: np.ndarray
    edge_survive_above: np.ndarray
    ce_dies_down: np.ndarray
    ce_dies_up: np.ndarray
    ce_dies_both: np.ndarray
    slot_survive_down: np.ndarray
    slot_survive_up: np.ndarray

    @property
    def n_slots(self) -> int:
        """Size of the shared weight address space across all ``R_i``."""
        return int(self.node_offset[-1])


def build_profile(local: LocalInstance, forest: Forest) -> ClusterProfile:
    """
    Precompute every signature-independent table for one cluster.

    Parameters
    ----------
    local : LocalInstance
        The cluster's problem.
    forest : Forest
        The agreement forest, as a partition of the local leaves.

    Returns
    -------
    ClusterProfile
        The flattened edge, component-edge and ``R_i`` tables.

    Raises
    ------
    ForestPDInputError
        If the forest has more than :data:`MAX_COMPONENTS` components or is not an agreement
        forest of the local trees.
    """
    k = len(forest)
    if k > MAX_COMPONENTS:
        raise ForestPDInputError(
            f"agreement forest has k={k} components; at most {MAX_COMPONENTS} are supported "
            "(and 2^k signatures would be intractable long before that)"
        )
    n_leaves = local.n_leaves
    comp_of_leaf = np.empty(n_leaves, dtype=np.int64)
    for index, block in enumerate(forest):
        for leaf in block:
            comp_of_leaf[leaf] = index

    # The per-edge component masks are the input to *everything* below, so they are computed once
    # per tree here and threaded through rather than re-derived by each consumer.
    scans = [_scan_tree(tree, comp_of_leaf, k) for tree in local.trees]
    embeddings, comp_trees, node_offset = _build_component_trees(local, forest, scans)

    leaf_log = local.leaf_log_or_zero
    total = float(leaf_log.sum())
    edge_below_l: list[np.ndarray] = []
    edge_above_l: list[np.ndarray] = []
    edge_owner_l: list[np.ndarray] = []
    edge_cols_l: list[np.ndarray] = []
    ce_slot_l: list[np.ndarray] = []
    ce_own_l: list[np.ndarray] = []
    ce_down_l: list[np.ndarray] = []
    ce_up_l: list[np.ndarray] = []
    ce_cols_l: list[np.ndarray] = []

    for j, tree in enumerate(local.trees):
        rooted, below, above, owner = scans[j]
        only_below = entirely_below(below, above)
        only_above = entirely_above(below, above)
        weight = local.tree_weight[j] * tree.weight
        # Log side constants: the leaves below each edge, and the rest.
        log_below = below_logs(rooted, leaf_log)
        log_above = total - log_below
        edge_below_l.append(only_below)
        edge_above_l.append(only_above)
        edge_owner_l.append(_owner_bits(owner))
        edge_cols_l.append(_survive_columns(weight, log_below, log_above))

        idx = np.nonzero(owner >= 0)[0]
        if idx.size:
            comps = owner[idx].astype(np.int64)
            # +1: R_i's virtual root shifts every F_i node up by one.
            fnode = np.array(
                [embeddings[j][c].edge_fnode[e] for c, e in zip(comps.tolist(), idx.tolist())],
                dtype=np.int64,
            )
            slots = node_offset[comps] + fnode + 1
            own = np.left_shift(np.uint64(1), comps.astype(np.uint64))
            # R_i is rooted away from the component's smallest leaf, the host tree away from
            # node 0; the two "below" sides agree unless that leaf sits below the edge.
            anchor_of = np.array([int(embeddings[j][c].leaf[0]) for c in range(k)], dtype=np.int64)
            anchors = anchor_of[comps]
            ends = tree.edges[idx].astype(np.int64)
            lower = np.where(rooted.parent[ends[:, 1]] == ends[:, 0], ends[:, 1], ends[:, 0])
            flip = (rooted.tin[lower] <= rooted.tin[anchors]) & (
                rooted.tout[anchors] <= rooted.tout[lower]
            )
            ce_slot_l.append(slots)
            ce_own_l.append(own)
            ce_down_l.append(np.where(flip, only_above[idx], only_below[idx]))
            ce_up_l.append(np.where(flip, only_below[idx], only_above[idx]))
            ce_cols_l.append(weight[idx][:, None])

    # A signature sees an edge only through its masks, so edges with equal masks are merged by
    # summing their weights: the same d_S and lambda sums, regrouped. Across trees most edges
    # repeat, so the per-signature loops get far shorter. An edge that can never pay is dropped.
    # With one local tree there is nothing to merge; skipping it keeps small clusters cheap.
    merge = _merge_rows if len(local.trees) > 1 else _keep_rows
    edge_keys, edge_cols = merge(
        [
            _concat(edge_below_l, np.uint64),
            _concat(edge_above_l, np.uint64),
            _concat(edge_owner_l, np.uint64),
        ],
        _concat_columns(edge_cols_l, 3),
    )
    ce_keys, ce_cols = merge(
        [
            _concat(ce_slot_l, np.int64).astype(np.uint64),
            _concat(ce_own_l, np.uint64),
            _concat(ce_down_l, np.uint64),
            _concat(ce_up_l, np.uint64),
        ],
        _concat_columns(ce_cols_l, 1),
    )
    return _assemble(
        k,
        forest,
        comp_trees,
        node_offset,
        edge_keys,
        edge_cols,
        ce_keys,
        ce_cols,
        leaf_log=leaf_log,
    )


def build_profile_from_splits(
    n_leaves: int,
    split_masks: np.ndarray,
    split_weights: np.ndarray,
    forest: Forest,
    leaf_log: np.ndarray | None = None,
) -> ClusterProfile:
    """
    :func:`build_profile` from the cluster's weighted local split table instead of its trees.

    Every row of a profile depends on an edge only through its split of the local leaves: which
    components lie wholly on each side (the definitely-chosen test of Algorithm 2), which component
    has leaves on both sides (the edge then lies inside its ``H_{j,i}``), and, for that component,
    its leaves on the side away from its smallest leaf, which is the ``F_i`` edge ``pi_{j,i}`` maps
    the edge to. So the profile is the same sums as :func:`build_profile`'s, grouped by split, and
    ``F_i`` itself is read off the splits restricted to ``X_i``. No tree is visited.

    Unlike :func:`build_profile`, this cannot check vertex-disjointness, so ``forest`` must be an
    agreement forest of the trees the splits come from; only a cheap partial check is made.

    Parameters
    ----------
    n_leaves : int
        Number of local leaves.
    split_masks : numpy.ndarray
        ``uint64``; the distinct local splits, each as the mask of one side
        (:attr:`~forestpd.cluster.base.LocalPiece.split_masks`).
    split_weights : numpy.ndarray
        ``float64``; the summed scaled weight ``sum_j w_j * len_j(e)`` per split.
    forest : Forest
        An agreement forest, as a partition of the local leaves.
    leaf_log : numpy.ndarray | None, optional
        ``float64[n_leaves]``; the summed ``log(1 - a(x))`` per local leaf (see
        :attr:`~forestpd.algo.local.LocalInstance.leaf_log`). ``None`` means all zero.

    Returns
    -------
    ClusterProfile
        Equal to :func:`build_profile`'s up to row order, the orientation of the two edge masks
        (the definitely-chosen test is symmetric in them) and rows of weight zero.

    Raises
    ------
    ForestPDInputError
        If the forest has more than :data:`MAX_COMPONENTS` components, or the partial check fails:
        a split with leaves of two components on both sides, or a component whose restricted
        splits do not nest.
    """
    k = len(forest)
    if k > MAX_COMPONENTS:
        raise ForestPDInputError(
            f"agreement forest has k={k} components; at most {MAX_COMPONENTS} are supported "
            "(and 2^k signatures would be intractable long before that)"
        )
    full = np.uint64((1 << n_leaves) - 1)
    masks = split_masks.astype(np.uint64, copy=False)
    weights = split_weights.astype(np.float64, copy=False)
    log_a, log_b = split_logs(masks, n_leaves, leaf_log)
    comp_mask = np.array([sum(1 << leaf for leaf in block) for block in forest], dtype=np.uint64)
    smallest = np.array([min(block) for block in forest], dtype=np.uint64)

    # Per split and component: does the component have leaves on side A (the mask) / side B?
    in_a = (masks[:, None] & comp_mask[None, :]) != 0
    in_b = ((~masks & full)[:, None] & comp_mask[None, :]) != 0
    bits = np.left_shift(np.uint64(1), np.arange(k, dtype=np.uint64))
    zero = np.uint64(0)
    only_a = np.bitwise_or.reduce(np.where(in_a & ~in_b, bits, zero), axis=1)
    only_b = np.bitwise_or.reduce(np.where(in_b & ~in_a, bits, zero), axis=1)
    inside = in_a & in_b
    n_inside = inside.sum(axis=1)
    if np.any(n_inside > 1):
        raise ForestPDInputError(
            "an edge lies inside two MAF components; the supplied forest is not an agreement forest"
        )

    # Edges inside a component: the component, and its leaves on the side away from its smallest
    # leaf, which name the F_i edge (by its lower endpoint) that the edge subdivides.
    rows = np.flatnonzero(n_inside == 1)
    comps = np.argmax(inside[rows], axis=1)
    away_is_a = (masks[rows] & np.left_shift(np.uint64(1), smallest[comps])) == 0
    away = np.where(away_is_a, masks[rows], ~masks[rows] & full) & comp_mask[comps]

    comp_trees: list[ComponentTree] = []
    fnode = np.empty(rows.size, dtype=np.int64)
    for comp in range(k):
        sel = np.flatnonzero(comps == comp)
        clusters = np.unique(away[sel])
        f_children, f_leaf, f_of = _f_from_clusters(int(smallest[comp]), clusters, comp)
        comp_trees.append(_r_tree(f_children, f_leaf))
        fnode[sel] = f_of[np.searchsorted(clusters, away[sel])]
    node_offset = np.zeros(k + 1, dtype=np.int64)
    np.cumsum([tree.n_nodes for tree in comp_trees], out=node_offset[1:])
    # +1: R_i's virtual root shifts every F_i node up by one.
    slots = node_offset[comps] + fnode + 1
    down = np.where(away_is_a, only_a[rows], only_b[rows])
    up = np.where(away_is_a, only_b[rows], only_a[rows])
    own = np.left_shift(np.uint64(1), comps.astype(np.uint64))

    # The definitely-chosen test is symmetric in the two sides, so each pair is stored in a fixed
    # order (the survive columns swapped along) and rows that differ only in orientation merge.
    swap = only_a > only_b
    survive_a, survive_b = _survive_columns(weights, log_a, log_b)[:, 1:].T
    owner_bits = np.zeros(masks.size, dtype=np.uint64)
    owner_bits[rows] = own
    edge_keys, edge_cols = _merge_rows(
        [np.minimum(only_a, only_b), np.maximum(only_a, only_b), owner_bits],
        np.column_stack(
            [weights, np.where(swap, survive_b, survive_a), np.where(swap, survive_a, survive_b)]
        ),
    )
    ce_keys, ce_cols = _merge_rows([slots.astype(np.uint64), own, down, up], weights[rows][:, None])
    leaf_log = np.zeros(n_leaves) if leaf_log is None else leaf_log
    return _assemble(
        k,
        forest,
        comp_trees,
        node_offset,
        edge_keys,
        edge_cols,
        ce_keys,
        ce_cols,
        leaf_log=leaf_log,
    )


def build_profiles_from_splits(
    n_leaves: int,
    split_masks: np.ndarray,
    split_weights: np.ndarray,
    forest: Forest,
    star_component: int | None = None,
    leaf_log: np.ndarray | None = None,
) -> tuple[ClusterProfile, ClusterProfile | None]:
    """
    :func:`build_profile_from_splits`, and the plain profile with it, compiled when possible.

    Parameters
    ----------
    n_leaves : int
        Number of local leaves.
    split_masks, split_weights : numpy.ndarray
        The split table, as for :func:`build_profile_from_splits`.
    forest : Forest
        An agreement forest of the local instance.
    star_component : int | None, default None
        Index of the ``{x_C^*}`` component (the last), to also derive the plain profile of
        ``I_C`` as :func:`plain_profile_from_plus` would.
    leaf_log : numpy.ndarray | None, optional
        As for :func:`build_profile_from_splits`.

    Returns
    -------
    tuple[ClusterProfile, ClusterProfile | None]
        The profile for ``forest``, and the plain profile (``None`` without ``star_component``).

    Raises
    ------
    ForestPDInputError
        As :func:`build_profile_from_splits`.
    """
    if not (USE_COMPILED and _profile_kernel.HAVE_NUMBA and n_leaves <= 64):
        profile = build_profile_from_splits(n_leaves, split_masks, split_weights, forest, leaf_log)
        plain = None if star_component is None else plain_profile_from_plus(profile, star_component)
        return profile, plain

    k = len(forest)
    if k > MAX_COMPONENTS:
        raise ForestPDInputError(
            f"agreement forest has k={k} components; at most {MAX_COMPONENTS} are supported "
            "(and 2^k signatures would be intractable long before that)"
        )
    comp_mask = np.array([sum(1 << leaf for leaf in block) for block in forest], dtype=np.uint64)
    smallest = np.array([min(block) for block in forest], dtype=np.int64)
    leaf_log = np.zeros(n_leaves) if leaf_log is None else leaf_log
    ok, edge_keys, edge_weight, comp_keys, comp_weight = _profile_kernel.split_rows(
        split_masks.astype(np.uint64, copy=False),
        split_weights.astype(np.float64, copy=False),
        comp_mask,
        smallest,
        np.uint64((1 << n_leaves) - 1),
        np.ascontiguousarray(leaf_log, dtype=np.float64),
        float(leaf_log.sum()),
    )
    if not ok:
        raise ForestPDInputError(
            "an edge lies inside two MAF components; the supplied forest is not an agreement forest"
        )
    edge_keys, edge_weight = _profile_kernel.merge_rows(edge_keys, edge_weight)
    comp_keys, comp_weight = _profile_kernel.merge_rows(comp_keys, comp_weight)
    comp_leaves = tuple(tuple(sorted(block)) for block in forest)

    # F_i from the distinct (component, away) pairs, which are contiguous and sorted. The kernels
    # return column-major arrays, so every column below is a contiguous view.
    owner = comp_keys[0].astype(np.int64)
    away = comp_keys[1]
    bounds = np.searchsorted(owner, np.arange(k + 1))
    comp_trees: list[ComponentTree] = []
    clusters_of: list[np.ndarray] = []
    f_of_list: list[np.ndarray] = []
    for comp in range(k):
        clusters = np.unique(away[bounds[comp] : bounds[comp + 1]])
        f_children, f_leaf, f_of = _f_from_clusters(int(smallest[comp]), clusters, comp)
        comp_trees.append(_r_tree(f_children, f_leaf))
        clusters_of.append(clusters)
        f_of_list.append(f_of)
    node_offset = np.zeros(k + 1, dtype=np.int64)
    np.cumsum([tree.n_nodes for tree in comp_trees], out=node_offset[1:])
    survive_down, survive_up = _slot_survival(comp_trees, comp_leaves, node_offset, leaf_log)

    def slots_of(keys: np.ndarray) -> np.ndarray:
        """``R_i`` slot of each (component, away) row; +1 for ``R_i``'s virtual root."""
        own = keys[0].astype(np.int64)
        out = np.empty(keys.shape[1], dtype=np.int64)
        edges = np.searchsorted(own, np.arange(k + 1))
        for comp in range(k):
            part = slice(edges[comp], edges[comp + 1])
            index = np.searchsorted(clusters_of[comp], keys[1, part])
            out[part] = node_offset[comp] + f_of_list[comp][index] + 1
        return out

    def assemble(
        n_comp: int,
        e_keys: np.ndarray,
        e_cols: np.ndarray,
        c_keys: np.ndarray,
        c_cols: np.ndarray,
    ) -> ClusterProfile:
        n_slots = int(node_offset[n_comp])
        return ClusterProfile(
            n_components=n_comp,
            comp_leaves=comp_leaves[:n_comp],
            comp_trees=tuple(comp_trees[:n_comp]),
            node_offset=node_offset[: n_comp + 1],
            edge_only_below=e_keys[0],
            edge_only_above=e_keys[1],
            edge_weight=e_cols[0],
            ce_slot=slots_of(c_keys),
            ce_own_bit=np.left_shift(np.uint64(1), c_keys[0]),
            ce_only_down=c_keys[2],
            ce_only_up=c_keys[3],
            ce_weight=c_cols[0],
            edge_owner=e_keys[2],
            edge_survive_below=e_cols[1],
            edge_survive_above=e_cols[2],
            ce_dies_down=c_cols[1],
            ce_dies_up=c_cols[2],
            ce_dies_both=c_cols[3],
            slot_survive_down=survive_down[:n_slots],
            slot_survive_up=survive_up[:n_slots],
        )

    profile = assemble(k, edge_keys, edge_weight, comp_keys, comp_weight)
    if star_component is None:
        return profile, None
    assert star_component == k - 1, "x_C^* must be the last component"
    plain_keys = _profile_kernel.plain_rows(
        edge_keys, edge_weight, comp_keys, comp_weight, np.uint64(1) << np.uint64(star_component)
    )
    return profile, assemble(star_component, *plain_keys)


def _f_from_clusters(
    root_leaf: int, clusters: np.ndarray, comp: int
) -> tuple[list[list[int]], list[int], np.ndarray]:
    """
    ``F_i`` under the canonical numbering of :func:`~forestpd.maf.embedding.extract_embedding`.

    ``F_i`` rooted at its smallest leaf has one non-root node per cluster (its leaf set below);
    the parent of a cluster is the smallest cluster strictly containing it. Nodes are numbered in
    preorder, children ordered by their smallest leaf.

    Parameters
    ----------
    root_leaf : int
        The component's smallest leaf, ``F_i``'s root.
    clusters : numpy.ndarray
        ``uint64``, sorted and distinct; the leaf sets below the non-root nodes.
    comp : int
        Component index (for the error message).

    Returns
    -------
    tuple[list[list[int]], list[int], numpy.ndarray]
        Children per ``F_i`` node; leaf id per node (``-1`` if internal); and the ``F_i`` node of
        each cluster, aligned with ``clusters``.

    Raises
    ------
    ForestPDInputError
        If the clusters do not nest (the forest is then not an agreement forest).
    """
    sets = [int(c) for c in clusters.tolist()]
    if len(sets) > 1:
        inter = clusters[:, None] & clusters[None, :]
        nested = (inter == 0) | (inter == clusters[:, None]) | (inter == clusters[None, :])
        if not nested.all():
            raise ForestPDInputError(
                f"MAF component {comp} has different topologies in different trees; "
                "the supplied forest is not an agreement forest"
            )
    # Larger sets first, so every set's ancestors come before it; its parent is the last one seen
    # that contains it (ancestors form a chain, smallest last).
    by_size = sorted(range(len(sets)), key=lambda i: -sets[i].bit_count())
    kids: dict[int, list[int]] = {-1: []}
    chain: list[int] = []
    for i in by_size:
        kids[i] = []
        parent = -1
        for j in reversed(chain):
            if sets[j] & sets[i] == sets[i]:
                parent = j
                break
        kids[parent].append(i)
        chain.append(i)

    def low(i: int) -> int:
        return (sets[i] & -sets[i]).bit_length() - 1

    f_children: list[list[int]] = [[]]
    f_leaf = [root_leaf]
    f_of = np.empty(len(sets), dtype=np.int64)
    stack = [(i, 0) for i in sorted(kids[-1], key=low, reverse=True)]
    while stack:
        i, up = stack.pop()
        node = len(f_leaf)
        f_of[i] = node
        f_children[up].append(node)
        f_children.append([])
        f_leaf.append(low(i) if sets[i] & (sets[i] - 1) == 0 else -1)
        stack.extend((c, node) for c in sorted(kids[i], key=low, reverse=True))
    return f_children, f_leaf, f_of


def plain_profile_from_plus(plus: ClusterProfile, star_component: int) -> ClusterProfile:
    """
    The profile of ``I_C`` derived from that of ``I_C^+``, without scanning the trees again.

    ``I_C^+`` is ``I_C`` with the leaf ``x_C^*`` hanging off one edge, and its forest is ``I_C``'s
    with ``{x_C^*}`` appended as the last component. Removing that leaf leaves every other
    component, its ``R_i`` and its slots unchanged; an edge of ``I_C`` is either an edge of
    ``I_C^+`` or the two edges ``x_C^*`` subdivides, which differ only by the ``x_C^*`` bit. So
    clearing that bit from every mask and merging equal rows gives ``I_C``'s tables exactly: the two
    halves of a subdivided edge sum to its weight, and the pendant edge of ``x_C^*`` drops out (one
    side becomes empty). The side constants stay those of ``I_C^+``'s edges, which is right: the
    taxa outside the cluster are not selected in ``I_C``, but they may still survive.

    Parameters
    ----------
    plus : ClusterProfile
        The profile of ``I_C^+``.
    star_component : int
        Index of the ``{x_C^*}`` component; it must be the last.

    Returns
    -------
    ClusterProfile
        The profile of ``I_C``, as :func:`build_profile` would compute it up to the order of
        summation.
    """
    assert star_component == plus.n_components - 1, "x_C^* must be the last component"
    keep = ~(np.uint64(1) << np.uint64(star_component))
    edge_keys, edge_cols = _merge_rows(
        [plus.edge_only_below & keep, plus.edge_only_above & keep, plus.edge_owner],
        np.column_stack([plus.edge_weight, plus.edge_survive_below, plus.edge_survive_above]),
    )
    ce_keys, ce_cols = _merge_rows(
        [
            plus.ce_slot.astype(np.uint64),
            plus.ce_own_bit,
            plus.ce_only_down & keep,
            plus.ce_only_up & keep,
        ],
        np.column_stack([plus.ce_weight, plus.ce_dies_down, plus.ce_dies_up, plus.ce_dies_both]),
    )
    n_slots = int(plus.node_offset[star_component])
    return _assemble(
        star_component,
        None,
        plus.comp_trees[:star_component],
        plus.node_offset[: star_component + 1],
        edge_keys,
        edge_cols,
        ce_keys,
        ce_cols,
        comp_leaves=plus.comp_leaves[:star_component],
        slot_survive=(plus.slot_survive_down[:n_slots], plus.slot_survive_up[:n_slots]),
    )


def _keep_rows(
    columns: list[np.ndarray], weights: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray]:
    """The rows unmerged (the signature of :func:`_merge_rows`)."""
    return columns, weights


def _merge_rows(
    columns: list[np.ndarray], weights: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Merge the rows that agree in every key column, summing their weight columns.

    Parameters
    ----------
    columns : list[numpy.ndarray]
        Equal-length ``uint64`` key columns.
    weights : numpy.ndarray
        ``float64[n, m]``; the weight columns per row.

    Returns
    -------
    tuple[list[numpy.ndarray], numpy.ndarray]
        The distinct key rows, as columns, and their summed weight columns.
    """
    if weights.shape[0] < 2:
        return columns, weights
    order = np.lexsort(columns[::-1])
    sorted_columns = [column[order] for column in columns]
    starts = np.ones(weights.shape[0], dtype=bool)
    starts[1:] = np.any([np.diff(column) != 0 for column in sorted_columns], axis=0)
    first = np.flatnonzero(starts)
    summed = np.add.reduceat(weights[order], first, axis=0)
    return [column[first] for column in sorted_columns], summed


def _survive_columns(weight: np.ndarray, log_a: np.ndarray, log_b: np.ndarray) -> np.ndarray:
    """
    Edge-row weight columns ``(w, w (1 - q_a), w (1 - q_b))`` from the log side constants.

    Examples
    --------
    >>> _survive_columns(np.array([2.0]), np.array([np.log(0.25)]), np.array([0.0]))
    array([[2. , 1.5, 0. ]])
    """
    return np.column_stack(
        [weight, weight * (0.0 - np.expm1(log_a)), weight * (0.0 - np.expm1(log_b))]
    )


def _component_logs(comp_leaves: Sequence[Sequence[int]], leaf_log: np.ndarray) -> np.ndarray:
    """``float64[k]``; the summed ``log(1 - a(x))`` of each component's leaves."""
    return np.array([float(leaf_log[list(block)].sum()) for block in comp_leaves])


def _dies_columns(
    weight: np.ndarray, down: np.ndarray, up: np.ndarray, comp_log: np.ndarray
) -> np.ndarray:
    """
    Component-row weight columns ``(w, w q_down, w q_up, w q_down q_up)`` from the side masks.

    A side's taxa outside the row's own component are the components in its mask, so its constant
    is the product of theirs (a sum of logs, never a quotient of products, which could underflow).

    Examples
    --------
    >>> _dies_columns(
    ...     np.array([2.0]), np.array([1], dtype=np.uint64), np.array([0], dtype=np.uint64),
    ...     np.array([np.log(0.5)]),
    ... )
    array([[2., 1., 2., 1.]])
    """
    log_down = np.zeros(weight.size)
    log_up = np.zeros(weight.size)
    for comp, value in enumerate(comp_log.tolist()):
        if value != 0.0:
            bit = np.uint64(1) << np.uint64(comp)
            log_down += np.where((down & bit) != 0, value, 0.0)
            log_up += np.where((up & bit) != 0, value, 0.0)
    q_down = np.exp(log_down)
    q_up = np.exp(log_up)
    return np.column_stack([weight, weight * q_down, weight * q_up, weight * q_down * q_up])


def _slot_survival(
    comp_trees: Sequence[ComponentTree],
    comp_leaves: Sequence[Sequence[int]],
    node_offset: np.ndarray,
    leaf_log: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per ``R_i`` slot, ``pi_{X_i}`` below its parent edge and on the other side of it.

    Parameters
    ----------
    comp_trees : Sequence[ComponentTree]
        The ``R_i``.
    comp_leaves : Sequence[Sequence[int]]
        Local leaf ids per component.
    node_offset : numpy.ndarray
        ``int64[k+1]``; the shared slot offsets.
    leaf_log : numpy.ndarray
        ``float64[n_leaves]``; the summed ``log(1 - a(x))`` per local leaf.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``float64[n_slots]`` twice: the probability that an unselected leaf of the component
        survives below / above each node (``1 - prod (1 - a)``; zero without survival).
    """
    n_slots = int(node_offset[-1])
    below = np.zeros(n_slots)
    above = np.zeros(n_slots)
    if not leaf_log.any():  # every constant is 1, every survival probability 0
        return below, above
    for comp, tree in enumerate(comp_trees):
        offset = int(node_offset[comp])
        logs = np.zeros(tree.n_nodes)
        for node in tree.postorder.tolist():
            leaf = int(tree.leaf[node])
            if leaf >= 0:
                logs[node] = leaf_log[leaf]
            else:
                logs[node] = logs[int(tree.left[node])] + logs[int(tree.right[node])]
        total = float(leaf_log[list(comp_leaves[comp])].sum())
        below[offset : offset + tree.n_nodes] = logs
        above[offset : offset + tree.n_nodes] = total - logs
    return -np.expm1(below), -np.expm1(above)


def _useful_edges(
    below: np.ndarray, above: np.ndarray, survive_below: np.ndarray, survive_above: np.ndarray
) -> np.ndarray:
    """
    The edge rows that can pay under some signature.

    Both sides nonempty, or one side nonempty and the other side's survive column nonzero (the edge
    then pays when only the nonempty side is covered).
    """
    useful: np.ndarray = ((below != 0) & (above != 0)) | (
        ((below == 0) & (above != 0) & (survive_below != 0))
        | ((above == 0) & (below != 0) & (survive_above != 0))
    )
    return useful


def _assemble(
    k: int,
    forest: Forest | None,
    comp_trees: Sequence[ComponentTree],
    node_offset: np.ndarray,
    edge_keys: list[np.ndarray],
    edge_cols: np.ndarray,
    ce_keys: list[np.ndarray],
    ce_cols: np.ndarray,
    comp_leaves: tuple[tuple[int, ...], ...] | None = None,
    leaf_log: np.ndarray | None = None,
    slot_survive: tuple[np.ndarray, np.ndarray] | None = None,
) -> ClusterProfile:
    """
    A :class:`ClusterProfile` from merged rows, dropping edge rows that can never pay.

    Parameters
    ----------
    k : int
        Number of components.
    forest : Forest | None
        The agreement forest, for ``comp_leaves`` (or pass ``comp_leaves``).
    comp_trees : Sequence[ComponentTree]
        The ``R_i``.
    node_offset : numpy.ndarray
        ``int64[k+1]``; the shared slot offsets.
    edge_keys : list[numpy.ndarray]
        ``(only_below, only_above, owner)``.
    edge_cols : numpy.ndarray
        ``float64[E, 3]``; weight and the two survive columns.
    ce_keys : list[numpy.ndarray]
        ``(slot, own, down, up)``.
    ce_cols : numpy.ndarray
        ``float64[E_c, 4]``; weight and the three dies columns, or ``float64[E_c, 1]``, the
        weight only, for the dies columns to be computed here from the masks and ``leaf_log``.
    comp_leaves : tuple[tuple[int, ...], ...] | None, optional
        The components' leaves, when ``forest`` is not given.
    leaf_log : numpy.ndarray | None, optional
        ``float64[n_leaves]``; the summed ``log(1 - a(x))`` per local leaf, for the dies columns
        and the slot survival. Required unless both are given.
    slot_survive : tuple[numpy.ndarray, numpy.ndarray] | None, optional
        :attr:`ClusterProfile.slot_survive_down` and ``slot_survive_up``, if already known.

    Returns
    -------
    ClusterProfile
        The profile.
    """
    below, above, owner = edge_keys
    useful = _useful_edges(below, above, edge_cols[:, 1], edge_cols[:, 2])
    slot, own, down, up = ce_keys
    if comp_leaves is None:
        assert forest is not None
        comp_leaves = tuple(tuple(sorted(block)) for block in forest)
    if ce_cols.shape[1] == 1:
        assert leaf_log is not None
        ce_cols = _dies_columns(ce_cols[:, 0], down, up, _component_logs(comp_leaves, leaf_log))
    if slot_survive is None:
        assert leaf_log is not None
        slot_survive = _slot_survival(comp_trees, comp_leaves, node_offset, leaf_log)
    return ClusterProfile(
        n_components=k,
        comp_leaves=comp_leaves,
        comp_trees=tuple(comp_trees),
        node_offset=node_offset,
        edge_only_below=below[useful],
        edge_only_above=above[useful],
        edge_weight=edge_cols[useful, 0],
        ce_slot=slot.astype(np.int64),
        ce_own_bit=own,
        ce_only_down=down,
        ce_only_up=up,
        ce_weight=ce_cols[:, 0].copy(),
        edge_owner=owner[useful],
        edge_survive_below=edge_cols[useful, 1],
        edge_survive_above=edge_cols[useful, 2],
        ce_dies_down=ce_cols[:, 1].copy(),
        ce_dies_up=ce_cols[:, 2].copy(),
        ce_dies_both=ce_cols[:, 3].copy(),
        slot_survive_down=slot_survive[0],
        slot_survive_up=slot_survive[1],
    )


def _owner_bits(owner: np.ndarray) -> np.ndarray:
    """``uint64`` bit of each edge's owning component, ``0`` where it has none."""
    bits = np.left_shift(np.uint64(1), np.maximum(owner, 0).astype(np.uint64))
    out: np.ndarray = np.where(owner >= 0, bits, np.uint64(0)).astype(np.uint64)
    return out


#: One tree's precomputed geometry: its rooting, both component masks, and the owner array.
TreeScan = tuple[RootedTree, np.ndarray, np.ndarray, np.ndarray]


def _scan_tree(tree: PhyloTree, comp_of_leaf: np.ndarray, k: int) -> TreeScan:
    """
    One pass over a tree: its rooting, the per-edge component masks, and the owner array.

    Also checks that the components are vertex-disjoint in this tree.

    Parameters
    ----------
    tree : PhyloTree
        One local tree.
    comp_of_leaf : numpy.ndarray
        Leaf -> component index.
    k : int
        Number of components.

    Returns
    -------
    TreeScan
        ``(rooted, below, above, owner)``: the rooting at node 0, the per-edge component masks of
        :func:`~forestpd.maf.partition.component_edge_masks`, and the per-edge owning component.

    Raises
    ------
    ForestPDInputError
        If an edge lies inside two components, or two components share a vertex.
    """
    rooted = tree.root_at(0)
    below, above = component_edge_masks(rooted, comp_of_leaf, k)
    owner = split_component_of_edge(below, above)
    if not components_are_vertex_disjoint(tree, owner):
        raise ForestPDInputError(
            "two MAF components share a vertex; the supplied forest is not an agreement forest"
        )
    return rooted, below, above, owner


def _concat_columns(parts: list[np.ndarray], m: int) -> np.ndarray:
    """Stack ``float64[n_i, m]`` blocks, tolerating an empty list."""
    if not parts:
        return np.empty((0, m), dtype=np.float64)
    return np.concatenate(parts, axis=0)


def _concat(parts: list[np.ndarray], dtype: type | np.dtype) -> np.ndarray:
    """
    Concatenate, tolerating an empty list.

    Parameters
    ----------
    parts : list[numpy.ndarray]
        Arrays to join.
    dtype : numpy.dtype or type
        Result dtype.

    Returns
    -------
    numpy.ndarray
        The concatenation cast to ``dtype``; empty if ``parts`` is.
    """
    if not parts:
        return np.empty(0, dtype=dtype)
    joined: np.ndarray = np.concatenate(parts).astype(dtype, copy=False)
    return joined


def _build_component_trees(
    local: LocalInstance,
    forest: Forest,
    scans: list[TreeScan],
) -> tuple[list[dict[int, Embedding]], list[ComponentTree], np.ndarray]:
    """
    Extract each component's embedding in every tree and build the shared ``R_i`` structures.

    Parameters
    ----------
    local : LocalInstance
        The cluster's problem.
    forest : Forest
        The agreement forest.
    scans : list[TreeScan]
        Per local tree, the output of :func:`_scan_tree`.

    Returns
    -------
    tuple[list[dict[int, Embedding]], list[ComponentTree], numpy.ndarray]
        Per tree, component -> embedding; the ``R_i`` per component (built from the first tree's
        embeddings); and ``int64[k+1]`` node offsets into the shared address space.

    Raises
    ------
    ForestPDInputError
        If some component has different topologies in different trees.
    """
    k = len(forest)
    per_tree: list[dict[int, Embedding]] = []
    for tree, (_, _, _, owner) in zip(local.trees, scans):
        by_comp: dict[int, list[int]] = {comp: [] for comp in range(k)}
        for e in np.nonzero(owner >= 0)[0]:
            by_comp[int(owner[e])].append(int(e))
        per_tree.append(
            {comp: extract_embedding(tree, forest[comp], by_comp[comp]) for comp in range(k)}
        )

    for comp in range(k):
        shapes = {per_tree[j][comp].shape for j in range(len(local.trees))}
        if len(shapes) > 1:
            raise ForestPDInputError(
                f"MAF component {comp} has different topologies in different trees; "
                "the supplied forest is not an agreement forest"
            )

    comp_trees = [_component_tree(per_tree[0][comp]) for comp in range(k)]
    offsets = np.zeros(k + 1, dtype=np.int64)
    for comp in range(k):
        offsets[comp + 1] = offsets[comp] + comp_trees[comp].n_nodes
    return per_tree, comp_trees, offsets


def _component_tree(embedding: Embedding) -> ComponentTree:
    """
    Root and binarize one component's ``F_i`` into the DP's ``R_i``.

    Parameters
    ----------
    embedding : Embedding
        The component's ``F_i`` under its canonical numbering.

    Returns
    -------
    ComponentTree
        ``R_i``, with ``F_i`` node ``f`` at ``R_i`` node ``f + 1`` and spare binarization nodes
        appended after them.
    """
    return _r_tree(embedding.children, [int(x) for x in embedding.leaf])


def _r_tree(f_children: Sequence[Sequence[int]], f_leaf: Sequence[int]) -> ComponentTree:
    """
    ``R_i`` from ``F_i`` given as children and leaf id per node (node ``0`` the smallest leaf).

    Parameters
    ----------
    f_children : Sequence[Sequence[int]]
        Children per ``F_i`` node, in canonical order.
    f_leaf : Sequence[int]
        Leaf id per ``F_i`` node, ``-1`` for internal nodes.

    Returns
    -------
    ComponentTree
        As :func:`_component_tree`.
    """
    if len(f_leaf) == 1:
        return ComponentTree(
            n_nodes=1,
            left=np.array([-1], dtype=np.int32),
            right=np.array([-1], dtype=np.int32),
            leaf=np.array([f_leaf[0]], dtype=np.int32),
            postorder=np.array([0], dtype=np.int32),
            root=0,
        )

    # Node 0 is the virtual root; F node f becomes R node f + 1. F node 0 is the component's
    # smallest leaf and F node 1 the rest of it, so the virtual root's two children are 1 and 2 —
    # exactly the paper's "subdivide the pendant edge of a_0 with a new root r".
    n_f = len(f_leaf)
    children: list[list[int]] = [[] for _ in range(n_f + 1)]
    children[0] = [1, 2]
    for f in range(1, n_f):
        children[f + 1] = [c + 1 for c in f_children[f]]
    leaf_list: list[int] = [-1] + list(f_leaf)

    left_list = [-1] * (n_f + 1)
    right_list = [-1] * (n_f + 1)
    for node in range(n_f + 1):
        kids = children[node]
        if not kids:
            continue
        if len(kids) == 1:  # cannot arise from a suppressed F_i, but stay total
            left_list[node] = kids[0]
            continue
        # Binarize a polytomy into a chain of zero-weight spares (they own no lambda entries, so
        # their gamma is identically zero and the score is unchanged).
        current = node
        for extra in kids[:-2]:
            spare = len(left_list)
            left_list.append(-1)
            right_list.append(-1)
            leaf_list.append(-1)
            left_list[current] = extra
            right_list[current] = spare
            current = spare
        left_list[current] = kids[-2]
        right_list[current] = kids[-1]

    left = np.asarray(left_list, dtype=np.int32)
    right = np.asarray(right_list, dtype=np.int32)
    leaf = np.asarray(leaf_list, dtype=np.int32)
    n_nodes = len(left_list)
    postorder = _postorder(left, right, root=0)
    return ComponentTree(
        n_nodes=n_nodes,
        left=left,
        right=right,
        leaf=leaf,
        postorder=postorder,
        root=0,
    )


def _postorder(left: np.ndarray, right: np.ndarray, root: int) -> np.ndarray:
    """
    Children-before-parents ordering of a binary tree, iteratively.

    Parameters
    ----------
    left, right : numpy.ndarray
        Children per node, ``-1`` where absent.
    root : int
        Root node.

    Returns
    -------
    numpy.ndarray
        ``int32`` node ids in postorder.
    """
    out: list[int] = []
    stack = [root]
    while stack:
        v = stack.pop()
        out.append(v)
        for child in (int(left[v]), int(right[v])):
            if child >= 0:
                stack.append(child)
    out.reverse()
    return np.asarray(out, dtype=np.int32)


__all__ = [
    "ClusterProfile",
    "ComponentTree",
    "build_profile",
    "build_profile_from_splits",
    "build_profiles_from_splits",
    "plain_profile_from_plus",
]
