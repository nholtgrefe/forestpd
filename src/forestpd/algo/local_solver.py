"""
Algorithm 3: solve one cluster by enumerating signatures.

A signature ``S`` is a set of MAF components, meaning that exactly these components contain a
selected taxon. For each signature:

1. Definitely-chosen edges. An edge is in ``E_{j,C,S}`` exactly when some active component lies
   entirely on each of its sides (Lemma "definitely-chosen"). Their total weight ``d_S`` is a
   constant of the signature. With the precomputed masks this is one vectorized test over the
   edges of all trees, instead of the lemma's postorder per tree.
2. Two-sided weights. Every other edge that can count lies inside exactly one active ``H_{j,i}``,
   and whether it counts depends on which of its sides already holds another active component. The
   three cases are three ``bincount`` scatters into the ``lambda`` arrays.

With survival probabilities an edge also counts when one of its sides is *not* covered but keeps an
unprotected survivor, which happens with probability ``1 - q`` (``q`` the side's constant, see
:mod:`~forestpd.algo.profile`). Per edge, with ``w`` its weight:

* both sides covered: ``d_S += w``;
* one side covered, the edge not inside an active ``H``: ``d_S += w (1 - q)`` of the other side;
* inside an active ``H``, one side covered: ``d_S += w (1 - q_o)`` and ``lambda`` of the open
  side ``+= w q_o``;
* inside an active ``H``, neither side covered: ``d_S += w (1 - q_d)(1 - q_u)``,
  ``lambda_down += w q_d (1 - q_u)``, ``lambda_up += w q_u (1 - q_d)`` and
  ``lambda_both += w q_d q_u``.

For an edge inside an active component the side constant ``q`` here covers the taxa outside that
component only (the manuscript's ``1 - p``); its own taxa enter the component DP through
``pi_{X_i}`` (see :mod:`~forestpd.algo.component_dp`). In the manuscript's terms ``d_S`` is
``C_S^cross`` plus ``PD_i^{(S)}(0)`` of the inactive components, computed in one pass over the
merged rows instead of per tree; the inactive components' unselected pseudo-leaves add their
``f_2(x, 0)`` (``idle``) on top. Without survival probabilities every ``q`` is ``1``, the extra
terms vanish, and these are exactly the cases above.
3. Per-component tables ``PD_i^{(S)}`` (Lemma "compute-PD_i"), one DP per active component.
4. The budget is split across the active components (Lemma "running-time-K") and ``d_S`` is added.
   On DP' the inactive components save their whole upper bound cost, which shifts the table.

On DP, signatures with more components than the budget are skipped, since every active component
other than ``{x_C^*}`` needs at least one unit of budget. At small budgets this removes most of the
loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator

import numpy as np

from . import _signature_kernel
from .component_dp import component_table, node_offsets
from .convolve import fold_max_plus, max_plus, saved_shift
from .local import NEG_INF, LocalInstance
from .profile import ClusterProfile
from .witness import first_match

#: Run the signature loop compiled (:mod:`._signature_kernel`) when Numba is installed. The Python
#: loop below computes the same tables; tests switch this off to exercise it.
USE_COMPILED = True


@dataclass(frozen=True)
class LocalSolution:
    """
    One cluster's budget table, with enough breadcrumbs to reconstruct a witness.

    Attributes
    ----------
    table : numpy.ndarray
        ``float64[width]``; ``D(beta)``, the best value at entry ``beta`` of the budget axis (at
        most ``beta`` allocated on DP, at least ``beta`` saved on DP').
    best_signature : numpy.ndarray
        ``int64[width]``; the signature mask achieving each entry, ``-1`` where infeasible.
    """

    table: np.ndarray
    best_signature: np.ndarray


def signature_weights(
    profile: ClusterProfile, signature: int
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    The definitely-chosen weight and the two-sided weights of one signature.

    Parameters
    ----------
    profile : ClusterProfile
        The cluster's precomputed tables.
    signature : int
        Bitmask of the active components.

    Returns
    -------
    tuple[float, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(d_S, lam_down, lam_up, lam_both)``; the arrays are indexed by the shared ``R_i``
        address space (see :attr:`ClusterProfile.node_offset`).
    """
    mask = np.uint64(signature)
    has_below = (profile.edge_only_below & mask) != 0
    has_above = (profile.edge_only_above & mask) != 0
    # Edges inside an active component are left to its component rows below.
    free = (profile.edge_owner & mask) == 0
    d_s = float(profile.edge_weight[has_below & has_above].sum())
    d_s += float(profile.edge_survive_below[free & has_above & ~has_below].sum())
    d_s += float(profile.edge_survive_above[free & has_below & ~has_above].sum())

    slots = profile.n_slots
    if profile.ce_slot.size == 0:
        zero = np.zeros(slots)
        return d_s, zero, zero.copy(), zero.copy()

    # An active component wholly on a side means that side is occupied for free; the edge then pays
    # out iff the *other* side gets a taxon of the component the edge belongs to (or, with survival
    # probabilities, keeps an unprotected survivor).
    down_covered = (profile.ce_only_down & mask) != 0
    up_covered = (profile.ce_only_up & mask) != 0
    # Only edges of *active* components matter here; those of inactive ones are edge rows above.
    active = (profile.ce_own_bit & mask) != 0
    to_down = active & ~down_covered & up_covered
    to_up = active & down_covered & ~up_covered
    to_both = active & ~down_covered & ~up_covered
    # The remaining case (both covered) is definitely-chosen and already counted in d_S.

    weight = profile.ce_weight
    dies_down, dies_up, dies_both = profile.ce_dies_down, profile.ce_dies_up, profile.ce_dies_both
    d_s += float((weight[to_down] - dies_down[to_down]).sum())
    d_s += float((weight[to_up] - dies_up[to_up]).sum())
    d_s += float(
        (weight[to_both] - dies_down[to_both] - dies_up[to_both] + dies_both[to_both]).sum()
    )

    def scatter(selector: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Sum the selected component edges' values into their ``R_i`` slots."""
        return np.bincount(profile.ce_slot[selector], weights=values[selector], minlength=slots)

    lam_down = scatter(to_down, dies_down) + scatter(to_both, dies_down - dies_both)
    lam_up = scatter(to_up, dies_up) + scatter(to_both, dies_up - dies_both)
    return d_s, lam_down, lam_up, scatter(to_both, dies_both)


def solve_local(
    local: LocalInstance,
    profile: ClusterProfile,
    forced_component: int = -1,
) -> LocalSolution:
    """
    Compute one cluster's budget table by enumerating its signatures.

    Parameters
    ----------
    local : LocalInstance
        The cluster's problem.
    profile : ClusterProfile
        Its precomputed tables.
    forced_component : int, default -1
        Component that must be active (the artificial ``x_C^*`` when computing ``PD_C^+``).

    Returns
    -------
    LocalSolution
        The cluster's budget table ``D(beta)`` and the best signature per budget.
    """
    budget = local.budget
    comp_idle, comp_saved = component_constants(profile, local)
    if USE_COMPILED and _signature_kernel.HAVE_NUMBA and profile.n_components > 0:
        best, best_signature = _signature_kernel.signature_loop(
            profile.edge_only_below,
            profile.edge_only_above,
            profile.edge_owner,
            profile.edge_weight,
            profile.edge_survive_below,
            profile.edge_survive_above,
            profile.ce_slot,
            profile.ce_own_bit,
            profile.ce_only_down,
            profile.ce_only_up,
            profile.ce_weight,
            profile.ce_dies_down,
            profile.ce_dies_up,
            profile.ce_dies_both,
            profile.slot_survive_down,
            profile.slot_survive_up,
            *_packed_component_trees(profile, local),
            np.ascontiguousarray(local.f2, dtype=np.float64),
            local.idle_or_zero,
            comp_idle,
            comp_saved,
            budget,
            forced_component,
            local.complement,
        )
        return LocalSolution(table=best, best_signature=best_signature)

    best = np.full(budget + 1, NEG_INF)
    best_signature = np.full(budget + 1, -1, dtype=np.int64)

    for signature, active in _signatures(
        profile.n_components, forced_component, budget, local.complement
    ):
        d_s, lam_down, lam_up, lam_both = signature_weights(profile, signature)
        d_s += inactive_idle(comp_idle, signature)
        min_selected = 2 if len(active) == 1 else 1
        tables = []
        for comp in active:
            span = slice(int(profile.node_offset[comp]), int(profile.node_offset[comp + 1]))
            tables.append(
                component_table(
                    profile.comp_trees[comp],
                    lam_down[span],
                    lam_up[span],
                    lam_both[span],
                    profile.slot_survive_down[span],
                    profile.slot_survive_up[span],
                    local,
                    min_selected,
                )
            )
        fold = fold_max_plus(tables, budget)
        if local.complement:
            fold = saved_shift(fold, inactive_saved(comp_saved, signature))
        candidate = fold + d_s
        improved = candidate > best
        best[improved] = candidate[improved]
        best_signature[improved] = signature

    return LocalSolution(table=best, best_signature=best_signature)


def component_constants(
    profile: ClusterProfile, local: LocalInstance
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per component, the summed ``idle`` and (on DP') upper bound cost of its leaves.

    An inactive component selects nothing: its leaves are worth their ``idle`` value, and on DP'
    it saves its whole upper bound cost.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``float64[k]`` and ``int64[k]`` (zeros on DP).
    """
    k = profile.n_components
    comp_idle = np.zeros(k)
    comp_saved = np.zeros(k, dtype=np.int64)
    idle = local.idle
    if idle is not None and idle.any():
        comp_idle[:] = [float(idle[list(block)].sum()) for block in profile.comp_leaves]
    if local.complement:
        cost_max = local.cost_max_or_cost
        comp_saved[:] = [int(cost_max[list(block)].sum()) for block in profile.comp_leaves]
    return comp_idle, comp_saved


def inactive_idle(comp_idle: np.ndarray, signature: int) -> float:
    """The summed ``idle`` of the components outside ``signature``."""
    return float(sum(v for c, v in enumerate(comp_idle.tolist()) if not (signature >> c) & 1))


def inactive_saved(comp_saved: np.ndarray, signature: int) -> int:
    """What the components outside ``signature`` save on DP' (zero on DP)."""
    return int(sum(v for c, v in enumerate(comp_saved.tolist()) if not (signature >> c) & 1))


def _packed_component_trees(
    profile: ClusterProfile, local: LocalInstance
) -> tuple[np.ndarray, ...]:
    """
    All ``R_i`` in the profile's shared node address space, for the compiled signature loop.

    Parameters
    ----------
    profile : ClusterProfile
        The cluster's precomputed tables.
    local : LocalInstance
        The local instance, for the budget-axis offsets.

    Returns
    -------
    tuple[numpy.ndarray, ...]
        ``node_offset``; per global node its ``left`` and ``right`` child (global, ``-1`` at a
        leaf), local ``leaf`` id (``-1`` if internal) and budget-axis offset (see
        :func:`~forestpd.algo.component_dp.node_offsets`); the concatenated postorders (global)
        with their offsets; and each component's root (global).
    """
    n_slots = profile.n_slots
    left = np.full(n_slots, -1, dtype=np.int64)
    right = np.full(n_slots, -1, dtype=np.int64)
    leaf = np.full(n_slots, -1, dtype=np.int64)
    axis = np.zeros(n_slots, dtype=np.int64)  # all zero on DP
    posts = []
    post_offset = np.zeros(profile.n_components + 1, dtype=np.int64)
    roots = np.empty(profile.n_components, dtype=np.int64)
    for i, tree in enumerate(profile.comp_trees):
        offset = int(profile.node_offset[i])
        span = slice(offset, offset + tree.n_nodes)
        left[span] = np.where(tree.left >= 0, tree.left + offset, -1)
        right[span] = np.where(tree.right >= 0, tree.right + offset, -1)
        leaf[span] = tree.leaf
        if local.complement:
            axis[span] = node_offsets(tree, local)
        posts.append(tree.postorder.astype(np.int64) + offset)
        post_offset[i + 1] = post_offset[i] + tree.n_nodes
        roots[i] = tree.root + offset
    post = np.concatenate(posts) if posts else np.empty(0, dtype=np.int64)
    return profile.node_offset.astype(np.int64), left, right, leaf, axis, post, post_offset, roots


def _signatures(
    k: int, forced: int, budget: int, complement: bool = False
) -> Iterator[tuple[int, list[int]]]:
    """
    Enumerate the signatures worth trying, as ``(mask, sorted component list)``.

    Nonempty, containing ``forced`` when one is given, and on DP with at most ``budget`` components
    besides ``forced`` (``{x_C^*}`` is free; every other active component needs budget).

    Parameters
    ----------
    k : int
        Number of MAF components.
    forced : int
        Component that must be active, or ``-1`` for none.
    budget : int
        The last table index.
    complement : bool, default False
        Whether the tables are on the DP' axis, where the index does not bound the components.

    Yields
    ------
    tuple[int, list[int]]
        The signature's component bitmask and its sorted list of active components.
    """
    optional = [c for c in range(k) if c != forced]
    base_mask = 0 if forced < 0 else 1 << forced
    base_list: list[int] = [] if forced < 0 else [forced]
    low = 1 if forced < 0 else 0
    high = len(optional) if complement else min(len(optional), budget)
    for size in range(low, high + 1):
        for chosen in combinations(optional, size):
            mask = base_mask
            for c in chosen:
                mask |= 1 << c
            yield mask, sorted(base_list + list(chosen))


def split_budget(tables: list[np.ndarray], budget: int) -> list[int] | None:
    """
    Recover a per-component budget split achieving ``fold_max_plus(tables, budget)[budget]``.

    Parameters
    ----------
    tables : list[numpy.ndarray]
        One ``float64[budget+1]`` table per active component.
    budget : int
        The total budget to split.

    Returns
    -------
    list[int] | None
        One budget per table, or ``None`` when the target is infeasible.
    """
    return split_budget_with(tables, budget_folds(tables, budget), budget)


def budget_folds(tables: list[np.ndarray], budget: int) -> list[np.ndarray]:
    """
    The suffix folds ``fold_max_plus(tables[i:])`` up to ``budget``, for :func:`split_budget_with`.

    An entry of a fold depends only on the tables' entries up to it, so folds computed for a large
    budget serve every smaller one: several budgets can be split from one computation.

    Parameters
    ----------
    tables : list[numpy.ndarray]
        One ``float64`` table per active component.
    budget : int
        The largest total budget to split.

    Returns
    -------
    list[numpy.ndarray]
        ``m + 1`` arrays of length ``budget + 1``; the last is the empty fold (``0`` at budget 0).
    """
    m = len(tables)
    suffix: list[np.ndarray] = [np.array([])] * (m + 1)
    tail = np.full(budget + 1, NEG_INF)
    tail[0] = 0.0
    suffix[m] = tail
    for i in range(m - 1, -1, -1):
        suffix[i] = max_plus(tables[i], suffix[i + 1], budget + 1)
    return suffix


def split_budget_with(
    tables: list[np.ndarray], suffix: list[np.ndarray], budget: int
) -> list[int] | None:
    """
    :func:`split_budget`, given the folds from :func:`budget_folds` (for this or a larger budget).

    Parameters
    ----------
    tables : list[numpy.ndarray]
        One ``float64`` table per active component.
    suffix : list[numpy.ndarray]
        ``budget_folds(tables, b)`` for some ``b >= budget``.
    budget : int
        The total budget to split.

    Returns
    -------
    list[int] | None
        One budget per table, or ``None`` when the target is infeasible.
    """
    m = len(tables)
    if m == 0:
        return [] if budget == 0 else None
    if suffix[0][budget] == NEG_INF:
        return None

    out: list[int] = []
    remaining = budget
    for i in range(m):
        # Entry ``take`` is table i at ``take`` plus the rest's fold at ``remaining - take`` (NEG_INF
        # if either is); the smallest matching ``take`` is used.
        candidates = tables[i][: remaining + 1] + suffix[i + 1][remaining::-1]
        take = first_match(candidates, suffix[i][remaining])
        if take < 0:  # pragma: no cover - the fold guarantees a split exists
            return None
        out.append(take)
        remaining -= take
    return out


__all__ = [
    "LocalSolution",
    "solve_local",
    "signature_weights",
    "split_budget",
    "budget_folds",
    "split_budget_with",
]
