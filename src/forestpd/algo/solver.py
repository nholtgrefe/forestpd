"""
Algorithm 1: the bottom-up pass over the tree of clusters (Section "Cluster Reduction").

Each cluster ``C`` is solved twice:

``PD_C(beta)``
    everything selected lies at or below ``C``, and at least two leaves of ``C``'s local instance
    are selected. A feasible taxon set is counted at the lowest cluster containing all of it, where
    its projection has at least two leaves, so the answer is read from these tables.
``PD_C^+(beta)``
    something outside the subtree of ``C`` is selected as well. This table is used by the parent,
    through the pseudo-leaf ``x_C``. It is computed by attaching an artificial leaf ``x_C^*`` (the
    manuscript's ``star_C``) at the cut, with a zero-weight edge and cost zero, and forcing its
    singleton component into every signature, so the parent's ``f_2(x_C, beta) = PD_C^+(beta)``.

With survival probabilities (see :mod:`~forestpd.algo.survival`) unselected regions still count:
an unselected child ``D`` is worth ``Omega_D^0`` (its pseudo-leaf's ``f_2(x_D, 0)``), and the region
outside the subtree of the cluster holding a solution is worth ``Omega_C^1``, part of ``PD_C``.
The tables leave out the empty set and single taxa, which are computed directly. The answer is

    ``max(value(empty set), max_x value({x}), max_C PD_C(B))``

(Theorem "running-time"); without survival probabilities the first two are zero.

The tables exist on two budget axes (see :mod:`~forestpd.algo.local`): ``DP``, with at most ``beta``
allocated, gives every budget up to ``B`` and costs about ``B^2``; ``DP'``, with at least ``beta``
saved, gives every budget from ``B`` up to the total cost ``c(X)`` and costs about
``(c(X) - B)^2``. Its length ``B' = c(X) - B`` is global: every cluster's tables run up to it (capped
at what the cluster costs). :func:`choose_tables` picks DP, DP' or both, whichever is cheapest for
the budgets asked; the reduction, the agreement forests and the profiles are shared.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..cluster.base import ClusterTree, LocalPiece
from ..core.tree import PhyloTree, drop_leaf
from ..exceptions import ForestPDInputError
from ..maf.base import Forest, Rooting
from .local import NEG_INF, LocalInstance, ordinary_f2, pseudo_leaf_f2
from .component_dp import component_dp_table, root_table
from .local_solver import (
    LocalSolution,
    budget_folds,
    component_constants,
    inactive_saved,
    signature_weights,
    solve_local,
    split_budget_with,
)
from .profile import (
    ClusterProfile,
    build_profile,
    build_profiles_from_splits,
    plain_profile_from_plus,
)
from .survival import cluster_sides, leaf_logs, subtree_logs
from .witness import trace_selection

#: Force the choice of tables in :func:`choose_tables`: ``"DP"``, ``"DP'"`` or ``"both"``; ``None``
#: (default) takes the cheapest. For tests.
FORCE_TABLES: str | None = None


def choose_tables(budgets: Sequence[int], total: int) -> tuple[int, int | None]:
    """
    Which tables to compute for a set of budgets: DP up to some budget, DP' from some budget, or
    both, whichever is cheapest.

    A DP up to ``B`` costs about ``B^2``, a DP' from ``b`` about ``(total - b)^2`` (Lemma
    "running-time-K"). With both, the budgets up to a split are read from DP and the rest from DP';
    the split is placed at the cheapest gap between consecutive budgets. Ties go to fewer runs, then
    to DP.

    Parameters
    ----------
    budgets : Sequence[int]
        The budgets to read out, in cost units; at least one.
    total : int
        The total cost of all taxa, in the same units.

    Returns
    -------
    tuple[int, int | None]
        ``(dp_budget, complement_from)``: the DP's budget (``-1`` for no DP) and the smallest
        budget read from DP' (``None`` for no DP'). Every budget is at most ``dp_budget`` or at
        least ``complement_from``.
    """
    points = sorted({min(int(b), total) for b in budgets})
    options: list[tuple[int, int, int, int, int | None]] = [
        (points[-1] ** 2, 1, 0, points[-1], None),
        ((total - points[0]) ** 2, 1, 1, -1, points[0]),
    ]
    for j in range(1, len(points)):
        cost = points[j - 1] ** 2 + (total - points[j]) ** 2
        options.append((cost, 2, 2, points[j - 1], points[j]))
    if FORCE_TABLES is not None:
        kind = {"DP": 0, "DP'": 1, "both": 2}[FORCE_TABLES]
        options = [option for option in options if option[2] == kind]
        if not options:
            raise ValueError("both tables need at least two distinct budgets")
    _, _, _, dp_budget, complement_from = min(options)
    return dp_budget, complement_from


@dataclass
class TableRun:
    """
    One cluster's tables on one budget axis.

    The tables stop at the cluster's cap (``local.budget``): the cost of the taxa below it, if
    that is less than the run's budget. On DP every larger entry would equal the best one, so the
    readers take that entry instead (see :func:`_beyond_cap`); keeping the tables this short is
    what keeps the memory small when there are many clusters and a large budget.

    Attributes
    ----------
    plain : LocalInstance
        The local instance ``I_C``.
    plain_solution : LocalSolution
        Its table ``PD_C``.
    plus : LocalInstance | None
        The local instance ``I_C^+``; ``None`` at the root cluster.
    plus_solution : LocalSolution | None
        Its table ``PD_C^+``; ``None`` at the root cluster.
    """

    plain: LocalInstance
    plain_solution: LocalSolution
    plus: LocalInstance | None = None
    plus_solution: LocalSolution | None = None


@dataclass
class ClusterState:
    """
    Everything computed for one cluster, kept for the reconstruction pass.

    Attributes
    ----------
    plain_profile : ClusterProfile
        The signature-independent tables of ``I_C``.
    plus_profile : ClusterProfile | None
        Those of ``I_C^+``; ``None`` at the root cluster.
    below : int
        The cost ``c(X_C)`` of the taxa at or below the cluster.
    runs : dict[bool, TableRun]
        The tables per budget axis, keyed by ``complement`` (``False``: DP, ``True``: DP').
    up : float
        ``Omega_C^1``, the expected weight of the edges outside the region below ``C`` when the
        solution lies below ``C``; part of ``PD_C`` in the manuscript, added when reading out
        because it depends on the clusters above (zero without survival probabilities).
    """

    plain_profile: ClusterProfile
    plus_profile: ClusterProfile | None
    below: int
    runs: dict[bool, TableRun] = field(default_factory=dict)
    up: float = 0.0


@dataclass
class Solution:
    """
    The optimizer's answer.

    Attributes
    ----------
    value : float
        The optimum of ``sum_j w_j PD_{T_j}(S)`` over feasible ``S`` (its expected value with
        survival probabilities) at the largest budget read from DP, or at the smallest read from
        DP' when there is no DP.
    meta : dict
        Diagnostics: cluster count and sizes (leaves of each ``I_C``), MAF sizes, signature count,
        timings (``solve_seconds`` = ``maf_seconds`` + ``dp_seconds``), ``"tables"`` (``"DP"``,
        ``"DP'"`` or ``"both"``) and ``"frontier"``: ``PD(beta)`` for every budget ``beta`` up to
        the DP's budget, and from ``complement_from`` up to the total cost; ``NaN`` in between.
    witnesses : dict[int, set[int]]
        For each budget in ``witness_budgets``, global taxon ids achieving ``PD(beta)``.
    single_values : numpy.ndarray
        ``float64[n_taxa]``; the objective of selecting each taxon alone.
    """

    value: float
    meta: dict[str, Any] = field(default_factory=dict)
    witnesses: dict[int, set[int]] = field(default_factory=dict)
    single_values: np.ndarray = field(default_factory=lambda: np.zeros(0))


def solve(
    cluster_tree: ClusterTree,
    maf_computer: Any,
    costs: np.ndarray,
    tree_weights: np.ndarray,
    budget: int,
    witness_budgets: Sequence[int] = (),
    log_loss: np.ndarray | None = None,
    complement_from: int | None = None,
) -> Solution:
    """
    Run the whole pipeline over a prepared cluster tree.

    Parameters
    ----------
    cluster_tree : ClusterTree
        Output of the cluster-reduction stage.
    maf_computer : MAFComputer
        The agreement-forest strategy.
    costs : numpy.ndarray
        ``int64[n_taxa]``; cost per global taxon id.
    tree_weights : numpy.ndarray
        ``float64[t]``; the per-tree weights ``w_j``.
    budget : int
        The budget ``B`` of the DP, which gives every budget up to ``B``; ``-1`` for no DP.
    witness_budgets : Sequence[int], default ()
        The budgets to reconstruct an optimal taxon set for, from the same tables; each must be at
        most ``budget`` or at least ``complement_from``.
    log_loss : numpy.ndarray | None, optional
        ``float64[n_taxa]``; ``log(1 - a(x))`` per taxon id, for survival probabilities ``a``.
        ``None`` means all zero: no taxon survives unless selected.
    complement_from : int | None, optional
        Also compute DP' for every budget from this one up to the total cost (with
        ``B' = total - complement_from``). ``None`` (default): DP only.

    Returns
    -------
    Solution
        The optimum, an optimal taxon set per requested budget, and diagnostics.

    Raises
    ------
    ForestPDInputError
        If the MAF stage returns a partition that is not an agreement forest. A MAF stage marked
        ``trusted`` (see :class:`~forestpd.maf.base.MAFComputer`) is only partially checked: its
        clusters' profiles are built from their split tables, which cannot see vertex-disjointness.
    """
    started = time.perf_counter()
    use_splits = bool(getattr(maf_computer, "trusted", False))
    states: dict[int, ClusterState] = {}
    maf_sizes: list[int] = []
    cluster_sizes: list[int] = []
    signature_count = 0
    maf_seconds = 0.0
    n_taxa = costs.size
    total = int(costs.sum())
    # The budget axes to compute, each with the last index of its tables (before the per-cluster
    # caps): DP up to B, DP' up to the global B' = total - complement_from (review of v15, item 4).
    axes: dict[bool, int] = {}
    if budget >= 0:
        axes[False] = budget
    if complement_from is not None:
        axes[True] = total - complement_from
    assert axes, "nothing to compute"
    if log_loss is None:
        log_loss = np.zeros(n_taxa)
    total_log = float(log_loss.sum())
    subtree_log = subtree_logs(cluster_tree, log_loss)
    # Per cluster: Omega^0 (its value when unselected, the pseudo-leaf's idle value in its parent),
    # the sum of its children's, and the survival sums at its local leaves.
    down = [0.0] * cluster_tree.n_clusters
    children_down = [0.0] * cluster_tree.n_clusters
    at_leaf: dict[int, np.ndarray] = {}
    empty_value = 0.0

    # A cluster can never use more budget than the taxa below it cost (or save more than that),
    # so its tables stop there, and are kept that short (see TableRun).
    below = _subtree_costs(cluster_tree, costs)
    for cluster in cluster_tree.postorder:
        piece = cluster_tree.pieces[cluster]
        leaf_log = leaf_logs(piece, log_loss, subtree_log, cluster, total_log)
        sides = cluster_sides(piece, leaf_log, tree_weights)
        children_down[cluster] = sum(down[d] for d in cluster_tree.children[cluster])
        root_leaf = piece.star_leaf if piece.star_leaf >= 0 else 0
        down[cluster] = float(sides.at_leaf[root_leaf]) + children_down[cluster]
        at_leaf[cluster] = sides.at_leaf
        empty_value += sides.empty

        trees = plain_trees(piece)
        maf_started = time.perf_counter()
        forest = cluster_forest(maf_computer, piece, trees)
        maf_seconds += time.perf_counter() - maf_started
        maf_sizes.append(len(forest))
        cluster_sizes.append(piece.n_leaves - (piece.star_leaf >= 0))

        state: ClusterState | None = None
        star_component = len(forest)
        for complement, top in axes.items():
            plain, plus = _build_instances(
                piece, trees, states, cluster_tree, costs, tree_weights, top, below[cluster],
                down, below, leaf_log, complement,
            )  # fmt: skip
            if state is None:  # the profiles do not depend on the budget axis
                if plus is None:
                    plain_profile, _ = _profiles(piece, plain, forest, use_splits, None, leaf_log)
                    plus_profile = None
                else:
                    # Adding a pendant leaf cannot break the agreement forest, so the same
                    # components are reused with one extra singleton {x_C^*}, which is then forced
                    # into every signature. The plain profile follows from the plus one (see
                    # plain_profile_from_plus).
                    plus_profile, derived = _profiles(
                        piece, plus, forest + ((piece.star_leaf,),), use_splits, star_component,
                        leaf_log,
                    )  # fmt: skip
                    assert derived is not None
                    plain_profile = derived
                state = ClusterState(plain_profile, plus_profile, below[cluster])
            run = TableRun(plain, solve_local(plain, state.plain_profile))
            signature_count += 1 << len(forest)
            if plus is not None:
                assert state.plus_profile is not None
                run.plus = plus
                run.plus_solution = solve_local(plus, state.plus_profile, star_component)
                signature_count += 1 << len(forest)
            state.runs[complement] = run
        assert state is not None
        states[cluster] = state

    # Top-down: Omega^1(D) for a child D of C is Omega^1(C), plus its siblings' Omega^0, plus C's
    # own edges paid on their side away from x_D.
    single_values = np.zeros(n_taxa)
    for cluster in reversed(cluster_tree.postorder):
        piece = cluster_tree.pieces[cluster]
        up = states[cluster].up
        for leaf in range(piece.n_leaves):
            taxon = piece.taxon_of_leaf[leaf]
            child = piece.child_of_leaf[leaf]
            if taxon >= 0:
                single_values[taxon] = up + children_down[cluster] + float(at_leaf[cluster][leaf])
            elif child >= 0:
                siblings = sum(down[d] for d in cluster_tree.children[cluster] if d != child)
                states[child].up = up + siblings + float(at_leaf[cluster][leaf])

    # The answer per budget: the best of the empty set, the single taxa and every cluster's PD_C
    # (Theorem "running-time"); NaN for budgets neither table covers.
    top = total if complement_from is not None else budget
    baseline = _Baseline(empty_value, single_values, costs, top)
    frontier = np.full(top + 1, np.nan)
    if budget >= 0:
        frontier[: budget + 1] = _dp_frontier(states, baseline, budget)
    if complement_from is not None:
        frontier[complement_from:] = _complement_frontier(states, baseline, complement_from, total)

    witnesses = _witnesses(states, baseline, witness_budgets, cluster_tree, budget, total)

    solve_seconds = time.perf_counter() - started
    tables = "both" if len(axes) == 2 else ("DP'" if True in axes else "DP")
    meta = {
        "n_clusters": cluster_tree.n_clusters,
        "cluster_sizes": cluster_sizes,
        "max_maf_size": max(maf_sizes) if maf_sizes else 0,
        "maf_sizes": maf_sizes,
        "n_signatures": signature_count,
        "budget": budget,
        "tables": tables,
        "frontier": frontier,
        "solve_seconds": solve_seconds,
        "maf_seconds": maf_seconds,
        "dp_seconds": solve_seconds - maf_seconds,
    }
    at = budget if budget >= 0 else complement_from
    assert at is not None
    return Solution(
        value=float(frontier[at]),
        meta=meta,
        witnesses=witnesses,
        single_values=single_values,
    )


class _Baseline:
    """
    The selections the cluster tables leave out: the empty set and single taxa.

    Parameters
    ----------
    empty : float
        The value of the empty selection.
    singles : numpy.ndarray
        ``float64[n_taxa]``; the value of each taxon alone.
    costs : numpy.ndarray
        ``int64[n_taxa]``; their costs.
    budget : int
        The largest budget needed.

    Attributes
    ----------
    values : numpy.ndarray
        ``float64[budget+1]``; the best of these at a budget of at most ``beta``.
    choice : numpy.ndarray
        ``int64[budget+1]``; the taxon achieving it, ``-1`` for the empty set.
    """

    def __init__(self, empty: float, singles: np.ndarray, costs: np.ndarray, budget: int) -> None:
        values = np.full(budget + 1, empty)
        choice = np.full(budget + 1, -1, dtype=np.int64)
        # Exact-cost entries first, then a running maximum; strict improvements only, so the empty
        # set wins ties (as it always does without survival probabilities, where all are zero).
        for taxon in np.flatnonzero(costs <= budget).tolist():
            at = int(costs[taxon])
            if singles[taxon] > values[at]:
                values[at] = singles[taxon]
                choice[at] = taxon
        running = np.maximum.accumulate(values)
        rises = np.ones(values.size, dtype=bool)
        rises[1:] = values[1:] > running[:-1]
        first = np.maximum.accumulate(np.where(rises, np.arange(values.size), 0))
        self.values = running
        self.choice = choice[first]


def _dp_frontier(states: dict[int, ClusterState], baseline: _Baseline, budget: int) -> np.ndarray:
    """``PD(beta)`` for ``beta = 0..budget`` from the DP tables."""
    frontier: np.ndarray = baseline.values[: budget + 1].copy()
    for state in states.values():
        # The table up to the cluster's cap, then its best entry (see _beyond_cap), in place.
        table = state.runs[False].plain_solution.table
        within = min(table.size, frontier.size)
        np.maximum(frontier[:within], table[:within] + state.up, out=frontier[:within])
        np.maximum(frontier[within:], table.max() + state.up, out=frontier[within:])
    # An entry holds the best value with at most beta allocated, except that the paper's
    # f_2(x_D, B) = -inf can leave a last entry below the one before; the running maximum makes
    # every entry "at most beta". One solve yields the whole PD(beta) curve, not just PD(B).
    np.maximum.accumulate(frontier, out=frontier)
    return frontier


def _complement_frontier(
    states: dict[int, ClusterState], baseline: _Baseline, start: int, total: int
) -> np.ndarray:
    """
    ``PD(beta)`` for ``beta = start..total`` from the DP' tables.

    A solution below ``C`` with at least ``e`` of ``c(X_C)`` saved leaves everything outside
    ``C`` unselected, so it costs at most ``beta`` iff ``e >= c(X_C) - beta``: budget ``beta``
    reads entry ``max(0, c(X_C) - beta)`` of ``PD'_C``.
    """
    betas = np.arange(start, total + 1)
    frontier: np.ndarray = baseline.values[start:].copy()
    for state in states.values():
        table = state.runs[True].plain_solution.table
        entries = np.maximum(state.below - betas, 0)
        np.maximum(frontier, table[entries] + state.up, out=frontier)
    return frontier


def plain_trees(piece: LocalPiece) -> tuple[PhyloTree, ...]:
    """
    The trees of ``I_C``: the piece's trees without ``x_C^*`` (as they are at the root cluster).

    Parameters
    ----------
    piece : LocalPiece
        The cluster's geometry.

    Returns
    -------
    tuple[PhyloTree, ...]
        The local trees the MAF stage and the plain DP run on.
    """
    if piece.star_leaf < 0:
        return piece.trees
    return tuple(drop_leaf(t, piece.star_leaf) for t in piece.trees)


def cluster_forest(maf_computer: Any, piece: LocalPiece, trees: Sequence[PhyloTree]) -> Forest:
    """
    The agreement forest of one cluster, as the solver computes it.

    Parameters
    ----------
    maf_computer : MAFComputer
        The agreement-forest strategy.
    piece : LocalPiece
        The cluster's geometry.
    trees : Sequence[PhyloTree]
        The cluster's plain trees (:func:`plain_trees`).

    Raises
    ------
    ForestPDInputError
        If the computer does not return a partition of the leaves.

    Returns
    -------
    Forest
        The forest of ``trees``. A computer with ``uses_rooting`` also gets the cluster's
        :class:`~forestpd.maf.base.Rooting`: towards the top of ``K``, at ``x_C^*`` (in the piece's
        trees, as the last leaf), or at leaf 0 in the top cluster.
    """
    if not getattr(maf_computer, "uses_rooting", False):
        forest: Forest = maf_computer.compute(trees)
    else:
        if piece.star_leaf < 0:
            rooting = Rooting(trees, 0, extra=False)
        else:
            rooting = Rooting(piece.trees, piece.star_leaf, extra=True)
        forest = maf_computer.compute(trees, rooting=rooting)
    return _checked_partition(forest, trees[0].n_leaves)


def _checked_partition(forest: Forest, n_leaves: int) -> Forest:
    """
    ``forest`` as tuples of ints, after checking it partitions the leaves ``0..n_leaves-1``.

    Raises
    ------
    ForestPDInputError
        If a component is empty, a leaf is not an integer in range, or a leaf is missing or
        appears twice.
    """
    seen = [False] * n_leaves
    out = []
    for component in forest:
        leaves = tuple(component)
        if not leaves:
            raise ForestPDInputError("the MAF stage returned an empty component")
        for leaf in leaves:
            if isinstance(leaf, (bool, np.bool_)) or not isinstance(leaf, (int, np.integer)):
                raise ForestPDInputError(f"the MAF stage returned a non-integer leaf {leaf!r}")
            if not 0 <= leaf < n_leaves:
                raise ForestPDInputError(f"the MAF stage returned leaf {leaf} out of range")
            if seen[leaf]:
                raise ForestPDInputError(f"the MAF stage returned leaf {leaf} twice")
            seen[leaf] = True
        out.append(tuple(int(leaf) for leaf in leaves))
    if not all(seen):
        raise ForestPDInputError(
            f"the MAF stage left out {seen.count(False)} leaves; it must return a partition"
        )
    return tuple(out)


def _subtree_costs(cluster_tree: ClusterTree, costs: np.ndarray) -> list[int]:
    """Total cost of the taxa at or below each cluster."""
    total = [0] * cluster_tree.n_clusters
    for cluster in cluster_tree.postorder:
        piece = cluster_tree.pieces[cluster]
        own = sum(int(costs[t]) for t in piece.taxon_of_leaf if t >= 0)
        total[cluster] = own + sum(total[d] for d in cluster_tree.children[cluster])
    return total


def _beyond_cap(table: np.ndarray, width: int) -> np.ndarray:
    """
    A DP table kept up to its cluster's cap (see :class:`TableRun`), as ``width`` entries.

    Beyond the cap no more budget can be used, and an entry holds the best value with at most that
    much allocated, so every larger entry equals the best entry. Only for DP tables: on DP' an
    entry beyond the cap would save more than the cluster costs, which is infeasible.

    Parameters
    ----------
    table : numpy.ndarray
        The table, up to the cluster's cap.
    width : int
        The number of entries wanted.

    Returns
    -------
    numpy.ndarray
        ``table`` itself if it is long enough, else a copy extended by its best entry.

    Examples
    --------
    >>> _beyond_cap(np.array([-np.inf, 2.0, 5.0]), 5)
    array([-inf,   2.,   5.,   5.,   5.])
    """
    if table.size >= width:
        return table
    return np.concatenate([table, np.full(width - table.size, table.max())])


def _witnesses(
    states: dict[int, ClusterState],
    baseline: _Baseline,
    budgets: Sequence[int],
    cluster_tree: ClusterTree,
    dp_budget: int,
    total: int,
) -> dict[int, set[int]]:
    """
    An optimal selection at a budget of at most ``beta``, for every ``beta`` in ``budgets``.

    A budget up to ``dp_budget`` is read from the DP tables, a larger one from the DP' tables. Per
    budget, the best ``PD_C + Omega_C^1`` entry over all clusters wins only if it beats the best
    empty or single selection strictly, so ties resolve to the smaller selection (and between
    clusters to the first). The winning entries are then expanded together, top-down over the tree
    of clusters (:func:`_expand`), so that work shared by several budgets is done once.

    Parameters
    ----------
    states : dict[int, ClusterState]
        All solved clusters.
    baseline : _Baseline
        The empty and single selections.
    budgets : Sequence[int]
        The budgets.
    cluster_tree : ClusterTree
        The tree of clusters ``K``.
    dp_budget : int
        The DP's budget (``-1`` without DP).
    total : int
        The total cost of all taxa.

    Returns
    -------
    dict[int, set[int]]
        Global taxon ids per budget.
    """
    budgets = sorted(set(budgets))
    selected: dict[int, set[int]] = {beta: set() for beta in budgets}
    if not budgets:
        return selected
    clusters = list(states)
    # Requests to expand a table entry: (cluster, plus, complement) -> [(budget of the set,
    # entry), ...].
    pending: dict[tuple[int, bool, bool], list[tuple[int, int]]] = defaultdict(list)

    on_dp = [beta for beta in budgets if beta <= dp_budget]
    if on_dp:
        wanted = np.asarray(on_dp, dtype=np.int64)
        # Per cluster and budget: the best entry of PD_C + Omega_C^1 up to the budget, and its first
        # position, as ``np.argmax(table[: beta + 1])`` gives it.
        best = np.empty((len(clusters), wanted.size))
        where = np.empty((len(clusters), wanted.size), dtype=np.int64)
        for row, cluster in enumerate(clusters):
            state = states[cluster]
            table = state.runs[False].plain_solution.table + state.up
            running = np.maximum.accumulate(table)
            rises = np.ones(table.size, dtype=bool)
            rises[1:] = table[1:] > running[:-1]
            first = np.maximum.accumulate(np.where(rises, np.arange(table.size), 0))
            positions = np.minimum(wanted, table.size - 1)
            best[row] = running[positions]
            where[row] = first[positions]
        for column, beta in enumerate(on_dp):
            row = int(np.argmax(best[:, column]))
            if best[row, column] > float(baseline.values[beta]):
                pending[(clusters[row], False, False)].append((beta, int(where[row, column])))
            elif baseline.choice[beta] >= 0:
                selected[beta].add(int(baseline.choice[beta]))

    on_complement = [beta for beta in budgets if beta > dp_budget]
    if on_complement:
        # Budget beta reads entry max(0, c(X_C) - beta) of PD'_C (see _complement_frontier).
        betas = np.asarray(on_complement, dtype=np.int64)
        entries = np.empty((len(clusters), betas.size), dtype=np.int64)
        best = np.empty((len(clusters), betas.size))
        for row, cluster in enumerate(clusters):
            state = states[cluster]
            entries[row] = np.maximum(state.below - np.minimum(betas, total), 0)
            best[row] = state.runs[True].plain_solution.table[entries[row]] + state.up
        for column, beta in enumerate(on_complement):
            row = int(np.argmax(best[:, column]))
            if best[row, column] > float(baseline.values[min(beta, total)]):
                pending[(clusters[row], False, True)].append((beta, int(entries[row, column])))
            elif baseline.choice[min(beta, total)] >= 0:
                selected[beta].add(int(baseline.choice[min(beta, total)]))

    for cluster in reversed(cluster_tree.postorder):  # a parent before its children
        for plus in (False, True):
            for complement in (False, True):
                requests = pending.pop((cluster, plus, complement), None)
                if requests:
                    _expand(
                        cluster, plus, complement, requests, states, cluster_tree, selected,
                        pending,
                    )  # fmt: skip
    return selected


def _profiles(
    piece: LocalPiece,
    local: LocalInstance,
    forest: tuple[tuple[int, ...], ...],
    use_splits: bool,
    star_component: int | None,
    leaf_log: np.ndarray,
) -> tuple[ClusterProfile, ClusterProfile | None]:
    """
    The cluster's profile, and its plain profile when ``star_component`` is given.

    The profiles come from the cluster's split table when allowed and available, else from its
    trees; they are the same either way (up to row order and orientation).

    Parameters
    ----------
    piece : LocalPiece
        The cluster's geometry, possibly carrying its split table.
    local : LocalInstance
        The local instance (``I_C^+``, or ``I_C`` at the root cluster), whose leaves are the
        piece's leaves.
    forest : tuple[tuple[int, ...], ...]
        An agreement forest of ``local``.
    use_splits : bool
        Whether the forest is trusted, so that the split table may be used.
    star_component : int | None
        Index of the ``{x_C^*}`` component, to also derive the plain profile of ``I_C``.
    leaf_log : numpy.ndarray
        The piece's per-leaf log constants (see :func:`~forestpd.algo.survival.leaf_logs`).

    Returns
    -------
    tuple[ClusterProfile, ClusterProfile | None]
        The profile of ``local``, and the plain profile (or ``None``).
    """
    if use_splits and piece.split_masks is not None and piece.split_weights is not None:
        return build_profiles_from_splits(
            local.n_leaves, piece.split_masks, piece.split_weights, forest, star_component, leaf_log
        )
    profile = build_profile(local, forest)
    plain = None if star_component is None else plain_profile_from_plus(profile, star_component)
    return profile, plain


def _build_instances(
    piece: LocalPiece,
    trees: tuple[PhyloTree, ...],
    states: dict[int, ClusterState],
    cluster_tree: ClusterTree,
    costs: np.ndarray,
    tree_weights: np.ndarray,
    top: int,
    cap: int,
    down: Sequence[float],
    below: Sequence[int],
    leaf_log: np.ndarray,
    complement: bool,
) -> tuple[LocalInstance, LocalInstance | None]:
    """
    Build ``I_C`` and (for a non-root cluster) ``I_C^+`` from a piece and its solved children.

    ``I_C^+`` is the piece as-is; ``I_C`` is the piece with its parent-direction leaf ``x_C^*``
    dropped. The leaves get their tables as in the manuscript (see :mod:`~forestpd.algo.local`):
    a taxon costs ``c(x)``; a pseudo-leaf ``x_D`` costs nothing (``c'(x_D) = c(X_D)``), takes the
    child's ``PD_D^+`` when selected and ``Omega_D^0`` when not; ``x_C^*`` costs nothing and is
    always selected, its singleton component being forced into every signature.

    Parameters
    ----------
    piece : LocalPiece
        The cluster's local geometry.
    trees : tuple[PhyloTree, ...]
        The trees of ``I_C`` (:func:`plain_trees`).
    states : dict[int, ClusterState]
        Already-solved clusters; every child of this cluster must be present.
    cluster_tree : ClusterTree
        The tree of clusters ``K``.
    costs : numpy.ndarray
        ``int64[n_taxa]``; cost per global taxon id.
    tree_weights : numpy.ndarray
        ``float64[t]``; the per-tree weights ``w_j``.
    top : int
        The last table index before the cluster's cap: ``B`` on DP, ``B'`` on DP'.
    cap : int
        ``c(X_C)``, the cluster's cap: it can neither use nor save more.
    down : Sequence[float]
        ``Omega^0`` per cluster, filled in for the children.
    below : Sequence[int]
        ``c(X_D)`` per cluster.
    leaf_log : numpy.ndarray
        The piece's per-leaf log constants.
    complement : bool
        Whether to build the DP' tables.

    Returns
    -------
    tuple[LocalInstance, LocalInstance | None]
        ``(I_C, I_C^+)``, with ``I_C^+`` ``None`` at the root cluster.
    """
    n_leaves = piece.n_leaves
    if piece.tree_weights is not None:  # merged local trees carry their own weights
        tree_weights = np.asarray(piece.tree_weights, dtype=np.float64)
    budget = min(top, cap)
    width = budget + 1
    local_cost = np.zeros(n_leaves, dtype=np.int64)
    cost_max = np.zeros(n_leaves, dtype=np.int64)
    idle = np.zeros(n_leaves)
    f2 = np.full((n_leaves, width), NEG_INF)
    for leaf in range(n_leaves):
        taxon = piece.taxon_of_leaf[leaf]
        child = piece.child_of_leaf[leaf]
        if taxon >= 0:
            local_cost[leaf] = cost_max[leaf] = int(costs[taxon])
            f2[leaf] = ordinary_f2(local_cost[leaf : leaf + 1], budget, complement)[0]
        elif child >= 0:
            child_solution = states[child].runs[complement].plus_solution
            assert child_solution is not None, "a child cluster always has a PD^+ table"
            child_table = child_solution.table
            if not complement:  # the child's table stops at its own cap
                child_table = _beyond_cap(child_table, width)
            f2[leaf] = pseudo_leaf_f2(child_table, width, top, complement)
            cost_max[leaf] = below[child]
            idle[leaf] = down[child]
        else:  # the artificial x_C^*: free, and always selected (its idle value is never read)
            f2[leaf, 0] = 0.0

    def instance(leaves: int, trees: tuple[PhyloTree, ...]) -> LocalInstance:
        return LocalInstance(
            trees,
            tree_weights,
            f2[:leaves],
            local_cost[:leaves],
            budget,
            leaf_log=leaf_log[:leaves],
            idle=idle[:leaves],
            cost_max=cost_max[:leaves],
            complement=complement,
        )

    if piece.star_leaf < 0:
        return instance(n_leaves, piece.trees), None
    return instance(piece.star_leaf, trees), instance(n_leaves, piece.trees)


def _expand(
    cluster: int,
    plus: bool,
    complement: bool,
    requests: list[tuple[int, int]],
    states: dict[int, ClusterState],
    cluster_tree: ClusterTree,
    selected: dict[int, set[int]],
    pending: dict[tuple[int, bool, bool], list[tuple[int, int]]],
) -> None:
    """
    Expand table entries of one cluster into the taxa they select there.

    Each request ``(key, entry)`` asks for the selection realizing ``PD_C^+[entry]`` (or
    ``PD_C[entry]``) of the set ``key``. Its taxa at this cluster go to ``selected[key]``; entry
    ``b`` on a pseudo-leaf ``x_D`` becomes a request for the child's ``PD_D^+`` entry ``b`` on the
    same axis, added to ``pending``. Requests sharing a signature share its component DP tables
    and budget folds, so each table is built once, however many sets pass through it.

    Parameters
    ----------
    cluster : int
        The cluster whose table entries to expand.
    plus : bool
        Whether the entries are of ``PD_C^+`` rather than ``PD_C``.
    complement : bool
        Whether the entries are of the DP' tables.
    requests : list[tuple[int, int]]
        ``(key, entry)`` pairs.
    states : dict[int, ClusterState]
        All solved clusters.
    cluster_tree : ClusterTree
        The tree of clusters ``K``.
    selected : dict[int, set[int]]
        Global taxon ids per key, added to.
    pending : dict[tuple[int, bool, bool], list[tuple[int, int]]]
        Requests per ``(cluster, plus, complement)``, added to for the child clusters.
    """
    state = states[cluster]
    run = state.runs[complement]
    local = run.plus if plus else run.plain
    profile = state.plus_profile if plus else state.plain_profile
    solution = run.plus_solution if plus else run.plain_solution
    assert local is not None and profile is not None and solution is not None

    by_signature: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for key, entry in requests:
        if entry > local.budget:  # beyond a DP cluster's cap: its best entry within the cap
            entry = int(np.argmax(solution.table[: local.budget + 1]))
        signature = int(solution.best_signature[entry])
        if signature >= 0:
            by_signature[signature].append((key, entry))

    piece = cluster_tree.pieces[cluster]
    comp_saved = component_constants(profile, local)[1] if complement else None
    for signature, group in by_signature.items():
        active = [c for c in range(profile.n_components) if (signature >> c) & 1]
        _, lam_down, lam_up, lam_both = signature_weights(profile, signature)
        min_selected = 2 if len(active) == 1 else 1
        spans = [
            slice(int(profile.node_offset[c]), int(profile.node_offset[c + 1])) for c in active
        ]
        dps = [
            component_dp_table(
                profile.comp_trees[comp],
                lam_down[span],
                lam_up[span],
                lam_both[span],
                profile.slot_survive_down[span],
                profile.slot_survive_up[span],
                local,
                min_selected,
            )
            for comp, span in zip(active, spans)
        ]
        tables = [root_table(dp, profile.comp_trees[comp]) for comp, dp in zip(active, dps)]
        # On DP' the inactive components save their whole cost; the rest is split among the
        # active ones (see local_solver.solve_local).
        shift = 0 if comp_saved is None else inactive_saved(comp_saved, signature)
        targets = [(key, max(entry - shift, 0)) for key, entry in group]
        folds = budget_folds(tables, max(target for _, target in targets))
        for key, target in targets:
            shares = split_budget_with(tables, folds, target)
            if shares is None:  # pragma: no cover - the forward pass guarantees feasibility
                continue
            for comp, span, dp, share in zip(active, spans, dps, shares):
                picks = trace_selection(dp, profile.comp_trees[comp], share)
                for leaf, allocated in picks:
                    taxon = piece.taxon_of_leaf[leaf]
                    child = piece.child_of_leaf[leaf]
                    if taxon >= 0:
                        selected[key].add(taxon)
                    elif child >= 0:
                        pending[(child, True, complement)].append((key, allocated))
                    # the artificial x_C^* stands for a taxon outside this cluster; nothing to add


__all__ = [
    "solve",
    "Solution",
    "ClusterState",
    "TableRun",
    "NEG_INF",
    "choose_tables",
    "cluster_forest",
    "plain_trees",
]
