"""
The MaxMPD instance.

A :class:`MaxMPDInstance` holds the trees, per-tree weights, taxon costs and survival probabilities
of one MaxMPD instance, parsed and validated once. Its :meth:`~MaxMPDInstance.optimize` method
solves it for a given budget and :meth:`~MaxMPDInstance.evaluate_pd` evaluates the objective of any
taxon set, both without re-reading the
trees. Taxa are indexed by the ids of a :class:`~forestpd.core.taxa.TaxonRegistry`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .algo.solver import choose_tables, cluster_forest, plain_trees, solve
from .cluster import ClusterTree, cluster_reduction
from .io.convert import MissingBranchLength, Trees, coerce_trees
from .io.newick import ROOT_LABEL
from ._evaluate import SplitTable
from .core.taxa import TaxonRegistry
from .core.tree import PhyloTree
from ._validate import (
    taxon_ids,
    costs_to_array,
    survival_to_array,
    validate_budget,
    validate_uniform,
    weights_to_array,
)
from .exceptions import ForestPDInputError
from .maf import HeuristicMAF, MAFComputer


@dataclass(frozen=True)
class MaxMPDResult:
    """
    Result of a MaxMPD optimization.

    Attributes
    ----------
    value : float
        The optimal objective value: the expected ``sum_j w_j PD_{T_j}`` of the surviving taxa when
        ``S`` is protected (``sum_j w_j PD_{T_j}(S)`` without survival probabilities).
    taxa : set[str] | None
        An optimal taxon-label set, or ``None`` when ``return_taxa`` was not requested.
    meta : dict
        Diagnostics:

        - ``"frontier"``: the optimal value ``PD(beta)`` for the budgets ``beta = 0..B``, as
          ``float64[min(B, C)+1]`` with ``C`` the total cost of all taxa (a larger budget buys
          nothing more, so its value is the last entry). The budgets the computed tables do not
          cover are ``NaN``: see ``"tables"``;
        - ``"tables"``: which of the manuscript's tables were computed. ``"DP"`` (budget
          allocated, up to the largest budget) covers every budget up to it; ``"DP'"`` (budget
          saved, up to ``C`` minus the smallest budget) covers every budget from the smallest one
          on; ``"both"`` covers the requested budgets from one or the other. The cheapest is
          chosen: DP' when the budgets are more than half the total cost;
        - ``"budget"``, ``"n_trees"``, ``"n_taxa"``, ``"maf"`` (the MAF stage used);
        - ``"n_clusters"``, ``"cluster_sizes"``, ``"maf_sizes"``, ``"max_maf_size"``,
          ``"n_signatures"``: as in :class:`MaxMPDStructure` (the sizes as lists);
        - timings in seconds: ``"parse_seconds"`` (reading and validating the trees, done once
          when the instance was built, so the same in every result of that instance),
          ``"reduction_seconds"`` (about zero once the instance has cached its reduction),
          ``"maf_seconds"`` and ``"dp_seconds"``, whose sum is ``"solve_seconds"``.
    """

    value: float
    taxa: set[str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MaxMPDStructure:
    """
    The structure that determines the running time, from :meth:`MaxMPDInstance.structure`.

    The per-cluster tuples follow the solver's processing order (children before parents), the
    order of ``meta["maf_sizes"]`` in an optimization result.

    Attributes
    ----------
    n_clusters : int
        Number of clusters of the common-cluster reduction.
    cluster_sizes : tuple[int, ...]
        Per cluster, the leaves of its local instance ``I_C``: its own taxa and one pseudo-leaf
        per child cluster.
    maf_sizes : tuple[int, ...]
        Per cluster, the size ``k`` of its agreement forest.
    max_maf_size : int
        The largest ``k``: the exponent of the running time ``O(2^k * n * B^2)``.
    n_signatures : int
        ``sum_C 2^{k_C}``, counting the ``PD_C^+`` tables of non-root clusters as well: the number
        of signatures the DP visits at most (a small budget skips the larger ones).
    reduction_seconds : float
        Time of the cluster reduction (about zero when the instance had it cached).
    maf_seconds : float
        Time of the MAF stage.
    """

    n_clusters: int
    cluster_sizes: tuple[int, ...]
    maf_sizes: tuple[int, ...]
    max_maf_size: int
    n_signatures: int
    reduction_seconds: float
    maf_seconds: float


class MaxMPDInstance:
    """
    A MaxMPD instance: phylogenetic trees on a shared taxon set, tree weights, taxon costs and
    survival probabilities.

    The objective of a protected set ``S`` is the expected weighted PD of the taxa that survive:
    protected taxa survive surely, and every other taxon ``x`` independently with probability
    ``a(x)``. With all ``a(x) = 0`` (the default) it is the plain weighted PD of ``S``.

    The trees are parsed and validated once; :meth:`optimize` and :meth:`evaluate_pd` can then be
    called any number of times.

    Parameters
    ----------
    trees : str | os.PathLike | Sequence
        The path of a file holding one Newick tree per line, or a list of Newick strings and/or
        PhyloZoo ``SemiDirectedPhyNetwork`` / ``DirectedPhyNetwork`` trees (with the ``phylozoo``
        extra). Every Newick tree ends with ``;``. All trees must carry the same taxon set.
    costs : Mapping[str, int] | None, optional
        Integer cost per taxon label, at least 1. ``None`` means unit costs; missing labels cost 1.
    weights : Sequence[float] | Mapping[int, float] | None, optional
        Per-tree weight ``w_j``, in the order of the trees (the list, or the file), or keyed by
        tree index. ``None`` means all ``1.0``.
    survival : Mapping[str, float] | None, optional
        Probability ``a(x)`` in ``[0, 1)`` that taxon ``x`` survives when it is *not* protected.
        ``None`` means ``0`` for every taxon; missing labels have ``0``.
    missing_branch_length : str | float, default "error"
        Policy for edges without a branch length: ``"error"``, ``"zero"``, or a numeric default.
    rooted : bool, default False
        Use rooted PD: the PD of a set includes the path from its taxa to the root of each tree,
        and the stem above the root (the root's own branch length) when the set is non-empty.
        By default PD is unrooted and the rooting of the input is ignored.

    Attributes
    ----------
    trees : tuple[PhyloTree, ...]
        The trees, with leaf id == taxon id. With ``rooted=True`` they carry one extra leaf at the
        root, which never appears in results.
    registry : TaxonRegistry
        Taxon label <-> id mapping.
    costs : numpy.ndarray
        ``int64[n]``; cost per taxon id (by ``registry``; with ``rooted=True`` this includes the
        hidden root leaf, so pair it with ``registry.labels``, not with :attr:`taxa`).
    weights : numpy.ndarray
        ``float64[t]``; weight per tree.
    survival : numpy.ndarray
        ``float64[n]``; survival probability ``a(x)`` per taxon id.

    Raises
    ------
    ForestPDInputError
        If any input is malformed, or the trees do not share one taxon set.

    Warns
    -----
    ForestPDWarning
        If the trees look rooted (a ``[&R]`` comment, a PhyloZoo ``DirectedPhyNetwork``, or a
        binary root in every tree) but ``rooted`` is False, or a tree is marked ``[&U]`` while
        ``rooted`` is True.

    Examples
    --------
    >>> import forestpd as fpd
    >>> inst = fpd.MaxMPDInstance(["(A:1,B:1,(C:1,D:1):2);", "(A:1,C:1,(B:1,D:1):2);"])
    >>> inst.optimize(budget=2).value
    8.0
    >>> inst.evaluate_pd({"A", "B"})
    6.0
    >>> rooted = fpd.MaxMPDInstance(["((A:1,B:1):1,(C:1,D:1):1);"], rooted=True)
    >>> rooted.evaluate_pd({"A", "B"}), rooted.evaluate_pd({"A"})   # paths to the root count
    (3.0, 2.0)
    >>> risky = fpd.MaxMPDInstance(["(A:1,B:1,(C:1,D:1):2);"], survival={"C": 0.5, "D": 0.5})
    >>> risky.evaluate_pd({"A", "B"})    # 1 + 1, plus 2 * 3/4 (C or D survives) + 1/2 + 1/2
    4.5
    """

    __slots__ = (
        "trees",
        "registry",
        "costs",
        "weights",
        "survival",
        "_log_loss",
        "_root_leaf",
        "_root_offset",
        "_stem",
        "_cluster_tree",
        "_parse_seconds",
        "_split_table",
    )

    def __init__(
        self,
        trees: Trees,
        costs: Mapping[str, int] | None = None,
        weights: Sequence[float] | Mapping[int, float] | None = None,
        survival: Mapping[str, float] | None = None,
        *,
        missing_branch_length: MissingBranchLength = "error",
        rooted: bool = False,
    ) -> None:
        started = time.perf_counter()
        parsed, registry, stems = coerce_trees(trees, missing_branch_length, rooted)
        self._set(parsed, registry, costs, weights, survival)
        if rooted:
            self._attach_root(stems)
        self._parse_seconds = time.perf_counter() - started

    @classmethod
    def from_phylotrees(
        cls,
        trees: Sequence[PhyloTree],
        registry: TaxonRegistry,
        costs: Mapping[str, int] | Sequence[int] | None = None,
        weights: Sequence[float] | Mapping[int, float] | None = None,
        survival: Mapping[str, float] | Sequence[float] | None = None,
    ) -> MaxMPDInstance:
        """
        Build an instance from trees already numbered against ``registry``.

        Parameters
        ----------
        trees : Sequence[PhyloTree]
            The trees, with leaf id == taxon id of ``registry``.
        registry : TaxonRegistry
            The taxon registry the leaf ids refer to.
        costs : Mapping[str, int] | Sequence[int] | None, optional
            Cost per taxon label, or a sequence of costs indexed by taxon id. ``None`` means unit
            costs.
        weights : Sequence[float] | Mapping[int, float] | None, optional
            Per-tree weight ``w_j``. ``None`` means all ``1.0``.
        survival : Mapping[str, float] | Sequence[float] | None, optional
            Survival probability per taxon label, or a sequence indexed by taxon id. ``None``
            means all ``0``.

        Returns
        -------
        MaxMPDInstance
            The instance.

        Raises
        ------
        ForestPDInputError
            If a cost, weight or survival probability is invalid, the trees do not share one
            taxon set, or a tree has an internal vertex of degree 2 (suppress it first with
            :func:`~forestpd.core.tree.suppress_degree_two`).
        """
        instance = cls.__new__(cls)
        started = time.perf_counter()
        instance._set(tuple(trees), registry, costs, weights, survival)
        instance._parse_seconds = time.perf_counter() - started
        return instance

    def _set(
        self,
        trees: tuple[PhyloTree, ...],
        registry: TaxonRegistry,
        costs: Mapping[str, int] | Sequence[int] | None,
        weights: Sequence[float] | Mapping[int, float] | None,
        survival: Mapping[str, float] | Sequence[float] | None = None,
    ) -> None:
        """
        Validate and store the instance data.

        Parameters
        ----------
        trees : tuple[PhyloTree, ...]
            The trees, numbered against ``registry``.
        registry : TaxonRegistry
            The taxon registry.
        costs : Mapping[str, int] | Sequence[int] | None
            Costs by label or by taxon id.
        weights : Sequence[float] | Mapping[int, float] | None
            Per-tree weights.
        survival : Mapping[str, float] | Sequence[float] | None
            Survival probabilities by label or by taxon id.

        Raises
        ------
        ForestPDInputError
            If a cost, weight or survival probability is invalid, or the trees do not share one
            taxon set.
        """
        if not trees:
            raise ForestPDInputError("At least one tree is required.")
        validate_uniform(trees, registry)
        self.trees = trees
        self.registry = registry
        self.costs = costs_to_array(registry, costs)
        self.weights = weights_to_array(weights, len(trees))
        self.survival = survival_to_array(registry, survival)
        self._log_loss = np.log1p(-self.survival)
        self._root_leaf = -1
        self._root_offset = 0.0
        self._stem = 0.0
        self._cluster_tree: ClusterTree | None = None  # budget-independent; built on first use
        self._split_table: SplitTable | None = None  # for evaluate_pd; built on first use

    def _attach_root(self, stems: Sequence[float]) -> None:
        """
        Set up rooted PD by forcing the root leaf into every solution.

        The root leaf's edge gets a weight ``M`` larger than the total length of any tree, so every
        optimal set of at least one taxon contains it. Its cost is 1, so the solver is run with one
        extra unit of budget, and ``M * sum(w)`` is subtracted from the result. The root leaf never
        survives on its own (``a = 0``): it only marks the root.

        Parameters
        ----------
        stems : Sequence[float]
            The stem length of every tree.
        """
        self._root_leaf = self.registry.id_of(ROOT_LABEL)
        self.survival[self._root_leaf] = 0.0
        self._log_loss[self._root_leaf] = 0.0
        big = 1.0 + max(tree.total_weight() for tree in self.trees)
        for tree in self.trees:
            rows = np.nonzero((tree.edges == self._root_leaf).any(axis=1))[0]
            tree.weight[int(rows[0])] = big
        self._root_offset = big * float(self.weights.sum())
        self._stem = float(np.dot(self.weights, np.asarray(stems, dtype=np.float64)))

    @property
    def n_trees(self) -> int:
        """Number of trees."""
        return len(self.trees)

    @property
    def rooted(self) -> bool:
        """Whether PD is rooted."""
        return self._root_leaf >= 0

    @property
    def n_taxa(self) -> int:
        """Number of taxa."""
        return self.registry.n - (1 if self.rooted else 0)

    @property
    def taxa(self) -> tuple[str, ...]:
        """Taxon labels, in id order."""
        labels = self.registry.labels
        return tuple(label for label in labels if label != ROOT_LABEL) if self.rooted else labels

    def optimize(
        self,
        budget: int,
        *,
        return_taxa: bool = False,
        maf: MAFComputer | None = None,
    ) -> MaxMPDResult:
        """
        Find a taxon set ``S`` of cost at most ``budget`` maximizing the objective.

        The objective is that of :meth:`evaluate_pd`.

        Parameters
        ----------
        budget : int
            Total integer budget ``B`` (``>= 0``).
        return_taxa : bool, default False
            Also reconstruct an optimal taxon set.
        maf : MAFComputer | None, optional
            Agreement-forest stage. ``None`` (default) uses :class:`~forestpd.maf.HeuristicMAF`.

        Returns
        -------
        MaxMPDResult
            The optimal value, optionally an optimal taxon set, and diagnostics.

        Raises
        ------
        ForestPDInputError
            If ``budget`` is not a non-negative integer.
        ForestPDInputError
            If the MAF stage does not return an agreement forest (a partition of each cluster's
            leaves), or ``maf`` is not a MAF computer, or a forest has more than 63 components.
        ForestPDValueError
            From a MAF stage that cannot handle a cluster (e.g. maffe and a polytomy).
        """
        validate_budget(budget)
        result = self._solve([budget], return_taxa, maf)[0]
        return result

    def optimize_many(
        self,
        budgets: Iterable[int],
        *,
        return_taxa: bool = False,
        maf: MAFComputer | None = None,
    ) -> list[MaxMPDResult]:
        """
        :meth:`optimize` for several budgets at once, from a single solve.

        The DP tables computed for the largest budget contain the answer for every smaller one,
        so this costs one :meth:`optimize` call plus one witness reconstruction per budget. When
        the budgets are large, the complementary tables (DP', on the budget left unspent) are
        cheaper, or both tables split between small and large budgets (``meta["tables"]``).

        Parameters
        ----------
        budgets : Iterable[int]
            The budgets (each ``>= 0``).
        return_taxa : bool, default False
            Also reconstruct an optimal taxon set for each budget.
        maf : MAFComputer | None, optional
            Agreement-forest stage. ``None`` (default) uses :class:`~forestpd.maf.HeuristicMAF`.

        Returns
        -------
        list[MaxMPDResult]
            One result per budget, in the given order.

        Raises
        ------
        ForestPDInputError
            If a budget is not a non-negative integer.
        ForestPDInputError
            If the MAF stage does not return an agreement forest (a partition of each cluster's
            leaves), or ``maf`` is not a MAF computer, or a forest has more than 63 components.
        ForestPDValueError
            From a MAF stage that cannot handle a cluster (e.g. maffe and a polytomy).

        Examples
        --------
        >>> import forestpd as fpd
        >>> inst = fpd.MaxMPDInstance(["(A:1,B:1,(C:1,D:1):2);", "(A:1,C:1,(B:1,D:1):2);"])
        >>> [r.value for r in inst.optimize_many([1, 2, 3])]
        [0.0, 8.0, 10.0]
        """
        budgets = list(budgets)
        for budget in budgets:
            validate_budget(budget)
        return self._solve(budgets, return_taxa, maf) if budgets else []

    def _solve(
        self, budgets: list[int], return_taxa: bool, maf: MAFComputer | None
    ) -> list[MaxMPDResult]:
        """One solve covering every budget in ``budgets``, read out at each of them."""
        # Costs with a common divisor g are solved in units of g: budget b buys exactly what
        # floor(b / g) units do, and the DP's tables get g times shorter.
        real = np.delete(self.costs, self._root_leaf) if self.rooted else self.costs
        unit = int(np.gcd.reduce(real)) if real.size else 1
        costs = self.costs // unit
        # In rooted mode the root leaf costs one extra unit (see _attach_root).
        shift = 1 if self.rooted else 0
        if self.rooted:
            costs[self._root_leaf] = 1
        total_cost = int(costs.sum())
        internal = [min(b // unit + shift, total_cost) for b in budgets]
        # DP up to the largest budget, DP' from the smallest, or both: whichever is cheapest.
        dp_budget, complement_from = choose_tables(internal, total_cost)
        maf_computer = _maf_computer(maf)
        cluster_tree, reduction_seconds = self._reduction()
        solution = solve(
            cluster_tree,
            maf_computer,
            costs,
            self.weights,
            dp_budget,
            witness_budgets=sorted(set(internal)) if return_taxa else (),
            log_loss=self._log_loss,
            complement_from=complement_from,
        )
        frontier = solution.meta["frontier"]
        from_empty = np.zeros(frontier.size, dtype=bool)
        if self.rooted:
            frontier, from_empty = self._rooted_frontier(frontier, solution.single_values, costs)
        # Back to the original budgets 0 .. min(max budget, total cost of the taxa).
        span = np.arange(min(max(budgets), int(real.sum())) + 1) // unit
        frontier, from_empty = frontier[span], from_empty[span]
        # The frontier ends at the total cost of all taxa; a larger budget buys nothing more, so it
        # reads the last entry (and is not padded: a huge budget costs no memory).
        results = []
        for budget, at in zip(budgets, internal):
            # Every result gets its own copies, so changing one result's meta leaves the others.
            meta = {
                key: value.copy() if isinstance(value, (list, np.ndarray)) else value
                for key, value in solution.meta.items()
            }
            meta.update(
                n_trees=self.n_trees,
                n_taxa=self.n_taxa,
                maf=type(maf_computer).__name__,
                parse_seconds=self._parse_seconds,
                reduction_seconds=reduction_seconds,
                budget=budget,
                frontier=frontier[: budget + 1].copy(),
            )
            index = min(budget, frontier.size - 1)
            value = float(frontier[index])
            taxa = None
            if return_taxa:
                witness = solution.witnesses[at]
                if self.rooted:
                    witness = set() if from_empty[index] else witness - {self._root_leaf}
                taxa = self.registry.labels_of(witness)
            results.append(MaxMPDResult(value=value, taxa=taxa, meta=meta))
        return results

    def structure(self, maf: MAFComputer | None = None) -> MaxMPDStructure:
        """
        The clusters and agreement-forest sizes, without solving.

        Runs the first half of :meth:`optimize`: the cluster reduction (cached on the instance, so
        a later :meth:`optimize` does not repeat it) and the MAF stage on every cluster, on the
        same trees and with the same rooting as the solver. The values equal the corresponding
        entries of ``optimize(...).meta``. Use it to check how hard an instance is before solving.

        Parameters
        ----------
        maf : MAFComputer | None, optional
            Agreement-forest stage. ``None`` (default) uses :class:`~forestpd.maf.HeuristicMAF`.

        Returns
        -------
        MaxMPDStructure
            Cluster count and sizes, the MAF size per cluster, and the signature count.

        Raises
        ------
        ForestPDInputError
            If the MAF stage does not return an agreement forest (a partition of each cluster's
            leaves), or ``maf`` is not a MAF computer, or a forest has more than 63 components.
        ForestPDValueError
            From a MAF stage that cannot handle a cluster (e.g. maffe and a polytomy).

        Examples
        --------
        >>> import forestpd as fpd
        >>> s = fpd.MaxMPDInstance(["(A:1,B:1,(C:1,D:1):2);", "(A:1,C:1,(B:1,D:1):2);"]).structure()
        >>> s.n_clusters, s.maf_sizes, s.max_maf_size
        (1, (2,), 2)
        """
        maf_computer = _maf_computer(maf)
        cluster_tree, reduction_seconds = self._reduction()
        sizes: list[int] = []
        maf_sizes: list[int] = []
        n_signatures = 0
        maf_seconds = 0.0
        for cluster in cluster_tree.postorder:
            piece = cluster_tree.pieces[cluster]
            trees = plain_trees(piece)
            started = time.perf_counter()
            forest = cluster_forest(maf_computer, piece, trees)
            maf_seconds += time.perf_counter() - started
            sizes.append(trees[0].n_leaves)
            maf_sizes.append(len(forest))
            n_signatures += (1 << len(forest)) * (1 if piece.star_leaf < 0 else 2)
        return MaxMPDStructure(
            n_clusters=cluster_tree.n_clusters,
            cluster_sizes=tuple(sizes),
            maf_sizes=tuple(maf_sizes),
            max_maf_size=max(maf_sizes) if maf_sizes else 0,
            n_signatures=n_signatures,
            reduction_seconds=reduction_seconds,
            maf_seconds=maf_seconds,
        )

    def _reduction(self) -> tuple[ClusterTree, float]:
        """The cached cluster reduction (built on first use), and the seconds spent building it."""
        started = time.perf_counter()
        if self._cluster_tree is None:
            self._cluster_tree = cluster_reduction(self.trees, weights=self.weights)
        return self._cluster_tree, time.perf_counter() - started

    def _rooted_frontier(
        self, internal: np.ndarray, singles: np.ndarray, costs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Translate the solver's frontier (over budget + 1, with the root leaf) to rooted PD.

        Parameters
        ----------
        internal : numpy.ndarray
            ``PD(beta)`` of the internal problem for ``beta = 0..B+1``, ``NaN`` where no table
            covers it.
        singles : numpy.ndarray
            The internal problem's value of each single taxon; that of the root leaf is the
            nothing-protected value plus the root-leaf edge.
        costs : numpy.ndarray
            The costs the solver was run with (in units of their common divisor).

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Rooted ``PD(beta)`` for ``beta = 0..B``, and per budget whether protecting nothing is
            optimal: until the cheapest taxon is affordable, and wherever it is not beaten. Else
            the value is the internal one minus the root-leaf offset, plus the weighted stem.
        """
        real_costs = np.delete(costs, self._root_leaf)
        cheapest = int(real_costs.min()) if real_costs.size else 1
        # With nothing protected, the root leaf's edge and the stem count only if some taxon
        # survives, which happens with probability 1 - prod (1 - a(x)).
        survives = -math.expm1(float(self._log_loss.sum()))
        nothing = float(singles[self._root_leaf]) + (self._stem - self._root_offset) * survives
        frontier = internal[1:] - self._root_offset + self._stem
        low = frontier[: min(cheapest, frontier.size)]  # (budgets no table covered stay NaN)
        low[~np.isnan(low)] = -np.inf
        from_empty = frontier <= nothing
        return np.where(from_empty, nothing, frontier), from_empty

    def evaluate_pd(self, taxa: Iterable[str]) -> float:
        """
        Objective value of protecting a taxon set.

        The expected ``sum_j w_j PD_{T_j}`` of the surviving taxa: the given taxa survive surely,
        every other taxon ``x`` independently with probability ``a(x)``. Without survival
        probabilities this is ``sum_j w_j PD_{T_j}(taxa)``.

        The first call groups the edges of all trees by split once (compiled, linear in the size of
        the trees); every call is then linear in the size of that split table, independent of the
        number of trees.

        Parameters
        ----------
        taxa : Iterable[str]
            Taxon labels.

        Returns
        -------
        float
            The (expected) weighted sum of phylogenetic diversities over all trees.

        Raises
        ------
        ForestPDInputError
            If a label is unknown, or ``taxa`` is a single string rather than a collection.
        """
        ids = taxon_ids(self.registry, taxa)
        selected = bool(ids)
        if self.rooted:
            ids.add(self._root_leaf)
        if self._split_table is None:
            self._split_table = SplitTable(self.trees, self.weights, self._log_loss)
        value = self._split_table.value(sorted(ids))
        if not self.rooted:
            return value
        # The root leaf's edge (weight M) and the stem count iff some real taxon survives.
        survives = 1.0 if selected else -math.expm1(float(self._log_loss.sum()))
        return value + (self._stem - self._root_offset) * survives

    def __repr__(self) -> str:
        return f"MaxMPDInstance(n_trees={self.n_trees}, n_taxa={self.n_taxa})"


def _maf_computer(maf: MAFComputer | None) -> MAFComputer:
    """The MAF stage to use: ``maf``, or the heuristic for ``None``."""
    if maf is None:
        return HeuristicMAF()
    if isinstance(maf, type) or not callable(getattr(maf, "compute", None)):
        raise ForestPDInputError(
            "maf must be a MAF computer instance (an object with a compute method, e.g. "
            f"HeuristicMAF()), got {maf!r}"
        )
    return maf


__all__ = ["MaxMPDInstance", "MaxMPDResult", "MaxMPDStructure", "TaxonRegistry"]
