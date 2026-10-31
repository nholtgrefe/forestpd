"""
Baselines used to cross-check the real pipeline.

Neither belongs in the package: each is always correct and never fast, which is exactly what makes
it a reference. The optimum must not move when the cluster reduction is skipped (``"trivial"``:
the whole instance as one cluster), or when the heuristic forest is replaced by singletons, or computed
outside the trusted split-table path.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from forestpd.algo.solver import solve
from forestpd.instance import MaxMPDResult
from forestpd.cluster import single_cluster
from forestpd.core.tree import PhyloTree
from forestpd.maf import Forest, HeuristicMAF, MAFComputer, singleton_forest

from .random_instances import Case


class TrivialMAF:
    """One component per leaf: always a valid agreement forest, never a small one."""

    trusted = True

    def compute(self, trees: Sequence[PhyloTree]) -> Forest:
        """Return the all-singletons forest."""
        return singleton_forest(trees[0].n_leaves)


class UntrustedMAF:
    """The heuristic without ``trusted``: profiles are then built from the trees, not the splits."""

    def compute(self, trees: Sequence[PhyloTree]) -> Forest:
        """Return the heuristic forest."""
        return HeuristicMAF().compute(trees)


#: Short names the tests parametrize over -> fresh MAF computers.
MAFS: dict[str, Callable[[], MAFComputer]] = {
    "heuristic": HeuristicMAF,
    "singleton": TrivialMAF,
    "untrusted": UntrustedMAF,
}

#: Cluster-reduction modes: the package's reduction, or none at all.
REDUCTIONS = ("common", "trivial")


def optimize(
    case: Case,
    reduction: str = "common",
    maf: str = "heuristic",
    return_taxa: bool = False,
) -> MaxMPDResult:
    """
    Solve ``instance`` with the named stages.

    ``"common"`` goes through :meth:`MaxMPDInstance.optimize`; ``"trivial"`` skips the
    cluster reduction and hands the solver the whole instance as a single cluster.
    """
    instance, budget = case
    computer = MAFS[maf]()
    if reduction == "common":
        return instance.optimize(budget, maf=computer, return_taxa=return_taxa)
    if reduction != "trivial":
        raise ValueError(f"unknown reduction {reduction!r}")
    working = min(budget, int(instance.costs.sum()))
    solution = solve(
        single_cluster(instance.trees),
        computer,
        instance.costs,
        instance.weights,
        working,
        witness_budgets=[working] if return_taxa else (),
        log_loss=np.log1p(-instance.survival),
    )
    taxa = instance.registry.labels_of(solution.witnesses[working]) if return_taxa else None
    return MaxMPDResult(value=solution.value, taxa=taxa, meta=dict(solution.meta))
