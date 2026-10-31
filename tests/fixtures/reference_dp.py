"""
Direct evaluation of the paper's definitions, for testing the DP against.

These functions read straight from Definition "two-sided-PD" and the definitely-chosen
characterization, with no dynamic programming and no shared code with the optimizer. They are
exponential in the component size, which is fine on the small components tests use.
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np

from forestpd.algo.local import LocalInstance
from forestpd.algo.profile import ClusterProfile, ComponentTree


def subtree_leaves(comp_tree: ComponentTree, node: int) -> set[int]:
    """Local leaf ids beneath ``node`` in ``R_i``."""
    out: set[int] = set()
    stack = [node]
    while stack:
        v = stack.pop()
        leaf = int(comp_tree.leaf[v])
        if leaf >= 0:
            out.add(leaf)
        for child in (int(comp_tree.left[v]), int(comp_tree.right[v])):
            if child >= 0:
                stack.append(child)
    return out


def two_sided_pd(
    comp_tree: ComponentTree,
    lam_down: np.ndarray,
    lam_up: np.ndarray,
    lam_both: np.ndarray,
    selected: set[int],
    survive_down: np.ndarray | None = None,
    survive_up: np.ndarray | None = None,
) -> float:
    """
    ``PD^(2)`` of a leaf subset, straight from Definition "two-sided-PD".

    Every edge contributes ``lam_down alpha_down + lam_up alpha_up + lam_both alpha_down alpha_up``,
    where ``alpha`` of a side is ``1`` if it holds a selected leaf and otherwise ``pi_{X_i}``, the
    survival probability of the component's taxa on it (``0`` when not given).
    """
    everything = subtree_leaves(comp_tree, comp_tree.root)
    total = 0.0
    for node in range(comp_tree.n_nodes):
        if node == comp_tree.root:
            continue
        below = subtree_leaves(comp_tree, node)
        pi_down = 0.0 if survive_down is None else float(survive_down[node])
        pi_up = 0.0 if survive_up is None else float(survive_up[node])
        a_down = 1.0 if selected & below else pi_down
        a_up = 1.0 if selected & (everything - below) else pi_up
        total += lam_down[node] * a_down + lam_up[node] * a_up + lam_both[node] * a_down * a_up
    return total


def best_component_value(
    comp_tree: ComponentTree,
    lam_down: np.ndarray,
    lam_up: np.ndarray,
    lam_both: np.ndarray,
    survive_down: np.ndarray,
    survive_up: np.ndarray,
    local: LocalInstance,
    min_selected: int,
) -> np.ndarray:
    """
    ``PD_i^{(S)}`` by exhaustive enumeration of leaf subsets and table entries per selected leaf.

    The reference that :func:`forestpd.algo.component_dp.component_table` must reproduce, with the
    same arguments. A selected leaf takes some entry of its ``f2`` row, an unselected one its
    ``idle`` value. On DP the entries are allocated budgets and the table holds the best value with
    at most ``beta`` allocated; on DP' they are saved amounts, an unselected leaf saves its upper
    bound cost, and the table holds the best value with at least ``beta`` saved.
    """
    leaves = sorted(subtree_leaves(comp_tree, comp_tree.root))
    width = local.budget + 1
    idle = local.idle_or_zero
    cost_max = local.cost_max_or_cost
    table = np.full(width, -np.inf)
    for size in range(min_selected, len(leaves) + 1):
        for subset in combinations(leaves, size):
            chosen = set(subset)
            base = two_sided_pd(
                comp_tree, lam_down, lam_up, lam_both, chosen, survive_down, survive_up
            )
            base += sum(float(idle[leaf]) for leaf in leaves if leaf not in chosen)
            unselected_saving = sum(int(cost_max[leaf]) for leaf in leaves if leaf not in chosen)
            for entries in product(range(width), repeat=size):
                extra = sum(float(local.f2[leaf, at]) for leaf, at in zip(subset, entries))
                if extra == -np.inf:
                    continue
                value = base + extra
                if local.complement:
                    reach = min(sum(entries) + unselected_saving, width - 1)
                    table[: reach + 1] = np.maximum(table[: reach + 1], value)
                elif sum(entries) < width:
                    table[sum(entries) :] = np.maximum(table[sum(entries) :], value)
    return table


def definitely_chosen_weight(profile: ClusterProfile, signature: int) -> float:
    """``d_S`` read directly off the characterization, without the vectorized path."""
    mask = np.uint64(signature)
    total = 0.0
    for e in range(profile.edge_weight.shape[0]):
        if (profile.edge_only_below[e] & mask) and (profile.edge_only_above[e] & mask):
            total += float(profile.edge_weight[e])
    return total


__all__ = [
    "subtree_leaves",
    "two_sided_pd",
    "best_component_value",
    "definitely_chosen_weight",
]
