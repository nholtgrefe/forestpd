"""
Reconstructing an optimal taxon set.

The paper only computes optimal values. To recover a taxon set, the forward pass stores the best
signature per budget and cluster; everything else is recomputed on demand along the optimal path.
This avoids storing choices inside the ``2^k`` signature loop, which would need as much memory as
the DP itself, and costs one extra DP per component on the optimal path. When several sets are
reconstructed (one per budget), the solver builds each component's table once and traces every set
in it (:func:`trace_selection`).
"""

from __future__ import annotations

import numpy as np

from .component_dp import ComponentDP, component_dp_table
from .local import NEG_INF, LocalInstance
from .profile import ComponentTree

#: Relative slack allowed when matching a stored optimum against a recomputed one. It is relative
#: (scaled by the magnitude of the value, at least 1) because recomputing in a different order can
#: differ in the last bits, and with large branch lengths (e.g. in generations) one unit in the last
#: place already exceeds any fixed absolute slack.
TOLERANCE = 1e-9


def matches(value: float, goal: float) -> bool:
    """Whether a recomputed value equals a stored one up to :data:`TOLERANCE` (relative)."""
    return abs(value - goal) <= TOLERANCE * max(1.0, abs(goal))


def first_match(values: np.ndarray, goal: float) -> int:
    """
    The first index whose value :func:`matches` ``goal`` and is not ``NEG_INF``, or ``-1``.

    The vectorized form of scanning candidates in order with :func:`matches`, so it picks the same
    candidate (and the reconstruction the same set) as such a scan.
    """
    if values.size == 0:
        return -1
    hits = (values != NEG_INF) & (np.abs(values - goal) <= TOLERANCE * max(1.0, abs(goal)))
    index = int(hits.argmax())  # the first True, or 0 when there is none
    return index if hits[index] else -1


def component_selection(
    comp_tree: ComponentTree,
    lam_down: np.ndarray,
    lam_up: np.ndarray,
    lam_both: np.ndarray,
    survive_down: np.ndarray,
    survive_up: np.ndarray,
    local: LocalInstance,
    min_selected: int,
    target_budget: int,
) -> list[tuple[int, int]]:
    """
    Recover which leaves of one component are selected, and with what budget.

    Recomputes the component DP and walks it downwards from the occupied root (``s_up = 0``),
    following a regime that reproduces each stored value.

    Parameters
    ----------
    comp_tree : ComponentTree
        The rooted binary ``R_i``.
    lam_down, lam_up, lam_both : numpy.ndarray
        The two-sided weights of this component's edges, indexed by ``R_i`` node.
    survive_down, survive_up : numpy.ndarray
        ``pi_{X_i}`` below and above each edge, indexed by ``R_i`` node.
    local : LocalInstance
        The local instance.
    min_selected : int
        Minimum number of leaves the component must contribute (1 or 2).
    target_budget : int
        The entry ``beta`` of the component's table in the optimum being reconstructed.

    Returns
    -------
    list[tuple[int, int]]
        ``(local leaf id, entry)`` pairs: per selected leaf, the entry of its ``f2`` row used.

    Raises
    ------
    AssertionError
        If no regime reproduces a stored DP value (cannot happen for a consistent table).
    """
    dp = component_dp_table(
        comp_tree, lam_down, lam_up, lam_both, survive_down, survive_up, local, min_selected
    )
    return trace_selection(dp, comp_tree, target_budget)


def trace_selection(
    dp: ComponentDP, comp_tree: ComponentTree, target_budget: int
) -> list[tuple[int, int]]:
    """
    Walk a component's DP table downwards from its occupied root, as :func:`component_selection`.

    Parameters
    ----------
    dp : ComponentDP
        The table from :func:`~forestpd.algo.component_dp.component_dp_table`.
    comp_tree : ComponentTree
        The rooted binary ``R_i``.
    target_budget : int
        The entry ``beta`` of the component's table.

    Returns
    -------
    list[tuple[int, int]]
        ``(local leaf id, entry)`` pairs.

    Raises
    ------
    AssertionError
        If no regime reproduces a stored DP value (cannot happen for a consistent table).
    """
    picked: list[tuple[int, int]] = []
    # (node, s_up, beta); only occupied states are walked, an unselected subtree selects nothing.
    stack = [(comp_tree.root, 0, target_budget)]
    while stack:
        node, s, beta = stack.pop()
        leaf = int(comp_tree.leaf[node])
        if leaf >= 0:
            picked.append((leaf, beta))
            continue
        w1 = int(comp_tree.left[node])
        w2 = int(comp_tree.right[node])
        frames = _matching_regime(dp, w1, w2, s, beta, dp.occ[node, s, beta])
        if frames is None:  # pragma: no cover - the forward pass guarantees a matching regime
            raise AssertionError("no DP regime reproduces the stored optimum")
        stack.extend(frames)
    return picked


def _matching_regime(
    dp: ComponentDP, w1: int, w2: int, s: int, beta: int, goal: float
) -> list[tuple[int, int, int]] | None:
    """
    The occupied child states of the first regime that reaches ``goal``, or ``None``.

    The regimes are tried in a fixed order: only ``w1`` occupied, then only ``w2``, then both, with
    the splits ``left = 0 .. beta`` compared at once. An unselected child is not returned:
    it selects nothing.

    Parameters
    ----------
    dp : ComponentDP
        The filled table.
    w1, w2 : int
        The left and right children of the current node.
    s : int
        The current node's ``s_up``.
    beta : int
        The current node's entry.
    goal : float
        The stored optimum ``occ[node, s, beta]``.

    Returns
    -------
    list[tuple[int, int, int]] | None
        The ``(node, s_up, beta)`` states of the occupied children in the matching regime.
    """
    occ, empty, offset, gamma = dp
    g1, g2 = gamma[w1], gamma[w2]
    at = beta - int(offset[w2])
    at = at if at > 0 else 0
    value = occ[w1, s, at] + (g1[1][s] + empty[w2, 1] + g2[0][1])
    if value != NEG_INF and matches(value, goal):
        return [(w1, s, at)]
    at = beta - int(offset[w1])
    at = at if at > 0 else 0
    value = occ[w2, s, at] + (g2[1][s] + empty[w1, 1] + g1[0][1])
    if value != NEG_INF and matches(value, goal):
        return [(w2, s, at)]
    # Every split at once: entry i is occ[w1, 1, i] + ... + occ[w2, 1, beta - i] + ...
    splits = (occ[w1, 1, : beta + 1] + g1[1][1] + occ[w2, 1, beta::-1]) + g2[1][1]
    index = first_match(splits, goal)
    if index < 0:
        return None
    left = index
    return [(w1, 1, left), (w2, 1, beta - left)]


__all__ = ["component_selection", "trace_selection"]
