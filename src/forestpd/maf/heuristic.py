"""
Heuristic agreement forest by common-neighbourhood contraction.

This module provides a fast heuristic for an agreement forest. Its forest is always valid, though
not necessarily minimum, so it affects the running time (through ``k``) but not the result. It is
the default.

The method generalizes the cherry reduction to ``t`` trees and to polytomies:

1. Keep a working copy of every tree whose leaves are the current *blocks* (initially the
   singletons).
2. Group blocks by their anchor tuple: the vertex each block is attached to, in every tree. A group
   of two or more blocks attached to the same vertex in all trees is contracted into one block. The
   merged embedded subtree is then the members' subtrees joined at that vertex, identically in every
   tree.
3. If no group can be merged, cut one block off as a final component and return to step 2.
   Cutting only refines the forest, so it stays valid. The block to cut is chosen as follows:
   rank the blocks by the number of trees in which their neighbouring blocks differ from those in
   the first tree (smaller blocks first among ties), try cutting each of the first ``lookahead``
   blocks, and cut the one after which the most blocks can be merged.
4. Stop when at most two blocks remain. Two blocks are joined by a single edge in every working
   tree, so they form one final component.

Whole groups are merged rather than pairs because of polytomies. On a binary tree a vertex carries
at most two blocks, so the two coincide. At a polytomy, merging two disjoint pairs at the same
vertex would let both merged blocks pass through that vertex and violate vertex-disjointness.
Merging the whole group avoids this, and step 2 also rejects two groups that claim the same vertex
in any tree.

Each round costs ``O(t * n)`` and there are at most ``n`` rounds. After the cluster reduction the
local instances are small and the observed cost is close to linear.
"""

from __future__ import annotations

from typing import Sequence


from ..core.tree import PhyloTree
from .base import Forest, singleton_forest
from ..exceptions import ForestPDValueError

#: A block's attachment vertex in every tree at once — the key the contraction groups on.
AnchorKey = tuple[int, ...]


class HeuristicMAF:
    """
    Heuristic agreement forest by common-neighbourhood contraction.

    Parameters
    ----------
    lookahead : int | None, default 32
        When no blocks can be merged, how many of the highest-ranked blocks to try cutting before
        choosing the one that makes the most blocks mergeable. ``None`` tries every block: slower
        on large clusters, but it can give smaller forests.

    Attributes
    ----------
    trusted : bool
        ``True``: every forest is an agreement forest by construction (contraction merges only
        common neighbourhoods, and cutting only refines), so the solver may skip the full
        validation.

    Raises
    ------
    ForestPDValueError
        If ``lookahead`` is not a positive integer or ``None``.

    Examples
    --------
    >>> from forestpd.core.tree import PhyloTree
    >>> t = PhyloTree.from_edges(6, [(0, 4), (1, 4), (4, 5), (2, 5), (3, 5)],
    ...                          [1.0] * 5, {0: 0, 1: 1, 2: 2, 3: 3})
    >>> HeuristicMAF().compute([t, t])          # identical trees agree completely
    ((0, 1, 2, 3),)
    """

    trusted = True

    def __init__(self, lookahead: int | None = 32) -> None:
        if lookahead is not None and (
            isinstance(lookahead, bool) or not isinstance(lookahead, int) or lookahead < 1
        ):
            raise ForestPDValueError(
                f"lookahead must be a positive integer or None, got {lookahead!r}"
            )
        self.lookahead = lookahead

    def compute(self, trees: Sequence[PhyloTree]) -> Forest:
        """
        Compute a small valid agreement forest of ``trees``.

        Parameters
        ----------
        trees : Sequence[PhyloTree]
            The local trees of one cluster, all on leaves ``0..n_leaves-1``.

        Returns
        -------
        Forest
            The components, each a sorted tuple of leaf ids, ordered by smallest leaf.
        """
        trees = list(trees)
        n_leaves = trees[0].n_leaves
        if n_leaves <= 1:
            return singleton_forest(n_leaves)
        if len(trees) == 1:
            return (tuple(range(n_leaves)),)

        return _contract(trees, self.lookahead)


class _Working:
    """
    One input tree reduced to its current blocks, as a mutable adjacency.

    Parameters
    ----------
    tree : PhyloTree
        The input tree; its leaves become the initial singleton blocks.
    """

    __slots__ = ("adj", "block_at", "node_of_block", "_next")

    def __init__(self, tree: PhyloTree) -> None:
        self.adj: dict[int, set[int]] = {v: set() for v in range(tree.n_nodes)}
        for u, v in tree.edges.tolist():
            self.adj[u].add(v)
            self.adj[v].add(u)
        # Leaf ids double as node ids and as initial block ids (the PhyloTree invariant).
        self.block_at: dict[int, int] = {leaf: leaf for leaf in range(tree.n_leaves)}
        self.node_of_block: dict[int, int] = {leaf: leaf for leaf in range(tree.n_leaves)}
        self._next = tree.n_nodes

    def anchor_of(self, block: int) -> int:
        """
        The vertex a block hangs off, or ``-1`` if it is isolated.

        Parameters
        ----------
        block : int
            A live block id.

        Returns
        -------
        int
            The node id of the block's single neighbour, or ``-1``.
        """
        nbrs = self.adj[self.node_of_block[block]]
        return next(iter(nbrs)) if nbrs else -1

    def merge_group(self, blocks: list[int], merged: int) -> None:
        """
        Contract a set of blocks that share an attachment vertex into one block.

        Parameters
        ----------
        blocks : list[int]
            Block ids sharing one attachment vertex.
        merged : int
            The new block id to give the contracted block.
        """
        anchor = self.anchor_of(blocks[0])
        for block in blocks:
            node = self.node_of_block.pop(block)
            del self.block_at[node]
            self.adj.pop(node)
            self.adj[anchor].discard(node)

        if len(self.adj[anchor]) <= 1:
            # The anchor is now pendant (or isolated): it *is* the merged leaf.
            self.block_at[anchor] = merged
            self.node_of_block[merged] = anchor
            return
        # Other branches survive, so the merged block needs a fresh leaf of its own.
        new_node = self._next
        self._next += 1
        self.adj[new_node] = {anchor}
        self.adj[anchor].add(new_node)
        self.block_at[new_node] = merged
        self.node_of_block[merged] = new_node

    def detach(self, block: int) -> list[tuple[int, int]]:
        """
        Remove a block's leaf, suppressing the degree-2 vertex it may leave behind.

        Parameters
        ----------
        block : int
            The block to remove.

        Returns
        -------
        list[tuple[int, int]]
            ``(block, new anchor)`` for each block whose attachment vertex moved.
        """
        node = self.node_of_block.pop(block)
        del self.block_at[node]
        (anchor,) = self.adj.pop(node)
        self.adj[anchor].discard(node)
        moved: list[tuple[int, int]] = []
        if len(self.adj[anchor]) == 2 and anchor not in self.block_at:
            a, b = tuple(self.adj[anchor])
            self.adj.pop(anchor)
            self.adj[a].discard(anchor)
            self.adj[b].discard(anchor)
            self.adj[a].add(b)
            self.adj[b].add(a)
            moved = [(self.block_at[x], y) for x, y in ((a, b), (b, a)) if x in self.block_at]
        return moved


def _anchor_key(working: list[_Working], block: int) -> AnchorKey:
    """
    The block's attachment vertex in every tree.

    Parameters
    ----------
    working : list[_Working]
        The working trees.
    block : int
        A live block id.

    Returns
    -------
    AnchorKey
        The block's anchor in each working tree, in tree order.
    """
    return tuple(w.anchor_of(block) for w in working)


def _moved_keys(
    keys: dict[int, AnchorKey], moves: list[tuple[int, int, int]]
) -> dict[int, AnchorKey]:
    """
    New anchor keys for the blocks that moved: ``(tree, block, new anchor)`` per move.

    A block's anchor changes only in the trees where it moved, so only those positions of its
    cached key are replaced.
    """
    changed: dict[int, list[int]] = {}
    for j, block, anchor in moves:
        key = changed.get(block)
        if key is None:
            key = changed[block] = list(keys[block])
        key[j] = anchor
    return {block: tuple(key) for block, key in changed.items()}


def _mergeable_groups(
    working: list[_Working],
    alive: set[int],
    keys: dict[int, AnchorKey],
    dropped: list[int] | None = None,
) -> list[list[int]]:
    """
    Maximal sets of blocks sharing an attachment vertex in *every* tree, safe to merge together.

    A group qualifies only if, in every tree, its anchor is left with at most one other branch.
    Groups claiming the same vertex in some tree are never returned together; larger groups win.

    Parameters
    ----------
    working : list[_Working]
        The working trees.
    alive : set[int]
        The live block ids.
    keys : dict[int, AnchorKey]
        Cached anchor key per live block.
    dropped : list[int] | None, optional
        Per tree, a vertex whose degree counts as one less (a previewed cut), or ``-1``.

    Returns
    -------
    list[list[int]]
        Disjoint groups of block ids (each sorted, of size at least two), largest first.
    """
    groups: dict[AnchorKey, list[int]] = {}
    for block in alive:
        groups.setdefault(keys[block], []).append(block)

    claimed: set[tuple[int, int]] = set()
    out: list[list[int]] = []
    for key, blocks in sorted(groups.items(), key=lambda item: -len(item[1])):
        if len(blocks) < 2:
            continue
        if dropped is None:
            if any(len(w.adj[anchor]) - len(blocks) > 1 for w, anchor in zip(working, key)):
                continue
        elif any(
            len(w.adj[anchor]) - (anchor == drop) - len(blocks) > 1
            for w, anchor, drop in zip(working, key, dropped)
        ):
            continue
        slots = {(j, anchor) for j, anchor in enumerate(key)}
        if slots & claimed:
            continue
        claimed |= slots
        out.append(sorted(blocks))
    return out


def _contract(trees: Sequence[PhyloTree], lookahead: int | None) -> Forest:
    """
    Run the contraction loop and return the resulting blocks.

    Parameters
    ----------
    trees : Sequence[PhyloTree]
        The local trees, at least two, on the same leaves.
    lookahead : int | None
        Number of cut candidates to try (see :class:`HeuristicMAF`); ``None`` for all.

    Returns
    -------
    Forest
        The final blocks, each sorted, ordered by smallest leaf.
    """
    n_leaves = trees[0].n_leaves
    members: dict[int, tuple[int, ...]] = {leaf: (leaf,) for leaf in range(n_leaves)}
    alive = set(range(n_leaves))
    finished: list[tuple[int, ...]] = []
    working = [_Working(tree) for tree in trees]
    next_block = n_leaves

    # Anchor keys are cached and invalidated surgically. Rebuilding every block's key on every
    # query cost `alive * t` Python calls per lookahead candidate, which was the single hottest
    # function in the whole pipeline; a merge or a cut moves only a handful of anchors.
    keys = {block: _anchor_key(working, block) for block in alive}

    while len(alive) > 2:
        groups = _mergeable_groups(working, alive, keys)
        if groups:
            for blocks in groups:
                new_block = next_block
                next_block += 1
                members[new_block] = tuple(
                    sorted(leaf for block in blocks for leaf in members[block])
                )
                for w in working:
                    w.merge_group(blocks, new_block)
                alive.difference_update(blocks)
                alive.add(new_block)
                for block in blocks:
                    del keys[block]
                keys[new_block] = _anchor_key(working, new_block)
            continue

        victim = _choose_cut(working, alive, members, keys, lookahead)
        finished.append(members[victim])
        moves: list[tuple[int, int, int]] = []
        for j, w in enumerate(working):
            changed = w.detach(victim)
            moves.extend((j, block, anchor) for block, anchor in changed)
        alive.discard(victim)
        del keys[victim]
        keys.update(_moved_keys(keys, moves))

    if len(alive) == 2:
        a, b = sorted(alive)
        finished.append(tuple(sorted(members[a] + members[b])))
    elif alive:
        finished.append(members[next(iter(alive))])
    return tuple(sorted(finished, key=lambda block: block[0]))


def _choose_cut(
    working: list[_Working],
    alive: set[int],
    members: dict[int, tuple[int, ...]],
    keys: dict[int, AnchorKey],
    lookahead: int | None,
) -> int:
    """
    Pick the block to cut when nothing can be merged.

    Ranks blocks by disagreement score (smaller blocks first among equals), then, for the first
    ``lookahead`` of them (all if ``None``), tentatively cuts each and picks the one that lets the
    most blocks merge in the next round.

    Parameters
    ----------
    working : list[_Working]
        The working trees; left unchanged (each cut is only previewed).
    alive : set[int]
        The live block ids.
    members : dict[int, tuple[int, ...]]
        Leaf ids per block.
    keys : dict[int, AnchorKey]
        Cached anchor key per live block; left unchanged.
    lookahead : int | None
        Number of candidates to try; ``None`` for all.

    Returns
    -------
    int
        The block id to cut.
    """
    disagreement = _disagreement_scores(alive, keys)
    candidates = sorted(
        alive, key=lambda block: (-disagreement[block], len(members[block]), block)
    )[:lookahead]

    best = candidates[0]
    best_score = -1
    for block in candidates:
        # Score the cut from a preview, leaving the trees untouched. In each tree, detaching the
        # block (see _Working.detach) lowers its anchor's degree by one, or, if that leaves the
        # anchor with two neighbours, suppresses it, and the blocks on either side then anchor on
        # each other. This loop runs once per candidate and tree, so it is written inline.
        dropped: list[int] = []
        moves: list[tuple[int, int, int]] = []
        for j, w in enumerate(working):
            node = w.node_of_block[block]
            (anchor,) = w.adj[node]
            others = w.adj[anchor]
            if len(others) == 3 and anchor not in w.block_at:
                a, b = others - {node}
                at = w.block_at
                if a in at:
                    moves.append((j, at[a], b))
                if b in at:
                    moves.append((j, at[b], a))
                dropped.append(-1)
            else:
                dropped.append(anchor)
        trial = dict(keys)
        trial.update(_moved_keys(keys, moves))
        rest = alive - {block}
        score = sum(len(group) for group in _mergeable_groups(working, rest, trial, dropped))
        if score > best_score:
            best, best_score = block, score
    return best


def _disagreement_scores(alive: set[int], keys: dict[int, AnchorKey]) -> dict[int, int]:
    """
    Per block, in how many trees its set of co-attached blocks differs from the first tree's.

    Parameters
    ----------
    alive : set[int]
        The live block ids.
    keys : dict[int, AnchorKey]
        Cached anchor key per live block.

    Returns
    -------
    dict[int, int]
        Per live block, the number of trees disagreeing with the first.
    """
    n_trees = len(next(iter(keys.values()))) if keys else 0
    # A block's partners in two trees differ exactly when its co-attached groups (which both
    # contain the block) differ, so whole groups are compared, one frozenset per anchor.
    groups = [_groups_in_tree(alive, keys, j) for j in range(n_trees)]
    score = dict.fromkeys(alive, 0)
    if n_trees:
        first = {block: groups[0][keys[block][0]] for block in alive}
        for j in range(1, n_trees):
            other = groups[j]
            for block in alive:
                if other[keys[block][j]] != first[block]:
                    score[block] += 1
    return score


def _groups_in_tree(
    alive: set[int], keys: dict[int, AnchorKey], tree: int
) -> dict[int, frozenset[int]]:
    """
    For each attachment vertex in one tree, the live blocks hanging off it.

    Parameters
    ----------
    alive : set[int]
        The live block ids.
    keys : dict[int, AnchorKey]
        Cached anchor key per live block.
    tree : int
        Index of the working tree to read.

    Returns
    -------
    dict[int, frozenset[int]]
        Anchor -> the live blocks with that anchor in tree ``tree``.
    """
    by_anchor: dict[int, set[int]] = {}
    for block in alive:
        by_anchor.setdefault(keys[block][tree], set()).add(block)
    return {anchor: frozenset(blocks) for anchor, blocks in by_anchor.items()}


__all__ = ["HeuristicMAF"]
