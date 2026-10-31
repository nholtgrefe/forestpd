"""
Random and hand-built instances shared by the test suite.

Trees are constructed directly as :class:`~forestpd.core.tree.PhyloTree` objects, bypassing
PhyloZoo, so the algorithmic tests neither need nor exercise the frontend.
"""

from __future__ import annotations

import random
from typing import NamedTuple


from forestpd.core.tree import PhyloTree
from forestpd.core.taxa import TaxonRegistry
from forestpd.instance import MaxMPDInstance


def random_tree(n_leaves: int, rng: random.Random, max_weight: int = 9) -> PhyloTree:
    """
    A random unrooted binary tree on leaves ``0..n_leaves-1`` with random integer weights.

    Built by repeated random leaf insertion, which reaches every binary topology.
    """
    if n_leaves < 2:
        raise ValueError("need at least two leaves")
    edges: list[tuple[int, int]] = [(0, 1)]
    next_node = n_leaves  # internal nodes are numbered after the leaves
    for leaf in range(2, n_leaves):
        target = rng.randrange(len(edges))
        u, v = edges.pop(target)
        hub = next_node
        next_node += 1
        edges.extend([(u, hub), (hub, v), (hub, leaf)])
    weights = [float(rng.randint(0, max_weight)) for _ in edges]
    leaf_of_node = {leaf: leaf for leaf in range(n_leaves)}
    return PhyloTree.from_edges(next_node, edges, weights, leaf_of_node)


def perturbed_tree(
    tree: PhyloTree,
    rng: random.Random,
    moves: int = 1,
    leaves: list[int] | None = None,
) -> PhyloTree:
    """
    A tree that shares structure with ``tree`` but disagrees in a few places.

    Produced by detaching a leaf and reattaching it on a random edge — one subtree-prune-and-regraft
    move per step, which is exactly what raises the MAF size by roughly one.

    Passing ``leaves`` fixes *which* leaves move. Displacing the same leaves in every tree keeps the
    MAF size constant as trees are added (the forest is those leaves plus the agreeing remainder),
    which is the realistic regime of many similar trees; letting each tree move a different leaf instead
    makes ``k`` grow with the number of trees.
    """
    edges = [(int(tree.edges[e, 0]), int(tree.edges[e, 1])) for e in range(tree.n_edges)]
    weights = [float(tree.weight[e]) for e in range(tree.n_edges)]
    n_leaves = tree.n_leaves
    for step in range(moves):
        leaf = rng.randrange(n_leaves) if leaves is None else leaves[step % len(leaves)]
        pendant = next(i for i, (u, v) in enumerate(edges) if leaf in (u, v))
        u, v = edges[pendant]
        hub = v if u == leaf else u
        others = [i for i, (a, b) in enumerate(edges) if i != pendant and hub in (a, b)]
        if len(others) != 2:
            continue
        i1, i2 = others
        a1 = edges[i1][0] if edges[i1][1] == hub else edges[i1][1]
        a2 = edges[i2][0] if edges[i2][1] == hub else edges[i2][1]
        merged_weight = weights[i1] + weights[i2]
        pendant_weight = weights[pendant]
        for index in sorted([pendant, i1, i2], reverse=True):
            edges.pop(index)
            weights.pop(index)
        edges.append((a1, a2))
        weights.append(merged_weight)
        target = rng.randrange(len(edges))
        x, y = edges.pop(target)
        w = weights.pop(target)
        edges.extend([(x, hub), (hub, y), (hub, leaf)])
        weights.extend([w / 2.0, w / 2.0, pendant_weight])
    used = sorted({node for edge in edges for node in edge})
    remap = {node: i for i, node in enumerate(used)}
    return PhyloTree.from_edges(
        len(used),
        [(remap[u], remap[v]) for u, v in edges],
        weights,
        {remap[leaf]: leaf for leaf in range(n_leaves)},
    )


class Case(NamedTuple):
    """A test instance together with the budget to solve it at."""

    instance: MaxMPDInstance
    budget: int


def make_instance(
    trees: list[PhyloTree],
    budget: int,
    costs: list[int] | None = None,
    weights: list[float] | None = None,
    survival: list[float] | None = None,
) -> Case:
    """Wrap explicit trees into a :class:`Case` with labels ``t000, t001, ...``."""
    registry = TaxonRegistry([f"t{i:03d}" for i in range(trees[0].n_leaves)])
    return Case(MaxMPDInstance.from_phylotrees(trees, registry, costs, weights, survival), budget)


def random_survival(rng: random.Random, n_taxa: int) -> list[float]:
    """Survival probabilities in ``[0, 1)``: a mix of zeros, moderate values and values near 1."""
    return [
        rng.choice([0.0, rng.uniform(0.0, 0.6), rng.uniform(0.6, 0.999)]) for _ in range(n_taxa)
    ]


def with_survival(case: Case, survival: list[float]) -> Case:
    """The same instance and budget with the given survival probabilities."""
    instance = case.instance
    return Case(
        MaxMPDInstance.from_phylotrees(
            instance.trees, instance.registry, instance.costs.tolist(), instance.weights, survival
        ),
        case.budget,
    )


def random_instance(
    rng: random.Random,
    n_leaves: int = 6,
    n_trees: int = 3,
    budget: int | None = None,
    heterogeneous_costs: bool = False,
    varied_weights: bool = False,
) -> Case:
    """A random uniform MaxMPD instance, optionally with uneven costs and tree weights."""
    trees = [random_tree(n_leaves, rng) for _ in range(n_trees)]
    costs = [rng.randint(1, 3) for _ in range(n_leaves)] if heterogeneous_costs else [1] * n_leaves
    weights = (
        [round(rng.uniform(0.25, 3.0), 2) for _ in range(n_trees)]
        if varied_weights
        else [1.0] * n_trees
    )
    if budget is None:
        budget = rng.randint(1, sum(costs))
    return make_instance(trees, budget, costs, weights)


def related_instance(
    rng: random.Random,
    n_leaves: int = 8,
    n_trees: int = 3,
    moves: int = 1,
    budget: int | None = None,
) -> Case:
    """An instance whose trees are near-identical — the regime the FPT algorithm is built for."""
    base = random_tree(n_leaves, rng)
    trees = [base] + [perturbed_tree(base, rng, moves) for _ in range(n_trees - 1)]
    budget = budget if budget is not None else rng.randint(2, n_leaves)
    return make_instance(trees, budget)


def caterpillar(n_leaves: int, weight: float = 1.0) -> PhyloTree:
    """A fully unbalanced (caterpillar) tree on ``0..n_leaves-1``."""
    edges: list[tuple[int, int]] = []
    spine = n_leaves
    edges.append((0, spine))
    edges.append((1, spine))
    for leaf in range(2, n_leaves - 1):
        edges.append((spine, spine + 1))
        spine += 1
        edges.append((leaf, spine))
    edges.append((n_leaves - 1, spine))
    n_nodes = spine + 1
    return PhyloTree.from_edges(
        n_nodes, edges, [weight] * len(edges), {leaf: leaf for leaf in range(n_leaves)}
    )


def star(n_leaves: int, weight: float = 1.0) -> PhyloTree:
    """A star: every leaf attached to one centre (a maximal polytomy)."""
    centre = n_leaves
    edges = [(leaf, centre) for leaf in range(n_leaves)]
    return PhyloTree.from_edges(
        n_leaves + 1, edges, [weight] * len(edges), {leaf: leaf for leaf in range(n_leaves)}
    )


def balanced(n_leaves: int, weight: float = 1.0) -> PhyloTree:
    """A balanced binary tree; ``n_leaves`` must be a power of two."""
    level = list(range(n_leaves))
    edges: list[tuple[int, int]] = []
    next_node = n_leaves
    while len(level) > 2:
        nxt = []
        for i in range(0, len(level), 2):
            hub = next_node
            next_node += 1
            edges.append((level[i], hub))
            edges.append((level[i + 1], hub))
            nxt.append(hub)
        level = nxt
    edges.append((level[0], level[1]))
    return PhyloTree.from_edges(
        next_node, edges, [weight] * len(edges), {leaf: leaf for leaf in range(n_leaves)}
    )


__all__ = [
    "random_tree",
    "perturbed_tree",
    "make_instance",
    "random_survival",
    "with_survival",
    "random_instance",
    "related_instance",
    "caterpillar",
    "star",
    "balanced",
]


def _rooted_random(taxa: list[int], rng: random.Random, next_node: int):
    """A random rooted binary tree over ``taxa``; returns ``(root, next_node, edges)``."""
    if len(taxa) == 1:
        return taxa[0], next_node, []
    shuffled = taxa[:]
    rng.shuffle(shuffled)
    cut = rng.randint(1, len(shuffled) - 1)
    left, next_node, left_edges = _rooted_random(shuffled[:cut], rng, next_node)
    right, next_node, right_edges = _rooted_random(shuffled[cut:], rng, next_node)
    node = next_node
    next_node += 1
    return node, next_node, left_edges + right_edges + [(node, left), (node, right)]


def _caterpillar_backbone(roots: list[int], next_node: int) -> tuple[list[tuple[int, int]], int]:
    """
    Join the clade roots by a caterpillar backbone; returns ``(edges, next_node)``.

    Two clades are joined directly; ``k >= 3`` clades hang off ``k - 2`` spine nodes, so every
    spine node has degree 3.
    """
    if len(roots) < 2:
        raise ValueError("need at least two clades")
    if len(roots) == 2:
        return [(roots[0], roots[1])], next_node
    spine = next_node
    next_node += 1
    edges = [(roots[0], spine), (roots[1], spine)]
    for root in roots[2:-1]:
        edges.append((spine, next_node))
        spine = next_node
        next_node += 1
        edges.append((root, spine))
    edges.append((roots[-1], spine))
    return edges, next_node


def clade_structured_tree(
    clades: list[list[int]], rng: random.Random, max_weight: int = 9
) -> PhyloTree:
    """
    A tree built from independent clades hung off a shared caterpillar backbone.

    Two such trees that differ only *inside* one clade share every backbone split, so the common
    cluster reduction isolates the disagreement — the regime where reducing beats not reducing.
    """
    n_leaves = sum(len(clade) for clade in clades)
    next_node = n_leaves
    edges: list[tuple[int, int]] = []
    roots: list[int] = []
    for clade in clades:
        root, next_node, clade_edges = _rooted_random(clade, rng, next_node)
        edges.extend(clade_edges)
        roots.append(root)

    backbone, next_node = _caterpillar_backbone(roots, next_node)
    edges.extend(backbone)

    weights = [float(rng.randint(1, max_weight)) for _ in edges]
    return PhyloTree.from_edges(next_node, edges, weights, {leaf: leaf for leaf in range(n_leaves)})


def clade_localized_instance(
    rng: random.Random,
    n_clades: int = 6,
    clade_size: int = 12,
    n_trees: int = 6,
    budget: int = 15,
):
    """
    Trees agreeing on a backbone, each disagreeing inside exactly one clade.

    Globally the MAF size grows with the number of disagreeing clades; per cluster it stays at the
    size of a single clade's disagreement. This is what the cluster reduction is for.
    """
    clades = [list(range(c * clade_size, (c + 1) * clade_size)) for c in range(n_clades)]
    base_seed = rng.randrange(1 << 30)
    trees = [clade_structured_tree(clades, random.Random(base_seed))]
    for j in range(1, n_trees):
        # Re-randomize exactly one clade; every other clade keeps the base topology.
        seeds = [base_seed] * n_clades
        seeds[(j - 1) % n_clades] = rng.randrange(1 << 30)
        per_clade = random.Random(base_seed)
        tree = _tree_from_clade_seeds(clades, seeds, per_clade)
        trees.append(tree)
    return make_instance(trees, budget)


def _tree_from_clade_seeds(clades, seeds, backbone_rng):
    """Build a clade-structured tree where clade ``c``'s topology comes from ``seeds[c]``."""
    n_leaves = sum(len(clade) for clade in clades)
    next_node = n_leaves
    edges: list[tuple[int, int]] = []
    roots: list[int] = []
    for clade, seed in zip(clades, seeds):
        root, next_node, clade_edges = _rooted_random(clade, random.Random(seed), next_node)
        edges.extend(clade_edges)
        roots.append(root)
    backbone, next_node = _caterpillar_backbone(roots, next_node)
    edges.extend(backbone)
    weights = [float(backbone_rng.randint(1, 9)) for _ in edges]
    return PhyloTree.from_edges(next_node, edges, weights, {leaf: leaf for leaf in range(n_leaves)})


__all__ += ["clade_structured_tree", "clade_localized_instance"]
