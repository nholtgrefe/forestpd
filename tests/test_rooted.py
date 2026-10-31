"""
Rooted PD (``rooted=True``): agreement with a direct computation, root hints, and PhyloZoo input.

The oracle computes rooted PD straight from the Newick string, with its own small parser: the total
length of the union of root-to-leaf paths, plus the root's own branch length for a non-empty set.
It shares no code with the package.
"""

from __future__ import annotations

import itertools
import random
import re
import warnings

import numpy as np
import pytest

import forestpd as fpd
from forestpd.exceptions import ForestPDInputError, ForestPDWarning
from forestpd.io.newick import ROOT_LABEL


def rooted_pd(newick: str, chosen: set[str], survival: dict[str, float] | None = None) -> float:
    """
    Rooted PD of ``chosen`` in ``newick``, computed directly from the string.

    With ``survival`` (probability per label that an unchosen taxon survives) it is the expected
    rooted PD: each node's branch counts with the probability that some taxon below it survives.
    """
    tokens = re.findall(r"[(),;]|[^(),;:]+(?::[0-9.eE+-]+)?|:[0-9.eE+-]+", newick)
    parent: dict[int, int | None] = {0: None}
    length: dict[int, float] = {}
    label: dict[int, str] = {}
    stack: list[int] = []
    current, next_id = 0, 1
    for token in tokens:
        if token == "(":
            parent[next_id] = current
            stack.append(current)
            current, next_id = next_id, next_id + 1
        elif token == ",":
            parent[next_id] = stack[-1]
            current, next_id = next_id, next_id + 1
        elif token == ")":
            current = stack.pop()
        elif token != ";":
            name, _, value = token.partition(":")
            if name:
                label[current] = name
            if value:
                length[current] = float(value)
    survival = survival or {}
    # Per node: whether a chosen taxon lies below, and the probability that every taxon below dies.
    has_chosen: dict[int, bool] = {node: False for node in parent}
    all_lost: dict[int, float] = {node: 1.0 for node in parent}
    for node, name in label.items():
        chosen_here = name in chosen
        lost_here = 1.0 if chosen_here else 1.0 - survival.get(name, 0.0)
        while node is not None:
            has_chosen[node] = has_chosen[node] or chosen_here
            if not chosen_here:
                all_lost[node] *= lost_here
            node = parent[node]
    return sum(
        length.get(node, 0.0) * (1.0 if has_chosen[node] else 1.0 - all_lost[node])
        for node in parent
    )


def random_rooted(rng: random.Random, labels: list[str]) -> str:
    """A random rooted tree with polytomies, sometimes a stem or an outdegree-1 chain on top."""
    nodes = [f"{x}:{rng.randint(1, 9)}" for x in labels]
    while len(nodes) > rng.choice([2, 2, 3]):
        k = min(len(nodes) - 1, rng.choice([2, 2, 3]))
        picked = [nodes.pop(rng.randrange(len(nodes))) for _ in range(k)]
        nodes.append("(" + ",".join(picked) + f"):{rng.randint(0, 9)}")
    tree = "(" + ",".join(nodes) + ")"
    if rng.random() < 0.3:
        tree = f"({tree}:{rng.randint(1, 5)})"  # outdegree-1 root
    if rng.random() < 0.5:
        tree += f":{rng.randint(1, 5)}"  # the root's own branch length
    return tree + ";"


class TestAgainstDirectComputation:
    @pytest.mark.parametrize("trial", range(30))
    def test_optimum_and_frontier(self, trial):
        rng = random.Random(trial)
        labels = [f"t{i}" for i in range(rng.randint(3, 7))]
        trees = [random_rooted(rng, labels) for _ in range(rng.randint(1, 3))]
        unit = (1, 2, 5)[trial // 3 % 3]  # costs with a common divisor are solved in its units
        costs = {x: unit * rng.randint(1, 3) for x in labels} if trial % 3 == 0 else None
        weights = [round(rng.uniform(0.5, 2.0), 2) for _ in trees] if trial % 4 == 0 else None
        cost = costs or dict.fromkeys(labels, 1)
        w = weights or [1.0] * len(trees)
        budget = rng.randint(0, sum(cost.values()))

        def score(chosen):
            return sum(wj * rooted_pd(t, set(chosen)) for wj, t in zip(w, trees))

        def best(b):
            return max(
                score(subset)
                for r in range(len(labels) + 1)
                for subset in itertools.combinations(labels, r)
                if sum(cost[x] for x in subset) <= b
            )

        instance = fpd.MaxMPDInstance(trees, costs, weights, rooted=True)
        result = instance.optimize(budget, return_taxa=True)
        assert result.value == pytest.approx(best(budget))
        assert sum(cost[x] for x in result.taxa) <= budget
        assert score(result.taxa) == pytest.approx(result.value)
        assert instance.evaluate_pd(result.taxa) == pytest.approx(result.value)
        for b, value in enumerate(result.meta["frontier"]):
            if not np.isnan(value):  # NaN: not covered by the tables computed (DP')
                assert value == pytest.approx(best(b)), f"frontier at budget {b}"

        budgets = list(range(sum(cost.values()) + 2))
        for b, many in zip(budgets, instance.optimize_many(budgets, return_taxa=True)):
            assert many.value == pytest.approx(best(b)), f"optimize_many at budget {b}"
            assert sum(cost[x] for x in many.taxa) <= b
            assert score(many.taxa) == pytest.approx(many.value)

    def test_pd_of_arbitrary_sets(self):
        rng = random.Random(99)
        labels = list("ABCDEF")
        trees = [random_rooted(rng, labels) for _ in range(3)]
        instance = fpd.MaxMPDInstance(trees, rooted=True)
        for r in range(len(labels) + 1):
            for subset in itertools.combinations(labels, r):
                direct = sum(rooted_pd(t, set(subset)) for t in trees)
                assert instance.evaluate_pd(subset) == pytest.approx(direct)


class TestBehaviour:
    def test_single_taxon_counts_its_path_and_the_stem(self):
        instance = fpd.MaxMPDInstance(["((A:1,B:2):3,C:4):5;"], rooted=True)
        assert instance.evaluate_pd({"A"}) == pytest.approx(1 + 3 + 5)
        assert instance.evaluate_pd(set()) == 0.0
        assert instance.optimize(1, return_taxa=True).taxa == {"B"}  # 2 + 3 + 5 beats 4 + 5

    def test_root_leaf_never_appears(self):
        instance = fpd.MaxMPDInstance(["(A:1,B:1,C:1);"], rooted=True)
        assert instance.taxa == ("A", "B", "C") and instance.n_taxa == 3
        assert instance.optimize(3, return_taxa=True).taxa == {"A", "B", "C"}

    def test_nothing_affordable_gives_zero(self):
        instance = fpd.MaxMPDInstance(["(A:1,B:1):7;"], costs={"A": 3, "B": 3}, rooted=True)
        result = instance.optimize(2, return_taxa=True)
        assert result.value == 0.0 and result.taxa == set()

    def test_unrooted_mode_is_unchanged(self):
        trees = ["(A:1,B:1,(C:1,D:1):2);"]
        assert fpd.optimize_maxmpd(trees, budget=2).value == pytest.approx(4.0)


class TestHiddenRootLeaf:
    """The reserved root leaf is internal: users can neither see nor address it."""

    TREES = ["((A:1,B:1):1,(C:1,D:1):1);"]

    def test_labels(self):
        instance = fpd.MaxMPDInstance(self.TREES, rooted=True)
        assert instance.taxa == ("A", "B", "C", "D")
        assert ROOT_LABEL in instance.registry.labels and len(instance.registry.labels) == 5

    @pytest.mark.parametrize(
        "kwargs", [{"costs": {ROOT_LABEL: 2}}, {"survival": {ROOT_LABEL: 0.5}}], ids=str
    )
    def test_not_a_valid_key(self, kwargs):
        with pytest.raises(ForestPDInputError, match="(?i)unknown"):
            fpd.MaxMPDInstance(self.TREES, rooted=True, **kwargs)

    def test_not_a_valid_taxon(self):
        instance = fpd.MaxMPDInstance(self.TREES, rooted=True)
        with pytest.raises(ForestPDInputError, match="(?i)unknown"):
            instance.evaluate_pd({"A", ROOT_LABEL})

    def test_reserved_in_unrooted_input(self):
        with pytest.raises(ForestPDInputError, match="reserved"):
            fpd.MaxMPDInstance([f"(A:1,B:1,'{ROOT_LABEL}':1);"])


class TestRootHints:
    def test_marked_rooted_warns_when_unrooted(self):
        with pytest.warns(ForestPDWarning, match="appear to be rooted"):
            fpd.MaxMPDInstance(["[&R](A:1,B:1,C:1);"])

    def test_all_binary_roots_warn_when_unrooted(self):
        with pytest.warns(ForestPDWarning, match="appear to be rooted"):
            fpd.MaxMPDInstance(["((A:1,B:1):1,C:1);", "((A:1,C:1):1,B:1);"])

    def test_multifurcating_tops_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ForestPDWarning)
            fpd.MaxMPDInstance(["(A:1,B:1,C:1);", "((A:1,C:1):1,B:1);"])
            fpd.MaxMPDInstance(["(A:1,B:1,C:1);"], rooted=True)  # a polytomy at the root is fine

    def test_marked_unrooted_warns_when_rooted(self):
        with pytest.warns(ForestPDWarning, match=r"\[&U\]"):
            instance = fpd.MaxMPDInstance(["[&U](A:1,B:1,C:1);"], rooted=True)
        assert instance.evaluate_pd({"A"}) == pytest.approx(1.0)


class TestPhyloZoo:
    phylozoo = pytest.importorskip("phylozoo")

    def test_directed_network_matches_rooted_newick(self):
        trees = ["(((A:1,B:2):3,C:4):6,D:2);", "((A:2,(B:1,C:1):2):1,D:5);"]
        networks = [
            self.phylozoo.DirectedPhyNetwork.from_string(t, format="enewick") for t in trees
        ]
        from_newick = fpd.MaxMPDInstance(trees, rooted=True)
        from_networks = fpd.MaxMPDInstance(networks, rooted=True)
        for budget in range(5):
            assert from_networks.optimize(budget).value == pytest.approx(
                from_newick.optimize(budget).value
            )

    def test_directed_network_warns_when_unrooted(self):
        net = self.phylozoo.DirectedPhyNetwork.from_string("((A:1,B:1):1,C:1);", format="enewick")
        with pytest.warns(ForestPDWarning, match="appear to be rooted"):
            fpd.MaxMPDInstance([net])

    def test_semidirected_network_cannot_be_rooted(self):
        net = self.phylozoo.SemiDirectedPhyNetwork.from_string("(A:1,B:1,C:1);", format="enewick")
        with pytest.raises(ForestPDInputError, match="unrooted"):
            fpd.MaxMPDInstance([net], rooted=True)
