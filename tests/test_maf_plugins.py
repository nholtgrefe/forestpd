"""
The pluggable MAF computers: :class:`TieredMAF` and the maffe adapters.

maffe is optional and may not be installed, so the adapters are tested against a stand-in module
with maffe's signatures that answers with the heuristic's forest. That checks everything on
forestpd's side of the boundary: the Newick it writes, where it roots, the labels it maps back,
and the full validation of an untrusted forest. :class:`TestRealMaffe` runs only if maffe itself
is importable.
"""

from __future__ import annotations

import re
import sys
import types

import numpy as np
import pytest

from forestpd import MaxMPDInstance
from .fixtures.reference_pd import pd
from forestpd.core.tree import drop_leaf
from forestpd.core.taxa import TaxonRegistry
from forestpd.exceptions import ForestPDImportError, ForestPDInputError, ForestPDValueError
from forestpd.io.convert import to_phylotree
from forestpd.io.newick import ROOT_LABEL, parse_newick, to_newick
from forestpd.maf import HeuristicMAF, MaffeRMAF, MaffeUMAF, Rooting, TieredMAF
from forestpd.maf.maffe import rooted_newick
from forestpd.maf.partition import is_agreement_forest

from .fixtures.baselines import TrivialMAF
from .fixtures.brute_force import brute_force_optimize_pd
from .fixtures.random_instances import caterpillar, random_instance, random_tree, star


def _heuristic_answer(newicks):
    """The heuristic's forest of some Newicks, as maffe-style label components."""
    parsed = [parse_newick(text) for text in newicks]
    registry = TaxonRegistry(parsed[0].leaf_labels.values())
    local = [to_phylotree(p, registry, missing_branch_length="zero") for p in parsed]
    return [sorted(registry.labels_of(comp)) for comp in HeuristicMAF().compute(local)]


def stand_in_maffe(answer=None, unrooted=True):
    """
    A fake ``maffe`` module with maffe's signatures, answering with the heuristic's forest.

    It records every call, so the tests can check what forestpd sends.
    """
    calls = []

    def solve(trees, *, approx_factor=1.0, approx_offset=0, log_level=0):
        calls.append(("solve", trees, approx_factor, approx_offset, log_level))
        return answer if answer is not None else _heuristic_answer(trees)

    def solve_unrooted(trees, *, log_level=0):
        calls.append(("solve_unrooted", trees, log_level))
        if len(trees) < 2:
            raise ValueError("unrooted mode requires at least two trees")
        return answer if answer is not None else _heuristic_answer(trees)

    module = types.ModuleType("maffe")
    module.solve = solve
    if unrooted:
        module.solve_unrooted = solve_unrooted
    module.calls = calls
    return module


@pytest.fixture
def maffe(monkeypatch):
    module = stand_in_maffe()
    monkeypatch.setitem(sys.modules, "maffe", module)
    return module


def root_label(newick):
    """The leaf a :func:`rooted_newick` string hangs from: ``(...,<label>);``."""
    return newick.rsplit(",", 1)[1].rstrip(");")


def rooted_input(n_leaves, n_trees, rng):
    """Rooted binary Newicks with branch lengths missing, each rooted at a random leaf's edge."""
    labels = [f"t{leaf:02d}" for leaf in range(n_leaves)]
    newicks = []
    for _ in range(n_trees):
        text = rooted_newick(random_tree(n_leaves, rng), rng.randrange(n_leaves))
        newicks.append(re.sub(r"\d+", lambda m: labels[int(m.group()) - 1], text))
    return newicks


class Spy:
    """Wraps a MAF computer, recording every call's rooting and forest."""

    uses_rooting = True  # always ask for it, to record it

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def compute(self, trees, rooting=None):
        if rooting is None or not getattr(self.inner, "uses_rooting", False):
            forest = self.inner.compute(trees)
        else:
            forest = self.inner.compute(trees, rooting=rooting)
        self.calls.append((trees, rooting, forest))
        return forest


def as_comp_of_leaf(forest, n_leaves):
    comp_of_leaf = np.empty(n_leaves, dtype=np.int32)
    for index, component in enumerate(forest):
        comp_of_leaf[list(component)] = index
    return comp_of_leaf


class TestToNewick:
    @pytest.mark.parametrize("n_leaves", [2, 3, 5, 12])
    def test_round_trip_keeps_the_tree(self, rng, n_leaves):
        """Writing and reading back gives the same PD for every leaf subset sampled (as bitmasks)."""
        tree = random_tree(n_leaves, rng) if n_leaves >= 3 else star(n_leaves)
        labels = [f"t{leaf:02d}" for leaf in range(n_leaves)]  # sorted, as the registry numbers
        parsed = parse_newick(to_newick(tree, labels))
        back = to_phylotree(parsed, TaxonRegistry(labels))
        assert back.n_leaves == n_leaves
        assert back.total_weight() == pytest.approx(tree.total_weight())
        for _ in range(10):
            subset = sum(1 << leaf for leaf in range(n_leaves) if rng.random() < 0.5)
            assert pd(back, subset) == pytest.approx(pd(tree, subset))

    def test_single_leaf(self):
        assert to_newick(star(1), ["a"]) == "a;"

    def test_awkward_labels_are_quoted(self):
        labels = ["a b", "it's", "(x)", "plain_1"]
        parsed = parse_newick(to_newick(caterpillar(4), labels))
        assert sorted(parsed.leaf_labels.values()) == sorted(labels)

    def test_wrong_label_count(self):
        with pytest.raises(ForestPDInputError, match="Expected 3 labels"):
            to_newick(star(3), ["a", "b"])

    def test_duplicate_labels(self):
        with pytest.raises(ForestPDInputError, match="distinct"):
            to_newick(star(3), ["a", "b", "a"])

    def test_labels_with_newick_syntax_survive_the_instance(self, rng):
        """Labels holding Newick's own syntax characters are quoted and read back unchanged."""
        labels = ["a:b", "c;d", "e,f", "[g]", "h'i", "(j)"]
        trees = [random_tree(6, rng) for _ in range(2)]
        instance = MaxMPDInstance([to_newick(tree, labels) for tree in trees])
        assert sorted(instance.taxa) == sorted(labels)
        for tree, back in zip(trees, instance.trees):
            assert back.total_weight() == pytest.approx(tree.total_weight())
        reference = MaxMPDInstance.from_phylotrees(trees, TaxonRegistry(labels))
        assert instance.optimize(3).value == pytest.approx(reference.optimize(3).value)


class Recorder:
    """A MAF computer that records the cluster sizes it gets and answers with a fixed rule."""

    def __init__(self, answer=None, trusted=True, fails=False):
        self.sizes = []
        self.answer = answer
        self.trusted = trusted
        self.fails = fails

    def compute(self, trees):
        self.sizes.append(trees[0].n_leaves)
        if self.fails:
            raise ForestPDValueError("cannot handle this cluster")
        if self.answer is not None:
            return self.answer
        return TrivialMAF().compute(trees)


def disagreeing(n_leaves, rng):
    """Two random binary trees on ``n_leaves`` leaves that are not identical."""
    first = random_tree(n_leaves, rng)
    while True:
        second = random_tree(n_leaves, rng)
        if len(HeuristicMAF().compute([first, second])) > 1:
            return [first, second]


class FakeMaffe:
    """Stands in for TieredMAF's maffe stage: records the rootings, answers with a fixed rule."""

    def __init__(self, answer=None, fails=False):
        self.calls = []
        self.answer = answer
        self.fails = fails

    def compute(self, trees, rooting=None):
        self.calls.append(rooting)
        if self.fails:
            raise ForestPDValueError("cannot handle this cluster")
        if self.answer is not None:
            return self.answer
        return HeuristicMAF().compute(trees)


def tiered(maffe_stage, k_threshold=2, singletons_first=True):
    """A TieredMAF whose stages are replaced: singletons first, so the second stage matters."""
    out = TieredMAF(k_threshold=k_threshold)
    if singletons_first:
        out.heuristic = TrivialMAF()
    out.maffe = maffe_stage
    return out


class TestTieredMAF:
    def test_defaults(self, maffe):
        found = TieredMAF()
        assert found.k_threshold == 14
        assert found.trusted and found.uses_rooting
        assert isinstance(found.heuristic, HeuristicMAF) and isinstance(found.maffe, MaffeRMAF)

    def test_needs_maffe(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "maffe", None)
        with pytest.raises(ForestPDImportError, match="pip install"):
            TieredMAF()

    def test_below_threshold_the_heuristic_answers(self, maffe, rng):
        trees = disagreeing(8, rng)
        stage = FakeMaffe()
        found = tiered(stage, k_threshold=50, singletons_first=False)
        assert found.compute(trees) == HeuristicMAF().compute(trees)
        assert stage.calls == []

    def test_a_smaller_maffe_forest_is_used(self, maffe, rng):
        trees = disagreeing(8, rng)
        heuristic = HeuristicMAF().compute(trees)
        assert len(heuristic) < 8  # smaller than the singletons the tiered stage starts from
        stage = FakeMaffe()
        assert tiered(stage).compute(trees) == heuristic
        assert len(stage.calls) == 1

    def test_maffe_runs_from_k_threshold_on(self, maffe, rng):
        """``k_threshold`` itself already calls maffe; one less does not."""
        trees = disagreeing(8, rng)
        k = len(HeuristicMAF().compute(trees))
        at, above = FakeMaffe(), FakeMaffe()
        tiered(at, k_threshold=k, singletons_first=False).compute(trees)
        tiered(above, k_threshold=k + 1, singletons_first=False).compute(trees)
        assert len(at.calls) == 1 and above.calls == []

    def test_a_maffe_forest_that_is_not_smaller_is_not_used(self, maffe, rng):
        trees = disagreeing(8, rng)
        stage = FakeMaffe(answer=TrivialMAF().compute(trees))
        found = tiered(stage, singletons_first=False)
        assert found.compute(trees) == HeuristicMAF().compute(trees)
        assert len(stage.calls) == 1

    def test_failing_maffe_keeps_the_heuristic_forest(self, maffe, rng):
        trees = disagreeing(8, rng)
        assert tiered(FakeMaffe(fails=True)).compute(trees) == TrivialMAF().compute(trees)

    def test_component_masks_refuse_more_than_63_components(self, rng):
        trees = disagreeing(64, rng)
        with pytest.raises(ForestPDValueError, match="at most 63"):
            is_agreement_forest(trees, np.arange(64, dtype=np.int32), 64)

    def test_rooting_is_passed_to_maffe(self, maffe, rng):
        trees = disagreeing(8, rng)
        rooting = Rooting(trees, 0, extra=False)
        stage = FakeMaffe()
        tiered(stage).compute(trees, rooting=rooting)
        assert stage.calls == [rooting]

    @pytest.mark.parametrize("k_threshold", [1, 0, 2.5, True])
    def test_bad_threshold(self, maffe, k_threshold):
        with pytest.raises(ForestPDValueError):
            TieredMAF(k_threshold=k_threshold)


class TestRooting:
    """The solver passes each cluster's root leaf: x_C^* below the top cluster, leaf 0 at it."""

    def test_solver_roots_every_cluster_towards_the_top(self, rng):
        instance, budget = random_instance(rng, n_leaves=12, n_trees=3)
        spy = Spy(HeuristicMAF())
        instance.optimize(budget, maf=spy)
        tops = [rooting for _, rooting, _ in spy.calls if not rooting.extra]
        assert len(tops) == 1 and tops[0].leaf == 0
        for trees, rooting, _ in spy.calls:
            if rooting.extra:  # the trees plus x_C^*, which is their last leaf
                assert rooting.leaf == trees[0].n_leaves == rooting.trees[0].n_leaves - 1

    def test_computers_without_uses_rooting_get_only_the_trees(self, rng):
        """A computer without ``uses_rooting`` is called as ``compute(trees)``."""
        instance, budget = random_instance(rng, n_leaves=8, n_trees=2)
        recorder = Recorder()
        value = instance.optimize(budget, maf=recorder).value
        assert recorder.sizes  # it was called, without a rooting
        assert value == pytest.approx(brute_force_optimize_pd(instance, budget)[0])

    def test_rooted_input_roots_the_top_cluster_at_the_root_leaf(self, rng):
        instance = MaxMPDInstance(rooted_input(9, 3, rng), rooted=True, missing_branch_length=1)
        spy = Spy(HeuristicMAF())
        instance.optimize(3, maf=spy)
        tops = [rooting for _, rooting, _ in spy.calls if not rooting.extra]
        assert len(tops) == 1 and tops[0].leaf == instance.registry.id_of(ROOT_LABEL) == 0


class TestMaffeRMAF:
    def test_missing_maffe_explains_how_to_install(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "maffe", None)
        with pytest.raises(ForestPDImportError, match="pip install"):
            MaffeRMAF()

    def test_without_rooting_roots_at_leaf_0(self, maffe, rng):
        trees = disagreeing(8, rng)
        forest = MaffeRMAF(approx_factor=1.5, log_level="quiet").compute(trees + trees[:1])
        assert is_agreement_forest(trees, as_comp_of_leaf(forest, 8), len(forest))
        ((_, newicks, factor, offset, level),) = maffe.calls
        assert len(newicks) == 2  # the duplicate tree is sent once
        assert (factor, offset, level) == (1.5, 0, "quiet")
        assert {root_label(newick) for newick in newicks} == {"1"}

    def test_rooting_at_an_extra_leaf_drops_it(self, maffe, rng):
        """Rooted at x_C^* (the last leaf); the forest comes back without it."""
        plus = disagreeing(9, rng)
        plain = [drop_leaf(tree, 8) for tree in plus]
        forest = MaffeRMAF().compute(plain, rooting=Rooting(plus, 8, extra=True))
        assert sorted(leaf for component in forest for leaf in component) == list(range(8))
        assert is_agreement_forest(plain, as_comp_of_leaf(forest, 8), len(forest))
        assert {root_label(newick) for newick in maffe.calls[0][1]} == {"9"}

    def test_trivial_clusters_skip_maffe(self, maffe, rng):
        tree = random_tree(8, rng)
        assert MaffeRMAF().compute([tree, tree]) == (tuple(range(8)),)
        assert MaffeRMAF().compute([star(3), random_tree(3, rng)]) == ((0, 1, 2),)
        assert maffe.calls == []

    def test_polytomy_is_refused(self, maffe, rng):
        with pytest.raises(ForestPDValueError, match="binary"):
            MaffeRMAF().compute([star(5), random_tree(5, rng)])

    @pytest.mark.parametrize(
        "answer",
        [[["1", "2"], ["2", "3", "4"]], [["1", "2"]], [["1", "x"], ["2", "3", "4"]], [[], ["1"]]],
        ids=["overlap", "missing", "unknown-label", "empty"],
    )
    def test_bad_partition_is_rejected(self, monkeypatch, rng, answer):
        monkeypatch.setitem(sys.modules, "maffe", stand_in_maffe(answer))
        with pytest.raises(ForestPDInputError):
            MaffeRMAF().compute(disagreeing(4, rng))

    def test_alone_an_invalid_forest_is_rejected_by_the_solver(self, monkeypatch, rng):
        """Trusted, so only partially checked; a forest whose components disagree across trees
        still fails that check."""
        trees = disagreeing(6, rng)
        monkeypatch.setitem(sys.modules, "maffe", stand_in_maffe([[str(i + 1) for i in range(6)]]))
        newicks = [to_newick(tree, [f"t{i}" for i in range(6)]) for tree in trees]
        with pytest.raises(ForestPDInputError):
            MaxMPDInstance(newicks).optimize(3, maf=MaffeRMAF())

    @pytest.mark.parametrize("tiered_too", [False, True], ids=["alone", "tiered"])
    @pytest.mark.parametrize("trial", range(8))
    def test_matches_oracle(self, maffe, rng, trial, tiered_too):
        for _ in range(trial):
            rng.random()
        instance, budget = random_instance(
            rng, n_leaves=rng.randint(4, 7), n_trees=rng.randint(2, 3)
        )
        expected, _ = brute_force_optimize_pd(instance, budget)
        maf = MaffeRMAF()
        if tiered_too:
            # Singletons first, so maffe's forest is the one the DP actually uses.
            maf = tiered(MaffeRMAF())
        assert instance.optimize(budget, maf=maf).value == pytest.approx(expected)
        assert maffe.calls

    @pytest.mark.parametrize("trial", range(6))
    def test_rooted_input_with_survival_matches_oracle(self, maffe, rng, trial):
        for _ in range(trial):
            rng.random()
        newicks = rooted_input(rng.randint(4, 7), rng.randint(2, 3), rng)
        labels = sorted(parse_newick(newicks[0]).leaf_labels.values())
        survival = {x: rng.choice([0.0, rng.uniform(0.0, 0.9)]) for x in labels}
        instance = MaxMPDInstance(newicks, survival=survival, rooted=True, missing_branch_length=1)
        for budget in range(len(labels) + 1):
            expected, _ = brute_force_optimize_pd(instance, budget)
            for maf in (MaffeRMAF(), tiered(MaffeRMAF())):
                assert instance.optimize(budget, maf=maf).value == pytest.approx(expected)


class TestMaffeUMAF:
    def test_both_are_trusted(self, maffe):
        assert MaffeRMAF().trusted and MaffeUMAF().trusted
        assert TieredMAF().trusted

    def test_missing_unrooted_solver_explains_how_to_install(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "maffe", stand_in_maffe(unrooted=False))
        MaffeRMAF()  # the rooted solver is there
        with pytest.raises(ForestPDImportError, match="solve_unrooted"):
            MaffeUMAF()

    def test_sends_unrooted_topologies(self, maffe, rng):
        trees = disagreeing(8, rng)
        forest = MaffeUMAF(log_level=1).compute(trees + trees[:1])
        assert is_agreement_forest(trees, as_comp_of_leaf(forest, 8), len(forest))
        ((name, newicks, level),) = maffe.calls
        assert name == "solve_unrooted" and level == 1 and len(newicks) == 2
        assert all(":" not in newick for newick in newicks)  # no branch lengths

    def test_one_distinct_tree_skips_maffe(self, maffe, rng):
        tree = random_tree(8, rng)
        assert MaffeUMAF().compute([tree, tree]) == (tuple(range(8)),)
        assert maffe.calls == []

    def test_polytomy_is_refused(self, maffe, rng):
        with pytest.raises(ForestPDValueError, match="binary"):
            MaffeUMAF().compute([star(5), random_tree(5, rng)])

    @pytest.mark.parametrize("trial", range(8))
    def test_matches_oracle(self, maffe, rng, trial):
        for _ in range(trial):
            rng.random()
        instance, budget = random_instance(
            rng, n_leaves=rng.randint(4, 7), n_trees=rng.randint(2, 3)
        )
        expected, _ = brute_force_optimize_pd(instance, budget)
        assert instance.optimize(budget, maf=MaffeUMAF()).value == pytest.approx(expected)
        assert maffe.calls


def _real_maffe():
    """The real maffe module, or skip."""
    real = pytest.importorskip("maffe")
    if getattr(real, "calls", None) is not None:  # pragma: no cover - a stand-in leaked
        pytest.skip("maffe is a stand-in")
    return real


class TestRealMaffe:
    """Against maffe itself; skipped unless it is installed."""

    @pytest.mark.parametrize("trial", range(5))
    def test_both_match_oracle(self, rng, trial):
        _real_maffe()
        for _ in range(trial):
            rng.random()
        instance, budget = random_instance(rng, n_leaves=rng.randint(5, 7), n_trees=3)
        expected, _ = brute_force_optimize_pd(instance, budget)
        for large in (MaffeRMAF(), MaffeUMAF()):
            assert instance.optimize(budget, maf=large).value == pytest.approx(expected)
        assert instance.optimize(budget, maf=tiered(MaffeRMAF())).value == pytest.approx(expected)

    @pytest.mark.parametrize("trial", range(10))
    def test_rooted_input_k_at_most_rooted_maf(self, rng, trial):
        """With rooted=True, every cluster's k is at most the input's rooted MAF size m_rho."""
        _real_maffe()
        for _ in range(trial):
            rng.random()
        instance = MaxMPDInstance(
            rooted_input(rng.randint(8, 14), 3, rng), rooted=True, missing_branch_length=1
        )
        # Leaf 0 is the root leaf rho, so rooting the whole input there gives m_rho.
        m_rho = len(MaffeRMAF().compute(instance.trees))
        spy = Spy(MaffeRMAF())
        instance.optimize(3, maf=spy)
        assert max(len(forest) for _, _, forest in spy.calls) <= m_rho
        unrooted = Spy(MaffeUMAF())
        instance.optimize(3, maf=unrooted)
        assert max(len(forest) for _, _, forest in unrooted.calls) <= m_rho
