"""
:func:`forestpd.utils.prune`: restricting tree sets to a taxon set.

The contract that matters: unrooted PD of any set of kept taxa is the same in a pruned tree as in
the full one, trees missing a kept taxon are dropped together with their weight, and the output
goes into :meth:`MaxMPDInstance.from_phylotrees` as it is.
"""

from __future__ import annotations

import itertools
import re

import numpy as np
import pytest

import forestpd as fpd
from forestpd.core.tree import has_degree_two
from forestpd.exceptions import ForestPDInputError
from forestpd.io.newick import to_newick
from forestpd.utils import prune

from .fixtures.random_instances import random_tree


def labelled_newicks(rng, n_taxa, n_trees, missing=()):
    """Random trees on ``x0..``; tree ``j`` lacks taxon ``missing[j]`` when given."""
    out = []
    for j in range(n_trees):
        labels = [f"x{i}" for i in range(n_taxa)]
        if j < len(missing) and missing[j] is not None:
            labels.remove(f"x{missing[j]}")
        out.append(to_newick(random_tree(len(labels), rng), labels))
    return out


@pytest.mark.parametrize("trial", range(10))
def test_pd_of_kept_taxa_is_unchanged(rng, trial):
    for _ in range(trial):
        rng.random()
    n = rng.randint(5, 9)
    newicks = labelled_newicks(rng, n, 3)
    weights = [rng.uniform(0.5, 2.0) for _ in newicks]
    keep = set(rng.sample([f"x{i}" for i in range(n)], rng.randint(2, n - 1)))
    full = fpd.MaxMPDInstance(newicks, weights=weights)
    trees, kept_weights, registry = prune(newicks, keep, weights)
    instance = fpd.MaxMPDInstance.from_phylotrees(trees, registry, weights=kept_weights)
    assert set(instance.taxa) == keep
    for size in range(len(keep) + 1):
        for chosen in itertools.combinations(sorted(keep), size):
            assert instance.evaluate_pd(chosen) == pytest.approx(full.evaluate_pd(chosen))


def test_trees_without_all_taxa_are_dropped_with_their_weight(rng):
    newicks = labelled_newicks(rng, 6, 4, missing=(None, 2, None, 5))
    trees, _, _ = prune(newicks, {"x0", "x1", "x3"}, weights={1: 7.0, 2: 3.0})
    assert len(trees) == 4  # x2 and x5 are not kept, so every tree has the kept taxa
    _, weights, _ = prune(newicks, {"x0", "x2"}, weights={1: 7.0, 2: 3.0})
    assert weights.tolist() == [1.0, 3.0, 1.0]  # tree 1 lacks x2
    _, weights, _ = prune(newicks, {"x2", "x5"}, weights=[1.0, 2.0, 3.0, 4.0])
    assert weights.tolist() == [1.0, 3.0]  # tree 1 lacks x2, tree 3 lacks x5


def test_output_feeds_from_phylotrees_and_the_solver(rng):
    newicks = labelled_newicks(rng, 8, 3, missing=(None, None, 4))
    trees, weights, registry = prune(newicks, [f"x{i}" for i in range(6)])
    assert not any(has_degree_two(tree) for tree in trees)
    instance = fpd.MaxMPDInstance.from_phylotrees(trees, registry, weights=weights)
    result = instance.optimize(3, return_taxa=True)
    assert instance.evaluate_pd(result.taxa) == pytest.approx(result.value)


def test_reads_a_tree_file(rng, tmp_path):
    newicks = labelled_newicks(rng, 6, 3)
    path = tmp_path / "trees.nwk"
    path.write_text("\n".join(newicks) + "\n")
    from_file, _, _ = prune(path, {"x0", "x1", "x2"})
    from_list, _, _ = prune(newicks, {"x0", "x1", "x2"})
    for a, b in zip(from_file, from_list):
        np.testing.assert_array_equal(a.edges, b.edges)
        np.testing.assert_array_equal(a.weight, b.weight)


def test_missing_lengths_follow_the_policy():
    newicks = ["((A,B),(C,(D:1,E:1):1));"]
    with pytest.raises(ForestPDInputError):
        prune(newicks, {"A", "B", "C"})
    zero, _, _ = prune(newicks, {"A", "B", "C"}, missing_branch_length="zero")
    assert zero[0].total_weight() == 0.0
    # Pruned to A, B, D: the edges of A and B have no length (set to 5); the merged edge towards
    # D has one, the sum of the parts that do (1 + 1).
    kept, _, _ = prune(newicks, {"A", "B", "D"}, missing_branch_length=5.0)
    assert sorted(kept[0].weight.tolist()) == [2.0, 5.0, 5.0]


@pytest.mark.parametrize(
    "taxa,message",
    [
        ({"x0"}, "At least two taxa"),
        (set(), "At least two taxa"),
        ({"x0", "nope"}, "No tree contains all"),
        ({"x0", 3}, "must be labels"),
    ],
)
def test_invalid_input(rng, taxa, message):
    with pytest.raises(ForestPDInputError, match=message):
        prune(labelled_newicks(rng, 5, 2), taxa)


def test_wrong_weight_count(rng):
    with pytest.raises(ForestPDInputError, match="Expected 2 weights"):
        prune(labelled_newicks(rng, 5, 2), {"x0", "x1"}, weights=[1.0])


# ---------------------------------------------------------------------------------------------
# Split systems


def split_diversity(sides, weights, chosen: set[int]) -> float:
    """The summed weight of the splits separating two chosen taxa."""
    total = 0.0
    for side, weight in zip(sides, weights):
        inside = chosen & set(side.tolist())
        if inside and chosen - inside:
            total += float(weight)
    return total


def splits_as_dict(sides, weights, labels) -> dict[frozenset[str], float]:
    return {frozenset(labels[x] for x in side): float(w) for side, w in zip(sides, weights)}


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "python"])
@pytest.mark.parametrize("trial", range(8))
def test_split_diversity_equals_pd(rng, monkeypatch, compiled, trial):
    """For every taxon set, the split diversity of the split system is its sum_j w_j PD_j."""
    monkeypatch.setattr(fpd.io.splits, "USE_COMPILED", compiled)
    for _ in range(trial):
        rng.random()
    n = rng.randint(3, 8)
    base = labelled_newicks(rng, n, 1)
    newicks = labelled_newicks(rng, n, 3) + base + base  # with a duplicated topology
    weights = [rng.uniform(0.2, 2.0) for _ in newicks]
    sides, split_weights, labels = fpd.utils.split_system(newicks, weights)
    instance = fpd.MaxMPDInstance(newicks, weights=weights)
    assert labels == instance.registry.labels
    assert len({frozenset(side.tolist()) for side in sides}) == len(sides)  # distinct splits
    assert all(0 not in side.tolist() for side in sides)  # the side without taxon 0
    for size in range(n + 1):
        for chosen in itertools.combinations(range(n), size):
            expected = instance.evaluate_pd([labels[x] for x in chosen])
            assert split_diversity(sides, split_weights, set(chosen)) == pytest.approx(expected)


@pytest.mark.parametrize("trial", range(6))
def test_compiled_and_python_split_systems_agree(rng, monkeypatch, trial):
    for _ in range(trial):
        rng.random()
    newicks = labelled_newicks(rng, rng.randint(2, 30), rng.randint(1, 6))
    monkeypatch.setattr(fpd.io.splits, "USE_COMPILED", True)
    compiled = splits_as_dict(*fpd.utils.split_system(newicks))
    monkeypatch.setattr(fpd.io.splits, "USE_COMPILED", False)
    python = splits_as_dict(*fpd.utils.split_system(newicks))
    assert compiled.keys() == python.keys()
    for side in compiled:
        assert compiled[side] == pytest.approx(python[side], rel=1e-12)


def read_nexus(path) -> tuple[list[str], dict[frozenset[str], float]]:
    """Taxon labels and ``{side: weight}`` from a file written by to_splits_nexus."""
    text = path.read_text()
    taxa_block = text.split("TAXLABELS")[1].split(";")[0]
    labels = [label.replace("''", "'") for label in re.findall(r"'((?:[^']|'')*)'", taxa_block)]
    matrix = text.split("MATRIX")[1].split(";")[0]
    splits = {}
    for row in re.findall(r"\[\d+, size=(\d+)\]\s+(\S+)\s+([\d ]+),", matrix):
        size, weight, members = row
        side = frozenset(labels[int(i) - 1] for i in members.split())
        assert len(side) == int(size)
        splits[side] = float(weight)
    return labels, splits


def test_nexus_file_round_trip(rng, tmp_path):
    newicks = labelled_newicks(rng, 9, 4)
    weights = [0.25] * 4
    path = tmp_path / "splits.nex"
    n_splits = fpd.io.to_splits_nexus(newicks, path, weights)
    labels, splits = read_nexus(path)
    expected = splits_as_dict(*fpd.utils.split_system(newicks, weights))
    assert n_splits == len(splits) == len(expected)
    assert labels == list(fpd.MaxMPDInstance(newicks).registry.labels)
    assert f"DIMENSIONS ntax=9 nsplits={n_splits};" in path.read_text()
    for side, weight in expected.items():
        assert splits[side] == pytest.approx(weight, rel=1e-15)


def test_nexus_quotes_labels(tmp_path):
    path = tmp_path / "quoted.nex"
    fpd.io.to_splits_nexus(["('it''s':1,'a b':1,c:1);"], path)
    labels, splits = read_nexus(path)
    assert labels == ["a b", "c", "it's"]
    assert splits[frozenset({"c"})] == 1.0


def test_phylotree_input(rng):
    """Pruned PhyloTrees go in directly; labels come from the registry, else the leaf ids."""
    newicks = labelled_newicks(rng, 8, 3, missing=(None, 6, None))
    trees, weights, registry = prune(newicks, {"x0", "x1", "x2", "x3"}, [0.5, 0.2, 0.3])
    sides, split_weights, labels = fpd.utils.split_system(trees, weights, labels=registry.labels)
    assert labels == registry.labels
    instance = fpd.MaxMPDInstance.from_phylotrees(trees, registry, weights=weights)
    assert split_diversity(sides, split_weights, {0, 1, 2, 3}) == pytest.approx(
        instance.evaluate_pd(registry.labels)
    )
    _, by_id, ids = fpd.utils.split_system(trees, weights)
    assert ids == ("0", "1", "2", "3")
    np.testing.assert_array_equal(by_id, split_weights)


@pytest.mark.parametrize(
    "call,message",
    [
        (lambda t, n: fpd.utils.split_system(t, labels=["a"]), "Expected 4 labels"),
        (lambda t, n: fpd.utils.split_system(n, labels=["a"]), "only for PhyloTree input"),
        (lambda t, n: fpd.utils.split_system([t[0], n[0]]), "Do not mix"),
        (lambda t, n: fpd.utils.split_system(n, weights=[1.0]), "Expected 3 weights"),
    ],
    ids=["label-count", "labels-with-text", "mixed-input", "weight-count"],
)
def test_split_system_input_errors(rng, call, message):
    newicks = labelled_newicks(rng, 4, 3)
    trees, _, _ = prune(newicks, {"x0", "x1", "x2", "x3"})
    with pytest.raises(ForestPDInputError, match=message):
        call(trees, newicks)


def test_trees_on_different_taxa_are_rejected():
    with pytest.raises(ForestPDInputError, match="same taxon set"):
        fpd.utils.split_system(["(A:1,B:1,C:1);", "(A:1,B:1,(C:1,D:1):1);"])


def test_keeping_every_taxon_changes_nothing(rng):
    newicks = labelled_newicks(rng, 6, 3)
    full = fpd.MaxMPDInstance(newicks)
    trees, weights, registry = prune(newicks, full.taxa)
    assert len(trees) == 3 and weights.tolist() == [1.0, 1.0, 1.0]
    for tree, original in zip(trees, full.trees):
        assert tree.total_weight() == pytest.approx(original.total_weight())
    instance = fpd.MaxMPDInstance.from_phylotrees(trees, registry, weights=weights)
    assert instance.optimize(3).value == pytest.approx(full.optimize(3).value)


def test_two_kept_taxa_leave_one_edge(rng):
    newicks = labelled_newicks(rng, 6, 2)
    full = fpd.MaxMPDInstance(newicks)
    trees, _, registry = prune(newicks, {"x1", "x4"})
    assert registry.labels == ("x1", "x4")
    for tree in trees:
        assert tree.n_nodes == 2 and tree.n_edges == 1
    instance = fpd.MaxMPDInstance.from_phylotrees(trees, registry)
    assert instance.optimize(2).value == pytest.approx(full.evaluate_pd({"x1", "x4"}))


def test_a_rooted_tree_is_pruned_as_unrooted():
    """The stem goes, and the root's two edges become one."""
    trees, _, _ = prune(["((A:1,B:2):3,(C:1,D:1):1):9;"], {"A", "B", "C", "D"})
    assert sorted(trees[0].weight.tolist()) == [1.0, 1.0, 1.0, 2.0, 4.0]


def test_weight_keys_are_tree_indices(rng):
    with pytest.raises(ForestPDInputError, match="indexed 0..1"):
        prune(labelled_newicks(rng, 5, 2), {"x0", "x1"}, weights={2: 1.0})


def test_split_system_of_tiny_trees():
    sides, weights, labels = fpd.utils.split_system(["(A:1,B:2);"])
    assert labels == ("A", "B") and weights.tolist() == [3.0]  # one split, the whole tree
    assert split_diversity(sides, weights, {0, 1}) == 3.0
    sides, weights, labels = fpd.utils.split_system(["(A:1,B:2,C:3);"], weights=[2.0])
    pendant = {
        side if len(side) == 1 else frozenset(labels) - side: w
        for side, w in splits_as_dict(sides, weights, labels).items()
    }  # whichever side is stored
    assert pendant == {frozenset({"A"}): 2.0, frozenset({"B"}): 4.0, frozenset({"C"}): 6.0}


def test_nexus_from_phylotrees_with_labels_and_mapping_weights(rng, tmp_path):
    newicks = labelled_newicks(rng, 6, 3)
    trees, _, registry = prune(newicks, {f"x{i}" for i in range(6)})
    path = tmp_path / "trees.nex"
    fpd.io.to_splits_nexus(trees, path, weights={1: 2.0}, labels=registry.labels)
    labels, splits = read_nexus(path)
    assert labels == list(registry.labels)
    expected = splits_as_dict(*fpd.utils.split_system(newicks, [1.0, 2.0, 1.0]))
    assert splits.keys() == expected.keys()
    for side, weight in expected.items():
        assert splits[side] == pytest.approx(weight)
