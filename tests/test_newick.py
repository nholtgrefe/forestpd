"""
The Newick parser: dialect, unrooting, error handling, and agreement with PhyloZoo.

The cross-check is the main safety net for owning a parser: random trees, written with every
feature the dialect allows, must come out as the same weighted tree whether read by
:func:`parse_newick` or by PhyloZoo.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

from forestpd.io.convert import build_registry, to_phylotree
from forestpd.io.newick import _parse_plain, _parse_tokens, parse_newick
from forestpd.core.tree import PhyloTree, leaf_masks_below
from forestpd.exceptions import ForestPDInputError


def weighted_splits(tree: PhyloTree) -> Counter:
    """The tree as a multiset of (split away from leaf 0, branch length): equal iff same tree."""
    below = leaf_masks_below(tree.root_at(0))
    return Counter((below[e], round(float(tree.weight[e]), 9)) for e in range(tree.n_edges))


def read(text: str, missing: float | str = "error") -> PhyloTree:
    """Parse and build a tree with a fresh registry."""
    return to_phylotree(parse_newick(text), missing_branch_length=missing)


def random_newick(rng: random.Random, n_leaves: int, *, missing: float = 0.0) -> str:
    """
    A random Newick tree using polytomies, internal labels, comments and missing lengths.

    The root is binary or multifurcating at random, so both unrooting paths get exercised.
    """
    nodes = [f"t{i}" for i in range(n_leaves)]
    rng.shuffle(nodes)

    def length() -> str:
        if rng.random() < missing:
            return ""
        return f":{rng.choice([rng.randint(0, 9), round(rng.uniform(0, 3), 4)])}"

    pool = [name + length() for name in nodes]
    while len(pool) > rng.choice([2, 3]):
        k = min(len(pool) - 1, rng.choice([2, 2, 2, 3]))
        picked = [pool.pop(rng.randrange(len(pool))) for _ in range(k)]
        inner = ",".join(picked)
        # Internal labels kept unique: PhyloZoo rejects repeated ones (ours ignores them).
        label = rng.choice(["", "", f"{len(pool)}{len(inner)}", "[&support=1]"])
        pool.append(f"({inner}){label}{length()}")
    return "(" + ",".join(pool) + ");"


class TestDialect:
    def test_binary_root_is_merged(self):
        tree = read("((A:1,B:1):1,(C:1,D:1):1);")
        assert tree.n_edges == 5
        assert sorted(tree.weight) == [1.0, 1.0, 1.0, 1.0, 2.0]

    def test_missing_lengths_merge_to_the_present_one(self):
        parsed = parse_newick("((A:1,B:1):1,(C:1,D:1));")
        assert sorted(parsed.lengths) == [1.0] * 5

    def test_quotes_comments_and_extra_fields(self):
        parsed = parse_newick("[&R] ('O''Brien':1,\"B c\":2:90:0.5,C_d:3)[note];")
        assert sorted(parsed.leaf_labels.values()) == ["B c", "C_d", "O'Brien"]
        assert sorted(parsed.lengths) == [1.0, 2.0, 3.0]

    def test_internal_labels_and_root_length_are_ignored(self):
        parsed = parse_newick("((A:1,B:1)90:1,C:1,D:1)root:5;")
        assert sorted(parsed.leaf_labels.values()) == ["A", "B", "C", "D"]
        assert sorted(parsed.lengths) == [1.0] * 5

    def test_unary_nodes_are_merged(self):
        assert sorted(parse_newick("(A:1,B:1,(C:2):3);").lengths) == [1.0, 1.0, 5.0]
        assert parse_newick("((A,B,C));").n_nodes == 4

    def test_tiny_trees(self):
        assert parse_newick("A;").n_nodes == 1
        assert parse_newick("(A:1,B:2);").lengths == [3.0]

    def test_semicolon_is_required(self):
        with pytest.raises(ForestPDInputError, match="does not end with ';'"):
            parse_newick("(A,B,C)")

    def test_two_trees_without_semicolon_are_rejected(self):
        for text in ("(A,B)(C,D);", "(A,B,C)\n(D,E,F);"):
            with pytest.raises(ForestPDInputError, match="missing between"):
                parse_newick(text)

    @pytest.mark.parametrize(
        "text,message",
        [
            ("(A,A,B);", "duplicate leaf label"),
            ("(A,,B);", "no label"),
            ("((A,B)#H1,(#H1,C),D);", "Reticulation"),
            ("(A,B));", "unbalanced"),
            ("((A,B);", "unbalanced"),
            ("(A:x,B);", "malformed branch length"),
            ("(A,B); (C,D);", "holds one tree"),
            ("('A,B);", "unterminated"),
        ],
    )
    def test_errors(self, text, message):
        with pytest.raises(ForestPDInputError, match=message):
            parse_newick(text)


class TestFastPath:
    """The vectorised path for large plain trees must agree exactly with the token-by-token one."""

    @pytest.mark.parametrize("trial", range(60))
    def test_same_result_as_token_path(self, trial):
        rng = random.Random(trial)
        text = random_newick(rng, rng.randint(2, 300), missing=0.3 if trial % 2 else 0.0)
        text = text.replace("[&support=1]", "")  # comments are left to the token path
        if trial % 3 == 0:
            text = text.replace("t1", "té", 1).replace(");", "):2.5;")
        for rooted in (False, True):
            fast = _parse_plain(text, rooted)
            assert fast is not None, text
            assert fast == _parse_tokens(text, rooted), text

    @pytest.mark.parametrize(
        "text",
        [
            "(A,(B,C)'x y');",  # quotes
            "(A,(B,C)[c]);",  # comment
            "(A, B,C);",  # whitespace
            "(A,(B),C);",  # a unary node
            "(A,B,C)x(D,E);",  # a second group under the root
            "(A:1:2,B,C);",  # an ignored extra field
            "(A,A,B);",
            "(A,,B);",
            "(A:x,B);",
            "(A,B));",
            "((A,B);",
            "(A,B);(C,D);",
            "(A,B)",
        ],
    )
    def test_unusual_or_malformed_input_agrees(self, text):
        for rooted in (False, True):
            fast = _parse_plain(text, rooted)
            try:
                slow = _parse_tokens(text, rooted)
            except ForestPDInputError:
                assert fast is None
            else:
                assert fast is None or fast == slow

    def test_large_tree_goes_through_parse_newick(self):
        caterpillar = "(" * 1500 + "t0:1,t1:2)" + "".join(f",t{i}:0.5)" for i in range(2, 1501))
        for rooted in (False, True):
            parsed = parse_newick(caterpillar + ":3;", rooted)
            assert _parse_plain(caterpillar + ":3;", rooted) is not None
            assert parsed == _parse_tokens(caterpillar + ":3;", rooted)
            assert len(parsed.leaf_labels) == 1501 + rooted


class TestAgainstPhyloZoo:
    @pytest.mark.parametrize("trial", range(40))
    def test_same_tree_as_phylozoo(self, trial):
        phylozoo = pytest.importorskip("phylozoo")
        from forestpd.io.pz import parsed_from_phylozoo

        rng = random.Random(trial)
        text = random_newick(rng, rng.randint(2, 30), missing=0.3 if trial % 2 else 0.0)
        ours = parse_newick(text)
        theirs = parsed_from_phylozoo(
            phylozoo.SemiDirectedPhyNetwork.from_string(text, format="enewick")
        )
        registry = build_registry([ours])
        assert set(theirs.leaf_labels.values()) == set(registry.labels)
        # A distinctive default, so a merge that mishandles missing lengths shows up.
        a = to_phylotree(ours, registry, missing_branch_length=7.0)
        b = to_phylotree(theirs, registry, missing_branch_length=7.0)
        assert weighted_splits(a) == weighted_splits(b), text


class TestArrayParsedTree:
    """The array form of ParsedTree and the fast PhyloTree constructor."""

    @pytest.mark.parametrize("trial", range(20))
    def test_from_leaf_arrays_equals_from_edges(self, trial):
        from forestpd.core.tree import PhyloTree

        rng = random.Random(trial)
        parsed = parse_newick(random_newick(rng, rng.randint(2, 200)))
        registry = build_registry([parsed])
        ids = np.array([registry.id_of(name) for name in parsed.leaf_names], dtype=np.int64)
        weights = np.nan_to_num(parsed.lengths)
        fast = PhyloTree.from_leaf_arrays(
            parsed.n_nodes, parsed.edges, weights, parsed.leaf_nodes, ids
        )
        slow = PhyloTree.from_edges(
            parsed.n_nodes,
            [tuple(e) for e in parsed.edges.tolist()],
            weights.tolist(),
            dict(zip(parsed.leaf_nodes.tolist(), ids.tolist())),
        )
        assert (fast.n_nodes, fast.n_leaves) == (slow.n_nodes, slow.n_leaves)
        np.testing.assert_array_equal(fast.edges, slow.edges)
        np.testing.assert_array_equal(fast.weight, slow.weight)

    def test_leaf_labels_view(self):
        parsed = parse_newick("((A:1,B:1):1,(C:1,D:1):1);")
        assert parsed.leaf_labels == dict(zip(parsed.leaf_nodes.tolist(), parsed.leaf_names))
        assert list(parsed.leaf_nodes) == sorted(parsed.leaf_nodes)

    def test_check_tree_rejects_non_trees(self):
        from forestpd.io.newick import ParsedTree

        labels = {0: "A", 1: "B", 2: "C", 3: "D"}
        with pytest.raises(ForestPDInputError, match="not connected"):
            ParsedTree.from_lists(
                5, [(0, 4), (1, 4), (2, 3), (3, 2)], [1.0] * 4, labels, check_tree=True
            )
        with pytest.raises(ForestPDInputError, match="needs 4 edges"):
            ParsedTree.from_lists(5, [(0, 4), (1, 4)], [1.0] * 2, labels, check_tree=True)

    def test_literal_nan_is_present_and_rejected(self):
        parsed = parse_newick("(A:nan,B:1,C:1);")
        assert parsed.has_length.all()
        with pytest.raises(ForestPDInputError, match="invalid branch length nan"):
            to_phylotree(parsed, missing_branch_length="zero")

    def test_missing_length_follows_the_policy(self):
        parsed = parse_newick("(A,B:1,C:1);")
        assert not parsed.has_length.all()
        assert to_phylotree(parsed, missing_branch_length=2.5).total_weight() == 4.5
        with pytest.raises(ForestPDInputError, match="has no branch length"):
            to_phylotree(parsed)


class TestCompiledStructurePass:
    """The compiled structure pass gives the NumPy path's result, and declines the same inputs."""

    @staticmethod
    def mutate(rng, text):
        """A random small edit: often makes the tree malformed or unusual."""
        i = rng.randrange(len(text) - 1)
        choice = rng.randrange(6)
        if choice == 0:
            return text[:i] + text[i + 1 :]  # drop a character
        if choice == 1:
            return text[:i] + rng.choice("(),:") + text[i:]  # insert a separator
        if choice == 2:
            return text.replace("t1,", "t2,", 1)  # a duplicate label
        if choice == 3:
            return text.replace(":", "::", 1)  # an empty extra field
        if choice == 4:
            return text.replace(",t3", ",(t3)", 1)  # a unary node
        return text[:i] + ":0.5" + text[i:]

    @pytest.mark.skipif(
        not __import__("forestpd.io._newick_kernel").io._newick_kernel.HAVE_NUMBA,
        reason="Numba not installed",
    )
    @pytest.mark.parametrize("trial", range(150))
    def test_same_as_numpy_path(self, monkeypatch, trial):
        from forestpd.io import newick as newick_module

        rng = random.Random(1000 + trial)
        text = random_newick(rng, rng.randint(2, 300), missing=0.3 if trial % 2 else 0.0)
        text = text.replace("[&support=1]", "")
        if trial % 3:
            text = self.mutate(rng, text)
        for rooted in (False, True):
            monkeypatch.setattr(newick_module, "USE_COMPILED", True)
            compiled = _parse_plain(text, rooted)
            monkeypatch.setattr(newick_module, "USE_COMPILED", False)
            numpy = _parse_plain(text, rooted)
            assert (compiled is None) == (numpy is None), text
            if compiled is not None:
                assert compiled == numpy, text
                assert compiled == _parse_tokens(text, rooted), text


class TestEdgeCases:
    def test_comma_outside_parentheses(self):
        with pytest.raises(ForestPDInputError, match="outside parentheses"):
            parse_newick("A,B;")

    def test_single_leaf_rooted(self):
        parsed = parse_newick("A:2.5;", rooted=True)
        assert sorted(parsed.leaf_labels.values())[-1] == "A"
        assert parsed.stem == 2.5

    def test_two_leaf_newick_without_lengths(self):
        from forestpd.io.newick import to_newick

        tree = to_phylotree(parse_newick("(A:1,B:2);"))
        assert to_newick(tree, ["A", "B"], lengths=False) == "(A,B);"
        assert to_newick(tree, ["A", "B"]) == "(A:3.0,B:0.0);"
