"""
Public-API behaviour: coercion, validation, error messages, and the frontend/core firewall.

Everything here goes through the documented entry points, so it doubles as a check that the
README's quickstart and the docstring examples actually run.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys
import warnings

import numpy as np
import pytest

import forestpd as fpd
from forestpd.exceptions import ForestPDInputError
from forestpd.io.newick import ROOT_LABEL

from .fixtures.baselines import MAFS

# Unrooted trees written with a multifurcating top level: each totals 6 and
# PD({A,D}) = 1 + 2 + 1 = 4 per tree.
FOUR_TAXA = ["(A:1,B:1,(C:1,D:1):2);", "(A:1,C:1,(B:1,D:1):2);"]
FOUR_TAXA_OPTIMUM = 8.0


class TestOptimizeMaxMPD:
    def test_quickstart(self):
        result = fpd.optimize_maxmpd(FOUR_TAXA, budget=2, return_taxa=True)
        assert result.value == pytest.approx(FOUR_TAXA_OPTIMUM)
        assert result.taxa in ({"A", "D"}, {"B", "C"})

    def test_timings(self):
        instance = fpd.MaxMPDInstance(FOUR_TAXA)
        first, second = instance.optimize(2).meta, instance.optimize(3).meta
        assert first["parse_seconds"] == second["parse_seconds"] > 0  # once, at construction
        for meta in (first, second):
            assert meta["dp_seconds"] == pytest.approx(meta["solve_seconds"] - meta["maf_seconds"])
            assert meta["dp_seconds"] >= 0
        built = fpd.MaxMPDInstance.from_phylotrees(instance.trees, instance.registry)
        assert built.optimize(2).meta["parse_seconds"] > 0  # the validation, timed

    def test_meta_is_populated(self):
        meta = fpd.optimize_maxmpd(FOUR_TAXA, budget=2).meta
        assert meta["cluster_sizes"] == [4]
        assert meta["n_trees"] == 2
        assert meta["n_taxa"] == 4
        assert meta["max_maf_size"] >= 1
        assert meta["solve_seconds"] >= 0.0
        assert meta["maf"] == "HeuristicMAF"

    def test_costs_and_weights_are_honoured(self):
        cheap = fpd.optimize_maxmpd(FOUR_TAXA, budget=2, costs={"A": 1, "B": 1, "C": 1, "D": 1})
        dear = fpd.optimize_maxmpd(FOUR_TAXA, budget=2, costs={"A": 2, "B": 2, "C": 2, "D": 2})
        assert cheap.value > dear.value  # nothing is affordable in pairs at cost 2

    def test_return_taxa_defaults_to_none(self):
        assert fpd.optimize_maxmpd(FOUR_TAXA, budget=2).taxa is None

    def test_pd_matches_the_optimizer(self):
        result = fpd.optimize_maxmpd(FOUR_TAXA, budget=2, return_taxa=True)
        assert fpd.MaxMPDInstance(FOUR_TAXA).evaluate_pd(result.taxa) == pytest.approx(result.value)

    def test_maf_stage_can_be_swapped(self):
        for maf in MAFS.values():
            value = fpd.optimize_maxmpd(FOUR_TAXA, budget=2, maf=maf()).value
            assert value == pytest.approx(FOUR_TAXA_OPTIMUM)


class TestOneCallFunctions:
    """Each one-call function equals building the instance and calling the method."""

    SURVIVAL = {"A": 0.3, "D": 0.8}

    def test_optimize_many(self):
        instance = fpd.MaxMPDInstance(FOUR_TAXA, survival=self.SURVIVAL)
        expected = instance.optimize_many([0, 1, 2, 3], return_taxa=True)
        results = fpd.optimize_many_maxmpd(
            FOUR_TAXA, [0, 1, 2, 3], survival=self.SURVIVAL, return_taxa=True
        )
        assert [r.value for r in results] == [r.value for r in expected]
        assert [r.taxa for r in results] == [r.taxa for r in expected]

    def test_evaluate(self):
        instance = fpd.MaxMPDInstance(FOUR_TAXA, weights=[1.0, 2.0], survival=self.SURVIVAL)
        for taxa in (set(), {"A"}, {"B", "C"}, {"A", "B", "C", "D"}):
            value = fpd.evaluate_maxmpd(FOUR_TAXA, taxa, weights=[1.0, 2.0], survival=self.SURVIVAL)
            assert value == instance.evaluate_pd(taxa)

    def test_evaluate_rejects_unknown_taxa(self):
        with pytest.raises(ForestPDInputError):
            fpd.evaluate_maxmpd(FOUR_TAXA, {"Z"})

    def test_optimize_takes_survival(self):
        result = fpd.optimize_maxmpd(FOUR_TAXA, 2, survival=self.SURVIVAL)
        assert (
            result.value == fpd.MaxMPDInstance(FOUR_TAXA, survival=self.SURVIVAL).optimize(2).value
        )


class TestMaxMPDInstance:
    def test_one_instance_many_budgets(self):
        instance = fpd.MaxMPDInstance(FOUR_TAXA)
        for budget in range(5):
            assert instance.optimize(budget).value == pytest.approx(
                fpd.optimize_maxmpd(FOUR_TAXA, budget=budget).value
            )

    def test_frontier_matches_separate_solves(self):
        instance = fpd.MaxMPDInstance(FOUR_TAXA, costs={"A": 2})
        # Every budget 0..5 at once: DP for the small ones, DP' for the large ones.
        result = instance.optimize_many(range(6))[-1]
        assert result.meta["tables"] == "both"
        frontier = result.meta["frontier"]
        assert not np.isnan(frontier).any()
        for budget, value in enumerate(frontier):
            assert instance.optimize(budget).value == pytest.approx(value)

    def test_large_single_budget_uses_the_complement(self):
        """B > C - B: DP' alone, covering only the budgets from B on (the rest NaN)."""
        result = fpd.MaxMPDInstance(FOUR_TAXA).optimize(3)
        assert result.meta["tables"] == "DP'"
        assert np.isnan(result.meta["frontier"][:3]).all()
        assert result.meta["frontier"][3] == result.value

    def test_properties(self):
        instance = fpd.MaxMPDInstance(FOUR_TAXA, weights=[1.0, 2.0])
        assert (instance.n_trees, instance.n_taxa) == (2, 4)
        assert instance.taxa == ("A", "B", "C", "D")
        assert list(instance.weights) == [1.0, 2.0]

    def test_costs_by_taxon_id(self):
        instance = fpd.MaxMPDInstance(FOUR_TAXA)
        same = fpd.MaxMPDInstance.from_phylotrees(instance.trees, instance.registry, [2, 1, 1, 1])
        assert list(same.costs) == [2, 1, 1, 1]
        with pytest.raises(ForestPDInputError, match="Expected 4 costs"):
            fpd.MaxMPDInstance.from_phylotrees(instance.trees, instance.registry, [1, 1])

    def test_degree_two_vertices_are_rejected(self):
        from forestpd.core.tree import PhyloTree, suppress_degree_two

        # 0 - 4 - 5 - (1, 2, 3): node 4 has degree 2.
        tree = PhyloTree.from_edges(
            6, [(0, 4), (4, 5), (1, 5), (2, 5), (3, 5)], [1.0] * 5, {i: i for i in range(4)}
        )
        registry = fpd.TaxonRegistry(["A", "B", "C", "D"])
        with pytest.raises(ForestPDInputError, match="Tree 1 has an internal vertex of degree 2"):
            fpd.MaxMPDInstance.from_phylotrees([suppress_degree_two(tree), tree], registry)
        fixed = fpd.MaxMPDInstance.from_phylotrees([suppress_degree_two(tree)] * 2, registry)
        assert fixed.evaluate_pd({"A", "B"}) == pytest.approx(2 * 3.0)  # 0-5 is 2 after merging

    def test_budget_is_validated_per_call(self):
        instance = fpd.MaxMPDInstance(FOUR_TAXA)
        with pytest.raises(ForestPDInputError, match="non-negative"):
            instance.optimize(-1)


class TestTreeFile:
    def test_one_tree_per_line(self, tmp_path):
        path = tmp_path / "trees.nwk"
        path.write_text("\n".join(FOUR_TAXA) + "\n")
        assert fpd.optimize_maxmpd(path, budget=2).value == pytest.approx(FOUR_TAXA_OPTIMUM)
        assert fpd.optimize_maxmpd(str(path), budget=2).value == pytest.approx(FOUR_TAXA_OPTIMUM)

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "trees.nwk"
        path.write_text("\n" + FOUR_TAXA[0] + "\n\n   \n" + FOUR_TAXA[1] + "\n\n")
        assert fpd.MaxMPDInstance(path).n_trees == 2

    def test_one_tree_per_line_is_required(self, tmp_path):
        path = tmp_path / "trees.nwk"
        path.write_text(FOUR_TAXA[0] + " " + FOUR_TAXA[1] + "\n")  # two trees on one line
        with pytest.raises(ForestPDInputError, match="line 1.*text after the end"):
            fpd.MaxMPDInstance(path)
        path.write_text("(A:1,B:1,\n(C:1,D:1):2);\n")  # one tree over two lines
        with pytest.raises(ForestPDInputError, match="line 1.*unbalanced"):
            fpd.MaxMPDInstance(path)

    def test_weights_follow_file_order(self, tmp_path):
        path = tmp_path / "trees.nwk"
        path.write_text("\n".join(FOUR_TAXA))
        from_file = fpd.optimize_maxmpd(path, budget=2, weights=[2.0, 0.5]).value
        from_list = fpd.optimize_maxmpd(FOUR_TAXA, budget=2, weights=[2.0, 0.5]).value
        assert from_file == pytest.approx(from_list)

    def test_lines_without_semicolons_are_rejected(self, tmp_path):
        path = tmp_path / "trees.nwk"
        path.write_text("(A:1,B:1,(C:1,D:1):2)\n(A:1,C:1,(B:1,D:1):2)\n")
        with pytest.raises(ForestPDInputError, match="line 1.*does not end with ';'"):
            fpd.MaxMPDInstance(path)

    def test_errors_name_the_tree(self, tmp_path):
        path = tmp_path / "trees.nwk"
        path.write_text(FOUR_TAXA[0] + "\n(A:1,B:1,(C:1,D:1):x);\n")
        with pytest.raises(ForestPDInputError, match="line 2"):
            fpd.MaxMPDInstance(path)

    def test_paths_inside_a_list_are_rejected(self, tmp_path):
        path = tmp_path / "trees.nwk"
        path.write_text(FOUR_TAXA[0])
        for element in (path, str(path)):
            with pytest.raises(ForestPDInputError, match="not inside a list"):
                fpd.MaxMPDInstance([element])

    def test_a_single_newick_string_is_rejected(self):
        with pytest.raises(ForestPDInputError, match="single Newick string"):
            fpd.MaxMPDInstance(FOUR_TAXA[0])

    def test_missing_and_empty_files(self, tmp_path):
        with pytest.raises(ForestPDInputError, match="not found"):
            fpd.MaxMPDInstance(tmp_path / "absent.nwk")
        empty = tmp_path / "empty.nwk"
        empty.write_text("\n   \n")
        with pytest.raises(ForestPDInputError, match="contains no trees"):
            fpd.MaxMPDInstance(empty)

    @pytest.mark.parametrize("rooted", [False, True])
    def test_converting_while_reading_equals_reading_first(self, tmp_path, rooted):
        """Trees are converted one at a time as they are read; the result must be exactly that of
        reading every tree first (registry, trees, stems), from a file (with blank lines and
        Windows line ends) and from a list."""
        from forestpd.io.convert import _coerce_all, coerce_trees

        newicks = [
            "((A:1,B:2):0.5,(C:1,(D:3,E:1):2):1,F:4);",
            "(((A:1,C:2):1,B:1):2,(D:1,F:1):1,E:2.5);",
            "(F:1,E:1,D:1,C:1,(A:0.5,B:0.5):3)[&R];",
        ]
        path = tmp_path / "trees.nwk"
        path.write_bytes(("\r\n".join(newicks[:2]) + "\r\n\r\n" + newicks[2] + "\n").encode())
        for source in (path, newicks):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", fpd.ForestPDWarning)
                fast = coerce_trees(source, "error", rooted)
                reference = _coerce_all(source, "error", rooted)
            assert fast[1].labels == reference[1].labels
            assert fast[2] == reference[2]
            for got, want in zip(fast[0], reference[0], strict=True):
                assert got.n_nodes == want.n_nodes and got.n_leaves == want.n_leaves
                assert np.array_equal(got.edges, want.edges)
                assert np.array_equal(got.weight, want.weight)

    @pytest.mark.parametrize(
        "second",
        [
            "((A:1,B:1):1,C:1,E:1);",  # other taxa, as many
            "((A:1,B:1):1,C:1,D:1,E:1);",  # an extra taxon
            "((A:1,B:1):1,C,D:1);",  # a missing branch length
            f"((A:1,B:1):1,C:1,{ROOT_LABEL}:1);",  # the reserved root label
        ],
    )
    def test_irregular_input_gives_the_same_result(self, second):
        """A later tree with another taxon set, a missing branch length or the reserved root label
        falls back to reading every tree first: the same error, or the same trees."""
        from forestpd.io.convert import _coerce_all, coerce_trees

        trees = ["((A:1,B:1):1,C:1,D:1);", second]
        try:
            reference = _coerce_all(trees, "error", False)
        except ForestPDInputError as exc:
            with pytest.raises(ForestPDInputError) as raised:
                coerce_trees(trees, "error", False)
            assert str(raised.value) == str(exc)
        else:
            fast = coerce_trees(trees, "error", False)
            assert fast[1].labels == reference[1].labels
            for got, want in zip(fast[0], reference[0], strict=True):
                assert np.array_equal(got.edges, want.edges)

    def test_rooting_warnings_point_at_the_caller(self):
        """On both paths, as before: the warning names this file, not forestpd's."""
        rooted_looking = ["((A:1,B:1):1,(C:1,D:1):1);", "((A:1,C:1):1,(B:1,D:1):1);"]
        with pytest.warns(fpd.ForestPDWarning) as record:
            fpd.MaxMPDInstance(rooted_looking)
        assert record[0].filename == __file__
        with pytest.warns(fpd.ForestPDWarning) as record:
            with pytest.raises(ForestPDInputError, match="no branch length"):
                fpd.MaxMPDInstance(rooted_looking + ["((A:1,B:1):1,(C,D:1):1);"])
        assert record[0].filename == __file__


class TestValidation:
    def test_negative_budget(self):
        with pytest.raises(ForestPDInputError, match="non-negative"):
            fpd.optimize_maxmpd(FOUR_TAXA, budget=-1)

    def test_non_integer_budget(self):
        with pytest.raises(ForestPDInputError, match="must be an integer"):
            fpd.optimize_maxmpd(FOUR_TAXA, budget=2.5)

    def test_zero_cost_is_rejected(self):
        """The DP equates "selected" with "given positive budget", so free taxa have no encoding."""
        with pytest.raises(ForestPDInputError, match="at least 1"):
            fpd.optimize_maxmpd(FOUR_TAXA, budget=2, costs={"A": 0})

    def test_unknown_cost_label(self):
        with pytest.raises(ForestPDInputError, match="unknown taxon"):
            fpd.optimize_maxmpd(FOUR_TAXA, budget=2, costs={"Z": 1})

    def test_wrong_weight_count(self):
        with pytest.raises(ForestPDInputError, match="Expected 2 weights"):
            fpd.optimize_maxmpd(FOUR_TAXA, budget=2, weights=[1.0])

    def test_no_trees(self):
        with pytest.raises(ForestPDInputError, match="At least one tree"):
            fpd.optimize_maxmpd([], budget=2)

    def test_missing_branch_length_policies(self):
        no_lengths = ["(A,B,(C,D));", "(A,C,(B,D));"]
        with pytest.raises(ForestPDInputError, match="no branch length"):
            fpd.optimize_maxmpd(no_lengths, budget=2)
        assert fpd.optimize_maxmpd(no_lengths, budget=2, missing_branch_length="zero").value == 0.0
        assert fpd.optimize_maxmpd(no_lengths, budget=2, missing_branch_length=1.0).value == 6.0

    def test_garbage_input(self):
        with pytest.raises(ForestPDInputError):
            fpd.optimize_maxmpd(["not a tree at all"], budget=2)

    def test_non_uniform_leaf_sets_are_rejected(self):
        """
        Differing leaf sets are a hardness barrier, not a missing feature.

        The message has to say so, or a user will reasonably assume it is a bug.
        """
        with pytest.raises(ForestPDInputError, match="same taxon set"):
            fpd.optimize_maxmpd(["(A,B,(C,D));", "(A,B,C);"], budget=2, missing_branch_length=1.0)

    @pytest.mark.parametrize(
        ("trees", "message"),
        [
            (["(A,B,(C,D));", "(A,B,(C,E));"], "Tree 0 lacks 1 of the 5 taxa"),  # as many, others
            (["(A,B,(C,D));", "(A,B,D);"], "Tree 1 lacks 1 of the 4 taxa"),  # fewer, not ids 0..2
            (["(A,B,D);", "(A,B,(C,D));"], "Tree 0 lacks 1 of the 4 taxa"),  # the first has fewer
        ],
    )
    def test_every_taxon_set_mismatch_gets_the_explanatory_error(self, trees, message):
        """
        Whatever the counts, a tree without some taxon of the others gets the explanatory error,
        naming the first such tree, not an internal one about leaf ids.
        """
        with pytest.raises(ForestPDInputError, match="same taxon set") as error:
            fpd.MaxMPDInstance(trees, missing_branch_length=1.0)
        assert message in str(error.value)


class TestFirewall:
    def test_only_io_pz_imports_phylozoo(self):
        """
        PhyloZoo (and NetworkX beneath it) is an optional extra; only ``io/pz.py`` may import it.
        """
        root = pathlib.Path(__file__).resolve().parents[1] / "src" / "forestpd"
        offenders = []
        for path in root.rglob("*.py"):
            if path.relative_to(root).as_posix() == "io/pz.py":
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                else:
                    continue
                if any(n.split(".")[0] in ("phylozoo", "networkx") for n in names):
                    offenders.append(str(path.relative_to(root)))
        assert not offenders, f"PhyloZoo leaked into the core: {offenders}"

    def test_runs_without_phylozoo_installed(self):
        """The whole Newick-to-optimum path works with PhyloZoo and NetworkX unimportable."""
        src = pathlib.Path(__file__).resolve().parents[1] / "src"
        script = f"""
import sys
sys.modules["phylozoo"] = None
sys.modules["networkx"] = None
sys.path.insert(0, {str(src)!r})
import forestpd as fpd
result = fpd.optimize_maxmpd({FOUR_TAXA!r}, budget=2, return_taxa=True)
assert abs(result.value - {FOUR_TAXA_OPTIMUM!r}) < 1e-9, result.value
try:
    import forestpd.io.pz
except fpd.ForestPDImportError:
    pass
else:
    raise AssertionError("io.pz imported without phylozoo")
"""
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
        )
        assert completed.returncode == 0, completed.stderr

    @pytest.mark.parametrize("numba", [True, False], ids=["with_numba", "without_numba"])
    def test_warns_on_import_only_without_numba(self, numba):
        """Without Numba, importing forestpd warns once and the pipeline still gives the optimum;
        with Numba, importing it warns about nothing."""
        src = pathlib.Path(__file__).resolve().parents[1] / "src"
        script = f"""
import sys, warnings
if not {numba!r}:
    sys.modules["numba"] = None
sys.path.insert(0, {str(src)!r})
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import forestpd as fpd
messages = [str(w.message) for w in caught if issubclass(w.category, fpd.ForestPDWarning)]
assert len(messages) == (0 if {numba!r} else 1), messages
assert all("Numba could not be imported" in m for m in messages), messages
result = fpd.optimize_maxmpd({FOUR_TAXA!r}, budget=2)
assert abs(result.value - {FOUR_TAXA_OPTIMUM!r}) < 1e-9, result.value
"""
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
        )
        assert completed.returncode == 0, completed.stderr


def test_package_exports_are_importable():
    for name in fpd.__all__:
        assert hasattr(fpd, name), name


class TestStructure:
    """``structure()`` reports what ``optimize`` would see, without the DP."""

    @staticmethod
    def instances(rng):
        from .fixtures.random_instances import clade_localized_instance, related_instance

        yield related_instance(rng, n_leaves=10, n_trees=3).instance
        yield clade_localized_instance(rng, n_clades=4, clade_size=4, n_trees=3, budget=5).instance
        yield fpd.MaxMPDInstance(
            ["((A:1,B:1):1,(C:1,(D:1,E:1):1):1);", "((A:1,C:1):1,(B:1,(D:1,E:1):1):1);"],
            rooted=True,
        )

    @pytest.mark.parametrize("maf", sorted(MAFS))
    def test_matches_optimize_meta(self, rng, maf):
        for instance in self.instances(rng):
            structure = instance.structure(maf=MAFS[maf]())
            meta = instance.optimize(3, maf=MAFS[maf]()).meta
            assert structure.n_clusters == meta["n_clusters"]
            assert list(structure.maf_sizes) == meta["maf_sizes"]
            assert structure.max_maf_size == meta["max_maf_size"]
            assert structure.n_signatures == meta["n_signatures"]
            assert list(structure.cluster_sizes) == meta["cluster_sizes"]

    def test_cluster_sizes_count_own_taxa_and_children(self, rng):
        from forestpd.cluster import cluster_reduction

        instance = next(self.instances(rng))
        structure = instance.structure()
        ktree = cluster_reduction(instance.trees, weights=instance.weights)
        expected = [
            sum(t >= 0 for t in ktree.pieces[c].taxon_of_leaf) + len(ktree.children[c])
            for c in ktree.postorder
        ]
        assert list(structure.cluster_sizes) == expected

    def test_reduction_is_cached_for_optimize(self, rng):
        instance = next(self.instances(rng))
        instance.structure()
        cached = instance._cluster_tree
        assert cached is not None
        instance.optimize(2)
        assert instance._cluster_tree is cached  # the same reduction object, not rebuilt


class TestInputEdgeCases:
    """Invalid inputs and degenerate instances at the API boundary."""

    @pytest.mark.parametrize("weights", [[-1.0, 1.0], [float("nan"), 1.0], [float("inf"), 1.0]])
    def test_invalid_tree_weights(self, weights):
        with pytest.raises(ForestPDInputError, match="finite and non-negative"):
            fpd.MaxMPDInstance(FOUR_TAXA, weights=weights)

    @pytest.mark.parametrize("cost", [1.5, True, "2"])
    def test_non_integer_costs(self, cost):
        with pytest.raises(ForestPDInputError, match="must be an integer"):
            fpd.MaxMPDInstance(FOUR_TAXA, costs={"A": cost})

    @pytest.mark.parametrize("policy", [float("nan"), float("inf"), "maybe", True])
    def test_invalid_missing_branch_length_policy(self, policy):
        with pytest.raises(ForestPDInputError, match="missing_branch_length"):
            fpd.MaxMPDInstance(["(A,B,(C,D));"], missing_branch_length=policy)

    def test_negative_branch_length(self):
        with pytest.raises(ForestPDInputError, match="invalid branch length"):
            fpd.MaxMPDInstance(["(A:1,B:-1,(C:1,D:1):1);"])

    def test_unsupported_tree_input(self):
        with pytest.raises(ForestPDInputError, match="Unsupported tree input"):
            fpd.MaxMPDInstance([42])

    def test_unreadable_tree_file(self, tmp_path):
        from forestpd.io.convert import _iter_file

        with pytest.raises(ForestPDInputError, match="not found"):
            fpd.MaxMPDInstance(tmp_path / "missing.nwk")
        with pytest.raises(ForestPDInputError, match="Could not read"):
            list(_iter_file(str(tmp_path)))  # a directory cannot be read as a file

    def test_from_phylotrees_needs_a_tree(self):
        with pytest.raises(ForestPDInputError, match="At least one tree"):
            fpd.MaxMPDInstance.from_phylotrees([], fpd.TaxonRegistry(["A", "B"]))

    def test_repr(self):
        assert repr(fpd.MaxMPDInstance(FOUR_TAXA)) == "MaxMPDInstance(n_trees=2, n_taxa=4)"

    def test_two_taxa(self):
        with pytest.warns(fpd.ForestPDWarning, match="appear to be rooted"):  # a binary root
            instance = fpd.MaxMPDInstance(["(A:1,B:2);", "(A:3,B:1);"], survival={"B": 0.5})
        assert instance.evaluate_pd(set()) == 0.0
        assert instance.evaluate_pd({"A"}) == pytest.approx(0.5 * 7)  # B survives half the time
        assert instance.optimize(2).value == pytest.approx(7.0)
        assert instance.structure().n_clusters >= 1


class FixedForest:
    """A MAF stage returning one fixed forest, with or without ``trusted``."""

    def __init__(self, forest, trusted):
        self.forest = forest
        self.trusted = trusted

    def compute(self, trees):
        return self.forest


class TestMAFStageOutputIsChecked:
    """Whatever a MAF stage returns must be a partition of the cluster's leaves (FOUR_TAXA is one
    cluster of four leaves)."""

    @pytest.mark.parametrize("trusted", [False, True], ids=["untrusted", "trusted"])
    @pytest.mark.parametrize(
        "forest,message",
        [
            (((0, 1), (2,)), "left out"),
            (((0, 1), (1, 2, 3)), "twice"),
            (((0, 1, 2), (3, 7)), "out of range"),
            (((0,), (1,), (2,), (3,), (-1,)), "out of range"),
            (((0, 1), (), (2, 3)), "empty component"),
            (((0, 1), (2.0, 3)), "non-integer"),
        ],
        ids=["missing", "overlap", "too-large", "negative", "empty", "float"],
    )
    def test_bad_partitions(self, trusted, forest, message):
        instance = fpd.MaxMPDInstance(FOUR_TAXA)
        stage = FixedForest(forest, trusted)
        with pytest.raises(ForestPDInputError, match=message):
            instance.optimize(2, maf=stage)
        with pytest.raises(ForestPDInputError, match=message):
            instance.structure(maf=stage)

    @pytest.mark.parametrize("maf", [object(), fpd.maf.HeuristicMAF], ids=["object", "a-class"])
    def test_maf_must_be_a_computer_instance(self, maf):
        instance = fpd.MaxMPDInstance(FOUR_TAXA)
        with pytest.raises(ForestPDInputError, match="MAF computer instance"):
            instance.optimize(2, maf=maf)
        with pytest.raises(ForestPDInputError, match="MAF computer instance"):
            instance.structure(maf=maf)


class TestSmallAndOddInput:
    def test_one_taxon(self):
        for rooted in (False, True):
            instance = fpd.MaxMPDInstance(["(A:1);"], survival={"A": 0.5}, rooted=rooted)
            result = instance.optimize(1, return_taxa=True)
            structure = instance.structure()
            assert structure.n_clusters == 1
            if rooted:  # the stem counts when A survives: surely if protected, else half the time
                assert result.value == pytest.approx(1.0) and result.taxa == {"A"}
                assert instance.evaluate_pd(()) == pytest.approx(0.5)
            else:  # a single taxon spans nothing
                assert result.value == 0.0 and result.taxa == set()

    def test_bare_single_taxon_newick(self):
        instance = fpd.MaxMPDInstance(["A;", "A;"])
        assert instance.taxa == ("A",) and instance.optimize(1).value == 0.0

    def test_single_leaf_newick_round_trip(self):
        tree = fpd.MaxMPDInstance(["A;"]).trees[0]
        assert fpd.MaxMPDInstance([fpd.io.to_newick(tree, ["A"])]).taxa == ("A",)

    @pytest.mark.parametrize("weights", [{5: 2.0}, {"0": 2.0}, {-1: 1.0}, {True: 1.0}])
    def test_weight_mapping_keys_must_be_tree_indices(self, weights):
        with pytest.raises(ForestPDInputError, match="indexed 0..1"):
            fpd.MaxMPDInstance(FOUR_TAXA, weights=weights)
        with pytest.raises(ForestPDInputError, match="indexed 0..1"):
            fpd.utils.prune(FOUR_TAXA, {"A", "B"}, weights)
        with pytest.raises(ForestPDInputError, match="indexed 0..1"):
            fpd.utils.split_system(FOUR_TAXA, weights)

    def test_numpy_integer_budgets(self):
        import numpy as np

        instance = fpd.MaxMPDInstance(FOUR_TAXA)
        assert instance.optimize(np.int64(2)).value == FOUR_TAXA_OPTIMUM
        assert [r.value for r in instance.optimize_many(np.arange(3))] == [0.0, 0.0, 8.0]
        with pytest.raises(ForestPDInputError, match="integer"):
            instance.optimize(np.float64(2.0))
        with pytest.raises(ForestPDInputError, match="integer"):
            instance.optimize(np.bool_(True))

    def test_a_string_is_not_a_set_of_labels(self):
        instance = fpd.MaxMPDInstance(FOUR_TAXA)
        with pytest.raises(ForestPDInputError, match="collection of taxon labels"):
            instance.evaluate_pd("AB")
        with pytest.raises(ForestPDInputError, match="collection of taxon labels"):
            fpd.utils.prune(FOUR_TAXA, "AB")
        assert instance.evaluate_pd(["A", "B"]) == 6.0

    def test_unknown_label_in_evaluate(self):
        with pytest.raises(ForestPDInputError, match="Unknown taxon"):
            fpd.MaxMPDInstance(FOUR_TAXA).evaluate_pd({"A", "Z"})


class TestBudgetsAndMeta:
    def test_frontier_ends_at_the_total_cost(self):
        instance = fpd.MaxMPDInstance(FOUR_TAXA)  # total cost 4
        everything = instance.evaluate_pd(instance.taxa)
        for budget, length in ((0, 1), (2, 3), (4, 5), (10, 5), (10**12, 5)):
            result = instance.optimize(budget)
            assert result.meta["budget"] == budget
            assert result.meta["frontier"].shape == (length,)
            if budget >= 4:  # a budget past the total cost buys everything, and no more
                assert result.value == result.meta["frontier"][-1] == everything

    def test_optimize_many_inputs(self):
        instance = fpd.MaxMPDInstance(FOUR_TAXA)
        assert instance.optimize_many([]) == []
        with pytest.raises(ForestPDInputError, match="non-negative"):
            instance.optimize_many([1, -1])
        results = instance.optimize_many(iter([2, 2, 0, 100, 100]), return_taxa=True)
        assert [r.meta["budget"] for r in results] == [2, 2, 0, 100, 100]
        assert [len(r.meta["frontier"]) for r in results] == [3, 3, 1, 5, 5]
        assert results[0].taxa in ({"A", "D"}, {"B", "C"}) and results[2].taxa == set()
        for result in results:
            cost = sum(int(instance.costs[instance.registry.id_of(x)]) for x in result.taxa)
            assert cost <= result.meta["budget"]
            assert instance.evaluate_pd(result.taxa) == pytest.approx(result.value)

    def test_results_do_not_share_meta(self):
        first, second = fpd.MaxMPDInstance(FOUR_TAXA).optimize_many([3, 3])
        first.meta["frontier"][-1] = 99.0
        first.meta["maf_sizes"].append(99)
        first.meta["cluster_sizes"].append(99)
        assert second.meta["frontier"][-1] == second.value != 99.0
        assert 99 not in second.meta["maf_sizes"] and 99 not in second.meta["cluster_sizes"]
