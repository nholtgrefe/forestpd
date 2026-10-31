"""Conversion to and from PhyloZoo (the optional ``phylozoo`` extra)."""

from __future__ import annotations

import pytest

import forestpd as fpd

phylozoo = pytest.importorskip("phylozoo")

from forestpd.io.pz import from_phylozoo, to_phylozoo  # noqa: E402

from .test_newick import weighted_splits  # noqa: E402

FOUR_TAXA = ["(A:1,B:1,(C:1,D:1):2);", "(A:1,C:1,(B:1,D:1):2);"]


def networks():
    return [
        phylozoo.SemiDirectedPhyNetwork.from_string(text, format="enewick") for text in FOUR_TAXA
    ]


def test_phylozoo_input_matches_newick_input():
    from_newick = fpd.optimize_maxmpd(FOUR_TAXA, budget=2)
    from_networks = fpd.optimize_maxmpd(networks(), budget=2)
    assert from_networks.value == pytest.approx(from_newick.value)


def test_mixed_inputs_share_one_registry():
    mixed = [FOUR_TAXA[0], networks()[1]]
    assert fpd.optimize_maxmpd(mixed, budget=2).value == pytest.approx(8.0)


def test_round_trip_preserves_the_tree():
    instance = fpd.MaxMPDInstance(FOUR_TAXA)
    for tree in instance.trees:
        back = from_phylozoo(to_phylozoo(tree, instance.registry), instance.registry)
        assert weighted_splits(back) == weighted_splits(tree)


def test_networks_with_reticulations_are_rejected():
    net = phylozoo.SemiDirectedPhyNetwork.from_string("((A,(B)#H1),(#H1,C),D);", format="enewick")
    with pytest.raises(fpd.ForestPDInputError, match="not a tree"):
        fpd.optimize_maxmpd([net], budget=2, missing_branch_length=1.0)
