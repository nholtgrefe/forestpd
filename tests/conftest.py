"""Shared fixtures."""

from __future__ import annotations

import random

import pytest


@pytest.fixture
def rng() -> random.Random:
    """A seeded RNG, so every random test is reproducible."""
    return random.Random(20240813)


@pytest.fixture
def paper_trees():
    """
    The two input trees of the manuscript's definitely-chosen figure.

    Taxa ``a..f`` are leaves ``0..5``; internal nodes ``6..9`` are ``v_1..v_4`` in the first tree
    and ``w_1..w_4`` in the second. Edge weights are exactly the figure's.
    """
    from forestpd.core.tree import PhyloTree

    first = PhyloTree.from_edges(
        10,
        [(0, 6), (1, 6), (6, 7), (2, 7), (7, 8), (3, 8), (8, 9), (4, 9), (5, 9)],
        [2, 5, 3, 1, 5, 2, 4, 3, 1],
        {i: i for i in range(6)},
    )
    second = PhyloTree.from_edges(
        10,
        [(4, 6), (2, 6), (6, 7), (5, 7), (7, 8), (0, 8), (8, 9), (1, 9), (3, 9)],
        [4, 2, 3, 5, 1, 5, 2, 1, 4],
        {i: i for i in range(6)},
    )
    return first, second
