"""
Getting trees in and out.

For users: :func:`to_newick` writes a tree as Newick text, :func:`to_splits_nexus` writes a set of
trees as one weighted split system (NEXUS, e.g. for PDA), and :mod:`forestpd.io.pz` converts to and
from PhyloZoo networks (with the optional ``phylozoo`` extra; not imported here).

Internally, :mod:`.newick` parses Newick text and :mod:`.convert` turns any accepted tree input
into :class:`~forestpd.core.tree.PhyloTree` objects sharing one
:class:`~forestpd.core.taxa.TaxonRegistry`.
"""

from __future__ import annotations

from .newick import to_newick
from .splits import to_splits_nexus

__all__ = ["to_newick", "to_splits_nexus"]
