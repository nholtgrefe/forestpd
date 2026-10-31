Input and output
================

.. module:: forestpd.io

forestpd reads Newick trees itself (from a file or a list of strings, see :doc:`../manual`).
These functions write trees out: as Newick, and as one weighted split system for split-diversity
tools such as PDA.

.. autofunction:: to_newick

.. autofunction:: to_splits_nexus

PhyloZoo
--------

.. module:: forestpd.io.pz

Conversion between forestpd trees and `PhyloZoo <https://github.com/nholtgrefe/phylozoo>`_
networks, for plotting and the rest of the PhyloZoo ecosystem. Needs the optional extra
(``pip install "forestpd[phylozoo]"``); this is the only module that imports PhyloZoo.

.. autofunction:: from_phylozoo

.. autofunction:: to_phylozoo
