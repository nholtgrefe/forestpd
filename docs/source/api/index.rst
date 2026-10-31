API reference
=============

The public API is small and kept stable. Import everything from the top level as usual
(``import forestpd as fpd``); the pages below document each name where users import it from.
Everything not listed here (``forestpd.core``, ``forestpd.cluster``, ``forestpd.algo``, and the
rest of ``forestpd.io`` and ``forestpd.maf``) is internal.

.. list-table:: Where to find what
   :widths: 45 55
   :header-rows: 1

   * - Names
     - Page
   * - ``MaxMPDInstance``, ``MaxMPDResult``, ``MaxMPDStructure``
     - :doc:`instance`
   * - ``optimize_maxmpd``, ``optimize_many_maxmpd``, ``evaluate_maxmpd``
     - :doc:`functions`
   * - ``utils.prune``, ``utils.split_system``
     - :doc:`utils`
   * - ``io.to_newick``, ``io.to_splits_nexus``, ``io.pz.from_phylozoo``, ``io.pz.to_phylozoo``
     - :doc:`io`
   * - ``maf.HeuristicMAF``, ``maf.TieredMAF``, ``maf.MaffeRMAF``, ``maf.MaffeUMAF``,
       ``maf.MAFComputer``, ``maf.Rooting``, ``maf.Forest``
     - :doc:`maf`
   * - ``PhyloTree``, ``TaxonRegistry``
     - :doc:`trees`
   * - ``ForestPDError`` and its subclasses, ``ForestPDWarning``
     - :doc:`exceptions`

.. toctree::
   :maxdepth: 2
   :caption: Modules

   instance
   functions
   utils
   io
   maf
   trees
   exceptions
