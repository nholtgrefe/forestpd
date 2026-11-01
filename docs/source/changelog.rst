Changelog
=========

0.1.1
-----

* maffe is an optional extra: ``pip install "forestpd[maffe]"`` installs it from PyPI (Python
  3.11 to 3.14, Linux and Windows on x86_64). On other platforms it is built from source, from
  maffe's release ``v0.1.1``. The error raised without maffe gives this command.

0.1.0
-----

First release.

* Exact solver for MaxMPD: the weighted PD summed over many trees on the same taxa, with integer
  taxon costs and a budget, and optional survival probabilities (expected PD of the survivors).
* Rooted and unrooted PD.
* Optimal values and optimal taxon sets for many budgets from one solve
  (:meth:`~forestpd.MaxMPDInstance.optimize_many`).
* Cluster reduction, and agreement forests from a built-in heuristic
  (:class:`~forestpd.maf.HeuristicMAF`) or, with the optional maffe package, minimum agreement
  forests (:class:`~forestpd.maf.MaffeRMAF`, :class:`~forestpd.maf.MaffeUMAF`,
  :class:`~forestpd.maf.TieredMAF`).
* Newick input from files or strings; export as Newick and as a weighted split system in NEXUS
  format for PDA; conversion to and from PhyloZoo trees (optional).
* Pruning trees to a common taxon set (:func:`forestpd.utils.prune`).
