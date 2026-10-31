User guide
==========

This page describes how to set up an instance, solve it for one or several budgets and score taxon
sets, and the options for rooted PD, pruning, agreement forests and export. :doc:`problem` defines
the problem; :doc:`examples` has a complete analysis. The code uses

.. code-block:: python

   import forestpd as fpd

Setting up an instance
----------------------

Trees
~~~~~

Every function that takes ``trees`` accepts either

* the **path of a file** with one Newick tree per line (blank lines are skipped), or
* a **list** of Newick strings and/or PhyloZoo trees (with the ``phylozoo`` extra, see
  :ref:`manual-phylozoo`).

.. code-block:: python

   instance = fpd.MaxMPDInstance("posterior.nwk")        # every tree in the file
   instance = fpd.MaxMPDInstance(["(A:1,B:1,(C:1,D:1):2);", "(A:1,C:1,(B:1,D:1):2);"])

Each Newick tree ends with ``;``. A list of file paths is not accepted; read the files into one
list of strings first. The trees must satisfy:

* **The same taxa in every tree.** Trees with additional taxa can be pruned to a common set first
  (see :ref:`manual-prune`).
* **Branch lengths on every edge**, by default. With ``missing_branch_length="zero"`` or a number
  (such as ``1.0``), missing lengths are filled in.
* **Polytomies are allowed.** Vertices with two neighbours are suppressed on reading (their two
  edges merged), which does not change any PD.
* **Rooting is ignored** unless ``rooted=True`` (see :ref:`manual-rooted`).

Costs, weights and survival probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~forestpd.MaxMPDInstance` holds the trees together with the tree weights, taxon costs and
survival probabilities. The trees are parsed and validated once; the instance can then be solved
and evaluated any number of times.

.. code-block:: python

   >>> import forestpd as fpd
   >>> trees = ["(A:1,B:1,(C:1,D:1):2);", "(A:1,C:1,(B:1,D:1):2);"]
   >>> instance = fpd.MaxMPDInstance(
   ...     trees,
   ...     costs={"A": 2, "C": 3},           # integer >= 1; missing labels cost 1
   ...     weights=[0.7, 0.3],               # per tree, in the order of the trees
   ...     survival={"A": 0.5, "D": 0.2},    # in [0, 1); missing labels 0
   ... )
   >>> instance
   MaxMPDInstance(n_trees=2, n_taxa=4)
   >>> instance.taxa
   ('A', 'B', 'C', 'D')

All three are optional:

* ``costs`` maps taxon labels to integer costs of at least 1. ``None`` means unit costs, so the
  budget is the number of taxa.
* ``weights`` gives one weight per tree, as a list in the order of the trees (the list, or the
  lines of the file) or as a dict from tree index to weight. ``None`` means weight 1 for every
  tree; use ``1 / t`` for each tree to get the average PD.
* ``survival`` maps taxon labels to the probability :math:`a(x)` that the taxon survives without
  protection. The objective is then the expected PD of the survivors (see :doc:`problem`).
  ``None`` means 0 for every taxon, which gives plain PD.

.. _manual-rooted:

Rooted PD
~~~~~~~~~

By default PD is unrooted: the total length of the subtree spanning the chosen taxa, regardless of
where the trees are rooted. With ``rooted=True`` it is rooted PD: the total length of the paths
from the chosen taxa to the root, plus the root's own branch (written as ``...):5;``, or the edges
above a root with a single child) once at least one taxon is chosen.

.. code-block:: python

   >>> rooted = fpd.MaxMPDInstance(["((A:1,B:1):1,(C:1,D:1):1);"], rooted=True)
   >>> rooted.evaluate_pd({"A"}), rooted.evaluate_pd({"A", "B"}), rooted.evaluate_pd({"A", "C"})
   (2.0, 3.0, 4.0)

Rooted mode accepts rooted Newick (any number of children at the root) and PhyloZoo
``DirectedPhyNetwork`` trees. A :class:`~forestpd.ForestPDWarning` is issued when the trees look
rooted but ``rooted=False`` (a tree marked ``[&R]``, a ``DirectedPhyNetwork``, or a binary root in
every tree), and when a tree is marked ``[&U]`` but ``rooted=True``. Rooted agreement forests are
usually somewhat larger than unrooted ones for the same trees, so rooted PD can take longer.

.. _manual-prune:

Pruning trees to a taxon set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`forestpd.utils.prune` restricts every tree to a given set of taxa, for example to remove
outgroups, extra accessions per species, or species outside the analysis. Trees that lack one of
the taxa are dropped, together with their weight. The PD of every set of kept taxa is unchanged.

.. code-block:: python

   >>> gene_trees = [
   ...     "((A:1,B:1):1,(C:1,(D:1,Out:3):1):1);",
   ...     "((A:1,C:1):1,(B:1,(D:1,Out:3):1):1);",
   ...     "((A:1,Out:2):1,(B:1,C:1):1);",            # lacks D: dropped
   ... ]
   >>> pruned, weights, registry = fpd.utils.prune(gene_trees, {"A", "B", "C", "D"})
   >>> len(pruned), registry.labels
   (2, ('A', 'B', 'C', 'D'))
   >>> instance = fpd.MaxMPDInstance.from_phylotrees(pruned, registry, weights=weights)
   >>> result = instance.optimize(budget=2, return_taxa=True)
   >>> result.value, sorted(result.taxa)
   (10.0, ['A', 'D'])

In the pruned trees, D's branch includes the edge it shared with ``Out`` (length 1 + 1), which
makes :math:`\{A, D\}` the best pair in both trees.
:meth:`~forestpd.MaxMPDInstance.from_phylotrees` builds an instance from pruned trees and takes
``costs`` and ``survival`` like the constructor. Pruning treats the trees as unrooted, since
removing taxa can move a root.

Solving
-------

One budget
~~~~~~~~~~

:meth:`~forestpd.MaxMPDInstance.optimize` returns the largest objective value over all sets of
taxa with cost at most the budget. With ``return_taxa=True`` it also returns an optimal set (one of
them, if several are optimal).

.. code-block:: python

   >>> instance = fpd.MaxMPDInstance(trees)
   >>> result = instance.optimize(budget=2, return_taxa=True)
   >>> result.value
   8.0
   >>> result.taxa in ({"A", "D"}, {"B", "C"})
   True

The result is a :class:`~forestpd.MaxMPDResult` with three fields:

* ``value``: the optimal objective value;
* ``taxa``: an optimal set of taxon labels, or ``None`` without ``return_taxa``;
* ``meta``: a dict of diagnostics:

  .. list-table::
     :widths: 30 70
     :header-rows: 1

     * - Key
       - Meaning
     * - ``"frontier"``
       - The optimal value for every budget from 0 up to the budget (see :ref:`manual-curves`).
     * - ``"tables"``
       - Which dynamic-programming tables were computed: ``"DP"``, ``"DP'"`` or ``"both"``.
     * - ``"n_clusters"``, ``"cluster_sizes"``
       - The clusters of the cluster reduction and their sizes.
     * - ``"maf_sizes"``, ``"max_maf_size"``
       - The agreement-forest size :math:`k` of each cluster, and the largest.
     * - ``"n_signatures"``
       - :math:`\sum_C 2^{k_C}`: the number of signatures the dynamic program visits at most.
     * - ``"maf"``
       - The agreement-forest stage used.
     * - ``"parse_seconds"``, ``"reduction_seconds"``, ``"maf_seconds"``, ``"dp_seconds"``,
         ``"solve_seconds"``
       - Timings in seconds: reading the trees (once per instance), the cluster reduction (once
         per instance, then cached), the agreement forests, the dynamic program with the
         reconstruction of the sets, and the solve in total.

.. _manual-curves:

Several budgets
~~~~~~~~~~~~~~~

The dynamic-programming tables for a budget contain the optimum for every smaller budget as well,
so a range of budgets takes about as long as its largest budget.
:meth:`~forestpd.MaxMPDInstance.optimize_many` returns one result per budget from a single solve:

.. code-block:: python

   >>> results = instance.optimize_many(range(5), return_taxa=True)
   >>> [r.value for r in results]
   [0.0, 0.0, 8.0, 10.0, 12.0]
   >>> len(results[3].taxa), instance.evaluate_pd(results[3].taxa)   # one optimal set of 3 taxa
   (3, 10.0)

With unrooted PD a single taxon spans no edges, so budget 1 gives 0 here.

There are two kinds of tables: one indexed by the budget spent, up to the largest budget, and a
complementary one indexed by the budget left unspent, up to the total cost minus the smallest
budget. ForestPD computes whichever is smaller for the requested budgets, or both, split between
small and large budgets; ``meta["tables"]`` reports which. A single budget above half the total
cost is therefore solved with the complementary table only.

``meta["frontier"]`` lists the optimal value for every budget from 0 up to the largest requested
budget (or the total cost, if that is smaller). Budgets that no computed table covers are ``NaN``.
For a complete curve, use :meth:`~forestpd.MaxMPDInstance.optimize_many`:

.. code-block:: python

   >>> instance.optimize(budget=2).meta["frontier"]
   array([0., 0., 8.])
   >>> instance.optimize(budget=3).meta["frontier"]     # 3 of 4: complementary table only
   array([nan, nan, nan, 10.])
   >>> instance.optimize_many(range(4))[-1].meta["frontier"]
   array([ 0.,  0.,  8., 10.])

Scoring a taxon set
~~~~~~~~~~~~~~~~~~~

:meth:`~forestpd.MaxMPDInstance.evaluate_pd` returns the objective value of a given set of taxa,
with the instance's weights and survival probabilities, for example to score an existing plan or a
plan made on a single tree:

.. code-block:: python

   >>> instance.evaluate_pd({"A", "B"})
   6.0
   >>> instance.evaluate_pd({"A", "D"})
   8.0

The first call groups the edges of all trees by split; later calls take time linear in the number
of distinct splits, independent of the number of trees.

Shortcut functions
~~~~~~~~~~~~~~~~~~

Three functions build the instance and call one method:

.. list-table::
   :widths: 45 55
   :header-rows: 1

   * - Function
     - Equivalent to
   * - :func:`fpd.optimize_maxmpd(trees, budget, ...) <forestpd.optimize_maxmpd>`
     - ``MaxMPDInstance(trees, ...).optimize(budget)``
   * - :func:`fpd.optimize_many_maxmpd(trees, budgets, ...) <forestpd.optimize_many_maxmpd>`
     - ``MaxMPDInstance(trees, ...).optimize_many(budgets)``
   * - :func:`fpd.evaluate_maxmpd(trees, taxa, ...) <forestpd.evaluate_maxmpd>`
     - ``MaxMPDInstance(trees, ...).evaluate_pd(taxa)``

.. code-block:: python

   >>> fpd.optimize_maxmpd(trees, budget=2).value
   8.0

They take the same options as :class:`~forestpd.MaxMPDInstance`. For several questions about the
same trees, build the instance once instead; reading large tree files is often the slowest step.

Running time
------------

Instance structure
~~~~~~~~~~~~~~~~~~

The running time grows exponentially with :math:`k`, the largest agreement-forest size over the
clusters (see :doc:`problem`). :meth:`~forestpd.MaxMPDInstance.structure` runs the cluster
reduction and the agreement-forest stage without solving:

.. code-block:: python

   >>> s = fpd.MaxMPDInstance(trees).structure()
   >>> s.n_clusters, s.maf_sizes, s.max_maf_size
   (1, (2,), 2)

Each additional component doubles the number of signatures of its cluster. ``n_signatures`` (here
and in ``meta``) is the total over all clusters, a direct measure of the work.

.. _manual-maf:

Agreement-forest stages
~~~~~~~~~~~~~~~~~~~~~~~

The agreement-forest stage is set with ``maf=`` in :meth:`~forestpd.MaxMPDInstance.optimize`,
:meth:`~forestpd.MaxMPDInstance.optimize_many` and :meth:`~forestpd.MaxMPDInstance.structure`.
Every agreement forest gives the exact optimum; a smaller one makes the dynamic program faster.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Stage
     - Description
   * - :class:`~forestpd.maf.HeuristicMAF`
     - The default: a fast heuristic without further dependencies. Its forests are valid and
       usually small, but not always minimum.
   * - :class:`~forestpd.maf.TieredMAF`
     - The heuristic, and on clusters where it leaves :math:`k \geq` ``k_threshold`` (default 14),
       a minimum rooted forest from maffe if that is smaller. Recommended when maffe is installed.
   * - :class:`~forestpd.maf.MaffeRMAF`
     - A minimum rooted agreement forest of every cluster, from maffe. Binary trees only.
   * - :class:`~forestpd.maf.MaffeUMAF`
     - A minimum unrooted agreement forest of every cluster, from maffe. Binary trees only.

.. code-block:: python

   from forestpd.maf import TieredMAF

   result = instance.optimize(budget=50, maf=TieredMAF())   # needs maffe

The maffe stages need the optional maffe package (see :doc:`installation`). Any object with a
``compute(trees)`` method that returns a partition of the cluster's leaves into agreement-forest
components can be passed as well; see :class:`~forestpd.maf.MAFComputer`.

Other factors
~~~~~~~~~~~~~

* **The budget in cost units.** The tables run over budgets in cost units, so fine-grained costs
  (say 1–1000) make budgets near half the total cost slower. Costs with a common divisor are
  divided by it first.
* **Numba compilation.** The compiled kernels are compiled the first time they run in a session,
  and cached on disk. When timing, solve a small instance first.
* **Reusing the instance.** Reading the trees and the cluster reduction are done once per
  instance, for any number of budgets.

Export and PhyloZoo
-------------------

Newick
~~~~~~

:func:`forestpd.io.to_newick` writes a tree of an instance as Newick:

.. code-block:: python

   >>> instance = fpd.MaxMPDInstance(trees)
   >>> fpd.io.to_newick(instance.trees[0], instance.registry.labels)
   '(A:1.0,B:1.0,(C:1.0,D:1.0):2.0);'

Split systems for PDA
~~~~~~~~~~~~~~~~~~~~~

:func:`forestpd.io.to_splits_nexus` writes the trees as one weighted split system in NEXUS format,
the input of split-diversity tools such as PDA. Each distinct split of the trees gets the weighted
sum of the lengths of the edges that induce it, so its split diversity equals the unrooted
objective without survival probabilities. :func:`forestpd.utils.split_system` returns the same in
memory.

.. code-block:: python

   n_splits = fpd.io.to_splits_nexus(trees, "splits.nex", weights=[0.7, 0.3])

.. _manual-phylozoo:

PhyloZoo trees and plotting
~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the ``phylozoo`` extra, lists of trees may contain PhyloZoo ``SemiDirectedPhyNetwork``
(unrooted) and ``DirectedPhyNetwork`` (rooted) trees, and :mod:`forestpd.io.pz` converts both ways.
This gives access to plotting and the other tools of
`PhyloZoo <https://github.com/nholtgrefe/phylozoo>`_:

.. code-block:: python

   from forestpd.io.pz import to_phylozoo
   from phylozoo.viz import plot

   instance = fpd.MaxMPDInstance(trees)
   plot(to_phylozoo(instance.trees[0], instance.registry))

Errors and warnings
-------------------

All errors raised by forestpd derive from :class:`~forestpd.ForestPDError`, and also from the
matching built-in exception, so ``except ValueError`` keeps working:

* :class:`~forestpd.ForestPDInputError` (a ``ValueError``): malformed input, such as unparsable
  Newick, a missing branch length, a cost below 1, a survival probability outside
  :math:`[0, 1)`, an unknown taxon label, or trees with different taxa.
* :class:`~forestpd.ForestPDValueError` (a ``ValueError``): another invalid value, such as a MAF
  stage that cannot handle a cluster.
* :class:`~forestpd.ForestPDImportError` (an ``ImportError``): an optional dependency (PhyloZoo,
  maffe) is needed but not installed.

:class:`~forestpd.ForestPDWarning` flags input that is valid but probably not intended, such as
rooted-looking trees with ``rooted=False``.
