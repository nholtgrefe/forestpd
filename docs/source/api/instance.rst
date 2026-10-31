Instances and results
=====================

.. currentmodule:: forestpd

A :class:`MaxMPDInstance` holds the trees, tree weights, taxon costs and survival probabilities,
parsed and validated once. Its methods optimize for one or many budgets, score taxon sets, and
report the structure that determines the running time.

.. autoclass:: MaxMPDInstance
   :members: optimize, optimize_many, evaluate_pd, structure, from_phylotrees, n_trees, n_taxa,
             taxa, rooted

.. autoclass:: MaxMPDResult

.. autoclass:: MaxMPDStructure
