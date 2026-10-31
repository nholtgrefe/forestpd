Trees and taxa
==============

.. currentmodule:: forestpd

forestpd's own tree and taxon types, as returned by :func:`forestpd.utils.prune` and held by a
:class:`MaxMPDInstance` (``instance.trees``, ``instance.registry``). Most analyses never need
them directly.

.. autoclass:: PhyloTree
   :members: from_edges, n_edges, total_weight

.. autoclass:: TaxonRegistry
   :members:
