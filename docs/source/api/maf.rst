Agreement forests
=================

.. module:: forestpd.maf

The agreement-forest stage, passed as ``maf=`` to
:meth:`~forestpd.MaxMPDInstance.optimize`, :meth:`~forestpd.MaxMPDInstance.optimize_many` and
:meth:`~forestpd.MaxMPDInstance.structure`. Any agreement forest gives the exact optimum; a
smaller one only makes the dynamic program faster (see :ref:`manual-maf`).

Built-in stages
---------------

.. autoclass:: HeuristicMAF
   :members:

.. autoclass:: TieredMAF
   :members:

.. autoclass:: MaffeRMAF
   :members:

.. autoclass:: MaffeUMAF
   :members:

Writing your own stage
----------------------

Any object with a ``compute`` method of the form below can be passed as ``maf=``.

.. autoclass:: MAFComputer
   :members:

.. autoclass:: Rooting

.. autodata:: Forest
   :no-value:
