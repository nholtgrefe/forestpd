One-call functions
==================

.. currentmodule:: forestpd

Each function builds a :class:`MaxMPDInstance` and calls one of its methods. For several
questions about the same trees, build the instance once instead, so the trees are read only once.

.. autofunction:: optimize_maxmpd

.. autofunction:: optimize_many_maxmpd

.. autofunction:: evaluate_maxmpd
