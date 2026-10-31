Exceptions and warnings
=======================

.. currentmodule:: forestpd

Every error raised by forestpd derives from :class:`ForestPDError`, and also from the matching
built-in exception (``ValueError``, ``ImportError``), so code that catches the built-in keeps
working.

.. autoexception:: ForestPDError
   :show-inheritance:

.. autoexception:: ForestPDValueError
   :show-inheritance:

.. autoexception:: ForestPDInputError
   :show-inheritance:

.. autoexception:: ForestPDImportError
   :show-inheritance:

.. autoexception:: ForestPDWarning
   :show-inheritance:
