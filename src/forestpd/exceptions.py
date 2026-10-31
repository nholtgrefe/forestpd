"""
Exceptions for forestpd.

All forestpd errors inherit from :class:`ForestPDError` and, where relevant, the matching
builtin (``ValueError``, ``ImportError``), so code catching the builtin keeps working. Raise the
most specific one.
"""

from __future__ import annotations


class ForestPDError(Exception):
    """
    Base exception for all forestpd errors.

    Catch this to handle any forestpd-specific error with a single ``except`` clause.
    """

    pass


class ForestPDValueError(ForestPDError, ValueError):
    """
    Raised when a value passed to a forestpd function is invalid.

    Inherits from both ForestPDError and ValueError.
    """

    pass


class ForestPDInputError(ForestPDValueError):
    """
    Raised when the input trees, weights, costs, survival probabilities or budget are invalid.

    This covers malformed input (unparsable Newick, missing branch lengths, non-positive costs,
    survival probabilities outside ``[0, 1)``, unknown taxon labels) as well as input outside the
    supported scope (differing leaf sets), and a MAF stage that does not return an agreement
    forest.
    """

    pass


class ForestPDImportError(ForestPDError, ImportError):
    """
    Raised when an optional dependency is needed but not installed.

    Converting to and from PhyloZoo objects requires the ``phylozoo`` extra
    (``pip install forestpd[phylozoo]``). Inherits from both ForestPDError and ImportError.
    """

    pass


class ForestPDWarning(UserWarning):
    """
    Warning for input that is valid but probably not what was intended.

    For example, trees that look rooted while PD is computed on unrooted trees.
    """

    pass


__all__ = [
    "ForestPDError",
    "ForestPDValueError",
    "ForestPDInputError",
    "ForestPDImportError",
    "ForestPDWarning",
]
