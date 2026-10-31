"""
Taxon registry: one shared integer id per taxon label.

Each tree is parsed with its own node numbering. The registry gives every taxon label one id (in
sorted label order), and every tree's leaves are renumbered to it, so leaf id == taxon id holds in
every tree, which the bitmasks of the algorithm rely on.
"""

from __future__ import annotations

from operator import itemgetter
from typing import Iterable, Sequence

import numpy as np

from ..exceptions import ForestPDInputError


class TaxonRegistry:
    """
    Bidirectional mapping between taxon labels and contiguous ids ``0..n-1``.

    Ids follow sorted label order, so the mapping is deterministic across runs.

    Parameters
    ----------
    labels : Iterable[str]
        The taxon labels (duplicates and order ignored).

    Examples
    --------
    >>> reg = TaxonRegistry(["B", "A", "C"])
    >>> reg.labels
    ('A', 'B', 'C')
    >>> reg.id_of("B")
    1
    >>> sorted(reg.labels_of(  [0, 2]))
    ['A', 'C']
    """

    def __init__(self, labels: Iterable[str]) -> None:
        self._labels: tuple[str, ...] = tuple(sorted(set(labels)))
        self._id: dict[str, int] = {label: i for i, label in enumerate(self._labels)}

    @property
    def labels(self) -> tuple[str, ...]:
        """Taxon labels, indexed by id."""
        return self._labels

    @property
    def n(self) -> int:
        """Number of taxa."""
        return len(self._labels)

    def id_of(self, label: str) -> int:
        """
        Integer id of ``label``.

        Parameters
        ----------
        label : str
            A taxon label.

        Returns
        -------
        int
            The taxon's id in ``0..n-1``.

        Raises
        ------
        ForestPDInputError
            If ``label`` is not registered.
        """
        try:
            return self._id[label]
        except KeyError:
            raise ForestPDInputError(f"Unknown taxon label: {label!r}") from None

    def id_array(self, labels: Sequence[str]) -> np.ndarray:
        """
        Ids of a sequence of labels, in order, as an array (one dictionary lookup in C).

        Parameters
        ----------
        labels : Sequence[str]
            Taxon labels.

        Returns
        -------
        numpy.ndarray
            ``int64`` id per label.

        Raises
        ------
        ForestPDInputError
            If a label is not registered (the first such label is named, as by :meth:`id_of`).

        Examples
        --------
        >>> TaxonRegistry(["B", "A", "C"]).id_array(["C", "A"]).tolist()
        [2, 0]
        """
        if len(labels) < 2:
            return np.array([self.id_of(label) for label in labels], dtype=np.int64)
        try:
            ids = itemgetter(*labels)(self._id)
        except KeyError:
            for label in labels:
                self.id_of(label)  # raises for the first unknown label
            raise  # pragma: no cover - unreachable
        return np.array(ids, dtype=np.int64)

    def label_of(self, taxon_id: int) -> str:
        """
        Label of ``taxon_id``.

        Parameters
        ----------
        taxon_id : int
            A taxon id in ``0..n-1``.

        Returns
        -------
        str
            The taxon's label.
        """
        return self._labels[taxon_id]

    def ids_of(self, labels: Iterable[str]) -> set[int]:
        """
        Ids of a set of labels.

        Parameters
        ----------
        labels : Iterable[str]
            Taxon labels.

        Returns
        -------
        set[int]
            Their ids.

        Raises
        ------
        ForestPDInputError
            If any label is not registered.
        """
        return {self.id_of(label) for label in labels}

    def labels_of(self, ids: Iterable[int]) -> set[str]:
        """
        Labels of a set of ids.

        Parameters
        ----------
        ids : Iterable[int]
            Taxon ids in ``0..n-1``.

        Returns
        -------
        set[str]
            Their labels.
        """
        return {self._labels[i] for i in ids}

    def __len__(self) -> int:
        return self.n

    def __contains__(self, label: object) -> bool:
        return label in self._id

    def __repr__(self) -> str:
        return f"TaxonRegistry(n={self.n})"


__all__ = ["TaxonRegistry"]
