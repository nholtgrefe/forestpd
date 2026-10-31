"""
The compiled structure pass of :func:`~forestpd.io.newick._parse_plain`, used when Numba is
installed.

One loop over the separators ``(),;:`` of a plain Newick string builds the tree exactly as the
token-by-token parser does: every ``(`` and ``,`` creates a node (numbered in order of
appearance), ``)`` returns to the parent, and the text after each separator belongs to the node
current after it, as a label (after ``(``, ``,`` or ``)``) or a branch length (after ``:``). Text
pieces are named by their index in ``text.split`` at the separators, so the caller can pick them
out without slicing.

Anything the vectorised path would not accept, including every malformed tree, makes the kernel
report failure, so that the caller falls back to the general parser, which also produces the error
messages.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: The compiled kernel, or ``None`` when Numba is unavailable.
plain_structure: Any

#: Separator codes, as in :mod:`~forestpd.io.newick`.
_OPEN, _CLOSE, _COMMA, _END, _COLON = 1, 2, 3, 4, 5

try:  # pragma: no cover - depends on the environment
    from numba import njit  # type: ignore[import-untyped,unused-ignore]

    @njit(cache=True)
    def plain_structure(classes):  # pragma: no cover - compiled
        """
        Parent, label piece and length piece of every node of a plain Newick tree.

        ``classes`` holds one code per character (0 for an ordinary character). Returns
        ``(ok, parent, label_piece, length_piece, empty_length)``: piece ``k`` is the text between
        separators ``k - 1`` and ``k``; ``label_piece`` and ``length_piece`` are ``-1`` where a
        node has none; ``empty_length`` marks a node whose length field is empty (a missing
        length). ``ok`` is ``False`` whenever the general parser must take over.
        """
        n_chars = classes.shape[0]
        n_sep = 0
        n_new = 0
        for i in range(n_chars):
            c = classes[i]
            if c != 0:
                n_sep += 1
                if c == _OPEN or c == _COMMA:
                    n_new += 1
        empty_parent = np.empty(0, dtype=np.int64)
        fail = (False, empty_parent, empty_parent, empty_parent, np.empty(0, dtype=np.bool_))
        if n_sep == 0 or classes[0] != _OPEN or classes[n_chars - 1] != _END:
            return fail
        n = n_new + 1
        parent = np.full(n, -1, dtype=np.int64)
        label_piece = np.full(n, -1, dtype=np.int64)
        length_piece = np.full(n, -1, dtype=np.int64)
        has_field = np.zeros(n, dtype=np.bool_)
        empty_length = np.zeros(n, dtype=np.bool_)
        current = 0
        created = 0
        depth = 0
        closed = False
        segment_field = False  # the current segment already had a ':' field
        k = -1  # ordinal of the separator just read
        for i in range(n_chars):
            c = classes[i]
            if c == 0:
                continue
            k += 1
            if c == _OPEN:
                if closed:
                    return fail
                created += 1
                parent[created] = current
                current = created
                depth += 1
                segment_field = False
            elif c == _COMMA:
                if depth == 0:
                    return fail
                created += 1
                parent[created] = parent[current]
                current = created
                segment_field = False
            elif c == _CLOSE:
                if depth == 0:
                    return fail
                current = parent[current]
                depth -= 1
                closed = depth == 0
                segment_field = False
            elif c == _END:
                if i != n_chars - 1 or depth != 0 or not closed:
                    return fail
                break
            else:  # ':'
                if segment_field or has_field[current]:
                    return fail  # an extra field, or a second length for one node
                segment_field = True
                has_field[current] = True
            # The piece after separator k is piece k + 1; it is empty if the next character is a
            # separator too.
            empty = classes[i + 1] != 0
            if c == _COLON:
                if empty:
                    empty_length[current] = True
                else:
                    length_piece[current] = k + 1
            elif not empty:
                if label_piece[current] >= 0:
                    return fail
                label_piece[current] = k + 1
        n_children = np.zeros(n, dtype=np.int64)
        for v in range(1, n):
            n_children[parent[v]] += 1
        for v in range(n):
            if n_children[v] == 1:
                return fail  # a unary node (the general parser merges it)
            if n_children[v] == 0 and label_piece[v] < 0:
                return fail  # an unlabelled leaf (the general parser reports it)
        return True, parent, label_piece, length_piece, empty_length

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - depends on the environment
    plain_structure = None
    HAVE_NUMBA = False


__all__ = ["HAVE_NUMBA", "plain_structure"]
