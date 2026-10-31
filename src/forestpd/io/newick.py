"""
Newick parsing.

Reads a Newick string into a :class:`ParsedTree`: an *unrooted* tree with leaf labels and
possibly missing branch lengths, ready to be numbered against a taxon registry and turned into a
:class:`~forestpd.core.tree.PhyloTree`.

Dialect:

* labels may be unquoted, ``'single-quoted'`` (``''`` escapes a quote) or ``"double-quoted"``;
  underscores are kept as written;
* ``[...]`` comments are skipped wherever they appear (so ``[&R]`` and rich-Newick annotations
  are harmless);
* a branch is ``:length``, optionally followed by further ``:field`` values (support, ``gamma``),
  which are ignored;
* internal node labels (bootstrap values, clade names) are ignored;
* every tree ends with ``;``, and a string holds exactly one tree.

By default the rooting is discarded: every internal node of degree two (a binary root, or a unary
node anywhere) is merged away, the merged branch length being the sum of the lengths that are
present (missing only if all are missing), and the root's own branch length is ignored.
Phylogenetic diversity is unchanged by this, since the merged edges are always in a spanning tree
together.

With ``rooted=True`` the root is kept for rooted PD. The *stem* (the root's own branch length plus
the edges of a chain of outdegree-1 nodes at the top) is recorded separately, and a reserved leaf
:data:`ROOT_LABEL` is attached at the top branching node. Rooted PD of a set ``S`` is then the
unrooted PD of ``S`` together with that leaf, plus the stem when ``S`` is non-empty.

The parser also records a *root hint*: ``"R"`` or ``"U"`` for a leading ``[&R]`` or ``[&U]``
comment, otherwise ``"binary"`` or ``"multifurcating"`` for the shape of the top branching node.

Rejected: unbalanced parentheses, unlabelled leaves, duplicate leaf labels, malformed lengths, and
reticulations (``#H`` nodes), since forestpd works on trees only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from operator import itemgetter
from typing import Mapping, Sequence, TypeVar

import numpy as np

from ..core.tree import PhyloTree
from ..exceptions import ForestPDInputError
from . import _newick_kernel

#: Build plain trees' structure with the compiled pass (:mod:`._newick_kernel`) when Numba is
#: installed. The NumPy path gives the same trees; set ``False`` to force it (tests).
USE_COMPILED = True

_T = TypeVar("_T")

#: Label of the leaf attached at the root in rooted mode. It is reserved: unrooted input carrying it
#: is rejected (see :func:`~forestpd.io.convert.coerce_trees`), and user-facing label lookups treat
#: it as unknown.
ROOT_LABEL = "\x00root"

#: Labels written without quotes by :func:`to_newick`; anything else is single-quoted.
_PLAIN_LABEL = re.compile(r"[A-Za-z0-9_.\-]+")

#: One token per match. The last alternative catches anything unexpected (including an unterminated
#: quote or comment), which is then reported as an error.
_TOKEN = re.compile(
    r"""
      (?P<space>\s+)
    | (?P<comment>\[[^\]]*\])
    | (?P<squote>'(?:[^']|'')*')
    | (?P<dquote>"[^"]*")
    | (?P<punct>[(),;])
    | (?P<field>:[^,():;\[\]\s]*)
    | (?P<label>[^,():;\[\]\s'"]+)
    | (?P<bad>.)
    """,
    re.VERBOSE | re.DOTALL,
)

#: Shorter texts always go token by token: the vectorised path only pays off on large trees.
_FAST_MIN_LENGTH = 1000

#: Characters that send a text to the token-by-token path: whitespace, quotes, comments and ``#``.
#: Without them, a Newick string is only punctuation, ``:`` and the labels and fields between them.
_NOT_PLAIN = re.compile(r"[\s'\"\[\]#]")

#: Character classes for the vectorised path, as a ``bytes.translate`` table over ASCII codes: the
#: separators "(),;:", ``_OTHER`` for the characters of ``_NOT_PLAIN`` (and other control
#: characters), 0 for everything else.
_OPEN, _CLOSE, _COMMA, _END, _COLON, _OTHER = 1, 2, 3, 4, 5, 6
_CLASS = bytearray(256)
_CLASS[:33] = bytes([_OTHER]) * 33
for _code, _char in enumerate("(),;:", start=1):
    _CLASS[ord(_char)] = _code
for _char in "'\"[]#":
    _CLASS[ord(_char)] = _OTHER
_CLASSES = bytes(_CLASS)
_TO_COLON = str.maketrans("(),;", "::::")


@dataclass(frozen=True, eq=False)
class ParsedTree:
    """
    An unrooted tree as read from Newick, before taxa are numbered.

    The edges always form a tree on ``0..n_nodes-1`` (they are built from parent pointers, or
    checked on construction from other sources), so :func:`~forestpd.io.convert.to_phylotree`
    does not check connectivity again.

    Attributes
    ----------
    n_nodes : int
        Number of nodes; ids are ``0..n_nodes-1``.
    edges : numpy.ndarray
        ``int64[E, 2]``; undirected edges as node pairs.
    lengths : numpy.ndarray
        ``float64[E]``; branch length per edge, ``NaN`` where the input gave none.
    has_length : numpy.ndarray
        ``bool[E]``; whether the input gave the edge a length (a literal ``nan`` length is
        present, and rejected later as invalid).
    leaf_nodes : numpy.ndarray
        ``int64[L]``; the leaf node ids, increasing.
    leaf_names : tuple[str, ...]
        The taxon label of each leaf, aligned with ``leaf_nodes``.
    stem : float
        Rooted mode only: the length above the top branching node (0 when absent or unrooted).
    root_hint : str | None
        ``"R"``, ``"U"``, ``"binary"``, ``"multifurcating"``, or ``None`` when unknown.
    """

    n_nodes: int
    edges: np.ndarray
    lengths: np.ndarray
    has_length: np.ndarray
    leaf_nodes: np.ndarray
    leaf_names: tuple[str, ...]
    stem: float = 0.0
    root_hint: str | None = None

    @property
    def leaf_labels(self) -> dict[int, str]:
        """Leaf node id -> taxon label."""
        return dict(zip(self.leaf_nodes.tolist(), self.leaf_names))

    @classmethod
    def from_lists(
        cls,
        n_nodes: int,
        edges: Sequence[tuple[int, int]],
        lengths: Sequence[float | None],
        leaf_labels: Mapping[int, str],
        stem: float = 0.0,
        root_hint: str | None = None,
        check_tree: bool = False,
    ) -> "ParsedTree":
        """
        Build a :class:`ParsedTree` from Python lists.

        Parameters
        ----------
        n_nodes : int
            Number of nodes.
        edges : Sequence[tuple[int, int]]
            Undirected edges as node pairs.
        lengths : Sequence[float | None]
            Branch length per edge, ``None`` where absent.
        leaf_labels : Mapping[int, str]
            Leaf node id -> taxon label.
        stem : float, default 0.0
            See the class attributes.
        root_hint : str | None, default None
            See the class attributes.
        check_tree : bool, default False
            Check that the edges form a tree (for sources that do not guarantee it).

        Returns
        -------
        ParsedTree
            The tree.

        Raises
        ------
        ForestPDInputError
            With ``check_tree``, if the edges do not form a tree on ``0..n_nodes-1``.
        """
        edge_array = np.array(edges, dtype=np.int64).reshape(-1, 2)
        if check_tree:
            _check_tree(n_nodes, edge_array)
        has_length = np.array([x is not None for x in lengths], dtype=bool)
        length_array = np.array(
            [np.nan if x is None else float(x) for x in lengths], dtype=np.float64
        )
        leaves = sorted(leaf_labels)
        return cls(
            n_nodes,
            edge_array,
            length_array,
            has_length,
            np.array(leaves, dtype=np.int64),
            tuple(leaf_labels[v] for v in leaves),
            stem,
            root_hint,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParsedTree):
            return NotImplemented
        return (
            self.n_nodes == other.n_nodes
            and np.array_equal(self.edges, other.edges)
            and np.array_equal(self.lengths, other.lengths, equal_nan=True)
            and np.array_equal(self.has_length, other.has_length)
            and np.array_equal(self.leaf_nodes, other.leaf_nodes)
            and self.leaf_names == other.leaf_names
            and self.stem == other.stem
            and self.root_hint == other.root_hint
        )

    __hash__ = None  # type: ignore[assignment]


def _check_tree(n_nodes: int, edges: np.ndarray) -> None:
    """
    Raise unless ``edges`` form a tree on ``0..n_nodes-1``.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    edges : numpy.ndarray
        ``int64[E, 2]`` node pairs.

    Raises
    ------
    ForestPDInputError
        If there are not ``n_nodes - 1`` edges, a node id is out of range, or some node is
        unreachable from node 0.
    """
    if len(edges) != max(n_nodes - 1, 0):
        raise ForestPDInputError(
            f"a tree on {n_nodes} nodes needs {n_nodes - 1} edges, got {len(edges)}"
        )
    if n_nodes <= 1:
        return
    if edges.min() < 0 or edges.max() >= n_nodes:
        raise ForestPDInputError("edge list refers to a node outside 0..n_nodes-1")
    neighbours: list[list[int]] = [[] for _ in range(n_nodes)]
    for u, v in edges.tolist():
        neighbours[u].append(v)
        neighbours[v].append(u)
    seen = [False] * n_nodes
    seen[0] = True
    stack = [0]
    while stack:
        for w in neighbours[stack.pop()]:
            if not seen[w]:
                seen[w] = True
                stack.append(w)
    if not all(seen):
        raise ForestPDInputError("edge list is not connected (not a tree)")


def parse_newick(text: str, rooted: bool = False) -> ParsedTree:
    """
    Parse one Newick tree into a :class:`ParsedTree`.

    Parameters
    ----------
    text : str
        A single Newick tree, ending with ``;``.
    rooted : bool, default False
        Keep the root: record the stem and attach the :data:`ROOT_LABEL` leaf at the root.

    Returns
    -------
    ParsedTree
        The tree as an unrooted tree, with degree-two nodes merged away.

    Raises
    ------
    ForestPDInputError
        If the string is not a single valid Newick tree ending with ``;``, has an unlabelled or
        duplicate leaf, a malformed branch length, or a reticulation.

    Examples
    --------
    >>> t = parse_newick("((A:1,B:1):1,(C:1,D:1):1);")
    >>> sorted(t.leaf_labels.values())
    ['A', 'B', 'C', 'D']
    >>> sorted(t.lengths.tolist())  # the binary root is merged: its two branches become length 2
    [1.0, 1.0, 1.0, 1.0, 2.0]
    >>> r = parse_newick("((A:1,B:1):1,(C:1,D:1):1):3;", rooted=True)
    >>> r.stem, r.root_hint, len(r.leaf_labels)  # the root leaf makes five leaves
    (3.0, 'binary', 5)
    """
    if len(text) >= _FAST_MIN_LENGTH:
        parsed = _parse_plain(text, rooted)
        if parsed is not None:
            return parsed
    return _parse_tokens(text, rooted)


def _parse_tokens(text: str, rooted: bool) -> ParsedTree:
    """
    Parse one Newick tree token by token: the general path of :func:`parse_newick`.

    Handles the whole dialect and produces every error message; :func:`_parse_plain` is a faster
    path for large plain trees that hands everything unusual back to this function.

    Parameters
    ----------
    text : str
        A single Newick tree, ending with ``;``.
    rooted : bool
        Keep the root (see :func:`parse_newick`).

    Returns
    -------
    ParsedTree
        The parsed tree.

    Raises
    ------
    ForestPDInputError
        As :func:`parse_newick`.
    """
    parent: list[int] = [-1]
    length: list[float | None] = [None]
    label: list[str | None] = [None]
    current = 0
    depth = 0
    started = False
    lead_comment: str | None = None  # a [&R] / [&U] comment before the tree
    closed = False  # the top-level ")" has been read; only the root's label, length and ";" follow
    finished = False
    has_length = False  # whether `current` already took its length (later fields are ignored)
    for match in _TOKEN.finditer(text):
        kind = match.lastgroup
        if kind == "space":
            continue
        token = match.group()
        if kind == "comment":
            if not started and token.upper() in ("[&R]", "[&U]"):
                lead_comment = token[2].upper()
            continue
        started = True
        if kind == "bad":
            raise ForestPDInputError(
                f"Invalid Newick: unexpected {token!r} at position {match.start()}"
                + (" (unterminated quote or comment)" if token in "'\"[" else "")
            )
        if finished:
            raise ForestPDInputError(
                "Invalid Newick: text after the end of the tree. A string or line holds one tree; "
                "to read several trees from a file, put each on its own line and pass the file's "
                "path as trees."
            )
        if kind == "field":
            if not has_length:
                length[current] = _parse_length(token[1:])
                has_length = True
        elif kind != "punct":
            name = _unquote(token, kind)
            if "#" in name and kind == "label":
                raise ForestPDInputError(
                    f"Reticulation node {name!r} found: forestpd accepts trees only"
                )
            label[current] = name
        elif token == ",":  # a new child of the current node's parent
            if depth == 0:
                raise ForestPDInputError("Invalid Newick: ',' outside parentheses")
            parent.append(parent[current])
            length.append(None)
            label.append(None)
            current = len(parent) - 1
            has_length = False
        elif token == "(":  # a new child of the current node
            if closed:
                raise ForestPDInputError(
                    "Invalid Newick: a second tree starts after the first; is a ';' missing "
                    "between them?"
                )
            parent.append(current)
            length.append(None)
            label.append(None)
            current = len(parent) - 1
            depth += 1
            has_length = False
        elif token == ")":
            if depth == 0:
                raise ForestPDInputError("Invalid Newick: unbalanced ')'")
            current = parent[current]
            depth -= 1
            has_length = False
            closed = depth == 0
        else:  # ";"
            finished = True
    if depth != 0:
        raise ForestPDInputError("Invalid Newick: unbalanced '('")
    if not finished:
        raise ForestPDInputError("Invalid Newick: the tree does not end with ';'")
    return from_parents(parent, length, label, rooted, lead_comment)


def _parse_plain(text: str, rooted: bool) -> ParsedTree | None:
    """
    Parse a large, plain Newick tree with array operations instead of a loop over tokens.

    "Plain" means no whitespace inside the tree, no quotes, comments or ``#``, every node given at
    most one label and one branch, and no node of outdegree one. That covers what tree-building
    software writes, and for such trees the result is exactly that of :func:`_parse_tokens`.
    Anything else, including every malformed tree, returns ``None`` so that the caller falls back
    to :func:`_parse_tokens`, which also produces the error messages.

    Parameters
    ----------
    text : str
        A single Newick tree.
    rooted : bool
        Keep the root (see :func:`parse_newick`).

    Returns
    -------
    ParsedTree | None
        The parsed tree, or ``None`` if the text is not a plain, valid tree.
    """
    text = text.strip()
    if not text.endswith(";"):
        return None
    # One class code per character. A non-ASCII character is an ordinary one, unless whitespace.
    if text.isascii():
        raw = text.encode("ascii")
    elif _NOT_PLAIN.search(text):
        return None
    else:  # UTF-32 keeps one code per character, so positions stay string positions
        wide = np.frombuffer(text.encode("utf-32-le", "surrogatepass"), dtype=np.uint32)
        raw = np.where(wide < 128, wide, ord("a")).astype(np.uint8).tobytes()
    classes = np.frombuffer(raw.translate(_CLASSES), dtype=np.int8)
    if classes.max() == _OTHER:
        return None
    if _newick_kernel.HAVE_NUMBA and USE_COMPILED:
        compiled = _parse_plain_compiled(text, classes, rooted)
        if compiled is not None:
            return compiled
    at = np.flatnonzero(classes != 0)  # positions of the separators "(),;:"
    kind = classes[at]
    # Piece k is the text between separators k-1 and k: a label (possibly empty) after "(),;", or
    # a field after ":".
    pieces = text.translate(_TO_COLON).split(":")
    punct = kind != _COLON
    if np.count_nonzero(kind == _END) != 1:  # the text ends with ";", so this is the last one
        return None

    # The structure. Every "(" and "," creates a node, numbered in order of appearance, which
    # becomes the current node; ")" returns to the node that was current before its "(".
    tokens = kind[punct][:-1]
    m = len(tokens)
    if m == 0:
        return None
    is_open = tokens == _OPEN
    is_close = tokens == _CLOSE
    depth = np.cumsum(is_open, dtype=np.int64) - np.cumsum(is_close, dtype=np.int64)
    # Balanced, with only the last ")" leaving the outer parentheses (so no second tree, and no
    # "," or ")" at depth 0).
    if tokens[-1] != _CLOSE or depth[-1] != 0 or (m > 1 and depth[:-1].min() < 1):
        return None
    # The "(" of every token's group: the token itself for "(", else the last "(" before it at the
    # group's depth (one deeper for ")"). In the tokens sorted stably by that depth, it is the last
    # "(" up to each token, since every depth's run starts with a "(".
    level = depth + is_close
    if level.max() < 1 << 16:  # stable sorts of 16-bit keys are radix sorts
        level = level.astype(np.uint16)
    order = np.argsort(level, kind="stable")
    last_open = np.where(is_open[order], np.arange(m), 0)
    np.maximum.accumulate(last_open, out=last_open)
    group = np.empty(m, dtype=np.int64)
    group[order] = order[last_open]
    opens = np.flatnonzero(is_open)
    # The owner of a group (the parent of its members) is the node current before its "(": the
    # node created by the previous token, or, after a ")", the owner of that token's group.
    creator = ~is_close
    created = np.cumsum(creator, dtype=np.int64)  # node id made by each "(" or ","
    n = int(created[-1]) + 1
    owner = np.full(m, -1, dtype=np.int64)
    owner[0] = 0  # the first token is "(", made under the root
    later, before = opens[1:], opens[1:] - 1
    direct = creator[before]
    owner[later[direct]] = created[before[direct]]
    link = np.full(m, -1, dtype=np.int64)
    pending = later[~direct]
    link[pending] = group[before[~direct]]
    while len(pending):  # pointer jumping along ")(" chains; usually no iteration at all
        found = owner[link[pending]]
        done = found >= 0
        owner[pending[done]] = found[done]
        pending = pending[~done]
        link[pending] = link[link[pending]]
    group_owner = owner[group]
    parent = np.empty(n, dtype=np.int64)
    parent[0] = -1
    parent[1:] = group_owner[creator]
    n_children = np.bincount(parent[1:], minlength=n)
    if np.any(n_children == 1):
        return None

    # The pieces. Each belongs to the node current after the separator that starts its segment;
    # only the first field of a segment is a branch length.
    current = np.concatenate(([0], np.where(creator, created, group_owner), [-1]))
    target = current[np.concatenate(([0], np.cumsum(punct)))]
    starts = np.concatenate(([True], punct))
    first_field = np.concatenate(([False], (kind == _COLON) & starts[:-1]))
    nonempty = np.diff(np.concatenate(([-1], at, [len(text)]))) > 1
    named = np.flatnonzero(starts & nonempty)
    measured = np.flatnonzero(first_field)
    if len(named) and np.bincount(target[named], minlength=n).max() > 1:
        return None
    if len(measured) and np.bincount(target[measured], minlength=n).max() > 1:
        return None
    filled = measured[nonempty[measured]]  # an empty field leaves the length missing
    try:
        values = list(map(float, _pick(pieces, filled)))
    except ValueError:
        return None
    value = np.full(n, np.nan)
    present = np.zeros(n, dtype=bool)
    value[target[filled]] = values
    present[target[filled]] = True

    leaves = np.flatnonzero(n_children == 0)
    label_piece = np.full(n, -1, dtype=np.int64)
    label_piece[target[named]] = named
    leaf_pieces = label_piece[leaves]
    if np.any(leaf_pieces < 0):
        return None
    names = _pick(pieces, leaf_pieces)
    if len(set(names)) != len(names):
        return None

    return _finish_plain(parent, n_children, value, present, leaves, names, rooted)


def _finish_plain(
    parent: np.ndarray,
    n_children: np.ndarray,
    value: np.ndarray,
    present: np.ndarray,
    leaves: np.ndarray,
    names: list[str],
    rooted: bool,
) -> ParsedTree:
    """
    The :class:`ParsedTree` of a plain tree, from its rooted structure (both fast paths).

    Parameters
    ----------
    parent : numpy.ndarray
        ``int64[n]``; parent per node, ``-1`` at the root (node 0).
    n_children : numpy.ndarray
        ``int64[n]``; children per node (the root has at least two, no node has one).
    value : numpy.ndarray
        ``float64[n]``; length of the edge above each node, ``NaN`` where missing.
    present : numpy.ndarray
        ``bool[n]``; whether that length is given.
    leaves : numpy.ndarray
        The leaf nodes, increasing.
    names : list[str]
        Their labels.
    rooted : bool
        Keep the root (see :func:`parse_newick`).

    Returns
    -------
    ParsedTree
        As :func:`from_parents` would build it.
    """
    n = parent.size
    # As from_parents and _unroot would build it: the root has at least two children here.
    hint = "binary" if n_children[0] == 2 else "multifurcating"
    children = np.arange(1, n, dtype=np.int64)
    if rooted:
        return ParsedTree(
            n + 1,
            np.vstack((np.column_stack((parent[1:], children)), [[0, n]])),
            np.append(value[1:], 0.0),
            np.append(present[1:], True),
            np.append(leaves, n),
            tuple(names) + (ROOT_LABEL,),
            float(value[0]) if present[0] else 0.0,
            hint,
        )
    if n_children[0] > 2:
        return ParsedTree(
            n,
            np.column_stack((parent[1:], children)),
            value[1:],
            present[1:],
            leaves.astype(np.int64),
            tuple(names),
            0.0,
            hint,
        )
    # A binary root is merged away: its children 1 and b become one edge (the lengths that are
    # present summed, missing if both are), the other nodes shift down by one.
    b = int(np.flatnonzero(parent[2:] == 0)[0]) + 2
    rest = np.flatnonzero(parent[1:] != 0) + 1
    both = np.array([value[1], value[b]])[np.array([present[1], present[b]])]
    merged = (both[0] + both[1]) if len(both) == 2 else (both[0] if len(both) else np.nan)
    return ParsedTree(
        n - 1,
        np.vstack(([[0, b - 1]], np.column_stack((parent[rest] - 1, rest - 1)))),
        np.concatenate(([merged], value[rest])),
        np.concatenate(([bool(present[1] or present[b])], present[rest])),
        (leaves - 1).astype(np.int64),
        tuple(names),
        0.0,
        hint,
    )


def _parse_plain_compiled(text: str, classes: np.ndarray, rooted: bool) -> ParsedTree | None:
    """
    :func:`_parse_plain` with the structure built by the compiled pass of :mod:`._newick_kernel`.

    Parameters
    ----------
    text : str
        The stripped tree text.
    classes : numpy.ndarray
        ``int8`` separator code per character.
    rooted : bool
        Keep the root (see :func:`parse_newick`).

    Returns
    -------
    ParsedTree | None
        The tree, or ``None`` where the NumPy path must decide.
    """
    ok, parent, label_piece, length_piece, _ = _newick_kernel.plain_structure(classes)
    if not ok:
        return None
    pieces = text.translate(_TO_COLON).split(":")
    n_children = np.bincount(parent[1:], minlength=parent.size)
    leaves = np.flatnonzero(n_children == 0)
    names = _pick(pieces, label_piece[leaves])
    if len(set(names)) != len(names):
        return None
    measured = np.flatnonzero(length_piece >= 0)
    try:
        values = list(map(float, _pick(pieces, length_piece[measured])))
    except ValueError:
        return None
    value = np.full(parent.size, np.nan)
    present = np.zeros(parent.size, dtype=bool)
    value[measured] = values
    present[measured] = True
    return _finish_plain(parent, n_children, value, present, leaves, names, rooted)


def _pick(items: list[_T], indices: np.ndarray) -> list[_T]:
    """
    Select list items by an array of indices, as a list (``[items[i] for i in indices]``, faster).

    Parameters
    ----------
    items : list
        The list to select from.
    indices : numpy.ndarray
        Integer positions in ``items``.

    Returns
    -------
    list
        The selected items, in the order of ``indices``.
    """
    if len(indices) < 2:
        return [items[i] for i in indices.tolist()]
    return list(itemgetter(*indices.tolist())(items))


def from_parents(
    parent: list[int],
    length: list[float | None],
    label: list[str | None],
    rooted: bool,
    hint: str | None = None,
) -> ParsedTree:
    """
    Build a :class:`ParsedTree` from a rooted tree given by parent pointers.

    Parameters
    ----------
    parent : list[int]
        Parent of every node; node 0 is the root (``parent[0] == -1``).
    length : list[float | None]
        Length of the edge above every node; ``length[0]`` is the root's own branch length.
    label : list[str | None]
        Label of every node (only leaf labels are used).
    rooted : bool
        Keep the root (see the module docstring) or discard it.
    hint : str | None, optional
        ``"R"`` or ``"U"`` if the input marked the rooting explicitly; otherwise the hint is
        derived from the shape of the top branching node.

    Returns
    -------
    ParsedTree
        The tree.

    Raises
    ------
    ForestPDInputError
        If a leaf is unlabelled or two leaves share a label.
    """
    parent, length, label = list(parent), list(length), list(label)
    n_children = [0] * len(parent)
    for p in parent:
        if p >= 0:
            n_children[p] += 1

    # The top branching node: follow the chain of outdegree-1 nodes below the root.
    top = 0
    stem = length[0] or 0.0
    while n_children[top] == 1:
        top = next(v for v in range(len(parent)) if parent[v] == top)
        stem += length[top] or 0.0
    if hint is None and n_children[top] >= 2:
        hint = "binary" if n_children[top] == 2 else "multifurcating"

    if not rooted:
        return replace(_unroot(parent, length, label, n_children), root_hint=hint)

    if n_children[top] == 0:  # a single leaf: rooted PD is just the stem
        name = label[top]
        if not name:
            raise ForestPDInputError("Invalid Newick: a leaf has no label")
        return ParsedTree.from_lists(2, [(0, 1)], [0.0], {0: name, 1: ROOT_LABEL}, stem, hint)
    parent.append(top)
    length.append(0.0)
    label.append(ROOT_LABEL)
    n_children.append(0)
    n_children[top] += 1
    return replace(_unroot(parent, length, label, n_children), stem=stem, root_hint=hint)


def _parse_length(raw: str) -> float | None:
    """
    Convert the text after ``:`` to a branch length.

    Parameters
    ----------
    raw : str
        The field text, without the colon.

    Returns
    -------
    float | None
        The length, or ``None`` for an empty field.

    Raises
    ------
    ForestPDInputError
        If the field is not a number.
    """
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ForestPDInputError(f"Invalid Newick: malformed branch length {raw!r}") from None


def _unquote(token: str, kind: str | None) -> str:
    """
    Strip quoting from a label token.

    Parameters
    ----------
    token : str
        The raw token.
    kind : str | None
        Its token kind: ``"squote"``, ``"dquote"`` or ``"label"``.

    Returns
    -------
    str
        The label text.
    """
    if kind == "squote":
        return token[1:-1].replace("''", "'")
    if kind == "dquote":
        return token[1:-1]
    return token


def _add_lengths(a: float | None, b: float | None) -> float | None:
    """
    Sum two branch lengths, treating a missing one as absent rather than zero.

    Parameters
    ----------
    a, b : float | None
        The lengths of two edges being merged.

    Returns
    -------
    float | None
        Their sum, the one that is present, or ``None`` if both are missing.
    """
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _unroot(
    parent: list[int],
    length: list[float | None],
    label: list[str | None],
    n_children: list[int],
) -> ParsedTree:
    """
    Turn the parsed rooted tree into an unrooted :class:`ParsedTree`.

    Merges away every internal node of degree two (including a binary root), drops a unary root,
    and validates the leaf labels.

    Parameters
    ----------
    parent : list[int]
        Parent of every node, ``-1`` at the root (node 0).
    length : list[float | None]
        Length of the edge to each node's parent.
    label : list[str | None]
        Label of every node.
    n_children : list[int]
        Number of children of every node.

    Returns
    -------
    ParsedTree
        The unrooted tree, with nodes renumbered to ``0..n_nodes-1``.

    Raises
    ------
    ForestPDInputError
        If a leaf is unlabelled or two leaves share a label.
    """
    n = len(parent)
    alive = [True] * n
    root = 0
    # A unary root is a pendant edge leading nowhere: drop it, and repeat while that holds.
    while n_children[root] == 1 and n > 1:
        child = next(v for v in range(n) if alive[v] and parent[v] == root)
        alive[root] = False
        parent[child] = -1
        length[child] = None
        root = child

    # Merge every unary non-root node into the edge below it.
    merged = [v for v in range(n) if n_children[v] == 1 and alive[v] and v != root]
    for v in merged:
        alive[v] = False
    for v in range(n) if merged else ():
        if not alive[v] or v == root:
            continue
        p = parent[v]
        while not alive[p]:
            length[v] = _add_lengths(length[v], length[p])
            p = parent[p]
        parent[v] = p

    edges: list[tuple[int, int]] = []
    lengths: list[float | None] = []
    root_children = [v for v in range(n) if alive[v] and v != root and parent[v] == root]
    merge_root = len(root_children) == 2
    if merge_root:
        alive[root] = False
        a, b = root_children
        edges.append((a, b))
        lengths.append(_add_lengths(length[a], length[b]))
    for v in range(n):
        if alive[v] and v != root and not (merge_root and parent[v] == root):
            edges.append((parent[v], v))
            lengths.append(length[v])

    has_child = [False] * n
    for v in range(n):
        if alive[v] and v != root and parent[v] >= 0:
            has_child[parent[v]] = True
    leaves = [v for v in range(n) if alive[v] and not has_child[v]]
    if merge_root:
        leaves = [v for v in leaves if v != root]

    leaf_labels: dict[int, str] = {}
    seen: set[str] = set()
    for v in leaves:
        name = label[v]
        if not name:
            raise ForestPDInputError("Invalid Newick: a leaf has no label")
        if name in seen:
            raise ForestPDInputError(f"Invalid Newick: duplicate leaf label {name!r}")
        seen.add(name)
        leaf_labels[v] = name

    index = {v: i for i, v in enumerate(v for v in range(n) if alive[v])}
    return ParsedTree.from_lists(
        n_nodes=len(index),
        edges=[(index[u], index[v]) for u, v in edges],
        lengths=lengths,
        leaf_labels={index[v]: name for v, name in leaf_labels.items()},
    )


def to_newick(tree: PhyloTree, labels: Sequence[str] | None = None, lengths: bool = True) -> str:
    """
    Write a :class:`~forestpd.core.tree.PhyloTree` as a Newick string.

    The tree is unrooted, so the string is rooted at an arbitrary internal node. Reading it back
    with :func:`parse_newick` gives the same unrooted tree.

    Parameters
    ----------
    tree : PhyloTree
        The tree to write.
    labels : Sequence[str] | None, optional
        Label per leaf id. ``None`` (default) writes the leaf ids themselves.
    lengths : bool, default True
        Write branch lengths; ``False`` writes the topology only.

    Returns
    -------
    str
        The Newick string, ending in ``;``. Labels that are not plain words are single-quoted.

    Raises
    ------
    ForestPDInputError
        If ``labels`` does not hold exactly one distinct label per leaf.

    Examples
    --------
    >>> t = PhyloTree.from_edges(6, [(0, 4), (1, 4), (4, 5), (2, 5), (3, 5)],
    ...                          [1.0, 2.0, 0.5, 1.0, 1.0], {i: i for i in range(4)})
    >>> to_newick(t)
    '(0:1.0,1:2.0,(2:1.0,3:1.0):0.5);'
    >>> to_newick(t, ["a", "b", "c d", "e"])
    "(a:1.0,b:2.0,('c d':1.0,e:1.0):0.5);"
    >>> to_newick(t, lengths=False)
    '(0,1,(2,3));'
    """
    n_leaves = tree.n_leaves
    if labels is None:
        names = [str(leaf) for leaf in range(n_leaves)]
    else:
        names = list(labels)
        if len(names) != n_leaves:
            raise ForestPDInputError(f"Expected {n_leaves} labels, got {len(names)}")
        if len(set(names)) != len(names):
            raise ForestPDInputError("Leaf labels must be distinct")
    names = [_quote(name) for name in names]
    if n_leaves == 1:
        return f"{names[0]};"
    if tree.n_nodes == 2:
        # A single edge: hang both leaves off a degree-two root, which the parser merges away.
        if not lengths:
            return f"({names[0]},{names[1]});"
        return f"({names[0]}:{float(tree.weight[0])!r},{names[1]}:0.0);"

    rooted = tree.root_at(n_leaves)
    weight = tree.weight.tolist()
    parent = rooted.parent.tolist()
    parent_edge = rooted.parent_edge.tolist()
    children: list[list[int]] = [[] for _ in range(tree.n_nodes)]
    for v in rooted.order.tolist()[1:]:
        children[parent[v]].append(v)

    # Postorder, so every child's text exists before its parent's (no recursion-depth limit).
    text: list[str] = [""] * tree.n_nodes
    for v in reversed(rooted.order.tolist()):
        body = names[v] if v < n_leaves else "(" + ",".join(text[c] for c in children[v]) + ")"
        if v == rooted.root or not lengths:
            text[v] = body
        else:
            text[v] = f"{body}:{weight[parent_edge[v]]!r}"
    return text[rooted.root] + ";"


def _quote(label: str) -> str:
    """Return ``label`` as written in Newick: bare if it is a plain word, else single-quoted."""
    if _PLAIN_LABEL.fullmatch(label):
        return label
    return "'" + label.replace("'", "''") + "'"


__all__ = ["ParsedTree", "parse_newick", "to_newick", "from_parents", "ROOT_LABEL"]
