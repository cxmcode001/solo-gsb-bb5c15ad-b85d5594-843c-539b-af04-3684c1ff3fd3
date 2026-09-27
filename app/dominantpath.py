"""Dominant (heaviest exclusive-time) root-to-leaf path of one trace.

Each span's weight is the length of its own half-open ``[start_ns, end_ns)``
interval minus the union of the intersections with its *direct* children's
intervals — the same exclusive-time rule as :mod:`app.servicetime`. Child
intervals that overlap each other or extend beyond the parent are subtracted
only once and only for the part inside the parent interval.

Spans whose ``parent_span_id`` is missing from the snapshot are treated as
roots, so the graph is a forest; the weight of every root-to-leaf path is the
sum of its node weights and the path with the largest total wins. Ties are
broken by the lexicographically smallest span-id sequence. Self-references
and parent/child cycles make the notion of a root-to-leaf path meaningless,
so they raise :class:`CyclicReferenceError` (the API maps it to 422). Pure
functions only — no I/O, no sampling policy.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

# Well-tested helpers shared with the integrity/service-time modules: cycle
# detection in the child->parent functional graph and interval-union length.
from .integrity import _canonical_cycle, _find_cycles
from .servicetime import _union_length


class CyclicReferenceError(ValueError):
    """The snapshot contains self-references and/or parent/child cycles.

    ``self_references`` holds the sorted self-referencing span ids;
    ``cycles`` holds every ring, each rotated to start at its smallest span
    id, the rings sorted among themselves.
    """

    def __init__(
        self, self_references: list[str], cycles: list[list[str]]
    ) -> None:
        self.self_references = self_references
        self.cycles = cycles
        involved = sorted(
            set(self_references) | {sid for ring in cycles for sid in ring}
        )
        super().__init__(
            "cyclic parent references among spans: " + ", ".join(involved)
        )


def dominant_path(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Heaviest root-to-leaf path over one trace's spans.

    ``rows`` are mappings with at least ``span_id``, ``parent_span_id``,
    ``start_ns`` and ``end_ns`` keys (``sqlite3.Row`` works). Returns
    ``{"span_count", "total_weight_ns", "path"}`` where ``path`` is a list of
    ``{"span_id", "weight_ns"}`` in root-to-leaf order. Raises
    :class:`CyclicReferenceError` when the snapshot has self-references or
    parent/child cycles.
    """
    data = [
        (r["span_id"], r["parent_span_id"], int(r["start_ns"]), int(r["end_ns"]))
        for r in rows
    ]
    known_ids = {sid for sid, _pid, _s, _e in data}

    # child -> parent, only for edges whose parent exists in the snapshot.
    # Self-references are collected separately: they are fatal, not edges.
    parent_of: dict[str, str] = {}
    self_refs: list[str] = []
    for sid, pid, _start, _end in data:
        if pid is None:
            continue
        if pid == sid:
            self_refs.append(sid)
        elif pid in known_ids:
            parent_of[sid] = pid

    cycles = sorted(_canonical_cycle(c) for c in _find_cycles(parent_of))
    if self_refs or cycles:
        raise CyclicReferenceError(sorted(self_refs), cycles)

    # What remains is a forest: spans without a parent edge are roots
    # (including orphans whose parent never arrived).
    children: dict[str, list[str]] = {}
    for child, parent in parent_of.items():
        children.setdefault(parent, []).append(child)
    for child_list in children.values():
        child_list.sort()
    roots = sorted(
        sid for sid, _pid, _s, _e in data if sid not in parent_of
    )

    intervals = {sid: (start, end) for sid, _pid, start, end in data}
    weight = {
        sid: _exclusive_ns(
            start, end, (intervals[c] for c in children.get(sid, ()))
        )
        for sid, _pid, start, end in data
    }

    best: list[tuple[str, int]] | None = None
    for root in roots:
        candidate = _heaviest(root, children, weight)
        if best is None or _better(candidate, best):
            best = candidate
    assert best is not None  # at least one span => at least one root
    return {
        "span_count": len(data),
        "total_weight_ns": sum(w for _sid, w in best),
        "path": [{"span_id": sid, "weight_ns": w} for sid, w in best],
    }


def _exclusive_ns(
    start_ns: int, end_ns: int, child_intervals: Iterable[tuple[int, int]]
) -> int:
    """Own interval length minus the union of direct-child intersections."""
    length = max(0, end_ns - start_ns)
    overlaps = []
    for c_start, c_end in child_intervals:
        lo = max(c_start, start_ns)
        hi = min(c_end, end_ns)
        if hi > lo:
            overlaps.append((lo, hi))
    return length - _union_length(overlaps)


def _heaviest(
    root: str,
    children: dict[str, list[str]],
    weight: dict[str, int],
) -> list[tuple[str, int]]:
    """Heaviest root-to-leaf path below ``root``, as (span_id, weight) pairs.

    Iterative post-order walk (traces can be deep); each node's best
    continuation is the child subtree with the largest total weight, ties
    broken by the lexicographically smaller span-id sequence.
    """
    memo: dict[str, list[tuple[str, int]]] = {}
    stack: list[tuple[str, bool]] = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if not expanded:
            stack.append((node, True))
            for child in children.get(node, ()):
                stack.append((child, False))
            continue
        best_child: list[tuple[str, int]] | None = None
        for child in children.get(node, ()):
            candidate = memo[child]
            if best_child is None or _better(candidate, best_child):
                best_child = candidate
        memo[node] = [(node, weight[node])] + (best_child or [])
    return memo[root]


def _better(
    candidate: list[tuple[str, int]], current: list[tuple[str, int]]
) -> bool:
    """True when ``candidate`` beats ``current``: heavier total, or equal
    total with a lexicographically smaller span-id sequence."""
    cand_total = sum(w for _sid, w in candidate)
    cur_total = sum(w for _sid, w in current)
    if cand_total != cur_total:
        return cand_total > cur_total
    return [sid for sid, _w in candidate] < [sid for sid, _w in current]
