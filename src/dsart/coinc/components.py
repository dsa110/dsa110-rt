"""Incremental union-find connected-components on in-window candidates.

Edge predicate (from ``docs/c1c2/C1C2_DESIGN.md`` §3.3):

    edge(i, j) iff |t_i - t_j| ≤ (w_i + w_j) / 2

with ``t`` in seconds (MJD × 86400) and ``w`` converted from sample
units via the per-entry ``sample_period_us``. DM / position / width
are *not* in the edge predicate — they characterise the cluster.

Because the window is monotone in MJD, every new add only ever creates
edges to entries already in the window with ``mjd_j ≤ mjd_new + dt_max``
and ``mjd_j ≥ mjd_new - dt_max`` (where ``dt_max`` is bounded by the
maximum legal pair half-sum), so the per-add cost is linear in the
number of neighbours, not in the window size.

Removal (age-out) is rare relative to add, and naively re-deriving the
components on remove is cheap enough at typical window depths
(< 100k entries) — we run the full union-find scan over the surviving
members rather than incrementally invalidate edges. The full rebuild
is O(n²) in the worst case but is bounded by ``coinc.window_s`` and
the per-half candidate rate; in practice the cluster scan dominates.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

from .window import WindowEntry

__all__ = [
    "CoincidenceGraph",
    "edge_predicate",
]


# A few constants kept module-level so unit tests can reach them.
_SECONDS_PER_DAY: float = 86400.0


def edge_predicate(a: WindowEntry, b: WindowEntry) -> bool:
    """Time-only edge: |Δt_sec| ≤ (w_a + w_b) / 2 (in seconds).

    Width is converted to seconds via the per-entry
    ``sample_period_us``; the two entries can in principle disagree on
    sample-period (different operating points), so we average over each
    side's own conversion.
    """
    dt_sec = abs(a.mjd - b.mjd) * _SECONDS_PER_DAY
    w_a_sec = a.width_samples * a.sample_period_us / 1.0e6
    w_b_sec = b.width_samples * b.sample_period_us / 1.0e6
    return dt_sec <= 0.5 * (w_a_sec + w_b_sec)


class _UnionFind:
    """Tiny union-find keyed by hashable id (object id of WindowEntry)."""

    __slots__ = ("_parent", "_rank")

    def __init__(self) -> None:
        self._parent: Dict[int, int] = {}
        self._rank: Dict[int, int] = {}

    def __contains__(self, x: int) -> bool:
        return x in self._parent

    def add(self, x: int) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: int) -> int:
        # iterative path compression
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            nxt = self._parent[x]
            self._parent[x] = root
            x = nxt
        return root

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        # union by rank
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        return ra


class CoincidenceGraph:
    """Maintains connected components over the current in-window set.

    Public API mirrors what the C2 service needs:

      * ``add(entry)`` — splice ``entry`` into the graph (and any new
        edges to existing entries) and return the *component id* that
        ``entry`` now belongs to. The component id is the python
        ``id()`` of the union-find root entry.
      * ``remove(entry)`` — drop ``entry`` and force a structural
        rebuild over the survivors. Cheap when the window depth is
        small.
      * ``components()`` — iterate over the current component
        membership lists.
      * ``component_of(entry)`` / ``components_touched(entries)`` —
        helpers the service uses to figure out which components a new
        batch perturbed.
    """

    def __init__(self) -> None:
        self._entries: Dict[int, WindowEntry] = {}
        self._uf = _UnionFind()

    # ----- structural ---------------------------------------------------

    def add(self, entry: WindowEntry) -> int:
        eid = id(entry)
        if eid in self._entries:
            return self._uf.find(eid)
        self._entries[eid] = entry
        self._uf.add(eid)
        # Edges to current members. We do a brute scan since we expect
        # < 100k entries; for production rates with window_s=5s the
        # window depth is typically O(10²)–O(10³).
        for other_id, other in self._entries.items():
            if other_id == eid:
                continue
            if edge_predicate(entry, other):
                self._uf.union(eid, other_id)
        return self._uf.find(eid)

    def remove(self, entry: WindowEntry) -> None:
        eid = id(entry)
        if eid not in self._entries:
            return
        del self._entries[eid]
        # Rebuild union-find from scratch over survivors. Splitting a
        # union-find on element removal is not free; a periodic rebuild
        # is the simplest correct option and matches the design's
        # complexity budget.
        self._uf = _UnionFind()
        members = list(self._entries.items())
        for eid_a, _ in members:
            self._uf.add(eid_a)
        # Rebuild edges. n² over the surviving set; window depth keeps
        # this bounded.
        n = len(members)
        for i in range(n):
            eid_a, ea = members[i]
            for j in range(i + 1, n):
                eid_b, eb = members[j]
                if edge_predicate(ea, eb):
                    self._uf.union(eid_a, eid_b)

    def remove_many(self, entries: Iterable[WindowEntry]) -> None:
        """Drop a batch of entries and rebuild only once afterwards."""
        any_removed = False
        for e in entries:
            eid = id(e)
            if eid in self._entries:
                del self._entries[eid]
                any_removed = True
        if not any_removed:
            return
        self._uf = _UnionFind()
        members = list(self._entries.items())
        for eid_a, _ in members:
            self._uf.add(eid_a)
        n = len(members)
        for i in range(n):
            eid_a, ea = members[i]
            for j in range(i + 1, n):
                eid_b, eb = members[j]
                if edge_predicate(ea, eb):
                    self._uf.union(eid_a, eid_b)

    # ----- query --------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def component_of(self, entry: WindowEntry) -> int:
        eid = id(entry)
        if eid not in self._entries:
            raise KeyError(f"entry {entry!r} not in graph")
        return self._uf.find(eid)

    def components_touched(self, entries: Iterable[WindowEntry]) -> Set[int]:
        """Return component ids touched by ``entries`` (all must be in
        the graph)."""
        ids: Set[int] = set()
        for e in entries:
            eid = id(e)
            if eid not in self._entries:
                continue
            ids.add(self._uf.find(eid))
        return ids

    def components(self) -> List[List[WindowEntry]]:
        """All current components, each a list of :class:`WindowEntry`.

        Order within each component is insertion order; order of the
        components themselves is by component-root id (arbitrary but
        deterministic for a given run).
        """
        buckets: Dict[int, List[WindowEntry]] = {}
        for eid, entry in self._entries.items():
            root = self._uf.find(eid)
            buckets.setdefault(root, []).append(entry)
        # Sort each component's members by MJD ascending for stable
        # downstream consumers.
        for root in buckets:
            buckets[root].sort(key=lambda e: e.mjd)
        return [buckets[root] for root in sorted(buckets.keys())]

    def component_members(self, component_id: int) -> List[WindowEntry]:
        """Members of the component identified by ``component_id``."""
        out: List[WindowEntry] = []
        for eid, entry in self._entries.items():
            if self._uf.find(eid) == component_id:
                out.append(entry)
        out.sort(key=lambda e: e.mjd)
        return out
