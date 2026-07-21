"""Equivalence tests for the sorted-sweep age-out rebuild.

``CoincidenceGraph._rebuild`` (used by ``remove`` / ``remove_many``)
replaced a brute O(N^2) all-pairs edge scan with a MJD-sorted sweep that
forward-probes only while the time gap can still satisfy the edge
predicate (drain-collapse guard, 2026-07-21). The sweep MUST produce the
same connected components as the brute reference for every input,
including the pathological all-same-instant case that stays O(N^2).
"""

from __future__ import annotations

import random
from typing import List, Set, FrozenSet

from dsart.coinc.components import CoincidenceGraph, edge_predicate
from dsart.coinc.window import WindowEntry

_SECONDS_PER_DAY = 86400.0


def _entry(t_sec: float, width_samples: int, *,
           sample_period_us: float = 1_000_000.0) -> WindowEntry:
    """A WindowEntry at MJD offset ``t_sec`` seconds from a fixed base.

    With sample_period_us = 1e6, one width_sample == 1 second, so the
    edge predicate |Δt| <= (w_a + w_b)/2 is easy to reason about.
    """
    return WindowEntry(
        mjd=60000.0 + t_sec / _SECONDS_PER_DAY,
        snr=9.0,
        l_rad=0.0, m_rad=0.0, l_pix=0, m_pix=0,
        dm_pc_cc=100.0, dm_idx_global=0, fine_dm_idx=0,
        event_specnum=0,
        width_samples=width_samples,
        kernel_id="unit",
        flags=0,
        search_node_id=1, gpu_half=0, cube_id=0,
        sample_period_us=sample_period_us,
    )


def _brute_components(entries: List[WindowEntry]) -> Set[FrozenSet[int]]:
    ids = [id(e) for e in entries]
    parent = {i: i for i in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    n = len(entries)
    for i in range(n):
        for j in range(i + 1, n):
            if edge_predicate(entries[i], entries[j]):
                union(ids[i], ids[j])
    comps: dict = {}
    for e in entries:
        comps.setdefault(find(id(e)), set()).add(id(e))
    return {frozenset(v) for v in comps.values()}


def _graph_components(g: CoincidenceGraph) -> Set[FrozenSet[int]]:
    return {frozenset(id(e) for e in comp) for comp in g.components()}


def _build_and_rebuild(entries: List[WindowEntry]) -> CoincidenceGraph:
    """Add all entries plus a throwaway, then remove the throwaway to
    force ``_rebuild`` over exactly ``entries``."""
    g = CoincidenceGraph()
    for e in entries:
        g.add(e)
    # Throwaway far in the future so it never edges to anything.
    dummy = _entry(1_000_000.0, 1)
    g.add(dummy)
    g.remove_many([dummy])
    return g


def _assert_matches(entries: List[WindowEntry]) -> None:
    g = _build_and_rebuild(entries)
    assert _graph_components(g) == _brute_components(entries)


def test_spread_storm_components_match_brute() -> None:
    # Entries spread ~1/s, widths 4 s -> overlapping chains.
    entries = [_entry(float(i), 4) for i in range(60)]
    _assert_matches(entries)


def test_isolated_points_match_brute() -> None:
    # Far apart relative to width -> all singletons.
    entries = [_entry(100.0 * i, 2) for i in range(40)]
    _assert_matches(entries)
    g = _build_and_rebuild(entries)
    assert len(g.components()) == 40


def test_all_same_instant_single_component() -> None:
    # Pathological worst case (stays O(N^2)): everything coincides.
    entries = [_entry(0.0, 4) for _ in range(50)]
    _assert_matches(entries)
    g = _build_and_rebuild(entries)
    assert len(g.components()) == 1


def test_mixed_widths_match_brute() -> None:
    entries = []
    for i in range(80):
        entries.append(_entry(float(i) * 0.5, width_samples=(i % 7) + 1))
    _assert_matches(entries)


def test_one_wide_bridge_entry() -> None:
    # A single very wide entry can bridge otherwise-separate clusters;
    # the forward-probe cutoff must still find those edges.
    entries = [_entry(0.0, 2), _entry(10.0, 2), _entry(5.0, 30)]
    _assert_matches(entries)


def test_randomised_storms_match_brute() -> None:
    rng = random.Random(20260721)
    for _ in range(50):
        n = rng.randint(1, 120)
        entries = [
            _entry(rng.uniform(0.0, 30.0), rng.randint(1, 12))
            for _ in range(n)
        ]
        _assert_matches(entries)


def test_remove_single_triggers_rebuild() -> None:
    # Exercise the remove() path (not just remove_many()).
    entries = [_entry(float(i), 4) for i in range(20)]
    g = CoincidenceGraph()
    for e in entries:
        g.add(e)
    victim = entries[5]
    g.remove(victim)
    survivors = [e for e in entries if e is not victim]
    assert _graph_components(g) == _brute_components(survivors)
