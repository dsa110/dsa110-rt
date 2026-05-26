"""Tests for :mod:`dsart.coinc.components` (connected components on
the half-window time edge)."""

from __future__ import annotations

from typing import List

import pytest

from dsart.coinc.components import CoincidenceGraph, edge_predicate
from dsart.coinc.window import WindowEntry


def _entry(
    *,
    mjd: float,
    width_samples: int = 4,
    sample_period_us: float = 1048.576,
    snr: float = 10.0,
    search_node_id: int = 1,
    gpu_half: int = 0,
    cube_id: int = 0,
    event_specnum: int = 0,
) -> WindowEntry:
    return WindowEntry(
        mjd=mjd,
        snr=snr,
        l_rad=0.0,
        m_rad=0.0,
        l_pix=0,
        m_pix=0,
        dm_pc_cc=100.0,
        dm_idx_global=0,
        fine_dm_idx=0,
        event_specnum=event_specnum,
        width_samples=width_samples,
        kernel_id="unit:d1:b4",
        flags=0,
        search_node_id=search_node_id,
        gpu_half=gpu_half,
        cube_id=cube_id,
        sample_period_us=sample_period_us,
    )


def test_edge_predicate_self_always_true() -> None:
    e = _entry(mjd=60781.0, width_samples=4, sample_period_us=1_000_000.0)
    assert edge_predicate(e, e) is True


def test_edge_predicate_within_half_sum() -> None:
    # 1 second per sample, both width=4 samples → half-sum = 4 s.
    a = _entry(mjd=60781.0, width_samples=4, sample_period_us=1_000_000.0)
    b_in = _entry(mjd=60781.0 + 3.0 / 86400.0, width_samples=4,
                  sample_period_us=1_000_000.0)
    # b_just_in: 3.999 s away (1 ms slack avoids float-precision drift
    # in the 60781.0 + Δs/86400 round-trip; we only care that the
    # predicate is monotone in Δt, not that the boundary is exactly 4s).
    b_just_in = _entry(mjd=60781.0 + 3.999 / 86400.0, width_samples=4,
                       sample_period_us=1_000_000.0)
    b_out = _entry(mjd=60781.0 + 4.01 / 86400.0, width_samples=4,
                   sample_period_us=1_000_000.0)
    assert edge_predicate(a, b_in) is True
    assert edge_predicate(a, b_just_in) is True
    assert edge_predicate(a, b_out) is False


def test_single_component_singleton() -> None:
    g = CoincidenceGraph()
    e = _entry(mjd=60781.0)
    cid = g.add(e)
    assert g.component_of(e) == cid
    comps = g.components()
    assert len(comps) == 1
    assert len(comps[0]) == 1


def test_two_close_entries_form_one_component() -> None:
    g = CoincidenceGraph()
    a = _entry(mjd=60781.0, width_samples=4, sample_period_us=1_000_000.0)
    b = _entry(mjd=60781.0 + 1.0 / 86400.0, width_samples=4,
               sample_period_us=1_000_000.0)
    g.add(a)
    g.add(b)
    # add()'s returned cid is only valid at call-time (unions can
    # promote a new root later); the durable invariant is that
    # component_of() agrees after the unions settle.
    assert g.component_of(a) == g.component_of(b)
    comps = g.components()
    assert len(comps) == 1
    assert {id(m) for m in comps[0]} == {id(a), id(b)}


def test_two_disjoint_components() -> None:
    g = CoincidenceGraph()
    a = _entry(mjd=60781.0, width_samples=4, sample_period_us=1_000_000.0)
    b = _entry(mjd=60781.0 + 100.0 / 86400.0, width_samples=4,
               sample_period_us=1_000_000.0)
    g.add(a)
    g.add(b)
    comps = g.components()
    assert len(comps) == 2
    # Touched components: only the one b is in.
    touched = g.components_touched([b])
    assert len(touched) == 1
    assert g.component_of(b) in touched


def test_chain_of_three_merges_via_transitivity() -> None:
    # A — B within half-sum, B — C within half-sum, A — C outside.
    # Connected components should still join all three.
    g = CoincidenceGraph()
    a = _entry(mjd=60781.0, width_samples=4, sample_period_us=1_000_000.0)
    b = _entry(mjd=60781.0 + 3.0 / 86400.0, width_samples=4,
               sample_period_us=1_000_000.0)
    c = _entry(mjd=60781.0 + 6.0 / 86400.0, width_samples=4,
               sample_period_us=1_000_000.0)
    # AB: |Δt|=3s ≤ 4s. BC: |Δt|=3s ≤ 4s. AC: |Δt|=6s > 4s.
    assert edge_predicate(a, b)
    assert edge_predicate(b, c)
    assert not edge_predicate(a, c)
    g.add(a)
    g.add(b)
    g.add(c)
    comps = g.components()
    assert len(comps) == 1
    assert len(comps[0]) == 3


def test_fork_two_neighbours_share_root() -> None:
    # Central node connected to two others; the two others are not
    # directly connected.
    g = CoincidenceGraph()
    center = _entry(mjd=60781.0, width_samples=4,
                    sample_period_us=1_000_000.0)
    left = _entry(mjd=60781.0 - 3.0 / 86400.0, width_samples=4,
                  sample_period_us=1_000_000.0)
    right = _entry(mjd=60781.0 + 3.0 / 86400.0, width_samples=4,
                   sample_period_us=1_000_000.0)
    # |Δt|=3s ≤ 4s for left/center and center/right.
    # |Δt|=6s > 4s for left/right.
    g.add(center)
    g.add(left)
    g.add(right)
    assert g.component_of(left) == g.component_of(right)
    assert g.component_of(center) == g.component_of(left)
    assert len(g.components()) == 1


def test_remove_splits_chain_into_two_components() -> None:
    g = CoincidenceGraph()
    a = _entry(mjd=60781.0, width_samples=4, sample_period_us=1_000_000.0)
    b = _entry(mjd=60781.0 + 3.0 / 86400.0, width_samples=4,
               sample_period_us=1_000_000.0)
    c = _entry(mjd=60781.0 + 6.0 / 86400.0, width_samples=4,
               sample_period_us=1_000_000.0)
    g.add(a); g.add(b); g.add(c)
    assert len(g.components()) == 1
    g.remove(b)  # cuts the chain
    assert len(g.components()) == 2
    assert g.component_of(a) != g.component_of(c)


def test_remove_many_idempotent_for_missing() -> None:
    g = CoincidenceGraph()
    a = _entry(mjd=60781.0, width_samples=4, sample_period_us=1_000_000.0)
    b = _entry(mjd=60781.0 + 3.0 / 86400.0, width_samples=4,
               sample_period_us=1_000_000.0)
    g.add(a); g.add(b)
    ghost = _entry(mjd=60781.5, width_samples=4,
                   sample_period_us=1_000_000.0)
    # ghost is not in the graph; remove_many should ignore it
    g.remove_many([ghost])
    assert len(g) == 2


def test_components_touched_after_add() -> None:
    g = CoincidenceGraph()
    a = _entry(mjd=60781.0, width_samples=4, sample_period_us=1_000_000.0)
    g.add(a)
    new = _entry(mjd=60781.0 + 1.0 / 86400.0, width_samples=4,
                 sample_period_us=1_000_000.0)
    g.add(new)
    touched = g.components_touched([new])
    assert touched == {g.component_of(a)}


def test_components_membership_sorted_by_mjd() -> None:
    g = CoincidenceGraph()
    entries: List[WindowEntry] = []
    for i in range(5):
        e = _entry(mjd=60781.0 + (i * 0.5) / 86400.0,
                   width_samples=4, sample_period_us=1_000_000.0,
                   event_specnum=i)
        g.add(e)
        entries.append(e)
    comps = g.components()
    assert len(comps) == 1
    mjds = [e.mjd for e in comps[0]]
    assert mjds == sorted(mjds)
