"""Unit test for the wave-1 spawn-stagger added 2026-06-09.

Pins the contract that ``RtOrchestrator._spawn_routines`` honors each
``RoutineSpec.spawn_delay_s`` BETWEEN successive wave-1 spawns (not
after the last one, and never inside wave-2 where the gate already
serialises). Used to keep two CUDA workers on the same search node
from simultaneously racing the kernel for ~17 GiB of mlock'd pinned
host pages — the failure mode that OOM-killed n09 + n13 on the first
2026-06-09 restart even though each half's steady state fit fine.

The test stubs ``RtOrchestrator._spawn_one_routine`` (the bit that
calls subprocess.Popen on real argv) and a fake sleep so the test is
deterministic and ~ms-fast.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from dsart.services.dsart_rt import PipelineConfig, RoutineSpec  # noqa: E402


def _routine(name: str, delay: float = 0.0) -> dict:
    """Build a RoutineSpec source dict with the minimum keys
    ``PipelineConfig.from_dict`` accepts."""
    return {
        "name": name,
        "cmd": "echo",
        "args": name,
        "spawn_delay_s": delay,
    }


def test_routine_spec_round_trips_spawn_delay() -> None:
    cfg = PipelineConfig.from_dict(
        {
            "schema_version": 1,
            "routines": [
                _routine("a", 4.0),
                _routine("b", 8.0),
                _routine("c"),
            ],
        }
    )
    by_name = {r.name: r for r in cfg.routines}
    assert by_name["a"].spawn_delay_s == 4.0
    assert by_name["b"].spawn_delay_s == 8.0
    assert by_name["c"].spawn_delay_s == 0.0


def test_spawn_delay_defaults_to_zero_when_omitted() -> None:
    """Older YAML without the field must keep its previous behavior
    (no stagger, no surprise sleeps)."""
    cfg = PipelineConfig.from_dict(
        {
            "schema_version": 1,
            "routines": [{"name": "a", "cmd": "echo", "args": "a"}],
        }
    )
    assert cfg.routines[0].spawn_delay_s == 0.0


@dataclass
class _SpawnTrace:
    """Records ``(action, name, t_relative)`` tuples so the test can
    assert on the order + timing of spawns / sleeps without depending
    on wall-clock."""
    events: List[Tuple[str, str, float]]
    t: float = 0.0


def _build_orchestrator(routines: list) -> tuple[object, _SpawnTrace]:
    """Construct a minimal ``RtOrchestrator`` with all real I/O stubbed
    out, returning the orchestrator and the trace buffer the test
    asserts on. The orchestrator's ``_spawn_routines`` is what we're
    testing; everything else (``_spawn_one_routine``, ``_stop_evt``)
    is replaced with cheap fakes.
    """
    from dsart.services.dsart_rt import RtOrchestrator

    cfg = PipelineConfig.from_dict(
        {"schema_version": 1, "routines": routines}
    )
    trace = _SpawnTrace(events=[])

    # Build the orchestrator WITHOUT touching DsaStore / etcd / sockets
    # by bypassing __init__ and setting only the bits _spawn_routines
    # touches.
    orch = RtOrchestrator.__new__(RtOrchestrator)
    orch._config = cfg
    orch._stop_evt = threading.Event()
    orch._children = {}

    def _fake_spawn_one_routine(r, val):
        trace.events.append(("spawn", r.name, trace.t))

    def _fake_substitute(p, val):
        return p

    def _fake_wait(timeout):
        # Advance the trace clock by the sleep amount (so the test
        # can assert "delay seen between A and B" without wall time).
        trace.t += float(timeout)
        trace.events.append(("sleep", f"{float(timeout):.1f}", trace.t))
        return False  # not interrupted

    orch._spawn_one_routine = _fake_spawn_one_routine
    orch._substitute = _fake_substitute
    orch._stop_evt.wait = _fake_wait  # type: ignore[method-assign]

    def _fake_select(routines):
        return tuple(routines)

    orch._select_routines = _fake_select  # type: ignore[method-assign]
    return orch, trace


def test_wave1_sleeps_only_between_routines_with_positive_delay() -> None:
    """Wave-1 spawns in YAML order, sleeping ``spawn_delay_s`` AFTER
    each routine except the last. Zero-delay routines emit no sleep
    (orchestrator stays snappy for the no-stagger path)."""
    orch, trace = _build_orchestrator(
        [
            _routine("rx", 4.0),
            _routine("half0", 8.0),
            _routine("half1", 0.0),
        ],
    )
    orch._spawn_routines(orch._config.routines, val=None)
    # Expect: spawn rx, sleep 4, spawn half0, sleep 8, spawn half1.
    # half1 is last, so no trailing sleep.
    expected_kinds = ["spawn", "sleep", "spawn", "sleep", "spawn"]
    assert [e[0] for e in trace.events] == expected_kinds, trace.events
    assert [e[1] for e in trace.events] == [
        "rx", "4.0", "half0", "8.0", "half1",
    ]
    # Spawn order is rx → half0 → half1.
    spawn_names = [e[1] for e in trace.events if e[0] == "spawn"]
    assert spawn_names == ["rx", "half0", "half1"]


def test_wave1_zero_delay_emits_no_sleeps() -> None:
    """When no routine carries a stagger, the orchestrator must not
    block at all (preserves the pre-2026-06-09 fast-path behavior for
    YAML revisions that don't opt into staggering)."""
    orch, trace = _build_orchestrator(
        [_routine("a"), _routine("b"), _routine("c")],
    )
    orch._spawn_routines(orch._config.routines, val=None)
    assert all(e[0] == "spawn" for e in trace.events), trace.events
    assert [e[1] for e in trace.events] == ["a", "b", "c"]


def test_stop_signal_during_stagger_aborts_remaining_wave1() -> None:
    """If the stop event fires while we're mid-stagger, the
    orchestrator must abort the wave-1 loop and NOT spawn any further
    routines (otherwise a stop verb that lands during the stagger
    would silently spawn the second half a few seconds later)."""
    from dsart.services.dsart_rt import RtOrchestrator

    cfg = PipelineConfig.from_dict(
        {
            "schema_version": 1,
            "routines": [
                _routine("first", 10.0),
                _routine("second"),
            ],
        }
    )
    trace = _SpawnTrace(events=[])
    orch = RtOrchestrator.__new__(RtOrchestrator)
    orch._config = cfg
    orch._stop_evt = threading.Event()
    orch._children = {}

    def _spawn(r, val):
        trace.events.append(("spawn", r.name, trace.t))

    def _wait_interrupted(timeout):
        # Pretend the stop signal arrives during the sleep.
        trace.t += float(timeout) * 0.5
        trace.events.append(("sleep-interrupted", "", trace.t))
        return True

    orch._spawn_one_routine = _spawn
    orch._substitute = lambda p, val: p
    orch._stop_evt.wait = _wait_interrupted  # type: ignore[method-assign]
    orch._select_routines = lambda routines: tuple(routines)  # type: ignore

    orch._spawn_routines(orch._config.routines, val=None)
    spawn_names = [e[1] for e in trace.events if e[0] == "spawn"]
    assert spawn_names == ["first"], (
        f"expected only 'first' to be spawned after stop signal; "
        f"got {spawn_names}"
    )
