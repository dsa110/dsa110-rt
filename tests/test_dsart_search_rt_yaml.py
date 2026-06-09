"""Schema + memory-safety invariants for ``configs/dsart_search_rt.yaml``.

The search-side analog of :mod:`test_dsart_pipeline_rt_yaml`. Pins the
two 2026-06-09 hotfixes that, together, let all four search nodes' C1
halves co-spawn cleanly on the 93 GiB hardware:

* ``c1.cube_ring_depth`` capped at 12 (per-half pinned-pool fits in
  the available physical memory once both halves and the rxring shm
  are co-resident).
* The two ``search_compute`` routines and ``search_rx`` carry
  positive ``spawn_delay_s`` values so the orchestrator staggers the
  CUDA pinned-host-pool allocation storms (the simultaneous-spawn
  race OOM-killed n09 + n13 on the first 2026-06-09 restart with
  depth=16 / spawn_delay_s=0).

If a future edit reverts either of those, this test fails loudly so
the change author re-runs the memory math against the hardware budget
rather than silently rediscovering the OOM via a midnight on-call
page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from dsart.services.dsart_rt import PipelineConfig  # noqa: E402

_YAML_PATH = _REPO / "configs" / "dsart_search_rt.yaml"


def _load() -> PipelineConfig:
    with _YAML_PATH.open() as f:
        return PipelineConfig.from_dict(yaml.safe_load(f))


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return _load()


def _by_name(cfg: PipelineConfig) -> dict[str, object]:
    return {r.name: r for r in cfg.routines}


def test_yaml_parses(cfg: PipelineConfig) -> None:
    assert cfg.schema_version == 1
    assert len(cfg.routines) >= 3
    names = {r.name for r in cfg.routines}
    assert {"search_rx", "search_compute_0", "search_compute_1"} <= names


# ---------------------------------------------------------------------------
# c1.cube_ring_depth budget (2026-06-09 second-pass hotfix)
# ---------------------------------------------------------------------------


# Hardware envelope: each search node has 93 GiB physical memory and runs
# 1 × search_rx + 2 × search_compute halves. At the production op-point
# (t_det=256, n_fdm=34, n_grid=256, fp16) one cube retention slot is
# 256 * 34 * 256 * 256 * 2 bytes = 1.067 GiB. CUDA's pinned-host caching
# allocator rounds up in 2 GiB chunks, so depth=N slot data → ⌈N*1.067/2⌉
# * 2 GiB of pinned-pool pages per half. Plus the 10.5 GiB rxring shm
# (mapped per-process) and ~8 GiB per-process Python/torch base, the
# per-host total at depth=12 is ~65 GiB (~28 GiB headroom); at depth=16
# we observed only ~7.8 GiB headroom and OOM-killed two halves during
# the startup race.
_MAX_SAFE_DEPTH_93GIB_NODES = 12


def test_cube_ring_depth_within_93gib_budget(cfg: PipelineConfig) -> None:
    """``c1.cube_ring_depth`` must stay within the 93 GiB-node memory
    envelope. Re-growing it above 12 forfeits the headroom that lets
    the two halves co-spawn cleanly under the CUDA pinned-pool race —
    re-doing the math against the hardware budget (see 2026-06-09
    YAML comment) is required before bumping this number.
    """
    c1 = cfg.raw.get("c1") or {}
    depth = c1.get("cube_ring_depth")
    assert isinstance(depth, int), (
        f"c1.cube_ring_depth must be an int, got {depth!r}"
    )
    assert 1 <= depth <= _MAX_SAFE_DEPTH_93GIB_NODES, (
        f"c1.cube_ring_depth={depth}; values > "
        f"{_MAX_SAFE_DEPTH_93GIB_NODES} re-trigger the 2026-06-09 "
        "startup OOM on the 93 GiB nodes. Re-derive the memory "
        "budget (per-half pinned-pool 2 GiB granularity + rxring + "
        "torch base) before raising. If you genuinely need more "
        "retention, fix the upstream C2-trigger latency instead "
        "(T5 BoundedCubeUploader is the right knob)."
    )


# ---------------------------------------------------------------------------
# Spawn-stagger (2026-06-09 dsart_rt change)
# ---------------------------------------------------------------------------


def test_search_rx_has_positive_spawn_delay(cfg: PipelineConfig) -> None:
    """``search_rx`` must spawn before the search_compute halves AND
    sleep long enough afterwards for its rxring shm to materialise.
    Otherwise the halves race the rxring creation and either
    (a) hit "bad magic" on attach (mmap_attach_readonly tight loop)
    or (b) deepen the simultaneous CUDA pinned-pool transient that
    OOM-killed n09 + n13 on 2026-06-09.
    """
    routines = _by_name(cfg)
    r = routines["search_rx"]
    assert r.spawn_delay_s >= 2.0, (
        f"search_rx.spawn_delay_s={r.spawn_delay_s} < 2.0; the post-"
        "spawn sleep is what gives the rxring shm time to be created "
        "and the OOM-storm time to settle before the halves are "
        "spawned. Don't drop this without re-validating "
        "co-spawn on the 93 GiB search nodes."
    )


def test_search_compute_0_staggers_against_half_1(cfg: PipelineConfig) -> None:
    """The two search_compute halves must NOT simultaneously demand
    their ~14 GiB pinned-host pools from a cold start. Half-0 carries
    the inter-half spawn delay (dsart_rt sleeps after spawning half-0
    before spawning half-1 in wave-1 order)."""
    routines = _by_name(cfg)
    r0 = routines["search_compute_0"]
    assert r0.spawn_delay_s >= 5.0, (
        f"search_compute_0.spawn_delay_s={r0.spawn_delay_s} < 5.0; "
        "the inter-half stagger is what avoids the simultaneous "
        "CUDA pinned-pool allocation storm that OOM-killed n09 + "
        "n13 on 2026-06-09. The half-0 pinned-pool warmup takes "
        "~3-5 s on the production 2080 Ti's."
    )


def test_search_compute_1_no_spawn_delay(cfg: PipelineConfig) -> None:
    """search_compute_1 is the last routine in wave-1; its
    spawn_delay_s is never honored (the orchestrator only sleeps
    between routines, not after the last one). Default 0.0 is fine;
    pin it to catch a YAML typo that adds a useless trailing sleep
    or — worse — a stray gate that lands on the wrong routine.
    """
    routines = _by_name(cfg)
    r1 = routines["search_compute_1"]
    assert r1.spawn_delay_s == 0.0, (
        f"search_compute_1.spawn_delay_s={r1.spawn_delay_s}; "
        "this is the last routine in wave-1, the delay is a no-op "
        "but a non-zero value here usually means a copy/paste error "
        "from the half-0 block."
    )


# ---------------------------------------------------------------------------
# Wave-1 ordering (search_rx must spawn before the halves)
# ---------------------------------------------------------------------------


def test_search_rx_listed_before_search_compute_halves(cfg: PipelineConfig) -> None:
    """The orchestrator spawns wave-1 routines in YAML order. search_rx
    creates the rxring shm; the halves attach it read-only. If the
    halves are listed first they enter the 180 s shm-wait retry loop
    AND start their own pinned-pool allocation in parallel with
    search_rx's 10.5 GiB shm creation — exactly the race we're
    trying to avoid. Pin the listing order so a future edit can't
    silently regress.
    """
    order = [r.name for r in cfg.routines]
    i_rx = order.index("search_rx")
    i_h0 = order.index("search_compute_0")
    i_h1 = order.index("search_compute_1")
    assert i_rx < i_h0 < i_h1, (
        f"wave-1 routine order is {order}; expected search_rx "
        "before search_compute_0 before search_compute_1."
    )
