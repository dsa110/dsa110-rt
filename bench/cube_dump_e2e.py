#!/usr/bin/env python3
"""bench/cube_dump_e2e.py — M6 chunk 7 cube-dump end-to-end bench.

Drives the chunk-5 :class:`SearchComputeService` against a synthetic
RX-ring source for a configurable number of cubes and verifies the
two cube-dump trigger paths end-to-end:

  * **Auto-trigger path (M6 D7/D8).** Bursty bright injections at a
    configurable cube cadence (default: every 10th cube). Each
    injection produces detector candidates that the chunk-1 clusterer
    rolls up into a peak ``ClusterRecord``; the chunk-3
    :class:`BrightPulsePredicate` (configured permissively here —
    ``min_snr=auto_min_snr`` defaults to 0.1, ``holdoff_ms=0``) fires
    on every cluster, and the writer thread persists each accepted
    submit as an NPZ archive.

  * **UDP-trigger path (M6 D9/D12).** A per-cube list of UDP-fired
    cube indices (``--udp-cubes 10,30,50,70``) lets the bench inject
    one ``b"dump"`` datagram on the loopback interface immediately
    before the listed cube enters ``_process_one_cube``. The chunk-4
    listener arms its one-shot flag, the per-cube driver consumes it
    before pipeline.process, and the writer persists the cube tensor
    as ``trigger_source='udp'``.

Two stress paths gate the chunk-7 deliverable:

  * **Queue-backpressure (M6 D7).** Run with ``--queue-maxsize 1
    --inject-backpressure 50`` to clamp the writer's bounded queue +
    monkey-patch ``np.savez`` with a 50 ms sleep. Bursts of dumps
    arriving faster than the slow writer drains land on a full queue
    and bump ``CubeDumpWriter.n_dropped``; the dispatch hot path
    stays non-blocking (asserted by the bench's wall-clock per-cube
    metric, NOT including ``stop()``-side writer drain).

  * **Sustained throughput (M6 D7 closure).** With default knobs the
    writer drains every accepted submit before the next burst lands;
    ``n_dumps_written == n_auto_dumps_dispatched + n_udp_dumps_dispatched``
    and ``n_dumps_dropped == 0``.

The bench also captures writer-thread ``np.savez`` wall-clock time via
the chunk-7 ``CubeDumpWriter.recent_write_ms_ms`` ring buffer (the
minimal extension this chunk lands on top of chunk 3) so the operator-
facing ``writer_p50_ms`` / ``writer_p99_ms`` / ``writer_max_ms``
metrics fall out of the bench without extra instrumentation.

CLI surface (see ``--help``):

  python -m bench.cube_dump_e2e \\
      --report-dir bench/reports/M6/cube_dump_e2e \\
      --n-cubes 100 \\
      --t-det 32 --n-fdm 4 --n-grid 16 \\
      --queue-maxsize 4 \\
      --auto-min-snr 0.1 \\
      --enable-udp \\
      --udp-cubes 10,30,50,70 \\
      --rng-seed 42

Outputs (under ``--report-dir``):

  * ``report.json`` — schema documented in the chunk-7 spec; consumed
    by ``tests/test_cube_dump_e2e_bench.py``.
  * ``cube_dump/`` — directory holding the per-(search_node, gpu)
    NPZ dumps + their sidecar manifest fields. The writer thread
    composes the canonical filename from
    ``cube_s${sid}_g${g}_${event_specnum_start}.npz``.
  * ``logs/`` — chunk-2 T1/T2 ASCII log rows (one pair per hour).
  * ``bench.log`` — operator-facing progress log.

The bench ALSO accepts ``--voltage-run-id <id>`` as a metadata-only
tag so the M6 DoD path's existing
``python -m bench.cube_dump_e2e --voltage-run-id 250924mptq``
invocation stays smoke-clean. The flag is not used to drive the
synthetic source — chunk 7's gate operates entirely on synthetic
cubes per the chunk-7 plan ("Stick to ``SyntheticRxRingSource`` to
keep the bench fast and deterministic").

Defaults are sized for the DoD smoke path (``n_cubes=50``, small
geometry, enable-udp on with two trigger cubes). The spec example
above produces the full chunk-7 characterisation run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("DSART_TEST", "1")

import torch  # noqa: E402

from dsart.cluster.cands_logger import CandsLoggerConfig  # noqa: E402
from dsart.cluster.forward import (  # noqa: E402
    ClustererBackend,
    ClustererConfig,
)
from dsart.dump import cube_dump as _cube_dump_module  # noqa: E402
from dsart.dump.cube_dump import (  # noqa: E402
    BrightPulsePredicateConfig,
    CubeDumpWriterConfig,
)
from dsart.dump.udp_listener import UdpTriggerListenerConfig  # noqa: E402
from dsart.services.cube_pipeline import CubePipelineConfig  # noqa: E402
from dsart.services.rx_ring import (  # noqa: E402
    SyntheticInjection,
    SyntheticRxRingSource,
)
from dsart.services.search_compute import (  # noqa: E402
    SearchComputeConfig,
    SearchComputeService,
)


_LOG = logging.getLogger("bench.cube_dump_e2e")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


# Default geometry: small enough for h01-CPU smoke + the DoD path's
# `python -m bench.cube_dump_e2e --voltage-run-id ...` invocation.
# The chunk-7 spec example runs with --n-cubes 100; the DoD smoke
# default trims to 50 cubes for fast turnaround.
DEFAULT_N_CUBES: int = 50
DEFAULT_T_DET: int = 32
DEFAULT_N_FDM: int = 4
DEFAULT_N_GRID: int = 16
DEFAULT_QUEUE_MAXSIZE: int = 4
DEFAULT_AUTO_MIN_SNR: float = 0.1
DEFAULT_DETECTOR_THRESHOLD_SIGMA: float = 8.0
DEFAULT_INJECT_EVERY: int = 10
DEFAULT_INJECT_AMPLITUDE: float = 200.0
DEFAULT_INJECT_BACKPRESSURE_MS: float = 0.0
DEFAULT_RNG_SEED: int = 42
DEFAULT_UDP_DELIVERY_TIMEOUT_S: float = 0.5

DEFAULT_REPORT_DIR: Path = (
    REPO_ROOT / "bench" / "reports" / "M6" / "cube_dump_e2e"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Return the HEAD SHA, or "unknown" if git isn't available."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _build_dm_grids(n_fdm: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic coarse/fine DM grid for the synthetic source.

    Mirrors ``bench/voltage_fixture_search._build_dm_grids`` so the two
    benches see consistent grid scaffolding. The bench does NOT gate
    on dispersion correctness — that's chunk 1's job.
    """
    n_coarse = max(2, n_fdm // 2)
    n_fine_per_coarse = max(1, n_fdm // n_coarse)
    coarse = np.linspace(50.0, 200.0, n_coarse, dtype=np.float64)
    spacing = (
        (coarse[1] - coarse[0]) / n_fine_per_coarse if n_coarse > 1 else 1.0
    )
    fine = np.concatenate(
        [
            coarse[c] + np.arange(n_fine_per_coarse) * spacing
            for c in range(n_coarse)
        ]
    )
    fine = fine[:n_fdm]
    fine_to_coarse = np.repeat(
        np.arange(n_coarse, dtype=np.int64), n_fine_per_coarse
    )[:n_fdm]
    return coarse, fine, fine_to_coarse


def parse_udp_cubes(spec: Optional[str]) -> Tuple[int, ...]:
    """Parse the ``--udp-cubes`` comma-separated list to a tuple of ints.

    Empty / None returns an empty tuple. Non-integer entries raise
    ``ValueError`` (the argparse layer surfaces this as a usage error).
    Duplicates are de-duplicated and the result is sorted ascending so
    downstream lookup uses a stable order.
    """
    if not spec:
        return ()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    cubes = sorted({int(p) for p in parts})
    return tuple(cubes)


def _percentile_ms(values_ms: Sequence[float], q: float) -> float:
    """Return the q-th percentile of ``values_ms`` (ms), or 0.0 if empty."""
    if not values_ms:
        return 0.0
    return float(np.percentile(np.asarray(values_ms, dtype=np.float64), q))


@dataclass(frozen=True, slots=True)
class _BenchInjections:
    """Resolved injection plan for one bench run."""
    inject_cubes: Tuple[int, ...]
    udp_cubes: Tuple[int, ...]
    inject_amplitude: float
    t_in_cube: int
    l_pix: int
    m_pix: int


def _resolve_injections(
    *,
    n_cubes: int,
    inject_every: int,
    inject_amplitude: float,
    t_det: int,
    n_grid: int,
    enable_udp: bool,
    udp_cubes_raw: Optional[str],
) -> _BenchInjections:
    """Resolve the injection + UDP trigger plan.

    Auto-trigger injections land at every ``inject_every``-th cube
    (cubes 0, K, 2K, ...) at the cube centre cell; this is the cluster-
    aware bursty workload chunk 7's "1 dump every ~10 cubes" gate
    targets. UDP triggers are sourced verbatim from the CLI list.
    """
    if inject_every <= 0:
        raise ValueError(
            f"inject_every={inject_every}, expected > 0"
        )
    inject_cubes = tuple(range(0, n_cubes, inject_every))
    udp_cubes = parse_udp_cubes(udp_cubes_raw) if enable_udp else ()
    return _BenchInjections(
        inject_cubes=inject_cubes,
        udp_cubes=udp_cubes,
        inject_amplitude=float(inject_amplitude),
        t_in_cube=int(t_det // 2),
        l_pix=int(n_grid // 2),
        m_pix=int(n_grid // 2),
    )


def _install_backpressure_patch(sleep_ms: float) -> Tuple[Any, Any]:
    """Patch ``dsart.dump.cube_dump.np.savez`` with a sleep-then-savez.

    Returns the ``(module, original_savez)`` tuple so the caller can
    restore the original in a finally block. The patch is local to
    the cube_dump module's bound ``np`` reference (writer threads
    call ``np.savez`` resolved at module-import time), so it does not
    affect any other tensor I/O the bench runs.
    """
    if sleep_ms <= 0.0:
        return _cube_dump_module.np, _cube_dump_module.np.savez
    real_savez = _cube_dump_module.np.savez
    sleep_s = float(sleep_ms) / 1000.0

    def _slow_savez(*args: Any, **kwargs: Any) -> Any:
        time.sleep(sleep_s)
        return real_savez(*args, **kwargs)

    _cube_dump_module.np.savez = _slow_savez  # type: ignore[assignment]
    _LOG.info(
        "inject-backpressure patched: np.savez sleeps %.3f ms before each write",
        sleep_ms,
    )
    return _cube_dump_module.np, real_savez


def _restore_savez(np_mod: Any, real_savez: Any) -> None:
    """Reverse of :func:`_install_backpressure_patch`."""
    np_mod.savez = real_savez


# ---------------------------------------------------------------------------
# Bench main
# ---------------------------------------------------------------------------


def _setup_logging(out_dir: Path) -> None:
    """Configure module-level logging to stdout + ``bench.log``."""
    bench_log_path = out_dir / "bench.log"
    handler = logging.FileHandler(bench_log_path, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    _LOG.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) for h in _LOG.handlers):
        _LOG.addHandler(handler)
    if not any(isinstance(h, logging.StreamHandler) for h in _LOG.handlers):
        _LOG.addHandler(logging.StreamHandler(sys.stdout))


def _build_search_compute_config(
    args: argparse.Namespace,
    *,
    out_dir: Path,
) -> SearchComputeConfig:
    """Translate CLI args into the static service config."""
    pipeline_cfg = CubePipelineConfig(
        n_grid=int(args.n_grid),
        edge_mask_kernel_support=3,
        cube_dtype=torch.float32,
        device="cpu",
    )
    return SearchComputeConfig(
        pipeline=pipeline_cfg,
        n_fdm=int(args.n_fdm),
        detector_threshold_sigma=float(args.detector_threshold_sigma),
        detector_dtype=torch.float32,
        detector_device="cpu",
        detector_streaming=True,
        detector_streaming_tile_size=64,
        search_node_id=0,
        gpu_half=0,
        layer1_n_burnin_cubes=1,
        cube_cell_l_rad=1.5e-4,
        cube_cell_m_rad=1.5e-4,
        cube_sample_period_us=131.072,
        cube_sample_period_specnum=16,
        # Clusterer: DBSCAN sidesteps the optional hdbscan import path
        # and is the chunk-1 D5 fallback; behaviour for the chunk-7
        # gate (cluster-records flow → predicate fires) is identical.
        clusterer_config=ClustererConfig(backend=ClustererBackend.DBSCAN),
        bright_pulse_predicate_config=BrightPulsePredicateConfig(
            min_snr=float(args.auto_min_snr),
            holdoff_ms=0.0,
        ),
        cube_dump_writer_config=CubeDumpWriterConfig(
            dump_root=out_dir / "cube_dump",
            search_node_id=0,
            gpu_half=0,
            queue_maxsize=int(args.queue_maxsize),
        ),
        udp_trigger_listener_config=(
            UdpTriggerListenerConfig(host="127.0.0.1", port=0)
            if bool(args.enable_udp)
            else None
        ),
        cands_logger_config=CandsLoggerConfig(
            log_root=out_dir / "logs",
            search_node_id=0,
            gpu_half=0,
        ),
    )


async def _wait_for_udp_delivery(
    service: SearchComputeService,
    *,
    prev_n: int,
    timeout_s: float,
) -> bool:
    """Yield to the asyncio loop until the UDP listener has received
    one more datagram than ``prev_n``.

    Returns True iff the listener observed at least one new datagram
    before ``timeout_s`` elapsed. The chunk-4 listener's
    ``datagram_received`` callback runs on the asyncio event loop
    thread, so the bench MUST yield (await) between ``sk.sendto`` and
    the next ``consume_dump_next_cube_flag`` poll for the flag to
    actually be set; ``asyncio.sleep(0)`` is generally insufficient
    for loopback UDP delivery (the kernel queues the datagram but
    asyncio doesn't dispatch the reader until the next event loop
    iteration with a non-zero deadline).
    """
    listener = service.udp_listener
    if listener is None:  # pragma: no cover - guarded by caller
        return False
    deadline = time.monotonic() + float(timeout_s)
    while listener.n_datagrams_received <= prev_n:
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.001)
    return True


async def _drive_bench(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    inj: _BenchInjections,
) -> Dict[str, Any]:
    """Drive the SearchComputeService end-to-end.

    Runs the per-cube loop manually (rather than ``service.run()``)
    so the bench can interleave UDP datagram sends + cube processing
    with single-cube precision. The dispatch wall clock is measured
    over the cube loop only — ``service.stop()``'s writer-drain wait
    is excluded so the chunk-7 ``wall_s / n_cubes`` test gate
    correctly characterises dispatch latency, not writer-finish
    latency.
    """
    n_cubes = int(args.n_cubes)
    t_det = int(args.t_det)
    n_fdm = int(args.n_fdm)
    n_grid = int(args.n_grid)

    coarse_dm, fine_dm, fine_to_coarse = _build_dm_grids(n_fdm)
    rng = np.random.default_rng(int(args.rng_seed))
    injections = tuple(
        SyntheticInjection(
            cube_idx=int(c),
            t_in_cube=inj.t_in_cube,
            l_pix=inj.l_pix,
            m_pix=inj.m_pix,
            amplitude=inj.inject_amplitude,
        )
        for c in inj.inject_cubes
    )
    src = SyntheticRxRingSource(
        n_cubes=n_cubes,
        t_det=t_det,
        n_fdm=n_fdm,
        n_grid=n_grid,
        coarse_dm_pc_cm3=coarse_dm,
        fine_dm_pc_cm3=fine_dm,
        fine_to_coarse=fine_to_coarse,
        rng=rng,
        injections=injections,
    )

    config = _build_search_compute_config(args, out_dir=out_dir)
    service = SearchComputeService(config=config, source=src)

    # Optional backpressure patch — applied AFTER the search-compute
    # config is built (which captures the writer config) but BEFORE
    # service.start() (which spawns the writer thread).
    np_mod, real_savez = _install_backpressure_patch(
        float(args.inject_backpressure)
    )

    udp_sock: Optional[socket.socket] = None
    udp_port: int = 0
    udp_set = set(inj.udp_cubes)
    max_observed_depth = 0
    n_udp_arm_failures = 0
    # Cumulative time spent inside ``cube_dump.submit`` across all
    # cubes. This is the true chunk-7 "dispatch hot-path latency" — the
    # time the real-time path is held up by writer-queue management.
    # Each submit is non-blocking (queue.put_nowait), so this is bounded
    # by the per-call overhead regardless of how slow np.savez runs in
    # the writer thread. Exposed as ``submit_dispatch_total_ms`` /
    # ``submit_dispatch_p99_us`` in the report so the chunk-7
    # backpressure gate can assert on the writer-impact-only latency
    # rather than the pipeline-bound wall time.
    submit_call_times_ns: List[int] = []

    await service.start()
    # Wrap ``cube_dump.submit`` with a per-call timer so we can isolate
    # the writer-impacted dispatch hot path from the pipeline-bound
    # outer loop. The wrapper is installed AFTER service.start so the
    # service's internal reference (built from cube_dump_writer_config)
    # is the one we're decorating; subsequent ``service._cube_dump``
    # accesses still resolve to the same instance.
    if service.cube_dump is not None:
        _wrapped_writer = service.cube_dump
        _orig_submit = _wrapped_writer.submit

        def _timed_submit(*args_inner: Any, **kwargs_inner: Any) -> bool:
            t_call_start = time.perf_counter_ns()
            try:
                return _orig_submit(*args_inner, **kwargs_inner)
            finally:
                submit_call_times_ns.append(
                    time.perf_counter_ns() - t_call_start
                )

        _wrapped_writer.submit = _timed_submit  # type: ignore[assignment]
    if bool(args.enable_udp) and service.udp_listener is not None:
        udp_port = service.udp_listener.bound_port
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _LOG.info(
            "udp listener bound on 127.0.0.1:%d (udp_cubes=%s)",
            udp_port,
            list(inj.udp_cubes),
        )
    _LOG.info(
        "bench start: n_cubes=%d t_det=%d n_fdm=%d n_grid=%d "
        "queue_maxsize=%d auto_min_snr=%.3f inject_every=%d "
        "inject_backpressure_ms=%.1f rng_seed=%d",
        n_cubes, t_det, n_fdm, n_grid,
        int(args.queue_maxsize), float(args.auto_min_snr),
        int(args.inject_every), float(args.inject_backpressure),
        int(args.rng_seed),
    )

    dispatch_start_ns = time.perf_counter_ns()
    try:
        async for slot in src:
            cube_id = int(slot.cube_id)
            if udp_sock is not None and cube_id in udp_set:
                listener = service.udp_listener
                assert listener is not None  # narrowed by enable_udp
                prev_n = listener.n_datagrams_received
                udp_sock.sendto(b"dump", ("127.0.0.1", udp_port))
                delivered = await _wait_for_udp_delivery(
                    service,
                    prev_n=prev_n,
                    timeout_s=DEFAULT_UDP_DELIVERY_TIMEOUT_S,
                )
                if not delivered:
                    n_udp_arm_failures += 1
                    _LOG.warning(
                        "udp datagram for cube_id=%d did not deliver "
                        "within %.3fs (prev_n=%d, current=%d)",
                        cube_id, DEFAULT_UDP_DELIVERY_TIMEOUT_S,
                        prev_n, listener.n_datagrams_received,
                    )
            await service._process_one_cube(slot)
            await src.release(slot.cube_id)
            # Sample queue depth post-process; the writer may have
            # already drained the just-submitted item, so this is a
            # conservative lower bound on instantaneous peak depth.
            cube_dump = service.cube_dump
            if cube_dump is not None:
                depth = cube_dump.queue_depth
                if depth > max_observed_depth:
                    max_observed_depth = depth
            if (cube_id + 1) % max(1, n_cubes // 10) == 0:
                _LOG.info(
                    "cube=%d/%d cands=%d clusters=%d auto=%d udp=%d "
                    "writer_dumped=%d writer_dropped=%d depth=%d",
                    cube_id + 1, n_cubes,
                    service.candidates_emitted,
                    service.clusters_emitted,
                    service.auto_dumps_dispatched,
                    service.udp_dumps_dispatched,
                    cube_dump.n_dumped if cube_dump else 0,
                    cube_dump.n_dropped if cube_dump else 0,
                    cube_dump.queue_depth if cube_dump else 0,
                )
    finally:
        dispatch_wall_s = (
            time.perf_counter_ns() - dispatch_start_ns
        ) / 1.0e9
        if udp_sock is not None:
            udp_sock.close()
        # service.stop() drains the writer queue (with the slow-savez
        # patch still active, this can block — that is expected and
        # is the reason the chunk-7 wall_s metric measures dispatch,
        # not stop()).
        try:
            await service.stop()
        finally:
            _restore_savez(np_mod, real_savez)

    cube_dump = service.cube_dump
    n_dumps_written = cube_dump.n_dumped if cube_dump else 0
    n_dumps_dropped = cube_dump.n_dropped if cube_dump else 0
    n_dumps_failed = cube_dump.n_failed if cube_dump else 0
    write_ms_arr: Tuple[float, ...] = (
        cube_dump.recent_write_ms_ms if cube_dump else ()
    )

    submit_total_ms = (
        sum(submit_call_times_ns) / 1.0e6
        if submit_call_times_ns else 0.0
    )
    submit_p99_us = (
        float(np.percentile(
            np.asarray(submit_call_times_ns, dtype=np.float64) / 1.0e3,
            99.0,
        )) if submit_call_times_ns else 0.0
    )
    summary: Dict[str, Any] = {
        "n_cubes_processed": int(service.cubes_processed),
        "n_candidates_emitted": int(service.candidates_emitted),
        "n_clusters_emitted": int(service.clusters_emitted),
        "n_auto_dumps_dispatched": int(service.auto_dumps_dispatched),
        "n_udp_dumps_dispatched": int(service.udp_dumps_dispatched),
        "n_dumps_written": int(n_dumps_written),
        "n_dumps_dropped": int(n_dumps_dropped),
        "n_dumps_failed": int(n_dumps_failed),
        "writer_p50_ms": _percentile_ms(write_ms_arr, 50.0),
        "writer_p99_ms": _percentile_ms(write_ms_arr, 99.0),
        "writer_max_ms": (
            float(max(write_ms_arr)) if write_ms_arr else 0.0
        ),
        # ``wall_s`` is the dispatch loop wall (cube-by-cube
        # processing including pipeline.process + clustering); it does
        # NOT include service.stop()'s writer-drain wait. The chunk-7
        # spec example value (3.7 s for 100 cubes) reflects this.
        "wall_s": float(dispatch_wall_s),
        # ``submit_dispatch_total_ms`` is the cumulative time spent
        # inside CubeDumpWriter.submit across all cubes. Each submit
        # is non-blocking (queue.put_nowait), so this stays sub-ms
        # even under the chunk-7 backpressure gate (slow np.savez
        # in the writer thread cannot leak to the dispatch path).
        "submit_dispatch_total_ms": float(submit_total_ms),
        "submit_dispatch_p99_us": float(submit_p99_us),
        "submit_dispatch_n_calls": int(len(submit_call_times_ns)),
    }
    queue_backpressure: Dict[str, Any] = {
        "queue_maxsize": int(args.queue_maxsize),
        "max_observed_depth": int(max_observed_depth),
        "dropped_at_full": bool(n_dumps_dropped > 0),
    }

    return {
        "summary": summary,
        "queue_backpressure": queue_backpressure,
        "n_udp_arm_failures": int(n_udp_arm_failures),
    }


async def _bench_main(args: argparse.Namespace) -> int:
    """Bench entrypoint. Writes ``report.json`` and returns the exit code.

    Exit code semantics:
      * 0 — bench completed end-to-end. The bench does NOT enforce
        zero-drop / zero-fail invariants here; those are gated by
        ``tests/test_cube_dump_e2e_bench.py`` (the per-CLI test
        scenarios target distinct invariants — backpressure-test
        wants drops > 0, sustained-throughput wants drops == 0).
      * 1 — UDP delivery failed (datagram did not arrive within
        the loop's poll window). Surfaced separately because the
        chunk-7 spec gates UDP dump count against the requested
        cube list, and a delivery failure is an environmental
        problem (kernel UDP backlog, host-firewall) rather than
        a code defect.
    """
    out_dir = Path(args.report_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cube_dump").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    _setup_logging(out_dir)

    inj = _resolve_injections(
        n_cubes=int(args.n_cubes),
        inject_every=int(args.inject_every),
        inject_amplitude=float(args.inject_amplitude),
        t_det=int(args.t_det),
        n_grid=int(args.n_grid),
        enable_udp=bool(args.enable_udp),
        udp_cubes_raw=args.udp_cubes,
    )

    bench_start_ns = time.perf_counter_ns()
    result = await _drive_bench(args, out_dir=out_dir, inj=inj)
    total_wall_s = (time.perf_counter_ns() - bench_start_ns) / 1.0e9

    cli_config = {
        "report_dir": str(out_dir),
        "n_cubes": int(args.n_cubes),
        "t_det": int(args.t_det),
        "n_fdm": int(args.n_fdm),
        "n_grid": int(args.n_grid),
        "queue_maxsize": int(args.queue_maxsize),
        "auto_min_snr": float(args.auto_min_snr),
        "detector_threshold_sigma": float(args.detector_threshold_sigma),
        "enable_udp": bool(args.enable_udp),
        "udp_cubes": list(inj.udp_cubes),
        "inject_every": int(args.inject_every),
        "inject_amplitude": float(args.inject_amplitude),
        "inject_backpressure_ms": float(args.inject_backpressure),
        "rng_seed": int(args.rng_seed),
        "voltage_run_id": (
            str(args.voltage_run_id) if args.voltage_run_id else None
        ),
    }

    report = {
        "schema_version": 1,
        "bench": "cube_dump_e2e",
        "milestone": "M6",
        "chunk": 7,
        "git_sha": _git_sha(),
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "utc": datetime.now(timezone.utc).isoformat(),
        "config": cli_config,
        "summary": result["summary"],
        "queue_backpressure": result["queue_backpressure"],
        "total_wall_s": float(total_wall_s),
        "n_udp_arm_failures": result["n_udp_arm_failures"],
    }

    report_path = out_dir / "report.json"
    with report_path.open("w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    _LOG.info("wrote %s", report_path)
    s = result["summary"]
    qb = result["queue_backpressure"]
    _LOG.info(
        "bench done: n_cubes=%d cands=%d clusters=%d "
        "auto_dumps=%d udp_dumps=%d written=%d dropped=%d failed=%d "
        "writer_p50=%.2fms p99=%.2fms max=%.2fms "
        "queue_max_observed=%d wall_s=%.3f",
        s["n_cubes_processed"], s["n_candidates_emitted"],
        s["n_clusters_emitted"], s["n_auto_dumps_dispatched"],
        s["n_udp_dumps_dispatched"], s["n_dumps_written"],
        s["n_dumps_dropped"], s["n_dumps_failed"],
        s["writer_p50_ms"], s["writer_p99_ms"], s["writer_max_ms"],
        qb["max_observed_depth"], s["wall_s"],
    )
    if int(result["n_udp_arm_failures"]) > 0:
        return 1
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the chunk-7 cube-dump-e2e argument parser."""
    parser = argparse.ArgumentParser(
        prog="bench.cube_dump_e2e",
        description=(
            "M6 chunk 7 cube-dump end-to-end bench (auto-trigger + "
            "UDP-trigger paths + queue backpressure)."
        ),
    )
    parser.add_argument(
        "--report-dir", type=str, default=str(DEFAULT_REPORT_DIR),
        help="Directory under which report.json + cube_dump/ + logs/ "
             "land. Created if missing.",
    )
    parser.add_argument(
        "--n-cubes", type=int, default=DEFAULT_N_CUBES,
        help="Number of synthetic cubes to drive through the service. "
             "DoD smoke default 50; the chunk-7 spec example runs with "
             "100 to expose ~10 auto dumps + 4 UDP dumps.",
    )
    parser.add_argument("--t-det", type=int, default=DEFAULT_T_DET)
    parser.add_argument("--n-fdm", type=int, default=DEFAULT_N_FDM)
    parser.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID)
    parser.add_argument(
        "--queue-maxsize", type=int, default=DEFAULT_QUEUE_MAXSIZE,
        help="CubeDumpWriter bounded-queue depth. Set to 1 with "
             "--inject-backpressure to force drops in the chunk-7 "
             "backpressure gate.",
    )
    parser.add_argument(
        "--auto-min-snr", type=float, default=DEFAULT_AUTO_MIN_SNR,
        help="BrightPulsePredicate min SNR floor. Default 0.1 fires on "
             "every emitted cluster (the predicate's other gates — DM "
             "band, width, cntc — are left at their permissive defaults "
             "for the chunk-7 wiring gate; chunk 3's predicate-only "
             "tests cover the threshold-by-threshold filtering).",
    )
    parser.add_argument(
        "--detector-threshold-sigma", type=float,
        default=DEFAULT_DETECTOR_THRESHOLD_SIGMA,
        help="Detector candidate-emission threshold in sigma. Default "
             "matches chunk-5 production (8σ).",
    )
    parser.add_argument(
        "--enable-udp", action="store_true",
        help="Bind the UDP trigger listener (chunk 4) on 127.0.0.1 + "
             "send a datagram before each cube listed in --udp-cubes. "
             "Disabled by default for tests that only exercise the "
             "auto-trigger path.",
    )
    parser.add_argument(
        "--udp-cubes", type=str, default="",
        help="Comma-separated cube indices at which the bench fires a "
             "UDP datagram (1 datagram per listed cube, sent just "
             "before the cube enters _process_one_cube). Ignored when "
             "--enable-udp is not set.",
    )
    parser.add_argument(
        "--inject-every", type=int, default=DEFAULT_INJECT_EVERY,
        help="Auto-trigger injection cadence. A SyntheticInjection "
             "(amplitude=--inject-amplitude) is placed at every K-th "
             "cube starting at cube 0 — 'bursty' workload per the "
             "chunk-7 spec. Default 10 (~1 dump every 10 cubes).",
    )
    parser.add_argument(
        "--inject-amplitude", type=float, default=DEFAULT_INJECT_AMPLITUDE,
        help="Synthetic injection amplitude (post-imager units). The "
             "chunk-5 service-test default is 200; the bench reuses "
             "that to keep cluster-record SNR well above the 8σ "
             "detector threshold.",
    )
    parser.add_argument(
        "--inject-backpressure", type=float,
        default=DEFAULT_INJECT_BACKPRESSURE_MS,
        help="If > 0, monkey-patches np.savez (in dsart.dump.cube_dump) "
             "to sleep this many ms before each write. Used by the "
             "chunk-7 backpressure gate to artificially stall the "
             "writer thread so queue-full drops are observable. "
             "Default 0 (no patch).",
    )
    parser.add_argument(
        "--rng-seed", type=int, default=DEFAULT_RNG_SEED,
        help="Seed for the synthetic RX-ring noise generator.",
    )
    parser.add_argument(
        "--voltage-run-id", type=str, default=None,
        help="Metadata-only tag (e.g. '250924mptq') stamped into "
             "report.json.config.voltage_run_id. Not used to drive "
             "the synthetic source; chunk 7 stays on synthetic cubes "
             "to keep the bench fast + deterministic. Accepting this "
             "flag keeps the M6 DoD path's existing invocation "
             "(`--voltage-run-id 250924mptq`) smoke-clean.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_bench_main(args))


if __name__ == "__main__":
    sys.exit(main())
