"""Production-shaped real-time perf bench: dada_db + dada_junkdb + corr_fast.

Mirrors the production fast-corr deployment exactly:

  1. Brings up a dedicated PSRDADA fada-shaped ring (default key ``f3a3``
     to avoid colliding with anything stamped on by the M3 prefix
     mapping that aliases the legacy ``fada`` to ``fa3a``).
  2. Spawns ``dada_junkdb`` to fill the ring with random bytes at a
     configurable rate (default = native cadence = ~2253 MB/s,
     1 fada block / 134.218 ms).
  3. Attaches an in-process reader, replaces the synthetic
     ``_synth_voltage_block`` call in ``profile_fast_path_K1.py`` with
     the actual ``np.asarray(reader.getNextPage())`` view of the
     PSRDADA shm page, and runs the same phase-instrumented profiling
     loop.
  4. Tears down dada_db + dada_junkdb on exit.

Why this matters
================
The previous synthetic bench (``profile_fast_path_K1.py`` standalone)
allocated a fresh ~288 MB numpy buffer per block. Two artefacts of that
shape were masking real production behaviour:

  * The ``_synth_voltage_block`` call took ~300 ms of host CPU time
    *between* pipelined pushes, which acted as natural rate-limiting
    and made the apparent pipelined wall faster than steady-state.
  * Per-iter buffer churn caused the host-pin auto-registration to
    accumulate stale CUDA registrations as numpy GC'd the previous
    block's buffer (production PSRDADA pages live for the entire
    service lifetime so this never happens there).

Reading from a real dada_db ring fed by ``dada_junkdb`` removes both:
the data is paced exactly as the on-sky fada writer would pace it, and
the small set (default 4) of stable PSRDADA buffer pages is
auto-registered exactly once each.

CLI::

    python bench/profile_realtime_psrdada.py \\
        --report-dir /tmp/rt-prof \\
        --device cuda \\
        --t-int-fast-native 8 \\
        --n-coarse-dm 24 \\
        --warmup 4 \\
        --n-blocks 30 \\
        --rate-mb-s 2253 \\
        --pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.common.constants import (  # noqa: E402
    BLOCK_DURATION_S,
    FADA_BYTES_PER_BLOCK,
    NATIVE_SAMPLE_US,
)
from dsart.services.corr_fast_integration import (  # noqa: E402
    BlockPipeliner,
    FastIntegrationConfig,
    build_context,
    process_block,
)
from dsart.services.slow_corr_kernel import (  # noqa: E402
    SlowCorrKernel,
    pack_bada_block,
    unpack_int4_split,
)

# Re-use the phase-counter + wrapper machinery from the synthetic bench.
from bench.profile_fast_path_K1 import (  # noqa: E402
    _PhaseCounters,
    _build_synthetic_summed_plan,
    _install_phase_wrappers,
    _synth_antpos,
)

LOG = logging.getLogger("profile_realtime_psrdada")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_FADA_KEY = "f3a3"          # M3 prefix-safe; never collides with on-sky fada
DEFAULT_FADA_NUM_BLOCKS = 4        # bench-realistic ring depth
DADA_HDR_SIZE = 4096

# Block period in ms (134.218 at native cadence).
BLOCK_MS_NATIVE = BLOCK_DURATION_S * 1000.0

# Native byte rate that fully feeds the ring at 1 block / 134.218 ms.
NATIVE_RATE_MB_S = FADA_BYTES_PER_BLOCK / (BLOCK_DURATION_S * 1e6)


# ---------------------------------------------------------------------------
# dada_db lifecycle (lock+page so the junkdb writes hit RAM and the
# reader sees the same physical pages every cycle).
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, label: str) -> tuple[int, str]:
    LOG.debug("%s: %s", label, " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        LOG.warning(
            "%s exited rc=%d stderr=%s", label, proc.returncode,
            proc.stderr.strip(),
        )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _dada_db_destroy(key: str, *, ignore_missing: bool = True) -> None:
    rc, out = _run(["dada_db", "-d", "-k", key], label=f"dada_db destroy {key}")
    if rc != 0 and not ignore_missing:
        raise RuntimeError(f"dada_db destroy {key} failed: {out}")


def _dada_db_create(
    key: str, *, num_blocks: int, num_readers: int = 1,
) -> None:
    cmd = [
        "dada_db", "-k", key,
        "-b", str(FADA_BYTES_PER_BLOCK),
        "-n", str(num_blocks),
        "-a", str(DADA_HDR_SIZE),
        "-r", str(num_readers),
        "-l",                          # lock in RAM (mlock)
        "-p",                          # page all blocks into RAM
    ]
    rc, out = _run(cmd, label=f"dada_db create {key}")
    if rc != 0:
        raise RuntimeError(f"dada_db create {key} failed: {out}")


@contextmanager
def dada_ring(
    key: str, *, num_blocks: int, num_readers: int = 1,
) -> Iterator[None]:
    """Create + tear down a fada-shaped ring under ``key``."""
    _dada_db_destroy(key)              # nuke stale rings
    _dada_db_create(key, num_blocks=num_blocks, num_readers=num_readers)
    try:
        yield
    finally:
        _dada_db_destroy(key)


# ---------------------------------------------------------------------------
# dada_junkdb writer
# ---------------------------------------------------------------------------


_MIN_HEADER = """\
HDR_VERSION 1.0
HDR_SIZE    4096
INSTRUMENT  DSA-RT
TELESCOPE   DSA-110
SOURCE      JUNKDB_SYNTH
NBIT        4
NDIM        2
NPOL        2
NCHAN       384
NANT        96
TSAMP       3.2768e-05
DSART_PRODUCER profile_realtime_psrdada
"""


@contextmanager
def junkdb_writer(
    *,
    key: str,
    rate_mb_s: float,
    duration_s: float,
    log_path: Path | None = None,
) -> Iterator[subprocess.Popen]:
    """Spawn dada_junkdb in the background; terminate on exit.

    Picks ``-r rate_mb_s`` and ``-t duration_s``; junkdb fills the ring
    with random bytes at the requested rate. We pipe a minimal valid
    fada header through a temp file because junkdb requires one as
    a positional argument (it gets stamped on every block-page header
    region).
    """
    with tempfile.NamedTemporaryFile(
        "w", suffix=".header", delete=False,
    ) as hdr_f:
        hdr_f.write(_MIN_HEADER)
        hdr_path = Path(hdr_f.name)

    cmd = [
        "dada_junkdb",
        "-k", key,
        "-r", f"{rate_mb_s:.3f}",
        "-t", f"{int(math.ceil(duration_s))}",
        "-z",                           # zero-copy direct shm
        str(hdr_path),
    ]
    LOG.info(
        "starting dada_junkdb: %s (native_rate=%.0f MB/s, duration=%.1fs)",
        " ".join(cmd), NATIVE_RATE_MB_S, duration_s,
    )
    out_fd = open(log_path, "w") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(
        cmd,
        stdout=out_fd,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,           # own pgrp so we can clean-kill
    )
    try:
        yield proc
    finally:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=2)
        if log_path and out_fd not in (subprocess.DEVNULL, None):
            out_fd.close()
        try:
            hdr_path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# In-process reader loop
# ---------------------------------------------------------------------------


@dataclass
class _RuntimeStats:
    n_blocks: int = 0
    wall_per_block_ms: list[float] | None = None
    wait_per_block_ms: list[float] | None = None
    e2e_per_block_ms: list[float] | None = None
    block_period_ms: float = BLOCK_MS_NATIVE


def _key_to_int(key_str: str) -> int:
    if len(key_str) != 4:
        raise ValueError(f"buffer key must be 4 chars, got {key_str!r}")
    return int(f"0x{key_str}", 16)


def _slow_process_one(raw_arr: np.ndarray, *, kernel: SlowCorrKernel,
                      device: torch.device) -> None:
    """One slow-corr block: unpack → compute_split → pack_bada (mirrors the service)."""
    real_v, imag_v = unpack_int4_split(
        raw_arr, device=device, out_dtype=torch.float16,
    )
    vis = kernel.compute_split(real_v, imag_v)
    pack_bada_block(vis)


def run_loop(
    args: argparse.Namespace, *, ctx, counters: _PhaseCounters,
) -> _RuntimeStats:
    """Connect to the fada ring and run ``--n-blocks`` blocks through process.

    The first ``--warmup`` blocks are processed to warm caches + the F34
    sliding-window prev-buffer + auto-pin all PSRDADA pages (which may
    take more than one full ring revolution if num_blocks > warmup).
    """
    from psrdada import Reader

    fada_int = _key_to_int(args.fada_key)
    LOG.info("connecting reader to fada=0x%04x", fada_int)
    reader = Reader(fada_int)

    try:
        hdr = reader.getHeader()
        LOG.info(
            "fada header: %d keys (DSART_PRODUCER=%s, SOURCE=%s)",
            len(hdr), hdr.get("DSART_PRODUCER", "?"),
            hdr.get("SOURCE", "?"),
        )

        block_n = 0
        # Warmup
        for _ in range(args.warmup):
            block_n += 1
            page = reader.getNextPage()
            page_arr = np.asarray(page)
            if page_arr.nbytes != FADA_BYTES_PER_BLOCK:
                LOG.error(
                    "warmup block %d: wrong size %d (expected %d); skipping",
                    block_n, page_arr.nbytes, FADA_BYTES_PER_BLOCK,
                )
                reader.markCleared()
                continue
            if args.mode == "slow":
                _slow_process_one(page_arr, kernel=ctx, device=ctx.device)
            elif args.pipeline:
                ctx_obj.push_or_process(page_arr, block_n=block_n)
            else:
                process_block(page_arr, ctx=ctx, block_n=block_n)
            reader.markCleared()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        LOG.info("warmup done; profiling %d blocks", args.n_blocks)

        # Profile
        wall_ms: list[float] = []
        wait_ms: list[float] = []           # time spent in reader.getNextPage()
        e2e_ms: list[float] = []            # block-to-block wall (full loop iter)
        loop_t = time.perf_counter()
        for _ in range(args.n_blocks):
            block_n += 1
            t_wait0 = time.perf_counter()
            page = reader.getNextPage()
            page_arr = np.asarray(page)
            wait_ms.append((time.perf_counter() - t_wait0) * 1000.0)
            if page_arr.nbytes != FADA_BYTES_PER_BLOCK:
                LOG.error(
                    "profile block %d: wrong size %d", block_n, page_arr.nbytes,
                )
                reader.markCleared()
                continue
            counters.reset_block()
            if args.mode == "slow":
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _slow_process_one(page_arr, kernel=ctx, device=ctx.device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
            elif args.pipeline:
                t0 = time.perf_counter()
                ctx_obj.push_or_process(page_arr, block_n=block_n)
                t1 = time.perf_counter()
            else:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                process_block(page_arr, ctx=ctx, block_n=block_n)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
            wall_ms.append((t1 - t0) * 1000.0)
            counters.commit_block()
            reader.markCleared()
            now = time.perf_counter()
            e2e_ms.append((now - loop_t) * 1000.0)
            loop_t = now

        if args.pipeline:
            ctx_obj.flush()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        try:
            reader.disconnect()
        except Exception:
            LOG.exception("reader.disconnect failed (non-fatal)")

    return _RuntimeStats(
        n_blocks=args.n_blocks,
        wall_per_block_ms=wall_ms,
        wait_per_block_ms=wait_ms,
        e2e_per_block_ms=e2e_ms,
    )


# Tiny adapter so the same loop body works for both sequential and
# pipelined modes without an inner branch each iteration.
class _PipelineAdapter:
    def __init__(self, ctx, n_buffers: int = 2):
        self.pipe = BlockPipeliner(ctx, n_buffers=n_buffers)

    def push_or_process(self, raw, *, block_n):
        return self.pipe.push(raw, block_n=block_n)

    def flush(self):
        return self.pipe.flush()


class _Pipeline3SAdapter:
    def __init__(self, ctx, n_buffers: int = 3):
        from dsart.services.corr_fast_integration import BlockPipeliner3S
        self.pipe = BlockPipeliner3S(ctx, n_buffers=n_buffers)

    def push_or_process(self, raw, *, block_n):
        return self.pipe.push(raw, block_n=block_n)

    def flush(self):
        return self.pipe.flush()


class _SeqAdapter:
    def __init__(self, ctx):
        self.ctx = ctx

    def push_or_process(self, raw, *, block_n):
        return process_block(raw, ctx=self.ctx, block_n=block_n)

    def flush(self):
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # PSRDADA / junkdb plumbing
    p.add_argument("--fada-key", default=DEFAULT_FADA_KEY,
                   help=f"4-char shm key (default {DEFAULT_FADA_KEY})")
    p.add_argument("--num-fada-blocks", type=int, default=DEFAULT_FADA_NUM_BLOCKS,
                   help="ring depth; production uses 70")
    p.add_argument("--rate-mb-s", type=float, default=NATIVE_RATE_MB_S,
                   help=(f"junkdb -r rate (default native ~{NATIVE_RATE_MB_S:.0f} "
                         "MB/s = 1 block / 134.218 ms)"))
    p.add_argument("--junkdb-extra-s", type=float, default=15.0,
                   help="extra duration over (warmup+nblocks)*0.134s "
                        "to leave junkdb running while we drain")
    # Bench / pipeline shape (mirrors profile_fast_path_K1.py)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--n-blocks", type=int, default=20)
    p.add_argument("--t-int-fast-native", type=int, default=8)
    p.add_argument("--n-grid", type=int, default=256)
    p.add_argument("--n-coarse-dm", type=int, default=24)
    p.add_argument("--chan-sum-factor", type=int, default=8)
    p.add_argument("--dm-chunk-size", type=int, default=2)
    p.add_argument("--dm-truth", type=float, default=1500.0)
    p.add_argument("--pipeline", action="store_true",
                   help="use BlockPipeliner (2 streams) — fast mode only")
    p.add_argument("--pipeline-3s", action="store_true",
                   help="use BlockPipeliner3S (3 streams: unpack || compute || dedisp) — fast mode only")
    p.add_argument("--mode", default="fast", choices=("fast", "slow"),
                   help="which correlator pipeline to drive")
    p.add_argument("--keep-ring", action="store_true",
                   help="don't tear the ring down (debug)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    LOG.info("device=%s warmup=%d n_blocks=%d", device, args.warmup, args.n_blocks)

    if args.mode == "fast":
        plan = _build_synthetic_summed_plan(
            n_coarse=args.n_coarse_dm,
            dm_max=2.0 * args.dm_truth,
            chan_sum_factor=args.chan_sum_factor,
            t_int_fast_us=float(args.t_int_fast_native * NATIVE_SAMPLE_US),
        )
        antpos_e, antpos_n = _synth_antpos(seed=42)
        cfg = FastIntegrationConfig(
            chgroup=0,
            obs_dec_rad=math.radians(53.85),
            n_grid=args.n_grid,
            kernel_support=1,
            cell_lambda_mode="common",
            chan_sum_factor=args.chan_sum_factor,
            sliding_window=True,
            n_fv_chunk=None,
            t_int_fast_native=args.t_int_fast_native,
            rfi_enabled=False,
            static_sky_disabled=True,
            dm_chunk_size=args.dm_chunk_size,
        )
        ctx = build_context(
            cfg=cfg, device=device,
            antpos_e=antpos_e, antpos_n=antpos_n,
            dm_plan=plan,
        )
        LOG.info(
            "fast ready: n_filled=%d n_dm=%d n_fv_per_block=%d t_int=%.3f µs",
            ctx.gridder.pattern.n_filled,
            ctx.multi_dm_coarse_dm.n_dm,
            ctx.kernel.n_fast_vis_per_full_block,
            plan.t_int_fast_us,
        )
    else:
        ctx = SlowCorrKernel(device=device)
        LOG.info("slow ready: SlowCorrKernel on %s", device)

    counters = _PhaseCounters()
    if args.mode == "fast" and not args.pipeline and not args.pipeline_3s:
        unwind = _install_phase_wrappers(counters)
    else:
        unwind = lambda: None

    # Adapter chosen based on mode (both expose push_or_process / flush).
    global ctx_obj
    if args.mode == "fast":
        if args.pipeline_3s:
            ctx_obj = _Pipeline3SAdapter(ctx)
        elif args.pipeline:
            ctx_obj = _PipelineAdapter(ctx)
        else:
            ctx_obj = _SeqAdapter(ctx)
    else:
        ctx_obj = _SeqAdapter(ctx)  # not used in slow path but defined for safety

    # junkdb runs long enough to feed (warmup + n_blocks) plus a margin
    # for the GPU work to drain.
    expected_consume_s = (
        (args.warmup + args.n_blocks) * BLOCK_DURATION_S
        + args.junkdb_extra_s
    )

    rc = 0
    junkdb_log = args.report_dir / "junkdb.log"
    stats: _RuntimeStats | None = None
    try:
        with dada_ring(
            args.fada_key,
            num_blocks=args.num_fada_blocks,
            num_readers=1,
        ):
            # Brief delay so the ring is fully created before junkdb attaches.
            time.sleep(0.5)
            with junkdb_writer(
                key=args.fada_key,
                rate_mb_s=args.rate_mb_s,
                duration_s=expected_consume_s,
                log_path=junkdb_log,
            ) as junk_proc:
                # Brief delay so junkdb stamps the header + writes ≥1 block
                # before we attach as a reader (prevents a header timeout).
                time.sleep(2.0)
                if junk_proc.poll() is not None:
                    LOG.error(
                        "dada_junkdb died early rc=%d; check %s",
                        junk_proc.returncode, junkdb_log,
                    )
                    return 2
                stats = run_loop(args, ctx=ctx, counters=counters)
    finally:
        unwind()

    # Reporting
    if stats is None or not stats.wall_per_block_ms:
        LOG.error("no blocks profiled")
        return 1

    LOG.info("mode=%s pipeline=%s", args.mode, args.pipeline)

    block_period_ms = stats.block_period_ms
    counter_summary = counters.stats()

    print()
    print(f"{'phase':<45s} {'mean (ms)':>11s} {'p50 (ms)':>11s} {'p99 (ms)':>11s}  "
          f"{'% of p50 wall':>14s}")
    print("-" * 96)

    p50_wall = float(np.median(stats.wall_per_block_ms))
    pct_sum = 0.0
    json_out = {
        "block_period_ms": block_period_ms,
        "wall_per_block_ms": stats.wall_per_block_ms,
        "phases": counter_summary,
    }
    for name, vals in counter_summary.items():
        mean = vals["mean_ms"]; p50 = vals["p50_ms"]; p99 = vals["p99_ms"]
        pct = 100.0 * p50 / p50_wall
        pct_sum += pct
        print(f"{name:<45s} {mean:11.2f} {p50:11.2f} {p99:11.2f}  {pct:13.1f}%")
    print("-" * 96)
    print(f"{'sum of phase p50s':<45s} {'':>11s} "
          f"{sum(v['p50_ms'] for v in counter_summary.values()):11.2f} "
          f"{'':>11s}  {pct_sum:13.1f}%")
    rt_factor = p50_wall / block_period_ms
    print(f"realtime block period = {block_period_ms:.2f} ms; "
          f"wall p50 / block_period = {rt_factor:.2f}x")
    print()
    wall_arr = np.array(stats.wall_per_block_ms)
    print(f"wall per block (push/proc inner): "
          f"mean={wall_arr.mean():.2f} p50={p50_wall:.2f} "
          f"p99={np.percentile(wall_arr, 99):.2f} ms")
    if stats.wait_per_block_ms:
        wait_arr = np.array(stats.wait_per_block_ms)
        print(f"reader.getNextPage wait:    "
              f"mean={wait_arr.mean():.2f} p50={np.median(wait_arr):.2f} "
              f"p99={np.percentile(wait_arr, 99):.2f} ms  "
              f"(0 ⇒ ring is full / consumer-bound)")
    if stats.e2e_per_block_ms:
        e2e_arr = np.array(stats.e2e_per_block_ms[1:])  # drop first (loop init)
        if e2e_arr.size:
            p50_e2e = float(np.median(e2e_arr))
            rt_e2e = p50_e2e / block_period_ms
            print(f"end-to-end per block:       "
                  f"mean={e2e_arr.mean():.2f} p50={p50_e2e:.2f} "
                  f"p99={np.percentile(e2e_arr, 99):.2f} ms  "
                  f"({rt_e2e:.2f}x RT)")
    if rt_factor < 1.0:
        print(f"  → REAL-TIME OK (margin {(1-rt_factor)*100:.1f}%)")
    else:
        print(f"  → {rt_factor:.2f}x SLOWER THAN REAL-TIME")
    json_out["wait_per_block_ms"] = stats.wait_per_block_ms
    json_out["e2e_per_block_ms"] = stats.e2e_per_block_ms

    json_path = args.report_dir / "phase_breakdown.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    LOG.info("wrote %s", json_path)

    return rc


if __name__ == "__main__":
    sys.exit(main())
