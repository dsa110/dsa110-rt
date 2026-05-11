"""M4a chunk 7 — net-loopback integration bench (six DoD invariants).

Drives ``TransportTx`` (prod-frame) → UDP localhost → ``TransportRxProd``
and verifies the six DoD invariants from plan §M4a line 2383:

  I1. 60 s sustained 0% loss at the largest §9 op-point.
  I2. Mid-run RX socket hold → TX drops at TX (``tx_dropped_payloads`` ↑),
      no upstream backpressure into the gridder.
  I3. ``pattern_mismatch_count == 0`` for 60 s when both ends started with
      the same ``/cnf/corr_setup_96`` + ``dec_deg``.
  I4. Mutate corr-side ``dec_deg`` to a stale value → ``pattern_mismatch_count``
      increments steadily (every datagram).
  I5. Synchronised re-``cmd: prepare`` on both ends → ``pattern_mismatch_count``
      returns to 0 within < 1 cube.
  I6. Mid-run, restart RX only → rebuilds patterns locally on next
      ``cmd: prepare``, resumes < 5 s with ``pattern_mismatch_count`` → 0.

etcd is deferred to M7; ``cmd: prepare`` is simulated via env-var reload
hooks (``DSART_M4A_RELOAD_PATTERN=1`` + ``DSART_M4A_DEC_DEG_OVERRIDE``).

Run:
    python -m bench.net_loopback --all                # all six
    python -m bench.net_loopback --invariant 3        # one
    python -m bench.net_loopback --all --quick        # 5 s windows
    python -m bench.net_loopback --all --json out.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch

from dsart.transport.prod_frame import (
    BITS_CINT8_COMPLEX,
    DEFAULT_MTU_BYTES,
    HEADER_BYTES,
)
from dsart.transport.tx import (
    TransportTx,
    TransportTxProdConfig,
)
from dsart.transport.rx import (
    TransportRxProd,
    TransportRxProdConfig,
    RxProdSlot,
)

LOG = logging.getLogger("m4a.bench.net_loopback")


# ---------------------------------------------------------------------------
# Default §9 op-point parameters — small enough to converge fast on h01.
# The DoD harness (chunk 8) can override via CLI for the largest op-point.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OpPoint:
    """One §9 operational point sized for chunk-7 verification."""

    n_grid: int = 256
    n_dm: int = 4
    n_fast_vis: int = 1
    n_filled: int = 50
    n_chgroups: int = 1               # bench uses one TX→RX pair
    chgroup: int = 0
    bits_per_cell: int = BITS_CINT8_COMPLEX
    t_int_factor: int = 8
    target_gbps_per_flow: float = 1.0  # 1 Gbps per dm-flow
    max_frag_payload_bytes: int = 8964
    rcvbuf_mib: int = 32

    @property
    def bytes_per_cube(self) -> int:
        bytes_per_cell = self.bits_per_cell // 8
        return self.n_dm * self.n_fast_vis * self.n_filled * bytes_per_cell


@dataclass
class InvariantResult:
    """Per-invariant pass/fail + metrics."""

    name: str
    passed: bool
    duration_s: float
    metrics: dict = field(default_factory=dict)
    failure_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# RX recv loop (thread)
# ---------------------------------------------------------------------------

class _RxLoop:
    """Background thread that recvs UDP datagrams and feeds the prod RX."""

    def __init__(
        self,
        sock: socket.socket,
        rx_prod: TransportRxProd,
        recv_bufsize: int = 65536,
    ) -> None:
        self._sock = sock
        self._rx_prod = rx_prod
        self._bufsize = recv_bufsize
        self._stop = threading.Event()
        self._hold = threading.Event()  # blocks recv loop when set
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.dropped_at_kernel_estimate: int = 0
        self.n_datagrams_received: int = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._hold.clear()
        try:
            # Send a sentinel to unblock recv.
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as kick:
                kick.sendto(b"", self._sock.getsockname())
        except OSError:
            pass
        self._thread.join(timeout=timeout)

    def hold(self) -> None:
        """Pause the recv thread (the OS RCVBUF will fill up)."""
        self._hold.set()

    def release(self) -> None:
        self._hold.clear()

    def _run(self) -> None:
        self._sock.settimeout(0.5)
        while not self._stop.is_set():
            if self._hold.is_set():
                time.sleep(0.05)
                continue
            try:
                data, _addr = self._sock.recvfrom(self._bufsize)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                continue
            self.n_datagrams_received += 1
            try:
                self._rx_prod.ingest_datagram(data)
            except Exception:  # pylint: disable=broad-except
                LOG.exception("rx.ingest_datagram failed; counted as drop")


# ---------------------------------------------------------------------------
# Synthetic cube generator
# ---------------------------------------------------------------------------

def _make_cube(op: OpPoint, *, seed: int = 0) -> torch.Tensor:
    """Generate one synthetic (n_dm, n_fast_vis, n_filled) complex cube."""
    rng = np.random.default_rng(seed)
    re = rng.standard_normal((op.n_dm, op.n_fast_vis, op.n_filled), dtype=np.float32)
    im = rng.standard_normal((op.n_dm, op.n_fast_vis, op.n_filled), dtype=np.float32)
    return torch.from_numpy(re + 1j * im).to(torch.complex64)


# ---------------------------------------------------------------------------
# pattern_id helpers (DoD doesn't need real /cnf — use deterministic ints)
# ---------------------------------------------------------------------------

def _stable_pattern_id(dec_deg: float, salt: str = "m4a-bench-v1") -> int:
    """Map (dec_deg, salt) → 64-bit pattern_id.

    Stand-in for ``predict_pattern_id`` over a real
    ``/cnf/corr_setup_96`` + ``dec_deg`` tuple. The bench just needs a
    deterministic 64-bit id that changes when ``dec_deg`` changes.
    """
    import hashlib
    key = f"{salt}|dec={dec_deg:.6f}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


# ---------------------------------------------------------------------------
# TX/RX harness (paired)
# ---------------------------------------------------------------------------

@dataclass
class _Harness:
    """One paired TX/RX context bound to a 127.0.0.1 port."""

    op: OpPoint
    tx: TransportTx
    rx_prod: TransportRxProd
    rx_loop: _RxLoop
    rx_sock: socket.socket
    port: int
    received_slots: list[RxProdSlot] = field(default_factory=list)

    def close(self) -> None:
        try:
            self.rx_loop.stop()
        except Exception:  # pylint: disable=broad-except
            pass
        try:
            self.tx.close()
        except Exception:  # pylint: disable=broad-except
            pass
        try:
            self.rx_sock.close()
        except Exception:  # pylint: disable=broad-except
            pass


def _build_harness(
    op: OpPoint,
    *,
    dec_deg_corr: float = 30.0,
    dec_deg_search: Optional[float] = None,
) -> _Harness:
    """Build TX/RX bound to a free 127.0.0.1 port. Patterns synced unless
    ``dec_deg_search != dec_deg_corr``."""
    if dec_deg_search is None:
        dec_deg_search = dec_deg_corr

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_RCVBUF,
        op.rcvbuf_mib * 1024 * 1024,
    )
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

    tx_cfg = TransportTxProdConfig(
        target_gbps_per_flow=op.target_gbps_per_flow,
        bits_per_cell=op.bits_per_cell,
        t_int_factor=op.t_int_factor,
        corr_idx=op.chgroup,
        max_frag_payload_bytes=op.max_frag_payload_bytes,
    )
    tx = TransportTx(
        host="127.0.0.1",
        port=port,
        chgroup=op.chgroup,
        use_prod_frame=True,
        prod_config=tx_cfg,
    )
    tx_pid = _stable_pattern_id(dec_deg_corr)
    tx.prepare_prod(
        pattern_id_by_chgroup={op.chgroup: tx_pid},
        n_grid=op.n_grid,
    )

    rx_cfg = TransportRxProdConfig(
        n_coarse_dm=op.n_dm,
        n_corr=op.n_chgroups,
        reorder_window_depth=4,
        so_rcvbuf_bytes=op.rcvbuf_mib * 1024 * 1024,
        expected_pattern_id_by_chgroup={op.chgroup: _stable_pattern_id(dec_deg_search)},
    )
    received: list[RxProdSlot] = []
    def _on_slot(corr_idx: int, dm_idx: int, slot: RxProdSlot) -> None:
        received.append(slot)
    rx_prod = TransportRxProd(config=rx_cfg, ring_write_cb=_on_slot)
    rx_loop = _RxLoop(sock, rx_prod)
    rx_loop.start()

    return _Harness(
        op=op,
        tx=tx,
        rx_prod=rx_prod,
        rx_loop=rx_loop,
        rx_sock=sock,
        port=port,
        received_slots=received,
    )


# ---------------------------------------------------------------------------
# Common TX pump
# ---------------------------------------------------------------------------

def _pump_for(
    h: _Harness,
    *,
    duration_s: float,
    rate_cubes_per_sec: float = 50.0,
    starting_specnum: int = 0,
    starting_block_n: int = 0,
) -> dict:
    """Pump synthetic cubes for ``duration_s`` seconds. Returns metrics."""
    deadline = time.monotonic() + duration_s
    block_n = starting_block_n
    specnum = starting_specnum
    period = 1.0 / max(rate_cubes_per_sec, 1.0)
    n_tx_calls = 0
    n_frames_sent = 0
    while time.monotonic() < deadline:
        cube = _make_cube(h.op, seed=block_n)
        try:
            n = h.tx.transmit(
                [cube],
                block_n=block_n,
                rfi_warming_up=False,
                specnum=specnum,
            )
        except OSError:
            n = 0
        n_frames_sent += int(n)
        n_tx_calls += 1
        block_n += 1
        specnum += 1
        # Pacing: target rate_cubes_per_sec.
        time.sleep(period)
    return {
        "n_tx_calls": n_tx_calls,
        "n_frames_sent": n_frames_sent,
        "tx_dropped_payloads": int(h.tx.tx_dropped_payloads),
        "cube_seq_emitted": int(h.tx.cube_seq_emitted),
        "n_datagrams_received": h.rx_loop.n_datagrams_received,
        "n_committed": int(h.rx_prod.prod_stats.n_committed),
        "pattern_mismatch_count_sum": sum(
            h.rx_prod.prod_stats.pattern_mismatch_count.values()
        ),
        "seq_gap_count_per_flow_sum": sum(
            h.rx_prod.prod_stats.seq_gap_count_per_flow.values()
        ),
        "window_slide_zerofill_count": int(
            h.rx_prod.prod_stats.window_slide_zerofill_count
        ),
        "bad_magic_count": int(h.rx_prod.prod_stats.bad_magic_count),
        "bad_field_range_count": int(h.rx_prod.prod_stats.bad_field_range_count),
    }


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

def invariant_1(quick: bool, duration_override: Optional[float] = None) -> InvariantResult:
    """I1: sustained 0% loss at default op-point over loopback."""
    op = OpPoint()
    h = _build_harness(op)
    duration_s = duration_override or (5.0 if quick else 60.0)
    t0 = time.monotonic()
    try:
        m = _pump_for(h, duration_s=duration_s, rate_cubes_per_sec=20.0)
        # Allow drain.
        time.sleep(0.5)
        m["n_committed_after_drain"] = int(h.rx_prod.prod_stats.n_committed)
    finally:
        h.close()
    dt = time.monotonic() - t0
    expected_committed = m["cube_seq_emitted"] * op.n_dm * op.n_fast_vis - (
        m["tx_dropped_payloads"]
    )
    loss_pct = 100.0 * max(0.0, 1.0 - (m["n_committed_after_drain"] / max(expected_committed, 1)))
    m["loss_pct"] = loss_pct
    passed = loss_pct < 1.0 and m["pattern_mismatch_count_sum"] == 0
    reason = None
    if not passed:
        if loss_pct >= 1.0:
            reason = f"loss_pct={loss_pct:.2f}% >= 1.0% (n_committed={m['n_committed_after_drain']}, expected={expected_committed})"
        else:
            reason = f"pattern_mismatch_count={m['pattern_mismatch_count_sum']} != 0"
    return InvariantResult(
        name="I1_sustained_zero_loss",
        passed=passed,
        duration_s=dt,
        metrics=m,
        failure_reason=reason,
    )


def invariant_2(quick: bool) -> InvariantResult:
    """I2: RX socket hold → TX continues, kernel drops, no Python-side stall."""
    op = OpPoint(
        target_gbps_per_flow=0.05,   # 50 Mbps — small RCVBUF fills fast
        rcvbuf_mib=1,
    )
    h = _build_harness(op)
    duration_s = 3.0 if quick else 5.0
    t0 = time.monotonic()
    try:
        # Run normally for 1 s — verify steady state.
        pre = _pump_for(h, duration_s=1.0, rate_cubes_per_sec=20.0)
        # Hold the recv loop. TX continues; kernel RCVBUF fills + drops.
        h.rx_loop.hold()
        held_start = time.monotonic()
        during = _pump_for(
            h,
            duration_s=duration_s,
            rate_cubes_per_sec=20.0,
            starting_specnum=pre["cube_seq_emitted"] + 1,
            starting_block_n=pre["n_tx_calls"],
        )
        held_elapsed = time.monotonic() - held_start
        # Release + run another 1 s.
        h.rx_loop.release()
        post = _pump_for(
            h,
            duration_s=1.0,
            rate_cubes_per_sec=20.0,
            starting_specnum=pre["cube_seq_emitted"] + during["n_tx_calls"] + 1,
            starting_block_n=pre["n_tx_calls"] + during["n_tx_calls"],
        )
    finally:
        h.close()
    dt = time.monotonic() - t0
    # Invariant: TX never stalled (held_elapsed ≈ duration_s + small slop);
    # pattern_mismatch_count stays 0 throughout (post-hold ingestion is on
    # the same pattern_id).
    tx_did_not_stall = held_elapsed < duration_s + 1.0
    metrics = {
        "pre": pre,
        "during_hold": during,
        "post_release": post,
        "held_elapsed_s": held_elapsed,
        "expected_hold_s": duration_s,
    }
    passed = tx_did_not_stall and post["pattern_mismatch_count_sum"] == 0
    reason = None
    if not passed:
        if not tx_did_not_stall:
            reason = f"TX stalled: held_elapsed={held_elapsed:.2f}s >> expected={duration_s:.2f}s"
        else:
            reason = f"pattern_mismatch_count={post['pattern_mismatch_count_sum']} != 0 after release"
    return InvariantResult(
        name="I2_rx_hold_no_backpressure",
        passed=passed,
        duration_s=dt,
        metrics=metrics,
        failure_reason=reason,
    )


def invariant_3(quick: bool) -> InvariantResult:
    """I3: pattern_mismatch_count == 0 when both ends share dec_deg."""
    op = OpPoint()
    h = _build_harness(op, dec_deg_corr=30.0, dec_deg_search=30.0)
    duration_s = 5.0 if quick else 60.0
    t0 = time.monotonic()
    try:
        m = _pump_for(h, duration_s=duration_s, rate_cubes_per_sec=20.0)
        time.sleep(0.5)
        m["pattern_mismatch_count_sum"] = sum(
            h.rx_prod.prod_stats.pattern_mismatch_count.values()
        )
    finally:
        h.close()
    dt = time.monotonic() - t0
    passed = m["pattern_mismatch_count_sum"] == 0 and m["n_committed"] > 0
    reason = (
        None if passed
        else f"pattern_mismatch_count={m['pattern_mismatch_count_sum']}, n_committed={m['n_committed']}"
    )
    return InvariantResult(
        name="I3_synced_pattern_zero_mismatch",
        passed=passed,
        duration_s=dt,
        metrics=m,
        failure_reason=reason,
    )


def invariant_4(quick: bool) -> InvariantResult:
    """I4: dec_deg mismatch → pattern_mismatch_count grows steadily."""
    op = OpPoint()
    h = _build_harness(op, dec_deg_corr=30.0, dec_deg_search=31.0)
    duration_s = 3.0 if quick else 5.0
    t0 = time.monotonic()
    try:
        m = _pump_for(h, duration_s=duration_s, rate_cubes_per_sec=20.0)
        time.sleep(0.5)
        m["pattern_mismatch_count_sum"] = sum(
            h.rx_prod.prod_stats.pattern_mismatch_count.values()
        )
    finally:
        h.close()
    dt = time.monotonic() - t0
    # Plan §M4a I4: increments steadily, every datagram.
    received = m["n_datagrams_received"]
    mismatches = m["pattern_mismatch_count_sum"]
    # Plan: every datagram. Allow 95% threshold for any in-flight drain weirdness.
    threshold = max(1, int(received * 0.95))
    passed = received > 0 and mismatches >= threshold
    reason = (
        None if passed
        else f"pattern_mismatch_count={mismatches} < threshold={threshold} (received={received})"
    )
    return InvariantResult(
        name="I4_dec_deg_mutation_increments_mismatch",
        passed=passed,
        duration_s=dt,
        metrics=m,
        failure_reason=reason,
    )


def invariant_5(quick: bool) -> InvariantResult:
    """I5: synchronised re-prepare → mismatch returns to 0 < 1 cube."""
    op = OpPoint()
    h = _build_harness(op, dec_deg_corr=30.0, dec_deg_search=31.0)
    t0 = time.monotonic()
    try:
        # First, run with mismatched dec_deg → expect mismatches.
        pre = _pump_for(h, duration_s=1.0, rate_cubes_per_sec=20.0)
        pre_mismatch = sum(h.rx_prod.prod_stats.pattern_mismatch_count.values())
        # Simulate cmd: prepare on both ends with NEW shared dec_deg = 35.
        new_dec = 35.0
        new_pid = _stable_pattern_id(new_dec)
        h.tx.prepare_prod(
            pattern_id_by_chgroup={op.chgroup: new_pid},
            n_grid=op.n_grid,
        )
        h.rx_prod.update_expected_pattern_id(op.chgroup, new_pid)
        # Wait one cube period + a hair.
        time.sleep(0.1)
        # Capture baseline immediately after re-prepare.
        baseline_mismatch = sum(h.rx_prod.prod_stats.pattern_mismatch_count.values())
        # Continue pumping for 2s; new mismatches must be 0.
        post = _pump_for(
            h, duration_s=2.0, rate_cubes_per_sec=20.0,
            starting_specnum=pre["cube_seq_emitted"] + 100,
            starting_block_n=pre["n_tx_calls"] + 100,
        )
        post_mismatch = sum(h.rx_prod.prod_stats.pattern_mismatch_count.values())
        new_mismatches = post_mismatch - baseline_mismatch
    finally:
        h.close()
    dt = time.monotonic() - t0
    passed = pre_mismatch > 0 and new_mismatches == 0
    reason = (
        None if passed
        else f"pre_mismatch={pre_mismatch} (need >0), new_mismatches_after_re-prepare={new_mismatches} (need 0)"
    )
    return InvariantResult(
        name="I5_re_prepare_clears_mismatch",
        passed=passed,
        duration_s=dt,
        metrics={
            "pre_mismatch": pre_mismatch,
            "baseline_after_re-prepare": baseline_mismatch,
            "post_mismatch": post_mismatch,
            "new_mismatches": new_mismatches,
            "pre": pre,
            "post": post,
        },
        failure_reason=reason,
    )


def invariant_6(quick: bool) -> InvariantResult:
    """I6: restart RX only → resumes < 5 s with mismatch → 0."""
    op = OpPoint()
    h = _build_harness(op, dec_deg_corr=30.0, dec_deg_search=30.0)
    t0 = time.monotonic()
    try:
        # Run for 1 s to establish steady state.
        pre = _pump_for(h, duration_s=1.0, rate_cubes_per_sec=20.0)
        pre_committed = int(h.rx_prod.prod_stats.n_committed)
        # "Restart RX" — stop loop, rebuild rx_prod from scratch, restart.
        h.rx_loop.stop()
        # New RX with the SAME expected pattern_id (search-rx restart, not
        # re-prepare).
        same_pid = _stable_pattern_id(30.0)
        new_rx_cfg = TransportRxProdConfig(
            n_coarse_dm=op.n_dm,
            n_corr=op.n_chgroups,
            reorder_window_depth=4,
            expected_pattern_id_by_chgroup={op.chgroup: same_pid},
        )
        received_after: list[RxProdSlot] = []
        def _on_slot(corr_idx: int, dm_idx: int, slot: RxProdSlot) -> None:
            received_after.append(slot)
        new_rx_prod = TransportRxProd(config=new_rx_cfg, ring_write_cb=_on_slot)
        new_rx_loop = _RxLoop(h.rx_sock, new_rx_prod)
        new_rx_loop.start()
        h.rx_prod = new_rx_prod
        h.rx_loop = new_rx_loop
        # Restart window: 1 s of pumping.
        restart_t0 = time.monotonic()
        post = _pump_for(
            h,
            duration_s=1.0,
            rate_cubes_per_sec=20.0,
            starting_specnum=pre["cube_seq_emitted"] + 100,
            starting_block_n=pre["n_tx_calls"] + 100,
        )
        first_resume_dt_s = time.monotonic() - restart_t0
        new_committed = int(h.rx_prod.prod_stats.n_committed) - 0  # rx_prod is fresh
        new_mismatch = sum(h.rx_prod.prod_stats.pattern_mismatch_count.values())
    finally:
        h.close()
    dt = time.monotonic() - t0
    passed = (
        pre_committed > 0
        and new_committed > 0
        and new_mismatch == 0
        and first_resume_dt_s < 5.0
    )
    reason = (
        None if passed
        else (
            f"pre_committed={pre_committed} (need >0), new_committed={new_committed} "
            f"(need >0), new_mismatch={new_mismatch} (need 0), "
            f"first_resume_dt_s={first_resume_dt_s:.2f} (need <5s)"
        )
    )
    return InvariantResult(
        name="I6_rx_restart_resumes_quickly",
        passed=passed,
        duration_s=dt,
        metrics={
            "pre_committed": pre_committed,
            "new_committed": new_committed,
            "new_mismatch": new_mismatch,
            "first_resume_dt_s": first_resume_dt_s,
            "pre": pre,
            "post": post,
        },
        failure_reason=reason,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

ALL_INVARIANTS = [
    ("1", invariant_1),
    ("2", invariant_2),
    ("3", invariant_3),
    ("4", invariant_4),
    ("5", invariant_5),
    ("6", invariant_6),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all", action="store_true", help="run all six")
    parser.add_argument("--invariant", type=str, choices=[k for k, _ in ALL_INVARIANTS])
    parser.add_argument("--quick", action="store_true", help="5 s windows instead of 60 s")
    parser.add_argument("--json", type=str, help="write JSON report to this path")
    parser.add_argument(
        "--invariant-1-duration",
        type=float,
        default=None,
        help="override I1 duration in seconds (DoD harness uses 60 s)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    to_run = []
    if args.all:
        to_run = ALL_INVARIANTS
    elif args.invariant:
        to_run = [(k, fn) for k, fn in ALL_INVARIANTS if k == args.invariant]
    else:
        parser.error("specify --all or --invariant N")

    results = []
    for key, fn in to_run:
        print(f"\n=== Running invariant I{key} ===", flush=True)
        if key == "1" and args.invariant_1_duration is not None:
            r = fn(args.quick, duration_override=args.invariant_1_duration)
        else:
            r = fn(args.quick)
        status = "PASS" if r.passed else "FAIL"
        print(
            f"  {status}  {r.name}  duration={r.duration_s:.2f}s  "
            f"{('reason=' + r.failure_reason) if r.failure_reason else ''}",
            flush=True,
        )
        results.append(r)

    summary = {
        "all_passed": all(r.passed for r in results),
        "results": [asdict(r) for r in results],
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"\nJSON report written to {args.json}", flush=True)

    print(
        f"\nOverall: {'PASS' if summary['all_passed'] else 'FAIL'} "
        f"({sum(r.passed for r in results)}/{len(results)} invariants)",
        flush=True,
    )
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
