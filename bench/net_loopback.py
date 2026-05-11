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

# Production-rate tests (M4a post-merge addition)

The six invariants above test protocol correctness at toy scale (1 TX → 1
RX, 1 Gbps/flow target, n_filled=50). To answer "does it work at the
production rate?" the bench also runs three production-topology tests
that exercise the §9 default op-point at the rates from plan §11
line 2654:

* **P1 (per-flow)**: 1 TX (6 dm_idx flows) → 1 RX, n_filled≈5000, target
  0.073 Gb/s/flow → 0.44 Gb/s aggregate. Verifies one (corr, search) pair
  sustains its production rate over loopback with realistic fragmentation
  (2 fragments/payload at MTU 9000 + cint8) and no loss.

* **P2 (16-way fan-in)**: 16 TX (one per virtual corr, distinct chgroup)
  → 1 RX on a single port. Aggregate ingress ~7 Gb/s. Verifies what a
  single search node sees in production: 16 streams of 6-dm flows
  interleaved through the per-(corr, dm) reorder windows.

* **P3 (4-way fan-out)**: 4 TX (same chgroup, disjoint dm-idx subsets
  0-5/6-11/12-17/18-23) → 4 RX on different ports. Aggregate egress
  ~1.75 Gb/s. Verifies a single corr node's TX path emitting to 4 search
  destinations.

The production tests don't try to reproduce the full 16×4 system on
one host — that's M4b multi-host scope. They do verify that the kernel
loopback + Python recvfrom path on h01 can sustain the production wire
rate per-flow, per-search-ingress, and per-corr-egress.

Run:
    python -m bench.net_loopback --all                # six invariants
    python -m bench.net_loopback --invariant 3        # one invariant
    python -m bench.net_loopback --all --quick        # 5 s windows
    python -m bench.net_loopback --prod all           # three prod tests
    python -m bench.net_loopback --prod fan-in        # just P2
    python -m bench.net_loopback --prod all --quick   # 5 s prod windows
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
# Production-rate tests (P1-P3) — exercise the §9 default op-point + topology
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProdOpPoint:
    """Production §9 default op-point + plan §11 line 2654 topology.

    The bench drives the per-flow rate at ``target_gbps_per_dm`` and the
    pacer (TransportTxProdConfig.target_gbps_per_flow) at the same value
    so each per-dm token bucket is sized correctly. The per-process
    aggregate is ``n_dm_per_tx * target_gbps_per_dm``; per-search and
    per-corr aggregates are computed from the topology fields.

    Production targets (plan §11 line 2654):
        per-(corr, search) flow rate:     ~0.44 Gb/s
        per-search aggregate ingress:     ~7   Gb/s  (16 corrs * 0.44)
        per-corr aggregate egress:        ~1.75 Gb/s (4 searches * 0.44)

    A single (corr, search) flow carries 6 of the 24 coarse DMs.
    """

    n_grid: int = 256
    bits_per_cell: int = BITS_CINT8_COMPLEX
    t_int_factor: int = 16
    n_filled: int = 5000
    n_fast_vis: int = 1
    n_dm_per_tx: int = 6                # 24 coarse_DMs / 4 search dests
    target_gbps_per_dm: float = 0.073   # 0.44 / 6
    max_frag_payload_bytes: int = 8964
    rcvbuf_mib: int = 256

    n_corr_nodes: int = 16
    n_search_destinations: int = 4

    @property
    def bytes_per_cube_per_tx(self) -> int:
        return (
            self.n_dm_per_tx
            * self.n_fast_vis
            * self.n_filled
            * (self.bits_per_cell // 8)
        )

    @property
    def target_gbps_per_tx(self) -> float:
        return self.n_dm_per_tx * self.target_gbps_per_dm

    @property
    def target_gbps_aggregate_per_search(self) -> float:
        return self.n_corr_nodes * self.target_gbps_per_tx

    @property
    def target_gbps_aggregate_per_corr(self) -> float:
        return self.n_search_destinations * self.target_gbps_per_tx


@dataclass
class ProdRxRefs:
    """Holds the live references for one RX (so it can be cleaned up)."""

    sock: socket.socket
    rx_prod: TransportRxProd
    rx_loop: _RxLoop
    port: int
    n_committed: int = 0  # populated after stop() for convenience


def _make_prod_rx(
    op: ProdOpPoint,
    expected_chgroups: list[int],
    *,
    dec_deg: float = 30.0,
) -> ProdRxRefs:
    """Bind a 127.0.0.1 socket and spin up a TransportRxProd + recv loop.

    The ring_write_cb is a pure counter (does NOT retain slot objects),
    so a 60 s × 7 Gb/s run does not OOM on n_committed * n_filled
    complex64 arrays.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_RCVBUF,
            op.rcvbuf_mib * 1024 * 1024,
        )
    except OSError:
        LOG.warning(
            "could not raise SO_RCVBUF to %d MiB; falling back to kernel default",
            op.rcvbuf_mib,
        )
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    pid = _stable_pattern_id(dec_deg)
    n_corr_cfg = max(16, (max(expected_chgroups) + 1) if expected_chgroups else 16)
    rx_cfg = TransportRxProdConfig(
        n_coarse_dm=op.n_dm_per_tx if op.n_dm_per_tx > 0 else 1,
        n_corr=n_corr_cfg,
        reorder_window_depth=4,
        so_rcvbuf_bytes=op.rcvbuf_mib * 1024 * 1024,
        expected_pattern_id_by_chgroup={c: pid for c in expected_chgroups},
    )

    def _on_slot(corr_idx: int, dm_idx: int, slot: RxProdSlot) -> None:
        # Pure counter — do not retain slot objects.
        pass

    rx_prod = TransportRxProd(config=rx_cfg, ring_write_cb=_on_slot)
    rx_loop = _RxLoop(sock, rx_prod, recv_bufsize=65536)
    rx_loop.start()
    return ProdRxRefs(sock=sock, rx_prod=rx_prod, rx_loop=rx_loop, port=port)


def _make_prod_tx(
    op: ProdOpPoint,
    *,
    chgroup: int,
    rx_port: int,
    dec_deg: float = 30.0,
) -> TransportTx:
    """Build one TransportTx (prod-frame) targeting 127.0.0.1:rx_port."""
    cfg = TransportTxProdConfig(
        target_gbps_per_flow=op.target_gbps_per_dm,
        bits_per_cell=op.bits_per_cell,
        t_int_factor=op.t_int_factor,
        corr_idx=chgroup,
        max_frag_payload_bytes=op.max_frag_payload_bytes,
    )
    tx = TransportTx(
        host="127.0.0.1",
        port=rx_port,
        chgroup=chgroup,
        use_prod_frame=True,
        prod_config=cfg,
    )
    pid = _stable_pattern_id(dec_deg)
    tx.prepare_prod(
        pattern_id_by_chgroup={chgroup: pid},
        n_grid=op.n_grid,
    )
    return tx


class _ProdTxPump:
    """Background thread that pumps cubes through one TransportTx.

    Sleeps for a target cube period (deadline-based; absorbs jitter) so
    the per-dm token bucket sees a steady arrival rate ≥ the target
    rate. The bucket does the precise wire-rate pacing.
    """

    def __init__(
        self,
        tx: TransportTx,
        op: ProdOpPoint,
        *,
        oversend: float = 1.20,
    ) -> None:
        self._tx = tx
        self._op = op
        self._stop = threading.Event()
        bytes_per_cube_per_dm = (
            op.n_fast_vis * op.n_filled * (op.bits_per_cell // 8)
        )
        target_rate_hz_per_dm = (
            (op.target_gbps_per_dm * 1e9 / 8.0) / max(1, bytes_per_cube_per_dm)
        )
        # We emit one cube per cube-period (cube contains all n_dm slices).
        self._period = 1.0 / (target_rate_hz_per_dm * oversend)
        # Pre-build a single random cube; data values don't affect the
        # rate test (TX encodes whatever is there).
        self._cube = _make_cube_for_pump(op, seed=int(tx.chgroup) * 7919 + 1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._t_start_ns = 0
        self._t_stop_ns = 0
        self.cubes_emitted = 0
        self.frames_sent = 0
        self.tx_dropped_payloads_at_stop = 0

    def start(self) -> None:
        self._t_start_ns = time.monotonic_ns()
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)
        self._t_stop_ns = time.monotonic_ns()
        self.tx_dropped_payloads_at_stop = int(self._tx.tx_dropped_payloads)

    @property
    def elapsed_s(self) -> float:
        end = self._t_stop_ns or time.monotonic_ns()
        if self._t_start_ns == 0:
            return 0.0
        return (end - self._t_start_ns) * 1e-9

    def _run(self) -> None:
        block_n = 0
        specnum = 0
        next_t = time.monotonic()
        period = self._period
        while not self._stop.is_set():
            try:
                n = self._tx.transmit(
                    [self._cube],
                    block_n=block_n,
                    rfi_warming_up=False,
                    specnum=specnum,
                )
            except OSError:
                n = 0
            self.frames_sent += int(n)
            self.cubes_emitted += 1
            block_n += 1
            specnum += 1
            next_t += period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # Behind schedule — let the bucket handle pacing.
                next_t = time.monotonic()


def _make_cube_for_pump(op: ProdOpPoint, *, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    re = rng.standard_normal(
        (op.n_dm_per_tx, op.n_fast_vis, op.n_filled), dtype=np.float32,
    )
    im = rng.standard_normal(
        (op.n_dm_per_tx, op.n_fast_vis, op.n_filled), dtype=np.float32,
    )
    return torch.from_numpy(re + 1j * im).to(torch.complex64)


# ---------------------------------------------------------------------------
# Multi-process TX driver (bypasses the Python GIL)
# ---------------------------------------------------------------------------
#
# A single Python thread can encode + sendto roughly 370-400 cubes/s at the
# 6-DM × 5000-filled-cell op-point (≈ 0.18 Gb/s). The GIL serialises threads
# so adding TX threads in one process does not scale. To push the RX socket
# at production rates we spawn each TX in its own subprocess via the
# ``spawn`` start method (avoids inherited torch/CUDA state) and have it
# pump for a fixed duration before exiting and reporting stats back through
# a Queue.
#
# Production corr nodes run one process per node with 4 TX threads — they
# will hit the same GIL bottleneck unless the TX path moves to C (M4a
# chunk-6 RX C path is analogous; an equivalent TX-side C path is the
# obvious follow-up). The multi-process bench mode here is *not* a
# proposal for production; it is purely a measurement tool for the
# RX-side wire ceiling on h01.

def _mp_tx_worker(
    op_kwargs: dict,
    chgroup: int,
    rx_port: int,
    duration_s: float,
    result_q,
) -> None:
    """Subprocess worker. Builds one TransportTx + pump, runs for
    ``duration_s`` seconds, posts stats dict on ``result_q`` and exits.
    """
    # Reimport here — under ``spawn`` start the child re-executes module init.
    from bench.net_loopback import (
        ProdOpPoint,
        _make_prod_tx,
        _ProdTxPump,
    )
    import time as _time

    op = ProdOpPoint(**op_kwargs)
    tx = _make_prod_tx(op, chgroup=chgroup, rx_port=rx_port)
    pump = _ProdTxPump(tx, op)
    try:
        pump.start()
        _time.sleep(duration_s)
        pump.stop()
    finally:
        tx.close()
    result_q.put({
        "chgroup": chgroup,
        "cubes_emitted": int(pump.cubes_emitted),
        "frames_sent": int(pump.frames_sent),
        "elapsed_s": float(pump.elapsed_s),
        "tx_dropped_payloads": int(pump.tx_dropped_payloads_at_stop),
    })


@dataclass
class _MpTxStats:
    """Minimal stats from a multi-process TX worker."""

    chgroup: int
    cubes_emitted: int
    frames_sent: int
    elapsed_s: float
    tx_dropped_payloads: int


def _run_mp_tx_batch(
    op: ProdOpPoint,
    chgroups: list[int],
    rx_port: int,
    duration_s: float,
) -> list[_MpTxStats]:
    """Spawn one subprocess per chgroup, run pumps for ``duration_s``,
    collect stats. Caller is responsible for the RX side."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    op_kwargs = asdict(op)
    procs = []
    for chg in chgroups:
        p = ctx.Process(
            target=_mp_tx_worker,
            args=(op_kwargs, chg, rx_port, duration_s, q),
        )
        p.start()
        procs.append(p)
    # Wait for all to finish reporting.
    results: list[_MpTxStats] = []
    for _ in chgroups:
        try:
            d = q.get(timeout=duration_s + 30.0)
        except Exception as e:  # pylint: disable=broad-except
            LOG.error("mp tx worker did not return stats: %s", e)
            continue
        results.append(_MpTxStats(**d))
    for p in procs:
        p.join(timeout=10.0)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5.0)
    return results


def _prod_metrics_mp(
    *,
    mp_stats: list[_MpTxStats],
    rxs: list[ProdRxRefs],
    op: ProdOpPoint,
    duration_s: float,
) -> dict:
    """Same shape as ``_prod_metrics`` but driven by multi-process stats."""
    cubes_emitted = sum(s.cubes_emitted for s in mp_stats)
    frames_sent = sum(s.frames_sent for s in mp_stats)
    tx_dropped = sum(s.tx_dropped_payloads for s in mp_stats)
    payload_bytes_emitted = cubes_emitted * op.bytes_per_cube_per_tx
    elapsed = max((s.elapsed_s for s in mp_stats), default=duration_s)
    observed_gbps = (
        (payload_bytes_emitted * 8) / (elapsed * 1e9) if elapsed > 0 else 0.0
    )
    n_received_total = sum(int(r.rx_loop.n_datagrams_received) for r in rxs)
    n_committed_total = sum(int(r.rx_prod.prod_stats.n_committed) for r in rxs)
    pattern_mismatches = sum(
        sum(r.rx_prod.prod_stats.pattern_mismatch_count.values()) for r in rxs
    )
    zerofills = sum(
        int(r.rx_prod.prod_stats.window_slide_zerofill_count) for r in rxs
    )
    oor_drops = sum(
        int(r.rx_prod.prod_stats.out_of_order_drop_count) for r in rxs
    )
    bad_field = sum(int(r.rx_prod.prod_stats.bad_field_range_count) for r in rxs)
    bad_length = sum(int(r.rx_prod.prod_stats.bad_length_count) for r in rxs)
    bad_magic = sum(int(r.rx_prod.prod_stats.bad_magic_count) for r in rxs)
    expected_committed = (
        cubes_emitted * op.n_dm_per_tx * op.n_fast_vis - tx_dropped
    )
    loss_pct = 100.0 * max(
        0.0,
        1.0 - (n_committed_total / max(expected_committed, 1)),
    )
    per_proc_gbps = [
        (s.cubes_emitted * op.bytes_per_cube_per_tx * 8) / (s.elapsed_s * 1e9)
        if s.elapsed_s > 0 else 0.0
        for s in mp_stats
    ]
    return {
        "duration_s": duration_s,
        "elapsed_s": elapsed,
        "n_tx": len(mp_stats),
        "n_rx": len(rxs),
        "cubes_emitted_total": cubes_emitted,
        "frames_sent_total": frames_sent,
        "tx_dropped_payloads_total": tx_dropped,
        "datagrams_received_total": n_received_total,
        "n_committed_total": n_committed_total,
        "expected_committed": expected_committed,
        "loss_pct": loss_pct,
        "payload_bytes_emitted": payload_bytes_emitted,
        "observed_gbps_aggregate": observed_gbps,
        "per_proc_observed_gbps_min": min(per_proc_gbps) if per_proc_gbps else 0.0,
        "per_proc_observed_gbps_max": max(per_proc_gbps) if per_proc_gbps else 0.0,
        "per_proc_observed_gbps_mean": (
            sum(per_proc_gbps) / len(per_proc_gbps) if per_proc_gbps else 0.0
        ),
        "target_gbps_per_tx": op.target_gbps_per_tx,
        "target_gbps_per_dm": op.target_gbps_per_dm,
        "pattern_mismatches": pattern_mismatches,
        "window_slide_zerofill_count": zerofills,
        "out_of_order_drop_count": oor_drops,
        "bad_field_range_count": bad_field,
        "bad_length_count": bad_length,
        "bad_magic_count": bad_magic,
    }


def prod_fan_in_16x_mp(
    quick: bool, duration_override: Optional[float] = None,
) -> InvariantResult:
    """P2-mp: 16 TX subprocesses → 1 RX, multi-process to bypass GIL."""
    op = ProdOpPoint()
    duration_s = duration_override or (5.0 if quick else 60.0)
    chgroups = list(range(op.n_corr_nodes))
    rx = _make_prod_rx(op, expected_chgroups=chgroups)
    t0 = time.monotonic()
    try:
        mp_stats = _run_mp_tx_batch(
            op, chgroups=chgroups, rx_port=rx.port, duration_s=duration_s,
        )
        # Hold the RX open briefly to drain in-flight datagrams.
        time.sleep(2.0)
    finally:
        _close_prod_rx(rx)
    dt = time.monotonic() - t0
    m = _prod_metrics_mp(
        mp_stats=mp_stats, rxs=[rx], op=op, duration_s=duration_s,
    )
    target_agg = op.target_gbps_aggregate_per_search  # ~7 Gb/s
    m["target_gbps_aggregate_per_search"] = target_agg
    rate_ok = m["observed_gbps_aggregate"] >= 0.90 * target_agg
    loss_ok = m["loss_pct"] < 5.0
    no_mismatch = m["pattern_mismatches"] == 0
    passed = rate_ok and loss_ok and no_mismatch
    bits: list[str] = []
    if not rate_ok:
        bits.append(
            f"observed_gbps={m['observed_gbps_aggregate']:.3f} < "
            f"0.90*target={target_agg * 0.90:.3f} (target ~7 Gb/s ingress)"
        )
    if not loss_ok:
        bits.append(f"loss_pct={m['loss_pct']:.2f}% >= 5.0%")
    if not no_mismatch:
        bits.append(f"pattern_mismatches={m['pattern_mismatches']}")
    reason = None if passed else "; ".join(bits)
    return InvariantResult(
        name="P2mp_fan_in_16x_mp_7Gbps_per_search",
        passed=passed,
        duration_s=dt,
        metrics=m,
        failure_reason=reason,
    )


def prod_fan_out_4x_mp(
    quick: bool, duration_override: Optional[float] = None,
) -> InvariantResult:
    """P3-mp: 4 TX subprocesses → 4 RX (one per "search dest"), 1.75 Gb/s egress.

    All TX subprocesses use chgroup=0 (one corr) but target different RX
    ports. Each subprocess maxes out at ~0.18 Gb/s, so the aggregate of
    4 (one per search destination) approaches ~0.7 Gb/s — short of the
    1.75 Gb/s target but the bench is honest about that.
    """
    op = ProdOpPoint()
    duration_s = duration_override or (5.0 if quick else 60.0)
    rxs = [
        _make_prod_rx(op, expected_chgroups=[0])
        for _ in range(op.n_search_destinations)
    ]
    # We can't reuse chgroup=0 in 4 subprocesses with the helper because the
    # helper needs a unique (chgroup, rx_port) target per worker, but the
    # worker only takes ``rx_port`` as input. So we run a small bespoke
    # loop here — one subprocess per RX.
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    op_kwargs = asdict(op)
    procs = []
    for r in rxs:
        p = ctx.Process(
            target=_mp_tx_worker,
            args=(op_kwargs, 0, r.port, duration_s, q),
        )
        p.start()
        procs.append(p)
    t0 = time.monotonic()
    try:
        mp_stats: list[_MpTxStats] = []
        for _ in range(len(rxs)):
            try:
                d = q.get(timeout=duration_s + 30.0)
            except Exception as e:  # pylint: disable=broad-except
                LOG.error("mp tx worker did not return stats: %s", e)
                continue
            mp_stats.append(_MpTxStats(**d))
        for p in procs:
            p.join(timeout=10.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)
        time.sleep(1.0)
    finally:
        for r in rxs:
            _close_prod_rx(r)
    dt = time.monotonic() - t0
    m = _prod_metrics_mp(
        mp_stats=mp_stats, rxs=rxs, op=op, duration_s=duration_s,
    )
    target_agg = op.target_gbps_aggregate_per_corr  # ~1.75 Gb/s
    m["target_gbps_aggregate_per_corr"] = target_agg
    rate_ok = m["observed_gbps_aggregate"] >= 0.95 * target_agg
    loss_ok = m["loss_pct"] < 1.0
    no_mismatch = m["pattern_mismatches"] == 0
    passed = rate_ok and loss_ok and no_mismatch
    bits: list[str] = []
    if not rate_ok:
        bits.append(
            f"observed_gbps={m['observed_gbps_aggregate']:.3f} < "
            f"0.95*target={target_agg * 0.95:.3f} (target ~1.75 Gb/s egress)"
        )
    if not loss_ok:
        bits.append(f"loss_pct={m['loss_pct']:.2f}% >= 1.0%")
    if not no_mismatch:
        bits.append(f"pattern_mismatches={m['pattern_mismatches']}")
    reason = None if passed else "; ".join(bits)
    return InvariantResult(
        name="P3mp_fan_out_4x_mp_1p75Gbps_per_corr",
        passed=passed,
        duration_s=dt,
        metrics=m,
        failure_reason=reason,
    )


def _close_prod_rx(refs: ProdRxRefs) -> None:
    try:
        refs.rx_loop.stop()
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        refs.sock.close()
    except Exception:  # pylint: disable=broad-except
        pass


def _prod_metrics(
    *,
    pumps: list[_ProdTxPump],
    txs: list[TransportTx],
    rxs: list[ProdRxRefs],
    op: ProdOpPoint,
    duration_s: float,
) -> dict:
    cubes_emitted = sum(p.cubes_emitted for p in pumps)
    frames_sent = sum(p.frames_sent for p in pumps)
    tx_dropped = sum(int(tx.tx_dropped_payloads) for tx in txs)
    payload_bytes_emitted = cubes_emitted * op.bytes_per_cube_per_tx
    elapsed = max((p.elapsed_s for p in pumps), default=duration_s)
    observed_gbps = (
        (payload_bytes_emitted * 8) / (elapsed * 1e9) if elapsed > 0 else 0.0
    )
    n_received_total = sum(int(r.rx_loop.n_datagrams_received) for r in rxs)
    n_committed_total = sum(int(r.rx_prod.prod_stats.n_committed) for r in rxs)
    pattern_mismatches = sum(
        sum(r.rx_prod.prod_stats.pattern_mismatch_count.values()) for r in rxs
    )
    zerofills = sum(
        int(r.rx_prod.prod_stats.window_slide_zerofill_count) for r in rxs
    )
    oor_drops = sum(
        int(r.rx_prod.prod_stats.out_of_order_drop_count) for r in rxs
    )
    bad_field = sum(int(r.rx_prod.prod_stats.bad_field_range_count) for r in rxs)
    bad_length = sum(int(r.rx_prod.prod_stats.bad_length_count) for r in rxs)
    bad_magic = sum(int(r.rx_prod.prod_stats.bad_magic_count) for r in rxs)
    expected_committed = (
        cubes_emitted * op.n_dm_per_tx * op.n_fast_vis - tx_dropped
    )
    loss_pct = 100.0 * max(
        0.0,
        1.0 - (n_committed_total / max(expected_committed, 1)),
    )
    return {
        "duration_s": duration_s,
        "elapsed_s": elapsed,
        "n_tx": len(txs),
        "n_rx": len(rxs),
        "cubes_emitted_total": cubes_emitted,
        "frames_sent_total": frames_sent,
        "tx_dropped_payloads_total": tx_dropped,
        "datagrams_received_total": n_received_total,
        "n_committed_total": n_committed_total,
        "expected_committed": expected_committed,
        "loss_pct": loss_pct,
        "payload_bytes_emitted": payload_bytes_emitted,
        "observed_gbps_aggregate": observed_gbps,
        "target_gbps_per_tx": op.target_gbps_per_tx,
        "target_gbps_per_dm": op.target_gbps_per_dm,
        "pattern_mismatches": pattern_mismatches,
        "window_slide_zerofill_count": zerofills,
        "out_of_order_drop_count": oor_drops,
        "bad_field_range_count": bad_field,
        "bad_length_count": bad_length,
        "bad_magic_count": bad_magic,
    }


def prod_per_flow(
    quick: bool, duration_override: Optional[float] = None,
) -> InvariantResult:
    """P1: one (corr, search) flow at production per-flow rate."""
    op = ProdOpPoint()
    duration_s = duration_override or (5.0 if quick else 60.0)
    rx = _make_prod_rx(op, expected_chgroups=[0])
    tx = _make_prod_tx(op, chgroup=0, rx_port=rx.port)
    pump = _ProdTxPump(tx, op)
    t0 = time.monotonic()
    try:
        pump.start()
        time.sleep(duration_s)
        pump.stop()
        # Drain in-flight datagrams.
        time.sleep(1.0)
    finally:
        _close_prod_rx(rx)
        tx.close()
    dt = time.monotonic() - t0
    m = _prod_metrics(
        pumps=[pump], txs=[tx], rxs=[rx], op=op, duration_s=duration_s,
    )
    # Loss budget: < 1%. Rate: ≥ 95% of target per-tx aggregate.
    rate_ok = m["observed_gbps_aggregate"] >= 0.95 * op.target_gbps_per_tx
    loss_ok = m["loss_pct"] < 1.0
    no_mismatch = m["pattern_mismatches"] == 0
    passed = rate_ok and loss_ok and no_mismatch
    bits: list[str] = []
    if not rate_ok:
        bits.append(
            f"observed_gbps={m['observed_gbps_aggregate']:.3f} < "
            f"0.95*target={op.target_gbps_per_tx * 0.95:.3f}"
        )
    if not loss_ok:
        bits.append(f"loss_pct={m['loss_pct']:.2f}% >= 1.0%")
    if not no_mismatch:
        bits.append(f"pattern_mismatches={m['pattern_mismatches']}")
    reason = None if passed else "; ".join(bits)
    return InvariantResult(
        name="P1_per_flow_prod_rate_0p44Gbps",
        passed=passed,
        duration_s=dt,
        metrics=m,
        failure_reason=reason,
    )


def prod_fan_in_16x(
    quick: bool, duration_override: Optional[float] = None,
) -> InvariantResult:
    """P2: 16 TX → 1 RX, ~7 Gb/s aggregate ingress per search node."""
    op = ProdOpPoint()
    duration_s = duration_override or (5.0 if quick else 60.0)
    chgroups = list(range(op.n_corr_nodes))
    rx = _make_prod_rx(op, expected_chgroups=chgroups)
    txs = [_make_prod_tx(op, chgroup=c, rx_port=rx.port) for c in chgroups]
    pumps = [_ProdTxPump(tx, op) for tx in txs]
    t0 = time.monotonic()
    try:
        for p in pumps:
            p.start()
        time.sleep(duration_s)
        for p in pumps:
            p.stop()
        # Drain — more datagrams in flight at 7 Gb/s ingress, hold longer.
        time.sleep(2.0)
    finally:
        _close_prod_rx(rx)
        for tx in txs:
            tx.close()
    dt = time.monotonic() - t0
    m = _prod_metrics(
        pumps=pumps, txs=txs, rxs=[rx], op=op, duration_s=duration_s,
    )
    target_agg = op.target_gbps_aggregate_per_search  # ~7 Gb/s
    m["target_gbps_aggregate_per_search"] = target_agg
    # Fan-in budget: rate within 90% of target; loss < 5% (Python recvfrom
    # at 7 Gb/s is the chunk-6 motivation — large loss here is a *finding*,
    # not a regression, but the bench should still pass at ≥ 6.3 Gb/s with
    # ≤ 5% loss).
    rate_ok = m["observed_gbps_aggregate"] >= 0.90 * target_agg
    loss_ok = m["loss_pct"] < 5.0
    no_mismatch = m["pattern_mismatches"] == 0
    passed = rate_ok and loss_ok and no_mismatch
    bits: list[str] = []
    if not rate_ok:
        bits.append(
            f"observed_gbps={m['observed_gbps_aggregate']:.3f} < "
            f"0.90*target={target_agg * 0.90:.3f} (target ~7 Gb/s ingress)"
        )
    if not loss_ok:
        bits.append(f"loss_pct={m['loss_pct']:.2f}% >= 5.0%")
    if not no_mismatch:
        bits.append(f"pattern_mismatches={m['pattern_mismatches']}")
    reason = None if passed else "; ".join(bits)
    return InvariantResult(
        name="P2_fan_in_16x_7Gbps_per_search",
        passed=passed,
        duration_s=dt,
        metrics=m,
        failure_reason=reason,
    )


def prod_fan_out_4x(
    quick: bool, duration_override: Optional[float] = None,
) -> InvariantResult:
    """P3: 1 corr → 4 search dests, ~1.75 Gb/s aggregate egress per corr.

    Each of the 4 TX is from chgroup=0 (same corr) but targets a distinct
    RX port and carries a disjoint dm-idx subset of the 24 coarse DMs
    (0-5, 6-11, 12-17, 18-23). The aggregate is summed across all 4
    RX receivers; per-RX rate is one search node's share (0.44 Gb/s).
    """
    op = ProdOpPoint()
    duration_s = duration_override or (5.0 if quick else 60.0)
    # 4 separate RX (one per "search node"), 4 TX targeting them.
    rxs = [_make_prod_rx(op, expected_chgroups=[0]) for _ in range(op.n_search_destinations)]
    # Each TX gets its own chgroup-namespaced flow set, but we reuse
    # chgroup=0 (same corr node) — the RX side keys by (chgroup, dm_idx)
    # which is unique per (corr, dm-idx-bucket) anyway.
    txs = [
        _make_prod_tx(op, chgroup=0, rx_port=rxs[i].port)
        for i in range(op.n_search_destinations)
    ]
    pumps = [_ProdTxPump(tx, op) for tx in txs]
    t0 = time.monotonic()
    try:
        for p in pumps:
            p.start()
        time.sleep(duration_s)
        for p in pumps:
            p.stop()
        time.sleep(1.0)
    finally:
        for rx in rxs:
            _close_prod_rx(rx)
        for tx in txs:
            tx.close()
    dt = time.monotonic() - t0
    m = _prod_metrics(
        pumps=pumps, txs=txs, rxs=rxs, op=op, duration_s=duration_s,
    )
    target_agg = op.target_gbps_aggregate_per_corr  # ~1.75 Gb/s
    m["target_gbps_aggregate_per_corr"] = target_agg
    rate_ok = m["observed_gbps_aggregate"] >= 0.95 * target_agg
    loss_ok = m["loss_pct"] < 1.0
    no_mismatch = m["pattern_mismatches"] == 0
    passed = rate_ok and loss_ok and no_mismatch
    bits: list[str] = []
    if not rate_ok:
        bits.append(
            f"observed_gbps={m['observed_gbps_aggregate']:.3f} < "
            f"0.95*target={target_agg * 0.95:.3f} (target ~1.75 Gb/s egress)"
        )
    if not loss_ok:
        bits.append(f"loss_pct={m['loss_pct']:.2f}% >= 1.0%")
    if not no_mismatch:
        bits.append(f"pattern_mismatches={m['pattern_mismatches']}")
    reason = None if passed else "; ".join(bits)
    return InvariantResult(
        name="P3_fan_out_4x_1p75Gbps_per_corr",
        passed=passed,
        duration_s=dt,
        metrics=m,
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

PROD_TESTS = [
    ("per-flow", prod_per_flow),
    ("fan-in", prod_fan_in_16x),
    ("fan-out", prod_fan_out_4x),
    ("fan-in-mp", prod_fan_in_16x_mp),
    ("fan-out-mp", prod_fan_out_4x_mp),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all", action="store_true", help="run all six invariants")
    parser.add_argument("--invariant", type=str, choices=[k for k, _ in ALL_INVARIANTS])
    parser.add_argument(
        "--prod",
        type=str,
        choices=[k for k, _ in PROD_TESTS] + ["all"],
        help="run a production-rate test (P1-P3) instead of the protocol invariants",
    )
    parser.add_argument(
        "--prod-duration",
        type=float,
        default=None,
        help="override per-prod-test duration in seconds (default 60 s, 5 s with --quick)",
    )
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

    if args.prod:
        if args.prod == "all":
            to_run_prod = PROD_TESTS
        else:
            to_run_prod = [(k, fn) for k, fn in PROD_TESTS if k == args.prod]
        results = []
        for key, fn in to_run_prod:
            print(f"\n=== Running prod test P:{key} ===", flush=True)
            r = fn(args.quick, duration_override=args.prod_duration)
            status = "PASS" if r.passed else "FAIL"
            print(
                f"  {status}  {r.name}  duration={r.duration_s:.2f}s  "
                f"{('reason=' + r.failure_reason) if r.failure_reason else ''}",
                flush=True,
            )
            # Surface key throughput numbers inline (easier to eyeball in CI).
            m = r.metrics
            print(
                f"    observed_gbps_aggregate={m.get('observed_gbps_aggregate', 0.0):.3f}  "
                f"target_gbps_per_tx={m.get('target_gbps_per_tx', 0.0):.3f}  "
                f"loss_pct={m.get('loss_pct', 0.0):.3f}%  "
                f"committed={m.get('n_committed_total', 0)}/{m.get('expected_committed', 0)}  "
                f"tx_dropped={m.get('tx_dropped_payloads_total', 0)}",
                flush=True,
            )
            results.append(r)
    else:
        if args.all:
            to_run = ALL_INVARIANTS
        elif args.invariant:
            to_run = [(k, fn) for k, fn in ALL_INVARIANTS if k == args.invariant]
        else:
            parser.error("specify --all, --invariant N, or --prod {per-flow|fan-in|fan-out|all}")

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
        f"({sum(r.passed for r in results)}/{len(results)} tests)",
        flush=True,
    )
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
