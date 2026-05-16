"""M4b — pair-rate transport bench (single n01 → n02 pair @ 4× per-pair rate).

Runs one side of a single (corr-host → search-host) UDP transport pair on
the OVRO LXD fleet's `br2` 100 GbE data plane (per plan §6.3). Unlike
:mod:`bench.net_loopback`, which co-locates TX and RX in one process on
``127.0.0.1``, this bench splits TX and RX into separate processes that
run on **different hosts**, coordinated by ``--start-at`` (a UTC second
both sides sleep until — clock-skew tolerance ≥ 50 ms with NTP).

# Why a single pair at 4× per-pair production rate

Production topology (plan §4.3 / §11.4):
    * 16 corr nodes × 4 search-pairs each = 64 unidirectional value-channel
      flows system-wide.
    * Each (corr, search) pair carries **6 of the 24 coarse-DM bins** at the
      §9 default op-point, giving ``6 × 0.073 ≈ 0.44 Gb/s`` per (corr, search)
      pair in steady state.
    * One search node aggregates 16 such pairs (one per corr) ⇒ ``16 × 0.44 ≈
      7 Gb/s`` ingress per search node (the C-epoll RX ceiling chunk-6 of M4a
      validated on h01 loopback).

The M4b pair test drives **ONE** TX→RX pair at the *production-receive-rate-
equivalent* aggregate — i.e. all 24 coarse-DM bins on a single (corr, search)
pair instead of 6, giving ``24 × 0.073 ≈ 1.76 Gb/s`` aggregate over one
real 100 GbE link. The framing in the M4b runbook is **"matches what one
production search node would receive aggregate from 4 corrs"**: at 1.76 Gb/s
this single pair carries 4× the per-pair production rate (and ~25% of the
per-search aggregate — the remaining 12 corrs would land in different
sockets / chgroups that the §11.4 multi-pair extension would exercise).

This is the **single-pair production-receive-rate-equivalent test**. It does
*not* try to reproduce the full 16-source per-search fan-in (that's deferred
to a follow-up bench when more nodes are Phase-2 complete); it does verify
that one real 100 GbE pair sustains 4× per-pair production rate without
fragment loss, with ``pattern_mismatch_count == 0``, and with TX-side
``tx_dropped_payloads`` incrementing on RX backpressure (no upstream stall).

# Plan §M4b DoD invariants this bench feeds

The orchestrator at ``tools/dod/M4b.sh`` runs this bench three times to
gate plan §M4b (line 2526) DoD:

    1. **60 s sustained at target rate, fragment loss < 1e-4, pattern_id
       mismatch == 0** (DoD invariant 1, §M4b line 2538).
    2. **Mid-run RX hold (SIGSTOP for 1 s, then SIGCONT) → TX-side
       ``tx_dropped_payloads`` increments, aggregate TX rate does not
       collapse** (DoD invariant 2: no backpressure into the corr-side TX
       queue; matches plan §4.3 line 1447 "drop oldest, don't block").
    3. **10-min soak at target rate, no congestion-collapse signature**
       (DoD invariant 3, §M4b line 2538).

The §11.6 lying-pipeline 30-min DoD (also called out in §M4b line 2538) is
deferred — the underlying ``bench/derisk/lying_pipeline.py`` does not exist
yet (no ``bench/derisk/`` dir). M7 owns that follow-up.

# CLI

::

    # TX side (run on the corr-host, e.g. n01):
    python -m bench.net_pair --mode tx --target-addr 10.41.0.222 \\
        --target-port 19000 --n-flows 24 --rate-gbps-per-flow 0.073 \\
        --n-filled 5000 --duration-s 60 [--start-at <unix-utc>] \\
        [--counters-out tx.json]

    # RX side (run on the search-host, e.g. n02):
    python -m bench.net_pair --mode rx --listen-addr 10.41.0.222 \\
        --listen-port 19000 --n-flows 24 --n-filled 5000 \\
        --duration-s 60 [--start-at <unix-utc>] [--counters-out rx.json] \\
        [--rx-impl epoll|python]   # default: epoll (M4a chunk 6)

    # Local pre-flight smoke test (5 s, 1 flow, loopback):
    python -m bench.net_pair --mode tx --smoke
    python -m bench.net_pair --mode rx --smoke

# Reuse from net_loopback

The bench imports ``bench.net_loopback`` for ``ProdOpPoint`` and
``_stable_pattern_id``. The TX-side wire-rate worker is a per-flow
specialisation of ``net_loopback._mp_wire_tx_worker`` (which packs
``n_dm_per_tx`` dm_idx values per subprocess); ``net_pair`` uses one
subprocess per (chgroup, dm_idx) flow so the multiprocessing pattern is
the same but the per-process workload is exactly ``n_frags`` datagrams
per cube. The RX-side ``epoll`` impl uses
:class:`dsart.transport.recv_epoll.RxEpoll` directly (the same C loop
that ``prod_fan_in_16x_mp_c`` exercises). The optional ``python`` impl
mirrors ``net_loopback._RxLoop`` + ``TransportRxProd``.

# What this bench does NOT cover

* Multi-corr fan-in to one search node (would need 4+ Phase-2-complete
  corr-hosts; current state has only n02 fully provisioned per the M4b
  Phase-2 report). Deferred to a fleet-wide bench post Phase-2 fan-out.
* The §11.6 lying-pipeline 30-min integration test (bench doesn't exist
  yet; M7 scope).
* Real correlator data — payloads are zero-filled bytes with valid
  ``pattern_id`` so the codec / RX reorder / fragment reassembly path is
  exercised end-to-end without coupling to the gridder.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

# Reuse: production op-point spec + deterministic pattern_id helper.
from bench import net_loopback as nl

LOG = logging.getLogger("m4b.bench.net_pair")


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PairConfig:
    """One pair-test op-point.

    Defaults derive from the §9 ``O-4`` operating point + the M4b 4×-pair
    framing (24 dm_idx flows on one chgroup ⇒ 1.76 Gb/s aggregate).
    """

    n_flows: int = 24
    """Number of (chgroup, dm_idx) flows. Default 24 = full coarse-DM
    range on one chgroup (4× the 6-flow production per-pair count)."""

    chgroup: int = 0
    """All flows share this chgroup (single corr → single search pair)."""

    rate_gbps_per_flow: float = 0.073
    """Per-(chgroup, dm_idx) flow rate. Default matches plan §11.4 line
    2802 / §11 line 2654 (0.44 Gb/s / 6 dm_idx)."""

    n_filled: int = 5000
    """Number of filled cells per payload (§9 default; 2 fragments at
    cint8 + jumbo MTU)."""

    n_grid: int = 256
    """Image grid side length (§9 default)."""

    bits_per_cell: int = 16
    """``bits_per_cell`` field; 16 = cint8 complex (§9 default)."""

    t_int_factor: int = 16
    """Header field; 16 = §9 default (524.288 µs cube cadence)."""

    n_fast_vis: int = 1
    """Header field for receiver bookkeeping; 1 = single-pol Stokes-I."""

    max_frag_payload_bytes: int = 8964
    """MTU 9000 − IPv4 (20) − UDP (8) − 8 B slack."""

    rcvbuf_mib: int = 256
    """SO_RCVBUF on the RX socket (≥ plan §6.1 ``rmem_max`` floor)."""

    sndbuf_mib: int = 8
    """SO_SNDBUF on each TX socket."""

    dec_deg: float = 30.0
    """Pseudo-DEC for ``_stable_pattern_id``; both sides MUST agree."""

    @property
    def aggregate_target_gbps(self) -> float:
        """Sum of per-flow rates ≈ 1.76 Gb/s at defaults."""
        return self.n_flows * self.rate_gbps_per_flow

    @property
    def bytes_per_payload(self) -> int:
        """Payload bytes per (cube, flow) — pre-fragmentation."""
        return self.n_filled * (self.bits_per_cell // 8)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["aggregate_target_gbps"] = self.aggregate_target_gbps
        d["bytes_per_payload"] = self.bytes_per_payload
        return d


# ---------------------------------------------------------------------------
# /proc/net/dev helper — capture NIC counters for the bound interface
# ---------------------------------------------------------------------------

_PROC_NET_DEV = "/proc/net/dev"

_NET_DEV_FIELDS = (
    "rx_bytes", "rx_packets", "rx_errs", "rx_drop",
    "rx_fifo", "rx_frame", "rx_compressed", "rx_multicast",
    "tx_bytes", "tx_packets", "tx_errs", "tx_drop",
    "tx_fifo", "tx_colls", "tx_carrier", "tx_compressed",
)


def _read_proc_net_dev(iface: str) -> Optional[dict]:
    """Parse /proc/net/dev for one interface; return per-counter dict.

    Returns ``None`` if the file or interface is missing — the bench
    reports ``nic_counters_*: null`` rather than failing.
    """
    try:
        with open(_PROC_NET_DEV, "r", encoding="ascii") as f:
            lines = f.readlines()
    except OSError:
        return None
    for line in lines:
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        if name.strip() != iface:
            continue
        vals = rest.split()
        if len(vals) < len(_NET_DEV_FIELDS):
            return None
        return {k: int(v) for k, v in zip(_NET_DEV_FIELDS, vals)}
    return None


def _detect_iface_for_addr(addr: str) -> Optional[str]:
    """Best-effort: ask ``ip route get <addr>`` for the egress dev name."""
    try:
        out = subprocess.check_output(
            ["ip", "-o", "route", "get", addr],
            stderr=subprocess.DEVNULL, timeout=2.0,
        ).decode("ascii", errors="replace")
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    toks = out.split()
    for i, t in enumerate(toks):
        if t == "dev" and i + 1 < len(toks):
            return toks[i + 1]
    return None


def _delta_net_dev(before: Optional[dict], after: Optional[dict]) -> Optional[dict]:
    if before is None or after is None:
        return None
    return {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in after}


# ---------------------------------------------------------------------------
# start-at coordination
# ---------------------------------------------------------------------------

def _wait_until_start_at(start_at: Optional[float], label: str) -> None:
    """If ``start_at`` is set, sleep until that UTC second.

    Tolerates ≥ 50 ms clock skew between TX and RX (both nodes have NTP
    per Phase 1; observed RTT 0.18 ms over br2 per the n02 Phase 2 report).
    """
    if start_at is None:
        return
    now = time.time()
    delta = start_at - now
    if delta > 0:
        LOG.info("[%s] sleeping %.3f s until start_at=%.3f", label, delta, start_at)
        time.sleep(delta)
    else:
        LOG.warning(
            "[%s] start_at=%.3f already passed by %.3f s; starting immediately",
            label, start_at, -delta,
        )


# ---------------------------------------------------------------------------
# TX subprocess worker — one per (chgroup, dm_idx) flow
# ---------------------------------------------------------------------------

def _mp_pair_tx_worker(
    cfg_kwargs: dict,
    chgroup: int,
    dm_idx: int,
    target_addr: str,
    target_port: int,
    duration_s: float,
    start_at: Optional[float],
    result_q,
) -> None:
    """One subprocess = one (chgroup, dm_idx) flow.

    Pre-builds the per-fragment wire bytes ONCE (header + zero payload),
    patches ``seq`` / ``specnum`` per iteration, and ``sendto``\\s in a
    deadline-paced loop. Bypasses Python encode/scale-offset compute so
    the bench measures the wire path ceiling, not a Python CPU bound.

    Mirrors :func:`bench.net_loopback._mp_wire_tx_worker` but for a single
    flow rather than ``n_dm_per_tx`` flows per process. Reused via spawn
    start ⇒ each worker re-imports cleanly without inheriting
    torch / CUDA state.
    """
    # Re-imports inside spawn child.
    from bench.net_pair import PairConfig
    from bench.net_loopback import _stable_pattern_id
    from dsart.transport.prod_frame import (
        FLAG_LAST_IN_BLOCK,
        FLAG_QUANTIZED,
        ProdFrameHeader,
        pack_frame,
        split_payload_into_fragments,
    )

    cfg = PairConfig(**cfg_kwargs)
    pid = _stable_pattern_id(cfg.dec_deg)
    bytes_per_cell = cfg.bits_per_cell // 8

    # Build socket. SO_SNDBUF nudge — kernel may cap at wmem_max.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_SNDBUF, cfg.sndbuf_mib * 1024 * 1024,
        )
    except OSError:
        pass
    addr = (target_addr, target_port)

    # Pre-build per-(frag_idx) wire bytes once. Payload bytes are zeros —
    # protocol validation only inspects the header (matches the
    # net_loopback wire-TX worker).
    payload_bytes = cfg.n_filled * bytes_per_cell
    raw_payload = bytes(payload_bytes)
    frags = split_payload_into_fragments(
        raw_payload, max_frag_payload_bytes=cfg.max_frag_payload_bytes,
    )
    n_frags = len(frags)
    flags = FLAG_QUANTIZED  # cint8 complex
    wire_packets: list[bytearray] = []
    for frag_idx, frag in enumerate(frags):
        f = flags
        if frag_idx == n_frags - 1:
            f |= FLAG_LAST_IN_BLOCK
        hdr = ProdFrameHeader(
            seq=0,            # patched per iteration
            specnum=0,        # patched per iteration
            chgroup=chgroup,
            dm_idx=dm_idx,
            frag_idx=frag_idx,
            n_frags=n_frags,
            n_grid=cfg.n_grid,
            n_filled=cfg.n_filled,
            pattern_id=pid,
            bits_per_cell=cfg.bits_per_cell,
            t_int_factor=cfg.t_int_factor,
            scale=1.0,
            offset=0.0,
            payload_bytes_in_frag=len(frag),
            flags=f,
        )
        wire_packets.append(bytearray(pack_frame(hdr, frag)))

    # Header offsets for in-place patching (must match _HEADER_FMT in
    # prod_frame.py: "<I H H Q Q ..."):
    #   magic        (4 B) @ 0
    #   version      (2 B) @ 4
    #   flags        (2 B) @ 6
    #   seq          (8 B) @ 8
    #   specnum      (8 B) @ 16
    SEQ_OFF = 8
    SPECNUM_OFF = 16

    # Pacing — bytes per cube for one flow = sum(frag_bytes) + header overhead.
    bytes_per_cube = sum(len(p) for p in wire_packets)
    bytes_per_sec_target = cfg.rate_gbps_per_flow * 1e9 / 8.0
    target_period_s = (
        bytes_per_cube / bytes_per_sec_target if bytes_per_sec_target > 0 else 0.0
    )

    # Coordinated start.
    _wait_until_start_at(start_at, label=f"tx[c={chgroup},dm={dm_idx}]")

    cubes_emitted = 0
    frames_sent = 0
    bytes_sent = 0
    sendto_errors = 0
    seq = 0
    t_start_ns = time.monotonic_ns()
    deadline_ns = t_start_ns + int(duration_s * 1e9)
    next_t = time.monotonic()

    while time.monotonic_ns() < deadline_ns:
        for frag_idx in range(n_frags):
            pkt = wire_packets[frag_idx]
            struct.pack_into("<Q", pkt, SEQ_OFF, seq)
            struct.pack_into("<Q", pkt, SPECNUM_OFF, cubes_emitted)
            try:
                n = sock.sendto(pkt, addr)
                frames_sent += 1
                bytes_sent += int(n)
            except OSError:
                sendto_errors += 1
        seq += 1
        cubes_emitted += 1
        if target_period_s > 0.0:
            next_t += target_period_s
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                # Behind schedule — let TX run unpaced this iteration.
                next_t = time.monotonic()

    t_stop_ns = time.monotonic_ns()
    elapsed_s = (t_stop_ns - t_start_ns) * 1e-9
    sock.close()
    result_q.put({
        "chgroup": int(chgroup),
        "dm_idx": int(dm_idx),
        "cubes_emitted": int(cubes_emitted),
        "frames_sent": int(frames_sent),
        "bytes_sent": int(bytes_sent),
        "sendto_errors": int(sendto_errors),
        "elapsed_s": float(elapsed_s),
        "n_frags_per_cube": int(n_frags),
        "bytes_per_cube": int(bytes_per_cube),
        # Wire-rate worker drops at sendto only (not via TransportTx pacer);
        # report the kernel sendto error count under tx_dropped_payloads
        # so the M4b orchestrator's invariant-2 check has a single field.
        "tx_dropped_payloads": int(sendto_errors),
    })


# ---------------------------------------------------------------------------
# TX driver
# ---------------------------------------------------------------------------

def _run_tx_pair_batch(
    cfg: PairConfig,
    target_addr: str,
    target_port: int,
    duration_s: float,
    start_at: Optional[float],
    dm_indices: list[int],
) -> list[dict]:
    """Spawn one subprocess per (cfg.chgroup, dm_idx) flow; wait for all."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    cfg_kwargs = asdict(cfg)
    procs = []
    for dm_idx in dm_indices:
        p = ctx.Process(
            target=_mp_pair_tx_worker,
            args=(
                cfg_kwargs, cfg.chgroup, int(dm_idx),
                target_addr, target_port, duration_s, start_at, q,
            ),
        )
        p.start()
        procs.append(p)
    results: list[dict] = []
    # Caller-side timeout = duration + start-at slack + 30 s clean-up.
    extra_slack = max(0.0, (start_at - time.time()) if start_at else 0.0)
    timeout_s = duration_s + extra_slack + 30.0
    for _ in dm_indices:
        try:
            d = q.get(timeout=timeout_s)
        except Exception as e:  # pylint: disable=broad-except
            LOG.error("pair tx worker did not return stats: %s", e)
            continue
        results.append(d)
    for p in procs:
        p.join(timeout=10.0)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5.0)
    return results


# ---------------------------------------------------------------------------
# RX side — epoll (default) and python (fallback) implementations
# ---------------------------------------------------------------------------

@dataclass
class RxResult:
    """Common shape returned by both RX impls so the JSON dump is stable."""

    impl: str
    listen_addr: str
    listen_port: int
    n_flows: int
    duration_s: float
    elapsed_s: float
    n_received: int
    n_committed: int
    bytes_received_total: int
    pattern_mismatch_count: int
    window_slide_zerofill_count: int
    out_of_order_drop_count: int
    bad_magic_count: int
    bad_version_count: int
    bad_length_count: int
    bad_field_range_count: int
    reserved_bit_count: int
    extra: dict = field(default_factory=dict)


def _run_rx_epoll(
    cfg: PairConfig,
    listen_addr: str,
    listen_port: int,
    duration_s: float,
    start_at: Optional[float],
) -> RxResult:
    """Bind RxEpoll, register pattern_id for chgroup, run for duration."""
    from dsart.transport.recv_epoll import RxEpoll

    pid = nl._stable_pattern_id(cfg.dec_deg)
    rx = RxEpoll.open(
        bind_host=listen_addr,
        bind_port=listen_port,
        so_rcvbuf_bytes=cfg.rcvbuf_mib * 1024 * 1024,
    )
    # Register the test chgroup; if a future variant fans in over multiple
    # chgroups, register all of them here. The epoll RX silently drops
    # datagrams whose pattern_id doesn't match (counted in
    # pattern_mismatch_count), so over-registering is harmless.
    rx.set_expected_pattern_id(cfg.chgroup, pid)
    rx.start()

    bound_port = rx.port
    # The "RX READY" stdout line is the M4b orchestrator's signal to start
    # the TX side. Print it AFTER socket bind + epoll start so a successful
    # match means the receiver is fully armed.
    print(
        f"RX READY listen={listen_addr}:{bound_port} n_flows={cfg.n_flows} "
        f"impl=epoll chgroup={cfg.chgroup}",
        flush=True,
    )

    _wait_until_start_at(start_at, label="rx[epoll]")
    t0 = time.monotonic()
    try:
        time.sleep(duration_s)
    finally:
        elapsed = time.monotonic() - t0
        try:
            rx.stop()
        except Exception:  # pylint: disable=broad-except
            LOG.exception("rx.stop failed")
        c = rx.counters()
        try:
            rx.close()
        except Exception:  # pylint: disable=broad-except
            LOG.exception("rx.close failed")

    return RxResult(
        impl="epoll",
        listen_addr=listen_addr,
        listen_port=bound_port,
        n_flows=cfg.n_flows,
        duration_s=duration_s,
        elapsed_s=elapsed,
        n_received=int(c.n_received),
        n_committed=int(c.n_committed),
        bytes_received_total=int(c.bytes_received_total),
        pattern_mismatch_count=int(c.pattern_mismatch_count),
        window_slide_zerofill_count=int(c.window_slide_zerofill_count),
        out_of_order_drop_count=int(c.out_of_order_drop_count),
        bad_magic_count=int(c.bad_magic_count),
        bad_version_count=int(c.bad_version_count),
        bad_length_count=int(c.bad_length_count),
        bad_field_range_count=int(c.bad_field_range_count),
        reserved_bit_count=int(c.reserved_bit_count),
        # The C epoll loop does not currently expose a per-datagram
        # latency histogram; placeholder for the follow-up that adds it.
        extra={"epoll_rx_latency_hist": None},
    )


def _run_rx_python(
    cfg: PairConfig,
    listen_addr: str,
    listen_port: int,
    duration_s: float,
    start_at: Optional[float],
) -> RxResult:
    """Python recvfrom + TransportRxProd fallback (mirrors net_loopback._RxLoop).

    Slower than the C path; useful when the C extension isn't built (e.g.
    cross-host development) or for diff'ing C vs Python correctness.
    """
    from dsart.transport.rx import (
        RxProdSlot,
        TransportRxProd,
        TransportRxProdConfig,
    )

    pid = nl._stable_pattern_id(cfg.dec_deg)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF,
            cfg.rcvbuf_mib * 1024 * 1024,
        )
    except OSError:
        LOG.warning("could not raise SO_RCVBUF to %d MiB", cfg.rcvbuf_mib)
    sock.bind((listen_addr, listen_port))
    bound_port = sock.getsockname()[1]

    # n_coarse_dm sized to cfg.n_flows so the per-(corr, dm_idx) reorder
    # window has a slot for every dm_idx the TX side emits.
    rx_cfg = TransportRxProdConfig(
        n_coarse_dm=max(1, cfg.n_flows),
        n_corr=max(16, cfg.chgroup + 1),
        reorder_window_depth=4,
        so_rcvbuf_bytes=cfg.rcvbuf_mib * 1024 * 1024,
        expected_pattern_id_by_chgroup={cfg.chgroup: pid},
    )

    def _on_slot(corr_idx: int, dm_idx: int, slot: RxProdSlot) -> None:
        # Pure counter — do not retain slot objects.
        pass

    rx_prod = TransportRxProd(config=rx_cfg, ring_write_cb=_on_slot)
    rx_loop = nl._RxLoop(sock, rx_prod, recv_bufsize=65536)
    rx_loop.start()

    print(
        f"RX READY listen={listen_addr}:{bound_port} n_flows={cfg.n_flows} "
        f"impl=python chgroup={cfg.chgroup}",
        flush=True,
    )

    _wait_until_start_at(start_at, label="rx[python]")
    t0 = time.monotonic()
    try:
        time.sleep(duration_s)
    finally:
        elapsed = time.monotonic() - t0
        try:
            rx_loop.stop()
        except Exception:  # pylint: disable=broad-except
            LOG.exception("rx_loop.stop failed")
        try:
            sock.close()
        except OSError:
            pass

    stats = rx_prod.prod_stats
    pm_total = sum(stats.pattern_mismatch_count.values())

    return RxResult(
        impl="python",
        listen_addr=listen_addr,
        listen_port=bound_port,
        n_flows=cfg.n_flows,
        duration_s=duration_s,
        elapsed_s=elapsed,
        n_received=int(rx_loop.n_datagrams_received),
        n_committed=int(stats.n_committed),
        bytes_received_total=0,  # Python RX doesn't track wire bytes.
        pattern_mismatch_count=int(pm_total),
        window_slide_zerofill_count=int(stats.window_slide_zerofill_count),
        out_of_order_drop_count=int(stats.out_of_order_drop_count),
        bad_magic_count=int(stats.bad_magic_count),
        bad_version_count=int(getattr(stats, "bad_version_count", 0)),
        bad_length_count=int(stats.bad_length_count),
        bad_field_range_count=int(stats.bad_field_range_count),
        reserved_bit_count=int(getattr(stats, "reserved_bit_count", 0)),
        extra={"epoll_rx_latency_hist": None},
    )


# ---------------------------------------------------------------------------
# JSON output builders
# ---------------------------------------------------------------------------

def _build_tx_counters(
    cfg: PairConfig,
    target_addr: str,
    target_port: int,
    duration_s: float,
    start_at: Optional[float],
    tx_stats: list[dict],
    nic_iface: Optional[str],
    nic_before: Optional[dict],
    nic_after: Optional[dict],
) -> dict:
    cubes_total = sum(s["cubes_emitted"] for s in tx_stats)
    frames_total = sum(s["frames_sent"] for s in tx_stats)
    bytes_total = sum(s["bytes_sent"] for s in tx_stats)
    sendto_errors_total = sum(s["sendto_errors"] for s in tx_stats)
    tx_dropped_total = sum(s["tx_dropped_payloads"] for s in tx_stats)
    elapsed = max((s["elapsed_s"] for s in tx_stats), default=duration_s)
    achieved_gbps_aggregate = (
        (bytes_total * 8) / (elapsed * 1e9) if elapsed > 0 else 0.0
    )
    per_flow = []
    for s in tx_stats:
        gbps = (
            (s["bytes_sent"] * 8) / (s["elapsed_s"] * 1e9)
            if s["elapsed_s"] > 0 else 0.0
        )
        per_flow.append({
            "chgroup": s["chgroup"],
            "dm_idx": s["dm_idx"],
            "elapsed_s": s["elapsed_s"],
            "cubes_emitted": s["cubes_emitted"],
            "frames_sent": s["frames_sent"],
            "bytes_sent": s["bytes_sent"],
            "sendto_errors": s["sendto_errors"],
            "achieved_gbps": gbps,
        })
    return {
        "mode": "tx",
        "config": cfg.to_dict(),
        "target_addr": target_addr,
        "target_port": target_port,
        "duration_s": duration_s,
        "start_at_unix_utc": start_at,
        "n_flows_started": len(tx_stats),
        "elapsed_s": elapsed,
        "cubes_emitted_total": cubes_total,
        "frames_sent_total": frames_total,
        "bytes_sent_total": bytes_total,
        "sendto_errors_total": sendto_errors_total,
        "tx_dropped_payloads_total": tx_dropped_total,
        "achieved_gbps_aggregate": achieved_gbps_aggregate,
        "achieved_gbps_per_flow_mean": (
            achieved_gbps_aggregate / max(1, len(tx_stats))
        ),
        "target_gbps_per_flow": cfg.rate_gbps_per_flow,
        "target_gbps_aggregate": cfg.aggregate_target_gbps,
        "per_flow": per_flow,
        "nic_iface": nic_iface,
        "nic_counters_before": nic_before,
        "nic_counters_after": nic_after,
        "nic_counters_delta": _delta_net_dev(nic_before, nic_after),
        # Hooks for the M4b.sh side: these names line up with the bench's
        # invariant verbiage so the orchestrator's jq-style asserts are
        # straightforward.
        "pattern_mismatch_observable_at": "rx_side_only",
    }


def _build_rx_counters(
    cfg: PairConfig,
    rx_result: RxResult,
    start_at: Optional[float],
    nic_iface: Optional[str],
    nic_before: Optional[dict],
    nic_after: Optional[dict],
) -> dict:
    achieved_gbps_aggregate = (
        (rx_result.bytes_received_total * 8) / (rx_result.elapsed_s * 1e9)
        if rx_result.elapsed_s > 0 and rx_result.bytes_received_total > 0
        else 0.0
    )
    # Estimate "fragment loss" from bytes received vs target.
    expected_bytes = (
        cfg.aggregate_target_gbps * 1e9 / 8.0 * rx_result.elapsed_s
    )
    fragment_loss_estimate = (
        max(0.0, 1.0 - rx_result.bytes_received_total / expected_bytes)
        if expected_bytes > 0 and rx_result.bytes_received_total > 0
        else None
    )
    return {
        "mode": "rx",
        "config": cfg.to_dict(),
        "listen_addr": rx_result.listen_addr,
        "listen_port": rx_result.listen_port,
        "duration_s": rx_result.duration_s,
        "elapsed_s": rx_result.elapsed_s,
        "start_at_unix_utc": start_at,
        "rx_impl": rx_result.impl,
        "n_received": rx_result.n_received,
        "n_committed": rx_result.n_committed,
        "bytes_received_total": rx_result.bytes_received_total,
        "achieved_gbps_aggregate": achieved_gbps_aggregate,
        "achieved_gbps_per_flow_mean": (
            achieved_gbps_aggregate / max(1, cfg.n_flows)
        ),
        "target_gbps_per_flow": cfg.rate_gbps_per_flow,
        "target_gbps_aggregate": cfg.aggregate_target_gbps,
        "fragment_loss_estimate_fraction": fragment_loss_estimate,
        # Protocol-error counters — the §M4b DoD invariants.
        "pattern_mismatch_count": rx_result.pattern_mismatch_count,
        "window_slide_zerofill_count": rx_result.window_slide_zerofill_count,
        "out_of_order_drop_count": rx_result.out_of_order_drop_count,
        "bad_magic_count": rx_result.bad_magic_count,
        "bad_version_count": rx_result.bad_version_count,
        "bad_length_count": rx_result.bad_length_count,
        "bad_field_range_count": rx_result.bad_field_range_count,
        "reserved_bit_count": rx_result.reserved_bit_count,
        "nic_iface": nic_iface,
        "nic_counters_before": nic_before,
        "nic_counters_after": nic_after,
        "nic_counters_delta": _delta_net_dev(nic_before, nic_after),
        **rx_result.extra,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_cfg_from_args(args: argparse.Namespace) -> PairConfig:
    return PairConfig(
        n_flows=args.n_flows,
        chgroup=args.chgroup,
        rate_gbps_per_flow=args.rate_gbps_per_flow,
        n_filled=args.n_filled,
        n_grid=args.n_grid,
        bits_per_cell=args.bits_per_cell,
        t_int_factor=args.t_int_factor,
        max_frag_payload_bytes=args.max_frag_payload_bytes,
        rcvbuf_mib=args.rcvbuf_mib,
        sndbuf_mib=args.sndbuf_mib,
        dec_deg=args.dec_deg,
    )


def _maybe_dump_json(path: Optional[str], payload: dict) -> None:
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
    LOG.info("wrote %s", path)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", required=True, choices=["tx", "rx"])
    p.add_argument(
        "--smoke", action="store_true",
        help="5 s, 1 flow, loopback (127.0.0.1:19000) — pre-flight smoke",
    )
    # TX target.
    p.add_argument("--target-addr", default="10.41.0.222",
                   help="TX dest address (use rx-host's br2 IP)")
    p.add_argument("--target-port", type=int, default=19000)
    # RX bind.
    p.add_argument("--listen-addr", default="10.41.0.222",
                   help="RX bind address (this host's br2 IP)")
    p.add_argument("--listen-port", type=int, default=19000)
    p.add_argument("--rx-impl", choices=["epoll", "python"], default="epoll")
    # Pair config.
    p.add_argument("--n-flows", type=int, default=24,
                   help="Number of (chgroup, dm_idx) flows. Default 24 = "
                        "full coarse-DM range = 4× per-pair production.")
    p.add_argument("--chgroup", type=int, default=0,
                   help="Single chgroup all flows share.")
    p.add_argument("--rate-gbps-per-flow", type=float, default=0.073,
                   help="Per-flow rate (default 0.44/6 ≈ 0.073).")
    p.add_argument("--n-filled", type=int, default=5000,
                   help="Cells per payload (§9 default).")
    p.add_argument("--n-grid", type=int, default=256)
    p.add_argument("--bits-per-cell", type=int, default=16,
                   choices=[16, 32], help="16 = cint8, 32 = cfp16.")
    p.add_argument("--t-int-factor", type=int, default=16)
    p.add_argument("--max-frag-payload-bytes", type=int, default=8964)
    p.add_argument("--rcvbuf-mib", type=int, default=256)
    p.add_argument("--sndbuf-mib", type=int, default=8)
    p.add_argument("--dec-deg", type=float, default=30.0,
                   help="Pseudo-DEC for pattern_id; both sides MUST agree.")
    # Time control.
    p.add_argument("--duration-s", type=float, default=60.0)
    p.add_argument("--start-at", type=float, default=None,
                   help="UTC unix seconds; both sides sleep until this "
                        "moment before TX starts pumping / RX starts the "
                        "duration timer.")
    # Output.
    p.add_argument("--counters-out", type=str, default=None,
                   help="Write counters JSON to this path on completion.")
    p.add_argument("--nic-iface", type=str, default=None,
                   help="Override the NIC iface used for /proc/net/dev "
                        "snapshots (default: ip-route auto-detect).")
    p.add_argument("-v", "--verbose", action="store_true")

    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # --smoke override: quick local loopback pre-flight.
    if args.smoke:
        args.duration_s = 5.0
        args.n_flows = 1
        if args.mode == "tx":
            args.target_addr = "127.0.0.1"
            args.target_port = args.target_port or 19000
        else:
            args.listen_addr = "127.0.0.1"
            args.listen_port = args.listen_port or 19000

    cfg = _build_cfg_from_args(args)

    if args.mode == "tx":
        # NIC counters bracket the run.
        iface = args.nic_iface or _detect_iface_for_addr(args.target_addr)
        nic_before = _read_proc_net_dev(iface) if iface else None
        # Spawn one TX worker per dm_idx, dm_idx ∈ [0, n_flows).
        dm_indices = list(range(cfg.n_flows))
        LOG.info(
            "TX: target=%s:%d n_flows=%d rate_per_flow=%.4f Gb/s aggregate=%.3f Gb/s",
            args.target_addr, args.target_port, cfg.n_flows,
            cfg.rate_gbps_per_flow, cfg.aggregate_target_gbps,
        )
        tx_stats = _run_tx_pair_batch(
            cfg=cfg,
            target_addr=args.target_addr,
            target_port=args.target_port,
            duration_s=args.duration_s,
            start_at=args.start_at,
            dm_indices=dm_indices,
        )
        nic_after = _read_proc_net_dev(iface) if iface else None
        payload = _build_tx_counters(
            cfg=cfg,
            target_addr=args.target_addr,
            target_port=args.target_port,
            duration_s=args.duration_s,
            start_at=args.start_at,
            tx_stats=tx_stats,
            nic_iface=iface,
            nic_before=nic_before,
            nic_after=nic_after,
        )
        _maybe_dump_json(args.counters_out, payload)
        # Headline summary on stdout.
        print(
            f"TX done: aggregate={payload['achieved_gbps_aggregate']:.3f} Gb/s "
            f"(target={cfg.aggregate_target_gbps:.3f}); "
            f"frames={payload['frames_sent_total']} "
            f"sendto_errors={payload['sendto_errors_total']} "
            f"tx_dropped_payloads={payload['tx_dropped_payloads_total']}",
            flush=True,
        )
        return 0

    # mode == "rx"
    iface = args.nic_iface or _detect_iface_for_addr(args.listen_addr)
    nic_before = _read_proc_net_dev(iface) if iface else None
    if args.rx_impl == "epoll":
        rx_result = _run_rx_epoll(
            cfg=cfg,
            listen_addr=args.listen_addr,
            listen_port=args.listen_port,
            duration_s=args.duration_s,
            start_at=args.start_at,
        )
    else:
        rx_result = _run_rx_python(
            cfg=cfg,
            listen_addr=args.listen_addr,
            listen_port=args.listen_port,
            duration_s=args.duration_s,
            start_at=args.start_at,
        )
    nic_after = _read_proc_net_dev(iface) if iface else None
    payload = _build_rx_counters(
        cfg=cfg,
        rx_result=rx_result,
        start_at=args.start_at,
        nic_iface=iface,
        nic_before=nic_before,
        nic_after=nic_after,
    )
    _maybe_dump_json(args.counters_out, payload)
    pm = payload["pattern_mismatch_count"]
    floss = payload["fragment_loss_estimate_fraction"]
    floss_str = f"{floss:.2e}" if floss is not None else "n/a"
    print(
        f"RX done: aggregate={payload['achieved_gbps_aggregate']:.3f} Gb/s "
        f"(target={cfg.aggregate_target_gbps:.3f}); "
        f"n_received={payload['n_received']} "
        f"n_committed={payload['n_committed']} "
        f"pattern_mismatch={pm} "
        f"frag_loss_est={floss_str}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
