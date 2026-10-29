#!/usr/bin/env python
"""Synthetic corr -> search UDP: the M0 v1 loopback stub and the x2 v2 benches.

v1 (unchanged; no subcommand)::

    fake_corr_to_search.py [--host H] [--port-base P] [--chgroup G] [--self-test]

x2 frame v2 (CP1 runner contract, docs/x2/CP1_RUNBOOK.md; the DRIVER runs on
h23 and starts marked workers on the nodes over ssh)::

    sweep      --frame-version 2 --rates-gbps 9,11,..,18 --dwell-s 120
               [--record-pps] [--nic-counters] DRV      -> clean_gbps_per_node
    fleet-load --format 2x-... --load-factor 1.1 --plan <node path> --engine '<cmd>'
               [--minutes M] DRV                        -> search_rx v4 + owner engines
    snap-live  --sender c --format 2x-... [--minutes M] DRV
    fleet-replay --format 2x-... --plan <search node path> --plan-id 0x... [--replay 250924mptq]
               [--minutes M] DRV                        -> G1 full chain: the REAL 2x corr
               (bench/timeres/b8_corr2x.py live on replayed voltages, frame v2, C sender) on every
               corr node into search_rx v4 + owner engines; every node process under
               tools/x2/prod_guard.sh; fleet_replay.json + fleet_replay.md in --out (a directory)
    replay-report --in fleet_replay.json --out DIR      -> re-evaluate a fleet-replay result
    teardown   DRV [--corr-replay]                      -> kill marked workers, rm test shm
    DRV = --tx-nodes n03,... --rx-nodes n01,... --scratch <node scratch> --marker <tag>
          --out <file.json | dir>

Exit codes: 0 every gate passed, 1 a gate failed (named in the JSON), 2 error.
Every node process carries ``DSART_X2_CP=<marker>`` in its environment and
every test shm is ``/dev/shm/dsart-x2-*``; the driver tears them down on exit.

Wire rules: frame v2 only, every datagram <= 8972 B (72 B header + <= 8900
B payload: never IP-fragmented; the 9036 B v1 default was the bogus "88%
loss on n01"). Rates are Gb/s of UDP payload per SEARCH node (the JSON also
reports wire Gb/s = + 66 B per datagram for IP/UDP/Ethernet/preamble/IFG).

Loss: per (corr -> search) pair from the sender's ``sent`` counters and the
v4 ring's per-flow ``datagrams``; a search node is CLEAN at a rate when its
loss <= max(1e-4, its 1x baseline loss) and its NIC discard counters
(rx_discards_phy, rx_out_of_buffer) did not increase during the dwell. The
1x baseline comes from ``--baseline-loss`` (``n01=3e-4,n02=7e-4`` or a JSON
file ``{"n01": 3e-4}``); otherwise the node's own rx_crc_errors_phy /
rx_packets_phy fraction during the dwell stands in for its PHY loss.

Node workers (started by the driver; also usable by hand / in tests)::

    tx  ...  one corr node: generates frame-v2 datagrams, C sendmmsg sender
    rx  ...  one search node: RxEpoll v4 counting ring, per-step snapshots
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# =============================================================================
# v1 stub (M0 loopback) — unchanged behaviour
# =============================================================================

DSART_UDP_MAGIC = 0xD5A1107E
HEADER = struct.Struct("<IHHQQHHHHHHIQBBHffI8s")


def pack_datagram(
    *,
    seq: int,
    specnum: int,
    chgroup: int,
    dm_idx: int,
    frag_idx: int,
    n_frags: int,
    n_grid: int,
    n_filled: int,
    pattern_id: int,
    bits_per_cell: int,
    t_int_factor: int,
    scale: float,
    offset: float,
    payload: bytes,
) -> bytes:
    if len(payload) > 8964:
        raise ValueError("fragment payload exceeds conservative MTU headroom")
    hdr = HEADER.pack(
        DSART_UDP_MAGIC, 1, 0, seq, specnum, chgroup, dm_idx, frag_idx, n_frags, n_grid, 0,
        n_filled, pattern_id, bits_per_cell, t_int_factor, 0, scale, offset, len(payload),
        b"\0" * 8,
    )
    return hdr + payload


def verify_packet(pkt: bytes, expect_chgroup: int) -> None:
    if len(pkt) < HEADER.size:
        raise ValueError(f"short packet {len(pkt)}")
    fields = HEADER.unpack(pkt[: HEADER.size])
    magic, version, chgroup = fields[0], fields[1], fields[5]
    pay_len = fields[18]
    if magic != DSART_UDP_MAGIC:
        raise ValueError(f"bad magic {magic:#x}")
    if version != 1:
        raise ValueError(f"bad version {version}")
    if chgroup != expect_chgroup:
        raise ValueError(f"chgroup mismatch got {chgroup} expected {expect_chgroup}")
    if len(pkt) != HEADER.size + pay_len:
        raise ValueError("payload length mismatch")


def send_one_block(*, host: str, port: int, chgroup: int, sleep_s: float) -> bytes:
    payload = (0).to_bytes(2, "little", signed=True) * 8
    pkt = pack_datagram(seq=1, specnum=2048, chgroup=chgroup, dm_idx=0, frag_idx=0, n_frags=1,
                        n_grid=256, n_filled=8, pattern_id=0xCAFEBABE_DEADBEEF, bits_per_cell=8,
                        t_int_factor=8, scale=1.0, offset=0.0, payload=payload)
    if sleep_s > 0:
        time.sleep(sleep_s)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tx:
        tx.sendto(pkt, (host, port))
    return pkt


def main_v1(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="M0 v1 loopback stub")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port-base", type=int, default=9000)
    ap.add_argument("--chgroup", type=int, default=0)
    ap.add_argument("--rate", default="native", choices=("native", "fast"))
    ap.add_argument("--self-test", action="store_true")
    ns = ap.parse_args(argv)
    port = ns.port_base + ns.chgroup
    pace = 134.218e-3 if ns.rate == "native" else 0.0
    if ns.self_test:
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.bind((ns.host, port))
        rx.settimeout(3.0)
        pkt_sent = send_one_block(host=ns.host, port=port, chgroup=ns.chgroup, sleep_s=pace)
        data, _addr = rx.recvfrom(65536)
        rx.close()
        verify_packet(data, ns.chgroup)
        print(f"fake_corr_to_search: self-test PASS chgroup={ns.chgroup} port={port} "
              f"bytes={len(data)} (sent={len(pkt_sent)})")
        return 0
    send_one_block(host=ns.host, port=port, chgroup=ns.chgroup, sleep_s=pace)
    print(f"fake_corr_to_search: sent 1 datagram to {ns.host}:{port} chgroup={ns.chgroup}")
    return 0


# =============================================================================
# x2 v2: shared constants
# =============================================================================

NATIVE_S = 32.768e-6
BLOCK_NATIVE = 4096
BLOCK_S = BLOCK_NATIVE * NATIVE_S
WIRE_OVERHEAD_B = 66                    # IP 20 + UDP 8 + Ethernet 18 + preamble 8 + IFG 12
MAX_DATAGRAM = 8972
PORT_BASE = 19100
PY_NODE = "/home/ubuntu/miniforge3/envs/dsa110-rt/bin/python"
#: Data-network addresses (CLAUDE.md); override with --ip-map n01=10.41.0.205,...
IP_MAP = {
    "n01": "10.41.0.205", "n02": "10.41.0.222", "n09": "10.41.0.253", "n13": "10.41.0.238",
    "n03": "10.41.0.224", "n04": "10.41.0.138", "n05": "10.41.0.206", "n06": "10.41.0.203",
    "n07": "10.41.0.208", "n08": "10.41.0.209", "n10": "10.41.0.211", "n11": "10.41.0.216",
    "n12": "10.41.0.218", "n14": "10.41.0.212", "n15": "10.41.0.223", "n16": "10.41.0.220",
    "n18": "10.41.0.228", "n19": "10.41.0.227", "n21": "10.41.0.250", "n22": "10.41.0.233",
    "localhost": "127.0.0.1",
}
#: Synthetic sweep geometry: 8 coarse DMs (half d % 8 -> rx node (d % 8) // 2), n_sub 2.
SWEEP_DMS = tuple(range(8))
SWEEP_N_SUB = 2
NIC_KEYS = ("rx_packets_phy", "rx_bytes_phy", "rx_discards_phy", "rx_out_of_buffer",
            "rx_crc_errors_phy", "tx_packets_phy", "tx_discards_phy", "tx_pause_ctrl_phy",
            "rx_pause_ctrl_phy")
#: fleet-replay: the real corr routes dm -> SEARCH_NODES[owner_half[dm] // 2] (dsart.x2.plan) and
#: sends to port_base + chgroup, chgroup = index of the corr node in CORR_NODES (b8_corr2x)
SEARCH_NODE_ORDER = ("n01", "n02", "n09", "n13")
CORR_NODE_ORDER = ("n03", "n04", "n05", "n06", "n07", "n08", "n10", "n11", "n12", "n14", "n15",
                   "n16", "n18", "n19", "n21", "n22")
#: the x2 TEST DADA key of bench/timeres/b8_corr2x.py live --replay, and its corr ready sentinel
REPLAY_KEY = "d2f0"
CORR_READY_SENTINEL = "/tmp/dsart-x2-corr-fast-live.ready"
#: array pointing (DEC phase centre) of a replayed fixture = b8_corr2x FIXTURE_POINTING_DEC_DEG: the
#: ring spec and the engines must build their value maps at the corr's DEC or every stream mismatches
REPLAY_POINTING_DEC = {"250924mptq": 54.5737}
BLOCK_MS = BLOCK_S * 1e3
CYCLE_S = 6144 * NATIVE_S


def guarded(cmd: str, timeout_s: float) -> str:
    """Rev 5 rule for node processes: ``cmd`` (a shell word string, run from the
    scratch copy) under tools/x2/prod_guard.sh in STOPPED mode (refuses if a
    production process runs, kills the run if one starts) with the timeout
    INSIDE the guarded command."""
    return f"env GUARD_REQUIRE_STOPPED=1 bash tools/x2/prod_guard.sh timeout {int(timeout_s)} {cmd}"


def _now() -> float:
    return time.time()


def _sleep_until(t: float) -> None:
    while True:
        d = t - time.time()
        if d <= 0:
            return
        time.sleep(min(d, 0.5))


def rx_node_of_dm(d: int, n_rx: int, owner_half=None) -> int:
    """Owner map: half = d % 8 (or the plan's owner_half[d]) -> node half // 2,
    folded onto the available rx nodes."""
    h = int(owner_half[d]) if owner_half is not None else d % 8
    return (h // 2) % n_rx


def detect_data_iface(ip: str) -> Optional[str]:
    """Interface holding ``ip`` (bridge -> physical member), like the sampler."""
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True,
                             text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[3].split("/")[0] == ip:
            iface = parts[1]
            brif = Path(f"/sys/class/net/{iface}/brif")
            if brif.is_dir():
                mem = sorted(p.name for p in brif.iterdir()
                             if p.name.startswith(("enp", "ens", "eth")))
                if mem:
                    iface = mem[0]
            return iface
    return None


def read_nic(iface: Optional[str]) -> dict:
    """NIC + UDP counters (read-only: ethtool -S, /proc/net/snmp)."""
    out: dict = {"t": time.time()}
    if iface:
        try:
            st = subprocess.run(["ethtool", "-S", iface], capture_output=True, text=True,
                                timeout=5).stdout
            for line in st.splitlines():
                k, _, v = line.strip().partition(":")
                if k in NIC_KEYS:
                    out[k] = int(v.strip())
        except Exception:  # noqa: BLE001
            pass
    try:
        lines = Path("/proc/net/snmp").read_text().splitlines()
        udp = [ln.split() for ln in lines if ln.startswith("Udp:")]
        if len(udp) >= 2:
            d = dict(zip(udp[0][1:], udp[1][1:]))
            out["udp_InErrors"] = int(d.get("InErrors", 0))
            out["udp_RcvbufErrors"] = int(d.get("RcvbufErrors", 0))
    except OSError:
        pass
    return out


def nic_delta(a: dict, b: dict) -> dict:
    return {k: int(b[k]) - int(a[k]) for k in b if k != "t" and k in a}


# =============================================================================
# Frame builders (frame v2; vectorised headers over a pooled payload)
# =============================================================================


class FlowFramer:
    """Datagrams of ONE (stream, dm) flow: k rows each, fixed payload pool,
    header fields seq_first / t0_native rewritten per datagram."""

    def __init__(self, *, stream: int, dm: int, n_sub: int, bits_main: int, n_values: int,
                 n_heavy: int, t_int_factor: int, pattern_id: int, n_grid: int, value_kind: int,
                 s_main: float, s_heavy: float, step_main: float, step_heavy: float,
                 payload_pool, k: int) -> None:
        import numpy as np
        from dsart.transport import prod_frame2 as f2

        self.np = np
        bh = 16 if n_heavy else 0
        self.bps = f2.bytes_per_sample(n_values, n_heavy, bits_main, bh)
        self.k = int(k)
        self.tif = int(t_int_factor)
        self.dlen = f2.HEADER_BYTES + self.k * self.bps
        if self.dlen > MAX_DATAGRAM:
            raise ValueError(f"datagram {self.dlen} B > {MAX_DATAGRAM}")
        flags = f2.FLAG_QUANTIZED | (f2.FLAG_HEAVY_PRESENT if n_heavy else 0)
        hdr = f2.Frame2Header(
            seq_first=0, t0_native=0, stream_id=stream, dm_idx=dm, k=self.k, n_sub=n_sub,
            bits_main=bits_main, bits_heavy=bh, n_grid=n_grid, t_int_factor=self.tif,
            value_kind=value_kind, n_values=n_values, pattern_id=pattern_id, n_heavy=n_heavy,
            step_main=step_main, step_heavy=step_heavy, payload_bytes=self.k * self.bps,
            s_main=s_main, s_heavy=s_heavy, flags=flags)
        self.template = np.frombuffer(f2.pack_header(hdr), np.uint8)
        pool = np.asarray(payload_pool, np.uint8).reshape(-1, self.bps)
        n_dg = max(1, pool.shape[0] // self.k)
        self.payloads = np.ascontiguousarray(pool[: n_dg * self.k].reshape(n_dg, self.k * self.bps))
        self._i = 0

    def build(self, seq_first: int, n_dg: int, out=None):
        """``n_dg`` consecutive datagrams from row ``seq_first`` -> uint8 [n, dlen]."""
        np = self.np
        buf = np.empty((n_dg, self.dlen), np.uint8) if out is None else out
        buf[:, :72] = self.template
        seqs = seq_first + self.k * np.arange(n_dg, dtype=np.uint64)
        buf[:, 8:16] = seqs.astype("<u8").view(np.uint8).reshape(n_dg, 8)
        buf[:, 16:24] = (seqs * np.uint64(self.tif)).astype("<u8").view(np.uint8).reshape(n_dg, 8)
        idx = (self._i + np.arange(n_dg)) % self.payloads.shape[0]
        self._i = int((self._i + n_dg) % self.payloads.shape[0])
        buf[:, 72:] = self.payloads[idx]
        return buf

    def block_parts(self, seq_first: int, rows: int):
        """One corr block of ``rows`` rows: ``(full [n, dlen] uint8, tail bytes)``;
        ceil(rows / k) datagrams, the last short if needed and LAST_IN_BLOCK —
        datagrams never straddle a block."""
        from dsart.transport import prod_frame2 as f2
        n_full = rows // self.k
        full = self.build(seq_first, n_full) if n_full else self.np.zeros((0, self.dlen), self.np.uint8)
        rem = rows - n_full * self.k
        tail = b""
        if rem:
            h = bytearray(self.template.tobytes())
            seq = seq_first + n_full * self.k
            struct.pack_into("<QQ", h, 8, seq, seq * self.tif)
            h[28] = rem
            struct.pack_into("<I", h, 60, rem * self.bps)
            fl = struct.unpack_from("<H", h, 6)[0] | f2.FLAG_LAST_IN_BLOCK
            struct.pack_into("<H", h, 6, fl)
            tail = bytes(h) + self.payloads[self._i % self.payloads.shape[0], : rem * self.bps].tobytes()
        elif n_full:
            fl = int(full[-1, 6]) | (int(full[-1, 7]) << 8) | f2.FLAG_LAST_IN_BLOCK
            full[-1, 6], full[-1, 7] = fl & 0xFF, fl >> 8
        return full, tail


def concat_parts(parts):
    """[(full [n, dlen], tail bytes), ...] -> (buf, offsets, lengths) for MmsgSender.submit."""
    import numpy as np

    chunks, lens = [], []
    for full, tail in parts:
        if full.shape[0]:
            chunks.append(full.reshape(-1))
            lens.extend([full.shape[1]] * full.shape[0])
        if tail:
            chunks.append(np.frombuffer(tail, np.uint8))
            lens.append(len(tail))
    lengths = np.asarray(lens, np.uint32)
    offsets = np.zeros(lengths.size, np.uint64)
    if lengths.size > 1:
        offsets[1:] = np.cumsum(lengths[:-1], dtype=np.uint64)
    buf = np.concatenate(chunks) if chunks else np.zeros(0, np.uint8)
    return buf, offsets, lengths


def synthetic_framer(stream: int, dm: int, frame_bytes: int, tif: int = 16) -> FlowFramer:
    """Max-size capacity frames: k = 1 row of ``frame_bytes - 72`` cint4 values."""
    import numpy as np
    nv = int(frame_bytes) - 72
    rng = np.random.default_rng(stream * 1000 + dm)
    pool = rng.integers(0, 256, size=(4, nv), dtype=np.uint8)
    return FlowFramer(stream=stream, dm=dm, n_sub=SWEEP_N_SUB, bits_main=8, n_values=nv,
                      n_heavy=0, t_int_factor=tif, pattern_id=0x5EED, n_grid=256, value_kind=0,
                      s_main=float(np.float32(0.3534)), s_heavy=float(np.float32(0.0773)),
                      step_main=0.3534, step_heavy=0.0773, payload_pool=pool, k=1)


def format_framers(fmt: str, chgroup: int, plan_path: str, antpos_npz: str, dec_deg: float,
                   dms=None, n_pool: int = 64, seed: int = 0) -> tuple[dict, dict]:
    """Realistic flows of one corr node (chgroup): WS-P value map + plan,
    row-rule codes of Gaussian noise. Returns ({(stream, dm): FlowFramer}, meta)."""
    import numpy as np
    from dsart.grid.baseline_grid import build_value_map_for_opoint
    from dsart.grid.sparsity_pattern import core_baseline_mask_from_station_numbers
    from dsart.transport import prod_frame2 as f2
    from dsart.x2 import quant as q
    from dsart.x2.opoint import opoint_from_id
    from dsart.x2.plan import TieredDmPlan

    op = opoint_from_id(fmt)
    plan = TieredDmPlan.from_npz(plan_path)
    z = np.load(antpos_npz)
    mask = core_baseline_mask_from_station_numbers(z["station"])
    # SPEC §3 production inputs: antpos_u = None, as corr_fast and the owner engine do
    vm = build_value_map_for_opoint(op, z["antpos_e"], z["antpos_n"],
                                    is_core_baseline_mask=mask, dec_deg=dec_deg)
    rule = q.RowRule.for_format(op.bits_main, op.n_sub, warmup_blocks=0)
    rng = np.random.default_rng(seed + chgroup)
    dms = list(range(len(plan.owner_half))) if dms is None else list(dms)
    flows = {}
    for sub in range(op.n_sub):
        s = chgroup * op.n_sub + sub
        sv = vm.streams[s]
        rq = q.RowQuantiser(rule, sv.n_contrib, sv.n_heavy)
        nn = np.sqrt(sv.n_contrib).astype(np.float32)[None, :]
        x = ((rng.normal(size=(n_pool, sv.n_values)) + 1j * rng.normal(size=(n_pool, sv.n_values)))
             .astype(np.complex64) * nn)
        bc = rq.process_block(x)
        pool = f2.pack_samples(bc.codes_heavy, bc.codes_main, op.bits_main)
        bps = pool.shape[1]
        k = f2.max_k(bps)
        for d in dms:
            flows[(s, d)] = FlowFramer(
                stream=s, dm=d, n_sub=op.n_sub, bits_main=op.bits_main, n_values=sv.n_values,
                n_heavy=sv.n_heavy, t_int_factor=int(plan.coarse_t_native[d]),
                pattern_id=int(vm.stream_pattern_ids[s]), n_grid=vm.n_grid,
                value_kind=0 if vm.kind == "gridded" else 1, s_main=float(rule.s_main_f32),
                s_heavy=float(rule.s_heavy_f32), step_main=float(bc.step_main[0]),
                step_heavy=float(bc.step_heavy[0]), payload_pool=pool, k=k)
    meta = {"owner_half": [int(x) for x in plan.owner_half],
            "t_native": [int(x) for x in plan.coarse_t_native], "n_sub": op.n_sub,
            "pattern_id": f"0x{vm.pattern_id:016x}"}
    return flows, meta


# =============================================================================
# Workers
# =============================================================================


def _write_json(path: str, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(p)


def _barrier(a) -> float:
    """Ready/go barrier: write ``<out>.ready``, wait for the driver's go file
    (JSON ``{"start_at": unix}``) and return the common start time. Without
    ``--go-file`` the worker uses ``--start-at``."""
    if not getattr(a, "go_file", None):
        if a.start_at is None:
            raise SystemExit("--start-at or --go-file is required")
        return float(a.start_at)
    _write_json(a.out + ".ready", {"t": time.time(), "host": socket.gethostname()})
    t_end = time.time() + a.go_timeout_s
    go = Path(a.go_file)
    while time.time() < t_end:
        if go.exists():
            try:
                return float(json.loads(go.read_text())["start_at"])
            except (ValueError, KeyError):
                pass
        time.sleep(0.05)
    raise TimeoutError(f"no go file {go} within {a.go_timeout_s} s")


def worker_tx(a) -> int:
    """One corr node. Sweep mode: synthetic max-size frames, per-step rates.
    Format mode: real value-map flows at the plan's block cadence x load factor."""
    from dsart.transport.send_mmsg import MmsgSender

    dests = [d.split(":") for d in a.dests.split(",")]
    n_dest = len(dests)
    src_ip = a.bind_ip if a.bind_ip else ""
    res: dict = {"role": "tx", "host": socket.gethostname(), "chgroup": a.chgroup,
                 "mode": a.mode, "steps": []}
    iface = detect_data_iface(a.bind_ip) if (a.nic and a.bind_ip) else None
    res["iface"] = iface
    if a.mode == "sweep":
        rates = [float(x) for x in a.rates.split(",")]
        n_tx = int(a.n_tx)
        flows = {}
        for sub in range(SWEEP_N_SUB):
            s = a.chgroup * SWEEP_N_SUB + sub
            for d in SWEEP_DMS:
                flows[(s, d)] = synthetic_framer(s, d, a.frame_bytes)
        dest_of = {d: rx_node_of_dm(d, n_dest) for d in SWEEP_DMS}
        tx = MmsgSender([(h, int(p), 0.0) for h, p in dests], routes=dest_of, bind_host=src_ip,
                        queue_slots=8192, batch=64, burst_bytes=a.burst_bytes)
        tx.start()
        seq = {f: 1 << 24 for f in flows}
        flows_of = {dd: [(f, fr) for f, fr in flows.items() if dest_of[f[1]] == dd]
                    for dd in range(n_dest)}
        dlen = next(iter(flows.values())).dlen
        start_at = _barrier(a)
        _sleep_until(start_at)
        for i, rate in enumerate(rates):
            t0 = start_at + i * (a.dwell_s + a.gap_s)
            t1 = t0 + a.dwell_s
            _sleep_until(t0)
            t_act = time.time()
            per_pair_bps = rate * 1e9 / n_tx
            dg_per_s = per_pair_bps / 8.0 / dlen
            for dd in range(n_dest):                    # C paces finely, a bit faster than
                tx.set_rate(dd, per_pair_bps * 1.15)    # the Python slices feed it
            c0 = tx.counters()
            nic0 = read_nic(iface)
            emitted = [0] * n_dest
            rr_i = [0] * n_dest
            while True:
                now = time.time()
                if now >= t1:
                    break
                parts = []
                for dd in range(n_dest):
                    due = min(int((now - t0) * dg_per_s) - emitted[dd], 4096)
                    fl = flows_of[dd]
                    for _ in range(max(0, due)):
                        (s_, d_), fr = fl[rr_i[dd] % len(fl)]
                        rr_i[dd] += 1
                        parts.append((fr.build(seq[(s_, d_)], 1), b""))
                        seq[(s_, d_)] += 1
                    emitted[dd] += max(0, due)
                if parts:
                    tx.submit(concat_parts(parts), block_timeout_s=0.0)
                time.sleep(0.004)
            tx.flush(max(0.2, a.gap_s * 0.6))
            c1 = tx.counters()
            res["steps"].append({
                "rate_gbps": rate, "t0": t0, "t1": t1, "late_s": t_act - t0,
                "sent_per_dest": [y["sent"] - x["sent"] for x, y in zip(c0["per_dest"], c1["per_dest"])],
                "bytes_per_dest": [y["bytes"] - x["bytes"] for x, y in zip(c0["per_dest"], c1["per_dest"])],
                "emitted_per_dest": emitted,
                "full_drops": sum(y["full_drops"] - x["full_drops"]
                                  for x, y in zip(c0["per_dest"], c1["per_dest"])),
                "errors": sum(y["errors"] - x["errors"] for x, y in zip(c0["per_dest"], c1["per_dest"])),
                "nic": nic_delta(nic0, read_nic(iface)) if iface else {},
            })
        tx.close()
    else:
        flows, meta = format_framers(a.format, a.chgroup, a.plan, a.antpos_npz, a.dec_deg,
                                     dms=None if not a.dm_subset else
                                     [int(x) for x in a.dm_subset.split(",")])
        res["meta"] = meta
        owner = meta["owner_half"]
        tnat = meta["t_native"]
        dms = sorted({d for (_, d) in flows})
        routes = {d: rx_node_of_dm(d, n_dest, owner) for d in dms}
        # expected bytes per block per dest -> pace at 1.25x that over the (scaled) block
        period = BLOCK_S / a.load_factor
        bytes_blk = [0] * n_dest
        for (s, d), fr in flows.items():
            rows = BLOCK_NATIVE // tnat[d]
            bytes_blk[routes[d]] += -(-rows // fr.k) * fr.dlen
        rate = [1.25 * 8 * b / period for b in bytes_blk]
        tx = MmsgSender([(h, int(p), r) for (h, p), r in zip(dests, rate)], routes=routes,
                        bind_host=src_ip, queue_slots=16384, batch=64, burst_bytes=a.burst_bytes)
        tx.start()
        start_at = _barrier(a)
        blk0 = int((start_at - 1.6e9) / BLOCK_S)              # common epoch for all corr nodes
        _sleep_until(start_at)
        c0 = tx.counters()
        nic0 = read_nic(iface)
        t_end = start_at + a.minutes * 60.0
        b = 0
        late = 0
        expected = 0
        while True:
            t_blk = start_at + b * period
            if t_blk >= t_end:
                break
            if time.time() > t_blk + period:
                late += 1                                     # generator behind real time
            _sleep_until(t_blk)
            parts = []
            for (s, d), fr in flows.items():
                rows = BLOCK_NATIVE // tnat[d]
                parts.append(fr.block_parts((blk0 + b) * rows, rows))
            batch = concat_parts(parts)
            expected += int(batch[2].size)
            tx.submit(batch, block_timeout_s=period)
            b += 1
        tx.flush(5.0)
        c1 = tx.counters()
        res["steps"].append({
            "blocks": b, "blocks_late": late, "period_s": period, "expected": expected,
            "rate_bps_per_dest": rate,
            "sent_per_dest": [y["sent"] - x["sent"] for x, y in zip(c0["per_dest"], c1["per_dest"])],
            "bytes_per_dest": [y["bytes"] - x["bytes"] for x, y in zip(c0["per_dest"], c1["per_dest"])],
            "full_drops": sum(y["full_drops"] - x["full_drops"] for x, y in zip(c0["per_dest"], c1["per_dest"])),
            "errors": sum(y["errors"] - x["errors"] for x, y in zip(c0["per_dest"], c1["per_dest"])),
            "unrouted": c1["unrouted"] - c0["unrouted"],
            "nic": nic_delta(nic0, read_nic(iface)) if iface else {},
        })
        tx.close()
    res["done_unix"] = time.time()
    _write_json(a.out, res)
    return 0


def worker_rx(a) -> int:
    """One search node: RxEpoll v4 counting ring over ports port_base + g;
    snapshots of per-stream datagram counts (+ NIC) at the given times."""
    import numpy as np
    from dsart.transport.recv_epoll import RxEpoll
    from dsart.transport.recv_ring import RingV4Spec, unlink_ring_v4

    dms = [int(x) for x in a.dms.split(",")]
    n_stream = int(a.n_tx) * SWEEP_N_SUB
    nv = int(a.frame_bytes) - 72
    spec = RingV4Spec(dms=tuple(dms), t_int_factor=(16,) * len(dms), t_buf=(a.t_buf,) * len(dms),
                      n_stream=n_stream, n_sub=SWEEP_N_SUB, bits_main=8, bits_heavy=16,
                      check_scales=False, row_bytes=(nv + 7) // 8 * 8, reorder_rows=64,
                      min_retention_s=0.0)
    iface = detect_data_iface(a.bind_ip) if a.nic else None
    rx = RxEpoll.open(bind_host=a.bind_ip, bind_port=a.port_base, so_rcvbuf_bytes=256 << 20)
    res: dict = {"role": "rx", "host": socket.gethostname(), "dms": dms, "iface": iface,
                 "snaps": []}
    try:
        for g in range(1, int(a.n_tx)):
            rx.add_port(bind_host=a.bind_ip, bind_port=a.port_base + g, so_rcvbuf_bytes=256 << 20)
        rx.attach_ring4(shm_name=a.ring, spec=spec, prefault=True, n_threads=a.threads,
                        strict_ports=True)
        rx.start()
        r = rx.ring4_reader(access="pread")
        if a.snap_at:
            snap_at = [float(x) for x in a.snap_at.split(",")]
        else:
            start_at = _barrier(a)
            step = a.dwell_s + a.gap_s
            snap_at = [start_at - a.gap_s / 2] + [start_at + i * step + a.dwell_s + 0.8 * a.gap_s
                                                  for i in range(a.n_steps)]
        for t in snap_at:
            _sleep_until(t)
            t_act = time.time()
            per_stream = np.zeros(n_stream, np.int64)
            holes = np.zeros(n_stream, np.int64)
            for d in dms:
                fs = r.flow_stats(d)
                per_stream += fs["datagrams"]
                holes += fs["rows_hole"]
            c = rx.counters()
            res["snaps"].append({
                "t": t, "t_actual": t_act, "datagrams_per_stream": per_stream.tolist(),
                "holes_per_stream": holes.tolist(),
                "n_received": c.n_received, "bytes": c.bytes_received_total,
                "writer": r.counters()["writer"], "nic": read_nic(iface) if iface else {},
            })
        r.close()
    finally:
        rx.close()
        unlink_ring_v4(a.ring)
    res["done_unix"] = time.time()
    _write_json(a.out, res)
    return 0


def worker_ring_stats(a) -> int:
    """Sample a v4 ring's counters (pread reader, read-only) every
    ``--every-s`` until ``--end-at``; used by fleet-load / snap-live /
    fleet-replay. ``--snap-before-go`` (fleet-replay): the first snapshot is
    taken BEFORE the ready/go barrier, so it precedes every datagram of the
    run (the senders start after the go); the end is then go + minutes +
    ``--tail-s``. Snapshots carry per-stream datagram / hole counts summed
    over the owned DMs; the last one also per (DM, stream)."""
    from dsart.transport.recv_ring import RxRingV4Reader
    iface = a.nic_iface or (detect_data_iface(a.bind_ip) if a.nic else None)
    r = RxRingV4Reader.open(a.ring, access="pread", attach_timeout_s=300.0)
    out = {"role": "ring-stats", "host": socket.gethostname(), "iface": iface, "snaps": []}
    dms = [i.dm for i in r.dms()]

    def snap(last: bool = False) -> None:
        import numpy as np
        c = r.counters()
        per_dm_streams = {d: r.flow_stats(d) for d in dms}
        dg = np.sum([v["datagrams"] for v in per_dm_streams.values()], axis=0) if dms else []
        holes = np.sum([v["rows_hole"] for v in per_dm_streams.values()], axis=0) if dms else []
        s = {"t": time.time(), "writer": c["writer"],
             "per_dm": {str(d): v["writer"] for d, v in c["per_dm"].items()},
             "datagrams_per_stream": [int(x) for x in dg],
             "rows_hole_per_stream": [int(x) for x in holes],
             "stall": r.stall_report(stall_s=2.0),
             "nic": read_nic(iface) if iface else {}}
        if last:
            s["dm_stream_datagrams"] = {str(d): [int(x) for x in v["datagrams"]]
                                        for d, v in per_dm_streams.items()}
        out["snaps"].append(s)

    try:
        if a.end_at is None:
            if a.snap_before_go:
                snap()
            a.end_at = _barrier(a) + a.minutes * 60.0 + a.tail_s
        while True:
            last = time.time() >= a.end_at
            snap(last=last)
            if last:
                break
            time.sleep(min(a.every_s, max(0.1, a.end_at - time.time())))
    finally:
        r.close()
    _write_json(a.out, out)
    return 0


# =============================================================================
# Driver (h23): node commands over ssh, or local subprocesses for "localhost"
# =============================================================================


class Driver:
    def __init__(self, a) -> None:
        self.a = a
        self.tx_nodes = [n for n in a.tx_nodes.split(",") if n]
        self.rx_nodes = [n for n in a.rx_nodes.split(",") if n]
        self.marker = a.marker
        self.scratch = a.scratch
        self.py = a.py_node
        self.ip = dict(IP_MAP)
        if a.ip_map:
            for kv in a.ip_map.split(","):
                k, v = kv.split("=")
                self.ip[k] = v
        self.results = Path(a.scratch) / "cp_results"
        self.procs: list[subprocess.Popen] = []
        self.replay_files: dict = {}

    def _is_local(self, node: str) -> bool:
        return node in ("localhost", "127.0.0.1")

    def _remote_cmd(self, args: list[str], log: str, guard_timeout_s: Optional[float] = None) -> str:
        env = (f"DSART_X2_CP={shlex.quote(self.marker)} PYTHONPATH={shlex.quote(self.scratch)}/src "
               f"DSART_CONFIG_DIR={shlex.quote(self.scratch)}/configs OMP_NUM_THREADS=1 "
               f"MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1")
        cmd = " ".join(shlex.quote(x) for x in args)
        py = f"{shlex.quote(self.py)} bench/fake_corr_to_search.py {cmd}"
        if guard_timeout_s is not None:
            py = guarded(py, guard_timeout_s)
        # braces: only the worker is backgrounded (its fds redirected), so the
        # ssh / bash that launched it returns at once
        return (f"mkdir -p {shlex.quote(str(self.results))} && cd {shlex.quote(self.scratch)} && "
                f"{{ env {env} nohup {py} "
                f"> {shlex.quote(log)} 2>&1 < /dev/null & }}")

    def launch(self, node: str, args: list[str], name: str,
               guard_timeout_s: Optional[float] = None) -> str:
        """Start a marked worker on ``node``; returns its result-JSON path
        (named after the marker: results of earlier runs are never read).
        ``guard_timeout_s``: run it under tools/x2/prod_guard.sh in stopped
        mode with ``timeout`` inside the guarded command (fleet-replay)."""
        name = f"{self.marker}_{name}"
        out = str(self.results / f"{name}.json")
        log = str(self.results / f"{name}.log")
        full = list(args) + ["--out", out]
        if guard_timeout_s is not None and not self._is_local(node):
            subprocess.run(["ssh", "-o", "BatchMode=yes", f"{node}.pro.pvt",
                            self._remote_cmd(full, log, guard_timeout_s)], check=True, timeout=60)
            return out
        if self._is_local(node):
            self.results.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, DSART_X2_CP=self.marker,
                       PYTHONPATH=str(Path(self.scratch) / "src"))
            p = subprocess.Popen([self.py, str(Path(self.scratch) / "bench" / "fake_corr_to_search.py")]
                                 + full, env=env, cwd=self.scratch,
                                 stdout=open(log, "w"), stderr=subprocess.STDOUT)
            self.procs.append(p)
        else:
            subprocess.run(["ssh", "-o", "BatchMode=yes", f"{node}.pro.pvt",
                            self._remote_cmd(full, log)], check=True, timeout=60)
        return out

    def run_on(self, node: str, shell: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
        if self._is_local(node):
            return subprocess.run(["bash", "-c", shell], capture_output=True, text=True,
                                  timeout=timeout)
        return subprocess.run(["ssh", "-o", "BatchMode=yes", f"{node}.pro.pvt", shell],
                              capture_output=True, text=True, timeout=timeout)

    def go_file(self) -> str:
        return str(self.results / f"{self.marker}_go.json")

    def barrier(self, launched: list[tuple[str, str]], lead_s: float, timeout_s: float) -> float:
        """Wait until every launched worker wrote ``<out>.ready``, then write
        the go file (start = now + lead) on every node; returns the start."""
        t_end = time.time() + timeout_s
        for node, out in launched:
            if self.fetch(node, out + ".ready", max(1.0, t_end - time.time())) is None:
                raise TimeoutError(f"worker {out} on {node} not ready within {timeout_s} s")
        start = time.time() + lead_s
        body = json.dumps({"start_at": start})
        for node in dict.fromkeys(n for n, _ in launched):
            r = self.run_on(node, f"mkdir -p {shlex.quote(str(self.results))} && printf %s "
                                  f"{shlex.quote(body)} > {shlex.quote(self.go_file())}.tmp && "
                                  f"mv {shlex.quote(self.go_file())}.tmp {shlex.quote(self.go_file())}")
            if r.returncode != 0:
                raise RuntimeError(f"go file on {node}: {r.stderr[-300:]}")
        return start

    def fetch(self, node: str, path: str, wait_s: float) -> Optional[dict]:
        t_end = time.time() + wait_s
        while True:
            r = self.run_on(node, f"cat {shlex.quote(path)} 2>/dev/null", timeout=60)
            if r.returncode == 0 and r.stdout.strip():
                try:
                    return json.loads(r.stdout)
                except json.JSONDecodeError:
                    pass
            if time.time() >= t_end:
                return None
            time.sleep(2.0)

    def teardown(self) -> dict:
        """Kill this marker's processes and remove /dev/shm/dsart-x2-* (every node)."""
        kill = ("me=$$; for p in $(pgrep -u \"$(id -u)\" .); do [ \"$p\" = \"$me\" ] && continue; "
                "tr '\\0' '\\n' < /proc/$p/environ 2>/dev/null | grep -qx "
                f"'DSART_X2_CP={self.marker}' && kill $p 2>/dev/null; done; sleep 2; "
                "for p in $(pgrep -u \"$(id -u)\" .); do [ \"$p\" = \"$me\" ] && continue; "
                "tr '\\0' '\\n' < /proc/$p/environ 2>/dev/null | grep -qx "
                f"'DSART_X2_CP={self.marker}' && kill -9 $p 2>/dev/null; done; "
                "rm -f /dev/shm/dsart-x2-*; true")
        out = {}
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
        local_done = False
        for n in dict.fromkeys(self.tx_nodes + self.rx_nodes):
            if self._is_local(n):
                if local_done:
                    continue
                local_done = True
                for p in self.procs:
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        p.kill()
                # same marker kill as on nodes; shm cleanup limited to this marker
                # (the CPU-test harness node also hosts other agents' test rings)
                subprocess.run(["bash", "-c", kill.replace("rm -f /dev/shm/dsart-x2-*",
                                                           f"rm -f /dev/shm/dsart-x2-{self.marker}*")],
                               capture_output=True, timeout=60)
                out[n] = "local"
                continue
            try:
                out[n] = self.run_on(n, kill, timeout=60).returncode
            except Exception as exc:  # noqa: BLE001
                out[n] = f"error: {exc}"
        return out

    # -- fleet-replay: marker-scoped teardown (other agents' test rings / buffers are left alone) --

    def _kill_marker(self) -> str:
        """TERM then KILL every process of this user carrying DSART_X2_CP=<marker>."""
        one = ("for p in $(pgrep -u \"$(id -u)\" .); do [ \"$p\" = \"$me\" ] && continue; "
               "tr '\\0' '\\n' < /proc/$p/environ 2>/dev/null | grep -qx "
               f"'DSART_X2_CP={self.marker}' && kill -{{sig}} $p 2>/dev/null; done")
        return "me=$$; " + one.format(sig="TERM") + "; sleep 3; " + one.format(sig="KILL") + "; sleep 1"

    def teardown_replay(self, corr_nodes: list[str]) -> dict:
        """Kill the marker's processes on every node, remove only this marker's
        /dev/shm/dsart-x2-<marker>* and, on corr nodes, the TEST DADA buffer
        0x{REPLAY_KEY} and the corr ready sentinel -- unless another (unmarked)
        b8 corr / replay writer still runs there (its buffer, not ours)."""
        out = {}
        for n in dict.fromkeys(self.tx_nodes + self.rx_nodes):
            sh = self._kill_marker() + f"; rm -f /dev/shm/dsart-x2-{self.marker}*"
            if n in corr_nodes:
                sh += ("; if pgrep -f '[b]8_corr2x[.]py|[x]2_fada_replay[.]py' > /dev/null; then "
                       "echo other-b8-running; else "
                       f"dada_db -d -k {REPLAY_KEY} > /dev/null 2>&1; rm -f {CORR_READY_SENTINEL}; fi")
            sh += "; true"
            try:
                r = self.run_on(n, sh, timeout=90)
                out[n] = r.returncode if not r.stdout.strip() else r.stdout.strip()[-200:]
            except Exception as exc:  # noqa: BLE001
                out[n] = f"error: {exc}"
        return out

    def leftovers(self, corr_nodes: list[str]) -> dict:
        """After teardown: this marker's processes, shm and (corr) the TEST DADA
        key / sentinel on every node -- all must be empty."""
        out = {}
        for n in dict.fromkeys(self.tx_nodes + self.rx_nodes):
            sh = ("me=$$; for p in $(pgrep -u \"$(id -u)\" .); do [ \"$p\" = \"$me\" ] && continue; "
                  "tr '\\0' '\\n' < /proc/$p/environ 2>/dev/null | grep -qx "
                  f"'DSART_X2_CP={self.marker}' && echo \"proc $p $(cat /proc/$p/comm)\"; done; "
                  f"ls /dev/shm/ | grep -- 'dsart-x2-{self.marker}' | sed 's/^/shm /'")
            if n in corr_nodes:
                sh += (f"; ipcs -m | grep -i '0x0000{REPLAY_KEY}\\|0x0000{int(REPLAY_KEY, 16) + 1:04x}' "
                       f"| sed 's/^/dada /'; ls {CORR_READY_SENTINEL} 2>/dev/null | sed 's/^/file /'")
            sh += "; true"
            try:
                r = self.run_on(n, sh, timeout=60)
                out[n] = [x for x in r.stdout.splitlines() if x.strip()]
            except Exception as exc:  # noqa: BLE001
                out[n] = [f"error: {exc}"]
        return out


def _baseline_loss(spec: Optional[str]) -> dict:
    if not spec:
        return {}
    p = Path(spec)
    if p.exists():
        return {k: float(v) for k, v in json.loads(p.read_text()).items()}
    return {k: float(v) for k, v in (kv.split("=") for kv in spec.split(","))}


def evaluate_sweep(tx_res: list[dict], rx_res: list[dict], rx_nodes: list[str], rates: list[float],
                   baseline: dict, *, nic: bool, frame_bytes: int) -> dict:
    """Per (rx node, step): loss = 1 - received / sent (datagrams), Gb/s and
    datagrams/s, NIC discard deltas, clean flag; clean_gbps_per_node."""
    n_rx = len(rx_nodes)
    per_node: dict = {n: [] for n in rx_nodes}
    for i, rate in enumerate(rates):
        for j, node in enumerate(rx_nodes):
            sent = sum(t["steps"][i]["sent_per_dest"][j] for t in tx_res)
            bytes_ = sum(t["steps"][i]["bytes_per_dest"][j] for t in tx_res)
            snaps = rx_res[j]["snaps"]
            a, b = snaps[i], snaps[i + 1]
            tx_late = max(t["steps"][i].get("late_s", 0.0) for t in tx_res)
            rx_late = max(s_.get("t_actual", s_.get("t", 0.0)) - s_.get("t", 0.0) for s_ in (a, b))
            recv = sum(b["datagrams_per_stream"]) - sum(a["datagrams_per_stream"])
            dwell = tx_res[0]["steps"][i]["t1"] - tx_res[0]["steps"][i]["t0"]
            loss = 1.0 - recv / sent if sent else 1.0
            nd = nic_delta(a["nic"], b["nic"]) if nic and a.get("nic") and b.get("nic") else {}
            crc_frac = (nd.get("rx_crc_errors_phy", 0) / nd["rx_packets_phy"]
                        if nd.get("rx_packets_phy") else 0.0)
            allowed = max(1e-4, baseline.get(node, crc_frac))
            discards = nd.get("rx_discards_phy", 0) + nd.get("rx_out_of_buffer", 0)
            timing_ok = tx_late < 0.1 * dwell and rx_late < 0.5
            clean = (loss <= allowed) and (discards == 0) and sent > 0 and timing_ok
            per_node[node].append({
                "rate_gbps_target": rate, "sent": sent, "received": recv, "loss": loss,
                "allowed_loss": allowed, "baseline_source": "given" if node in baseline else "crc",
                "gbps_payload": bytes_ * 8 / dwell / 1e9,
                "gbps_wire": (bytes_ + sent * WIRE_OVERHEAD_B) * 8 / dwell / 1e9,
                "datagrams_per_s": sent / dwell, "nic_delta": nd, "clean": bool(clean),
                "timing_ok": bool(timing_ok), "tx_late_s": tx_late, "rx_late_s": rx_late,
                "rx_writer": b.get("writer", {})})
    clean_gbps = {}
    for node, steps in per_node.items():
        c = 0.0
        for st in sorted(steps, key=lambda s: s["rate_gbps_target"]):
            if not st["clean"]:
                break
            c = st["rate_gbps_target"]
        clean_gbps[node] = c
    return {"per_node": per_node, "clean_gbps_per_node": clean_gbps, "frame_bytes": frame_bytes,
            "n_rx": n_rx}


def _out_path(out: str, name: str) -> Path:
    p = Path(out)
    if p.suffix == ".json":
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.json"


def cmd_sweep(a) -> int:
    if int(a.frame_version) != 2:
        raise SystemExit("sweep supports --frame-version 2 only")
    if not 72 < a.frame_bytes <= MAX_DATAGRAM:
        raise SystemExit(f"--frame-bytes must be in (72, {MAX_DATAGRAM}] (no IP fragmentation)")
    drv = Driver(a)
    a.lead_s = 3.0 if a.lead_s is None else a.lead_s
    rates = [float(x) for x in a.rates_gbps.split(",")]
    n_rx, n_tx = len(drv.rx_nodes), len(drv.tx_nodes)
    step = a.dwell_s + a.gap_s
    rx_out, tx_out = [], []
    try:
        for j, n in enumerate(drv.rx_nodes):
            dms = [d for d in SWEEP_DMS if rx_node_of_dm(d, n_rx) == j]
            rx_out.append(drv.launch(n, [
                "rx", "--bind-ip", drv.ip[n], "--port-base", str(a.port_base), "--n-tx", str(n_tx),
                "--dms", ",".join(map(str, dms)), "--frame-bytes", str(a.frame_bytes),
                "--threads", str(a.rx_threads), "--t-buf", "1024",
                "--ring", f"/dsart-x2-{a.marker}-sweep-{j}", "--go-file", drv.go_file(),
                "--n-steps", str(len(rates)), "--dwell-s", str(a.dwell_s), "--gap-s", str(a.gap_s)]
                + (["--nic"] if a.nic_counters else []),
                f"sweep_rx_{j}_{n}"))
        for g, n in enumerate(drv.tx_nodes):
            d_list = ",".join(f"{drv.ip[r]}:{a.port_base + g}" for r in drv.rx_nodes)
            tx_out.append(drv.launch(n, [
                "tx", "--mode", "sweep", "--chgroup", str(g), "--n-tx", str(n_tx),
                "--dests", d_list, "--bind-ip", drv.ip.get(n, ""), "--rates", a.rates_gbps,
                "--dwell-s", str(a.dwell_s), "--gap-s", str(a.gap_s), "--go-file", drv.go_file(),
                "--frame-bytes", str(a.frame_bytes), "--burst-bytes", str(a.burst_bytes)]
                + (["--nic"] if a.nic_counters else []),
                f"sweep_tx_{g}_{n}"))
        start = drv.barrier(list(zip(drv.rx_nodes, rx_out)) + list(zip(drv.tx_nodes, tx_out)),
                            a.lead_s, a.ready_timeout_s)
        wait = max(0.0, start + len(rates) * step - time.time()) + 60
        txr = [drv.fetch(n, p, wait) for n, p in zip(drv.tx_nodes, tx_out)]
        rxr = [drv.fetch(n, p, 60) for n, p in zip(drv.rx_nodes, rx_out)]
    finally:
        td = drv.teardown()
    missing = [n for n, r in zip(drv.tx_nodes + drv.rx_nodes, txr + rxr) if r is None]
    out = _out_path(a.out, "capacity")
    if missing:
        _write_json(str(out), {"error": f"no result from {missing}", "teardown": td})
        return 2
    ev = evaluate_sweep(txr, rxr, drv.rx_nodes, rates, _baseline_loss(a.baseline_loss),
                        nic=a.nic_counters, frame_bytes=a.frame_bytes)
    ev.update({"rates_gbps": rates, "dwell_s": a.dwell_s, "tx_nodes": drv.tx_nodes,
               "rx_nodes": drv.rx_nodes, "record_pps": True, "teardown": td,
               "tx_full_drops": sum(s["full_drops"] for t in txr for s in t["steps"]),
               "gates": {}})
    low = rates[0]
    failed = [n for n, c in ev["clean_gbps_per_node"].items() if c < low]
    ev["gates"]["clean_at_lowest_rate"] = {"passed": not failed, "nodes_failing": failed}
    _write_json(str(out), ev)
    print(json.dumps({"clean_gbps_per_node": ev["clean_gbps_per_node"], "out": str(out)}))
    return 0 if not failed else 1


def _ring_load_common(a, *, engines: bool, name: str) -> int:
    """fleet-load / snap-live: real-format TX on every corr node into
    search_rx v4 (+ owner engines) on every search node."""
    drv = Driver(a)
    n_rx = len(drv.rx_nodes)
    a.lead_s = 5.0 if a.lead_s is None else a.lead_s
    if not a.plan:
        raise SystemExit("--plan (v3 plan npz path on the nodes) is required")
    antpos = a.antpos_npz or f"{a.scratch}/tests/fixtures/x2/dsa110_antpos.npz"
    ring_of = {n: f"/dsart-x2-{a.marker}-rx4-{j}" for j, n in enumerate(drv.rx_nodes)}
    tx_out, st_out = [], []
    res: dict = {"format": a.format, "load_factor": a.load_factor, "minutes": a.minutes,
                 "tx_nodes": drv.tx_nodes, "rx_nodes": drv.rx_nodes, "gates": {}}
    try:
        for j, n in enumerate(drv.rx_nodes):
            spec = str(drv.results / f"{a.marker}_ring4_spec_{j}_{n}.json")
            pids = str(drv.results / f"{a.marker}_ring4_pids_{j}_{n}.json")
            mk = drv.run_on(n, (
                f"mkdir -p {drv.results} && cd {shlex.quote(a.scratch)} && env PYTHONPATH={a.scratch}/src "
                f"{shlex.quote(drv.py)} -m dsart.services.search_rx make-ring4-spec --plan {a.plan} "
                f"--opoint {a.format} --node-index {j % 4} --antpos-npz {antpos} --dec-deg {a.dec_deg} "
                f"--retention-s {a.retention_s} --out {spec} --pattern-ids-out {pids}"), timeout=300)
            if mk.returncode != 0:
                raise RuntimeError(f"make-ring4-spec on {n}: {mk.stderr[-400:]}")
            env = f"DSART_X2_CP={a.marker} PYTHONPATH={a.scratch}/src OMP_NUM_THREADS=1"
            rxcmd = (f"cd {a.scratch} && {{ env {env} nohup {drv.py} -m dsart.services.search_rx "
                     f"--ring-version 4 --ring4-spec {spec} --pattern-id-file {pids} "
                     f"--bind-host {drv.ip[n]} --port-base {a.port_base} --n-ports {len(drv.tx_nodes)} "
                     f"--rx-threads {a.rx_threads} --shm-name {ring_of[n]} --status-interval-s 10 "
                     f"{'--no-prefault ' if a.rx_no_prefault else ''}"
                     f"> {drv.results}/{a.marker}_{name}_search_rx_{j}_{n}.log 2>&1 < /dev/null & }}")
            drv.run_on(n, rxcmd, timeout=60)
            st_out.append(drv.launch(n, ["ring-stats", "--ring", ring_of[n], "--go-file", drv.go_file(),
                                         "--minutes", str(a.minutes), "--every-s", "10",
                                         "--bind-ip", drv.ip[n]]
                                     + (["--nic"] if not drv._is_local(n) else []),
                                     f"{name}_ringstats_{j}_{n}"))
            if engines and a.engine:
                ecmd = (f"cd {a.scratch} && {{ env {env} nohup {drv.py} {a.engine} --shm-name {ring_of[n]} "
                        f"--plan {a.plan} --opoint {a.format} --halves {2 * (j % 4)},{2 * (j % 4) + 1} "
                        f"--minutes {a.minutes} --out {drv.results}/{a.marker}_engine_{j}_{n} "
                        f"> {drv.results}/{a.marker}_{name}_engine_{j}_{n}.log 2>&1 < /dev/null & }}")
                drv.run_on(n, ecmd, timeout=60)
        for g, n in enumerate(drv.tx_nodes):
            dl = ",".join(f"{drv.ip[r]}:{a.port_base + g}" for r in drv.rx_nodes)
            tx_out.append(drv.launch(n, [
                "tx", "--mode", "format", "--format", a.format, "--plan", a.plan, "--chgroup", str(g),
                "--n-tx", str(len(drv.tx_nodes)), "--dests", dl, "--bind-ip", drv.ip.get(n, ""),
                "--load-factor", str(a.load_factor), "--minutes", str(a.minutes),
                "--go-file", drv.go_file(), "--antpos-npz", antpos, "--dec-deg", str(a.dec_deg),
                "--burst-bytes", str(a.burst_bytes)]
                + (["--dm-subset", a.dm_subset] if a.dm_subset else [])
                + (["--nic"] if not drv._is_local(n) else []), f"{name}_tx_{g}_{n}"))
        start = drv.barrier(list(zip(drv.rx_nodes, st_out)) + list(zip(drv.tx_nodes, tx_out)),
                            a.lead_s, a.ready_timeout_s)
        end = start + a.minutes * 60.0
        txr = [drv.fetch(n, p, max(0.0, end - time.time()) + 120) for n, p in zip(drv.tx_nodes, tx_out)]
        str_ = [drv.fetch(n, p, 120) for n, p in zip(drv.rx_nodes, st_out)]
        eng = {}
        if engines and a.engine:
            for j, n in enumerate(drv.rx_nodes):
                # the engines finish after the TX: wait (<= 300 s) for each one's final JSON, or a fast
                # node's result is read before a slower node's engine has written it (G1 smoke run)
                edir = f"{drv.results}/{a.marker}_engine_{j}_{n}"
                drv.run_on(n, f"for i in $(seq 150); do grep -qs '\"final\": true' {edir}/*.json && break; "
                              "sleep 2; done", timeout=330)
                r = drv.run_on(n, f"cat {edir}/*.json 2>/dev/null "
                                  "| head -c 200000")
                eng[n] = r.stdout[:200000] if r.returncode == 0 else None
    finally:
        res["teardown"] = drv.teardown()
    out = _out_path(a.out, name)
    if any(r is None for r in txr + str_):
        res["error"] = "missing worker results"
        _write_json(str(out), res)
        return 2
    baseline = _baseline_loss(a.baseline_loss)
    sent = [sum(t["steps"][0]["sent_per_dest"][j] for t in txr) for j in range(n_rx)]
    gates_ok = True
    per_node = {}
    for j, n in enumerate(drv.rx_nodes):
        s0, s1 = str_[j]["snaps"][0], str_[j]["snaps"][-1]
        recv = s1["writer"]["ok"] + s1["writer"]["pattern_mismatch"] - s0["writer"]["ok"] - s0["writer"]["pattern_mismatch"]
        rows = sum(v["rows_written"] for v in s1["per_dm"].values())
        holes = sum(v["rows_hole"] for v in s1["per_dm"].values())
        loss = 1.0 - recv / sent[j] if sent[j] else 1.0
        allowed = max(1e-4, baseline.get(n, 0.0))
        per_node[n] = {"sent": sent[j], "received": recv, "loss": loss, "allowed_loss": allowed,
                       "row_hole_frac": holes / (rows + holes) if rows + holes else 0.0,
                       "rejected": {k: v for k, v in s1["writer"].items()
                                    if k not in ("datagrams", "ok", "bytes", "sweeps") and v},
                       "stalled_at_end": s1["stall"]}
        ok = loss <= allowed and not per_node[n]["rejected"]
        gates_ok &= ok
        per_node[n]["passed"] = ok
    tx_ok = all(st["full_drops"] == 0 and st["errors"] == 0 and st["unrouted"] == 0
                and st["blocks_late"] <= max(2, 0.01 * st["blocks"]) for t in txr for st in t["steps"])
    res["per_node"] = per_node
    res["tx"] = [{"host": t["host"], **{k: t["steps"][0][k] for k in
                                        ("blocks", "blocks_late", "expected", "full_drops", "errors",
                                         "unrouted", "nic")}} for t in txr]
    tot_dg = sum(sent)
    res["datagrams_per_s_per_node"] = {n: s / (a.minutes * 60.0) for n, s in zip(drv.rx_nodes, sent)}
    res["gbps_per_node"] = {n: sum(t["steps"][0]["bytes_per_dest"][j] for t in txr) * 8 / (a.minutes * 60) / 1e9
                            for j, n in enumerate(drv.rx_nodes)}
    res["gates"]["loss"] = {"passed": bool(gates_ok)}
    res["gates"]["tx_kept_up"] = {"passed": bool(tx_ok)}
    if engines and a.engine:
        res["engine_results"] = eng
    res["total_datagrams"] = tot_dg
    _write_json(str(out), res)
    print(json.dumps({"gates": res["gates"], "out": str(out)}))
    return 0 if (gates_ok and tx_ok) else 1


def cmd_fleet_load(a) -> int:
    return _ring_load_common(a, engines=True, name="full_load")


def cmd_snap_live(a) -> int:
    if a.sender != "c":
        raise SystemExit("snap-live: only --sender c (send_mmsg) is implemented")
    if a.load_factor is None:
        a.load_factor = 1.0
    return _ring_load_common(a, engines=False, name="snap_live")


# =============================================================================
# fleet-replay: the REAL 2x corr (b8_corr2x live, replayed voltages, C sender)
# -> search_rx v4 -> owner engines (G1 full chain without SNAPs)
# =============================================================================


def _n_sub_of(fmt: str) -> int:
    m = re.search(r"-ns(\d+)-", fmt)
    return int(m.group(1)) if m else 4


def _preflight_replay(drv: Driver, a, chg: dict) -> dict:
    """Per node (read-only, plus one guarded ``true``): scratch copy, production
    stopped, no GPU compute process, no other x2 bench running, search: the
    plan; corr: the fixture sub-band (+ its own cal), no TEST DADA buffer."""
    out = {}
    sc = shlex.quote(a.scratch)
    for n in dict.fromkeys(drv.tx_nodes + drv.rx_nodes):
        sh = (f"test -f {sc}/tools/x2/prod_guard.sh && echo scratch_ok; "
              f"env GUARD_REQUIRE_STOPPED=1 bash {sc}/tools/x2/prod_guard.sh true > /dev/null 2>&1 "
              "&& echo stopped_ok; "
              "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null "
              "| sed 's/^/gpu_app /'; "
              "pgrep -af '[b]8_corr2x[.]py|[b]8_owner2x[.]py|[x]2_fada_replay[.]py|[f]ake_corr_to_search[.]py"
              "|[s]earch_rx --ring-version' | sed 's/^/busy /'; ")
        if n in drv.rx_nodes:
            sh += f"test -f {shlex.quote(a.plan)} && echo plan_ok; "
        if n in chg:
            rf = drv.replay_files.get(n)
            if rf:
                sh += f"test -r {shlex.quote(rf)} && echo fixture_ok; "
            sh += (f"for r in /home/ubuntu/x2_scratch/voltages /home/ubuntu/data/voltage_fixtures "
                   f"/home/ubuntu/data/voltages; do if [ -f $r/{a.replay}/manifest.yaml ]; then "
                   f"f=$(ls $r/{a.replay}/fl_*_chgroup{chg[n]}.out 2>/dev/null | head -1); "
                   f"[ -n \"$f\" ] && test -r \"$(readlink -f $f)\" && echo fixture_ok $r; "
                   f"test -s $r/{a.replay}/cals/antennas.out && echo cal_ok; break; fi; done; "
                   f"ipcs -m | grep -qi '0x0000{REPLAY_KEY}' && echo dada_key_exists; ")
        sh += "true"
        try:
            lines = drv.run_on(n, sh, timeout=90).stdout.splitlines()
        except Exception as exc:  # noqa: BLE001
            lines = [f"error {exc}"]
        fatal = []
        if "scratch_ok" not in lines:
            fatal.append("no scratch copy")
        if "stopped_ok" not in lines:
            fatal.append("production running (guard refuses) or guard error")
        fatal += [x for x in lines if x.startswith(("gpu_app", "busy", "error", "dada_key_exists"))]
        if n in drv.rx_nodes and "plan_ok" not in lines:
            fatal.append(f"no plan {a.plan}")
        warn = []
        if n in chg:
            if not any(x.startswith("fixture_ok") for x in lines):
                fatal.append(f"no {a.replay} chgroup {chg[n]} voltage file")
            if "cal_ok" not in lines:
                warn.append("no fixture cal: corr_fast falls back to the production cal (pattern ids may "
                            "differ from the ring's)")
        out[n] = {"lines": lines, "fatal": fatal, "warn": warn}
    return out


def _health(drv: Driver, chg: dict) -> dict:
    """Light liveness probe: corr b8 workers, search_rx and engines per node."""
    out = {}
    for n in dict.fromkeys(drv.tx_nodes + drv.rx_nodes):
        if n in chg:
            # SigCgt of the corr python: which signals have handlers (a handled signal makes the
            # blocking semop in psrdada's getNextPage fail -> corr_fast exits; G1 2026-10-29)
            sh = ("p=$(pgrep -f '^[^ ]*python[^ ]* bench/timeres/[b]8_corr2x[.]py live' | head -1); "
                  "echo corr=$(pgrep -fc '[b]8_corr2x[.]py live') "
                  f"sentinel=$(test -f {CORR_READY_SENTINEL} && echo 1 || echo 0) "
                  "sigcgt=$( [ -n \"$p\" ] && grep SigCgt /proc/$p/status | cut -f2)")
        else:
            sh = ("echo rx=$(pgrep -fc '[s]earch_rx --ring-version') "
                  "engine=$(pgrep -fc '[b]8_owner2x[.]py engine')")
        try:
            out[n] = drv.run_on(n, sh, timeout=30).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            out[n] = f"error: {exc}"
    return out


def _log_tail(drv: Driver, node: str, path: str, n: int = 25) -> str:
    try:
        return drv.run_on(node, f"tail -n {n} {shlex.quote(path)} 2>/dev/null", timeout=30).stdout[-3000:]
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def cmd_fleet_replay(a) -> int:
    """G1 full chain: b8_corr2x live (corr_fast 2x on replayed voltages, frame v2, C sender) on every
    --tx-nodes corr node, search_rx v4 + ring-stats + b8_owner2x engines on every --rx-nodes search
    node; common --start-at / --block0. Every node process runs under prod_guard (stopped mode)."""
    drv = Driver(a)
    drv.replay_files = dict(kv.split("=", 1) for kv in (a.replay_file or "").split(",") if kv)
    bad = ([n for n in drv.rx_nodes if n not in SEARCH_NODE_ORDER]
           + [n for n in drv.tx_nodes if n not in CORR_NODE_ORDER])
    if bad or len(set(drv.tx_nodes)) != len(drv.tx_nodes):
        raise SystemExit(f"fleet-replay: --rx-nodes from {SEARCH_NODE_ORDER}, --tx-nodes distinct from "
                         f"{CORR_NODE_ORDER} (bad: {bad})")
    if not a.plan or not a.plan_id:
        raise SystemExit("fleet-replay: --plan (npz on the search nodes) and --plan-id are required")
    dec = a.dec_deg if a.dec_deg is not None else REPLAY_POINTING_DEC.get(a.replay)
    if dec is None:
        raise SystemExit(f"fleet-replay: --dec-deg (the corr's pointing DEC for fixture {a.replay})")
    chg = {n: CORR_NODE_ORDER.index(n) for n in drv.tx_nodes}
    idx = {n: SEARCH_NODE_ORDER.index(n) for n in drv.rx_nodes}
    # every corr sends to all four search nodes (SEARCH_NODES order); a search node outside this run
    # gets a loopback address, so no test datagram reaches a node another agent may be using
    tx_hosts = ",".join(drv.ip[n] if n in drv.rx_nodes else a.absent_rx_host for n in SEARCH_NODE_ORDER)
    n_sub = _n_sub_of(a.format)
    lossy = [n for n in (a.lossy_tx_nodes or "").split(",") if n]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_s = a.lead_s + a.minutes * 60.0 + 2400.0           # guard timeouts (teardown ends earlier)
    sc, res_dir, py = a.scratch, drv.results, drv.py
    res: dict = {"marker": a.marker, "format": a.format, "plan": a.plan, "plan_id": a.plan_id,
                 "replay": a.replay, "dec_deg": dec, "minutes": a.minutes, "tx_nodes": drv.tx_nodes,
                 "rx_nodes": drv.rx_nodes, "chgroup": chg, "search_index": idx, "tx_hosts": tx_hosts,
                 "port_base": a.port_base, "burst_bytes": a.burst_bytes, "ab_mode": a.ab_mode,
                 "lossy_tx_nodes": lossy, "ring_spec": {}, "launch": {}, "health": [], "gates": {}}
    print(f"fleet-replay {a.marker}: preflight {len(drv.tx_nodes)} corr + {len(drv.rx_nodes)} search",
          flush=True)
    res["preflight"] = _preflight_replay(drv, a, chg)
    fatal = {n: v["fatal"] for n, v in res["preflight"].items() if v["fatal"]}
    if fatal and not a.skip_preflight:
        res["error"] = f"preflight: {fatal}"
        _write_json(str(out_dir / "fleet_replay.json"), res)
        print(json.dumps({"error": res["error"]}), flush=True)
        return 2
    corr_dir = {n: f"{res_dir}/{a.marker}_corr_{chg[n]}_{n}" for n in drv.tx_nodes}
    st_out: dict = {}
    corr_res: dict = {}
    ring_res: dict = {}
    eng_res: dict = {}
    t_start = None
    try:
        # 1. search side: ring spec at the corr's DEC, search_rx v4 on all 16 chgroup ports, ring-stats
        #    (first snapshot before any datagram), owner engines on both GPUs
        for n in drv.rx_nodes:
            j = idx[n]
            ring = f"/dsart-x2-{a.marker}-rx4-{j}"
            spec = f"{res_dir}/{a.marker}_ring4_spec_{j}_{n}.json"
            pids = f"{res_dir}/{a.marker}_ring4_pids_{j}_{n}.json"
            antpos = a.antpos_npz or f"{sc}/tests/fixtures/x2/dsa110_antpos.npz"
            mk = drv.run_on(n, (
                f"mkdir -p {res_dir} && cd {sc} && env DSART_X2_CP={a.marker} PYTHONPATH={sc}/src "
                + guarded(f"{py} -m dsart.services.search_rx make-ring4-spec --plan {a.plan} "
                          f"--opoint {a.format} --node-index {j} --antpos-npz {antpos} --dec-deg {dec} "
                          f"--retention-s {a.retention_s} --out {spec} --pattern-ids-out {pids}", 600)),
                timeout=660)
            if mk.returncode != 0:
                raise RuntimeError(f"make-ring4-spec on {n}: {(mk.stdout + mk.stderr)[-600:]}")
            try:
                res["ring_spec"][n] = json.loads(mk.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError):
                res["ring_spec"][n] = mk.stdout[-400:]
            env = f"DSART_X2_CP={a.marker} PYTHONPATH={sc}/src OMP_NUM_THREADS=1"
            rx = guarded(f"{py} -m dsart.services.search_rx --ring-version 4 --ring4-spec {spec} "
                         f"--pattern-id-file {pids} --bind-host {drv.ip[n]} --port-base {a.port_base} "
                         f"--n-ports {len(CORR_NODE_ORDER)} --rx-threads {a.rx_threads} --shm-name {ring} "
                         f"--status-interval-s 10", total_s)
            drv.run_on(n, f"cd {sc} && {{ env {env} nohup {rx} > {res_dir}/{a.marker}_replay_search_rx_{j}_{n}.log "
                          "2>&1 < /dev/null & }", timeout=60)
            st_out[n] = drv.launch(n, ["ring-stats", "--ring", ring, "--go-file", drv.go_file(),
                                       "--go-timeout-s", str(a.ready_timeout_s + a.lead_s + 600),
                                       "--minutes", str(a.minutes), "--every-s", "10", "--bind-ip", drv.ip[n],
                                       "--nic", "--snap-before-go", "--tail-s", str(a.tail_s)]
                                   + (["--nic-iface", a.nic_iface] if a.nic_iface else []),
                                   f"replay_ringstats_{j}_{n}", guard_timeout_s=total_s)
            edir = f"{res_dir}/{a.marker}_engine_{j}_{n}"
            eng = guarded(f"{py} {a.engine} --shm-name {ring} --plan {a.plan} --opoint {a.format} "
                          f"--halves {2 * j},{2 * j + 1} --minutes {a.minutes} --dec-deg {dec} --out {edir}",
                          total_s)
            drv.run_on(n, f"cd {sc} && {{ env {env} nohup {eng} > {res_dir}/{a.marker}_replay_engine_{j}_{n}.log "
                          "2>&1 < /dev/null & }", timeout=60)
            print(f"  search {n} (index {j}): ring {ring} {res['ring_spec'][n]}", flush=True)
        t_end_ready = time.time() + a.ready_timeout_s
        for n in drv.rx_nodes:
            if drv.fetch(n, st_out[n] + ".ready", max(1.0, t_end_ready - time.time())) is None:
                raise TimeoutError(f"ring-stats on {n} not ready: "
                                   + _log_tail(drv, n, st_out[n].replace(".json", ".log")))
        # 2. common start: a block boundary >= lead after the last corr launch (warm-up 60-75 s)
        t0 = time.time() + a.lead_s + 1.5 * len(drv.tx_nodes)
        block0 = int(-(-(t0 - 1.6e9) // BLOCK_S))
        t_start = 1.6e9 + block0 * BLOCK_S
        res.update(start_at=t_start, block0=block0,
                   start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)))
        body = json.dumps({"start_at": t_start})
        for n in drv.rx_nodes:                               # ring-stats: sample until start + minutes
            r = drv.run_on(n, f"printf %s {shlex.quote(body)} > {drv.go_file()}.tmp && "
                              f"mv {drv.go_file()}.tmp {drv.go_file()}")
            if r.returncode != 0:
                raise RuntimeError(f"go file on {n}: {r.stderr[-300:]}")
        # 3. the real corr on every corr node
        n_tx_timeout = int(a.lead_s + 1.5 * len(drv.tx_nodes) + a.minutes * 60 + 1200)
        for n in drv.tx_nodes:
            od = corr_dir[n]
            rf = drv.replay_files.get(n)
            cmd = guarded(
                f"env PYTHONPATH=src DSART_CONFIG_DIR=configs MKL_INTERFACE_LAYER=GNU,LP64 {py} "
                f"bench/timeres/b8_corr2x.py live --opoint {a.format} --minutes {a.minutes} "
                f"--replay {a.replay} --reader-identity corr_fast --trace --tx v2 --tx-hosts {tx_hosts} "
                f"--tx-port-base {a.port_base} --tx-bind-host {drv.ip[n]} --tx-burst-bytes {a.burst_bytes} "
                f"--start-at {t_start!r} --block0 {block0} --plan-id {a.plan_id} --ab-mode {a.ab_mode} "
                f"--out {od}" + (f" --replay-file {shlex.quote(rf)}" if rf else ""), n_tx_timeout)
            drv.run_on(n, f"mkdir -p {res_dir} && cd {sc} && {{ env DSART_X2_CP={a.marker} "
                          f"DSART_HOST_PIN_PERSIST=1 nohup {cmd} > {od}.log 2>&1 < /dev/null & }}", timeout=60)
            res["launch"][n] = time.time()
        print(f"  corr launched on {len(drv.tx_nodes)} nodes; start {res['start_utc']} block0 {block0} "
              f"(in {t_start - time.time():.0f} s)", flush=True)
        # 4. at the start: which corr nodes are warm (sentinel) / alive
        _sleep_until(t_start - 3.0)
        h = _health(drv, chg)
        res["health"].append({"t": time.time(), "at": "start", "nodes": h})
        res["not_ready_at_start"] = [n for n in drv.tx_nodes if "sentinel=1" not in h.get(n, "")]
        dead = [n for n in drv.tx_nodes if "corr=0" in h.get(n, "")]
        print(f"  at start: not warm {res['not_ready_at_start']} dead {dead}", flush=True)
        res["dead_at_start"] = dead
        if len(dead) > a.max_dead_corr:
            # a missing chgroup is not a real-time result: stop now instead of running the
            # whole duration on a broken fan-in quorum (G1 2026-10-29: 2 of 16 died at start-up)
            res["dead_log_tails"] = {n: _log_tail(drv, n, corr_dir[n] + ".log", 60) for n in dead}
            raise RuntimeError(f"corr workers dead at the start: {dead}")
        # 5. run: periodic liveness lines
        t_data_end = t_start + a.minutes * 60.0
        while time.time() < t_data_end:
            _sleep_until(min(t_data_end, time.time() + a.check_every_s))
            h = _health(drv, chg)
            res["health"].append({"t": time.time(), "nodes": h})
            down = sorted(n for n, v in h.items()
                          if re.search(r"\b(corr|rx|engine)=0\b", v) or v.startswith("error"))
            print(f"  t+{time.time() - t_start:6.0f} s  down: {down}", flush=True)
        # 6. results: corr live_2x.json, ring stats, engines
        t_wait = time.time() + a.collect_timeout_s
        for n in drv.tx_nodes:
            od = corr_dir[n]
            r = drv.fetch(n, f"{od}/live_2x.json", max(5.0, t_wait - time.time()))
            if r is None:
                r = drv.fetch(n, f"{od}/live_error.json", 5.0) or {}
                r["missing"] = True
                r["log_tail"] = _log_tail(drv, n, od + ".log", 40)
            sl = drv.run_on(n, f"grep -h x2_fada_replay_start_late_s {od}/writer_live.log 2>/dev/null | tail -1",
                            timeout=30).stdout.strip()
            try:
                r["writer_start_late_s"] = json.loads(sl)["x2_fada_replay_start_late_s"] if sl else 0.0
            except (ValueError, KeyError):
                r["writer_start_late_s"] = sl
            corr_res[n] = r
        for n in drv.rx_nodes:
            ring_res[n] = drv.fetch(n, st_out[n], max(0.0, t_data_end + a.tail_s - time.time()) + 300)
        for n in drv.rx_nodes:
            j = idx[n]
            edir = f"{res_dir}/{a.marker}_engine_{j}_{n}"
            drv.run_on(n, f"for i in $(seq 150); do grep -qs '\"final\": true' {edir}/b8_engine.json && break; "
                          "sleep 2; done", timeout=330)
            e = drv.fetch(n, f"{edir}/b8_engine.json", 5.0)
            halves = {}
            for hh in (2 * j, 2 * j + 1):
                hj = drv.run_on(n, f"cat {edir}/halves/*_half{hh}.json 2>/dev/null | head -c 400000", timeout=60)
                try:
                    halves[hh] = json.loads(hj.stdout) if hj.stdout.strip() else None
                except ValueError:
                    halves[hh] = None
            eng_res[n] = {"summary": e, "halves": halves,
                          "log_tail": None if e else _log_tail(drv, n, f"{res_dir}/{a.marker}_replay_engine_{j}_{n}.log")}
    except BaseException as exc:  # noqa: BLE001 -- record and tear down on any failure (also ^C)
        import traceback
        res["error"] = repr(exc)
        res["traceback"] = traceback.format_exc()[-3000:]
        print(f"fleet-replay {a.marker}: error {exc!r}", flush=True)
    finally:
        print(f"fleet-replay {a.marker}: teardown", flush=True)
        res["teardown"] = drv.teardown_replay(corr_nodes=drv.tx_nodes)
        res["leftovers"] = drv.leftovers(corr_nodes=drv.tx_nodes)
    res["corr_raw"] = corr_res
    res["engines_raw"] = eng_res
    # first + last ring snapshot: enough to re-evaluate (replay-report) without the 10 s series
    res["ring_raw"] = {n: ({"snaps": [r["snaps"][0], r["snaps"][-1]], "iface": r.get("iface")}
                           if r and r.get("snaps") else r) for n, r in ring_res.items()}
    ev = evaluate_replay(res, corr_res, ring_res, eng_res, n_sub=n_sub, lossy=lossy)
    res.update(ev)
    res["leftovers_clean"] = all(not v for v in res["leftovers"].values())
    _write_json(str(out_dir / "fleet_replay.json"), res)
    (out_dir / "fleet_replay.md").write_text(replay_markdown(res))
    print(json.dumps({"passed": res.get("passed"), "gates": res["gates"], "error": res.get("error"),
                      "leftovers_clean": res["leftovers_clean"], "out": str(out_dir)}), flush=True)
    if res.get("error"):
        return 2
    return 0 if res.get("passed") else 1


def _nic_d(s0: dict, s1: dict) -> dict:
    a, b = s0.get("nic") or {}, s1.get("nic") or {}
    return nic_delta(a, b) if a and b else {}


def evaluate_replay(res: dict, corr_res: dict, ring_res: dict, eng_res: dict, *, n_sub: int,
                    lossy: list) -> dict:
    """Gates (G1 full chain):
    corr per node: mean <= 85% of 134.218 ms, p99 <= 100%, zero overruns (no dropped block, the replay
      writer never late, its buffer never full, every block run), zero TX drops;
    corner turn per search node: loss <= max(1e-4, PHY CRC fraction over the same window) on the
      non-lossy flows, zero NIC discards, zero rejected datagrams, fan-in complete;
    search per half: engine status ok, duty <= 0.75, wall p99 <= 0.9 cycle, 0 lag-recovery seeks."""
    n_blocks_req = int(res["minutes"] * 60e3 / BLOCK_MS)
    corr = {}
    for n in res["tx_nodes"]:
        r = corr_res.get(n) or {}
        s = r.get("summary") or {}
        tr = r.get("trace") or {}
        tl = r.get("tx_loopback") or {}
        w = r.get("writer") or {}
        cnt = tl.get("counters") or {}
        sent = list(tl.get("dest_sent") or [])
        enq = list(tl.get("dest_enqueued") or [])
        row = {
            "chgroup": res["chgroup"][n], "missing": bool(r.get("missing") or not r), "rc": r.get("rc"),
            "b8_failed_gates": r.get("failed_gates"),
            "mean_ms": s.get("ms_per_block_mean"), "p50_ms": s.get("ms_per_block_p50"),
            "p99_ms": s.get("ms_per_block_p99"), "max_ms": s.get("ms_per_block_max"),
            "trace_blocks": tr.get("n_blocks"), "trace_mean_ms": tr.get("mean"), "trace_p99_ms": tr.get("p99"),
            "trace_p999_ms": tr.get("p999"), "trace_max_ms": tr.get("max"),
            "trace_hist": tr.get("hist"), "trace_edges_ms": tr.get("edges_ms"),
            "over_budget_blocks": (tr.get("hist") or [None])[-1],
            "n_blocks_req": n_blocks_req, "n_blocks_in": s.get("n_blocks_in"),
            "n_blocks_processed": s.get("n_blocks_processed"), "n_dropped": s.get("n_dropped"),
            "writer_blocks": w.get("n_blocks"), "writer_late_blocks": w.get("late_blocks"),
            "writer_start_late_s": r.get("writer_start_late_s"),
            "replay_max_nfull": r.get("max_nfull"), "replay_nbufs": r.get("nbufs"),
            "tx_sent_per_dest": sent, "tx_bytes_per_dest": tl.get("dest_bytes"),
            "tx_full_drops": sum(tl.get("dest_full_drops") or [0]), "tx_errors": sum(tl.get("dest_errors") or [0]),
            "tx_unrouted": int(cnt.get("unrouted", 0) or 0),
            "tx_unsent_at_close": int(sum(e - x for e, x in zip(enq, sent))) if enq and sent else None,
            "tx_rejected": (s.get("x2_tx") or {}).get("rejected"),
            # over the data window (the sender's own gbps_total also counts the warm-up / wait for the start)
            "tx_gbps": 8.0 * sum(tl.get("dest_bytes") or [0]) / (res["minutes"] * 60.0) / 1e9 if tl else None,
            "tx_drain_p99_ms": (tl.get("drain_ms") or {}).get("p99"),
            "tx_sender_cpu_frac": tl.get("sender_cpu_frac"),
            # corr_fast's CUDA-event stage timer: the GPU time of one block, and per stage
            "gpu_block_ms": ((s.get("stage_timer") or {}).get("stage_ms") or {}).get("block_gpu"),
            "stage_ms": (s.get("stage_timer") or {}).get("stage_ms"),
        }
        overruns = None
        if not row["missing"] and s:
            overruns = (int(s.get("n_dropped") or 0) + int(w.get("late_blocks") or 0)
                        + int(row["replay_max_nfull"] is None or row["replay_nbufs"] is None
                              or row["replay_max_nfull"] >= row["replay_nbufs"]))
        row["overruns"] = overruns
        drops = None
        if tl:
            drops = (row["tx_full_drops"] + row["tx_errors"] + row["tx_unrouted"]
                     + int(row["tx_unsent_at_close"] or 0) + int(row["tx_rejected"] or 0))
        row["tx_drops"] = drops
        g = {"mean_le_85pct": row["mean_ms"] is not None and row["mean_ms"] <= 0.85 * BLOCK_MS,
             "p99_le_100pct": row["p99_ms"] is not None and row["p99_ms"] <= BLOCK_MS,
             "zero_overruns": overruns == 0 and int(s.get("n_blocks_in") or 0) >= n_blocks_req,
             "zero_tx_drops": drops == 0 and sum(sent) > 0}
        row["gates"] = g
        row["passed"] = all(g.values())
        corr[n] = row
    search, halves = {}, {}
    for n in res["rx_nodes"]:
        j = res["search_index"][n]
        st = ring_res.get(n)
        if not st or not st.get("snaps"):
            search[n] = {"missing": True, "passed": False, "gates": {"ring_stats": False}}
            continue
        s0, s1 = st["snaps"][0], st["snaps"][-1]
        w0, w1 = s0["writer"], s1["writer"]
        recv_w = (w1["ok"] + w1["pattern_mismatch"]) - (w0["ok"] + w0["pattern_mismatch"])
        d0, d1 = s0.get("datagrams_per_stream") or [], s1.get("datagrams_per_stream") or []
        dstream = [b - a for a, b in zip(d0, d1)] if d0 else list(d1)
        pairs = {}
        for c in res["tx_nodes"]:
            g_ = res["chgroup"][c]
            sent_c = (corr.get(c) or {}).get("tx_sent_per_dest") or []
            sent_c = int(sent_c[j]) if len(sent_c) > j else None
            recv_c = int(sum(dstream[g_ * n_sub:(g_ + 1) * n_sub])) if dstream else None
            pairs[c] = {"sent": sent_c, "received": recv_c,
                        "loss": (1.0 - recv_c / sent_c) if sent_c and recv_c is not None else None}
        sent_all = sum(p["sent"] or 0 for p in pairs.values())
        ok_pairs = {c: p for c, p in pairs.items() if c not in lossy}
        sent_ex = sum(p["sent"] or 0 for p in ok_pairs.values())
        recv_ex = sum(p["received"] or 0 for p in ok_pairs.values())
        nd = _nic_d(s0, s1)
        crc = nd.get("rx_crc_errors_phy", 0) / nd["rx_packets_phy"] if nd.get("rx_packets_phy") else 0.0
        allowed = max(1e-4, crc)
        discards = nd.get("rx_discards_phy", 0) + nd.get("rx_out_of_buffer", 0)
        rejected = {k: v - w0.get(k, 0) for k, v in w1.items()
                    if k not in ("datagrams", "ok", "bytes", "sweeps") and v - w0.get(k, 0)}
        # fan-in: every stream of every corr in the run delivered to every owned DM, and the engines'
        # windows released complete / by quorum (never by deadline, never a timeout) >= 99.9%
        dsd = s1.get("dm_stream_datagrams") or {}
        exp_streams = [res["chgroup"][c] * n_sub + k for c in res["tx_nodes"] for k in range(n_sub)]
        silent = {d: [s_ for s_ in exp_streams if s_ < len(v) and v[s_] == 0] for d, v in dsd.items()}
        silent = {d: v for d, v in silent.items() if v}
        win = {"complete": 0, "fan_in": 0, "deadline": 0}
        timeouts = rows_missing = rows_present = 0
        for hh, hj in ((eng_res.get(n) or {}).get("halves") or {}).items():
            for d, f in ((hj or {}).get("fan_in") or {}).items():
                for k, v in (f.get("windows") or {}).items():
                    win[k] = win.get(k, 0) + int(v)
                timeouts += int(f.get("timeouts", 0))
                rows_missing += int(f.get("rows_missing", 0))
                rows_present += int(f.get("rows_present", 0))
        n_win = sum(win.values())
        fan_ok = (bool(dsd) and not silent and win.get("deadline", 0) == 0 and timeouts == 0 and n_win > 0
                  and (win["complete"] + win["fan_in"]) >= 0.999 * n_win)
        flows = {k: int(sum(int(v.get(k, 0)) for v in (s1.get("per_dm") or {}).values()))
                 for k in ("rows_written", "rows_hole", "rows_late", "rows_dup", "rows_mismatch", "resyncs")}
        loss_all = 1.0 - recv_w / sent_all if sent_all else None
        loss_ex = 1.0 - recv_ex / sent_ex if sent_ex else None
        g = {"loss_le_max_1e-4_crc": loss_ex is not None and loss_ex <= allowed,
             "zero_nic_discards": bool(nd) and discards == 0,
             "zero_rejected": not rejected,
             "fan_in_complete": fan_ok}
        search[n] = {"index": j, "sent": sent_all, "received_writer": recv_w, "loss_all": loss_all,
                     "sent_excl_lossy": sent_ex, "received_excl_lossy": recv_ex, "loss_excl_lossy": loss_ex,
                     "phy_crc_frac": crc, "allowed_loss": allowed, "nic_delta": nd, "discards": discards,
                     # diagnostic: CRC errors per datagram sent (the corruption is size dependent and the
                     # PHY packet count also holds small non-test frames)
                     "phy_crc_per_datagram": nd.get("rx_crc_errors_phy", 0) / sent_all if sent_all else None,
                     "rejected": rejected, "pairs": pairs, "silent_streams_by_dm": silent,
                     "fan_in_windows": win, "fan_in_timeouts": timeouts,
                     "rows_missing_frac": rows_missing / (rows_missing + rows_present)
                     if rows_missing + rows_present else None,
                     "flows": flows, "stall_at_end": s1.get("stall"),
                     "row_hole_frac": flows["rows_hole"] / (flows["rows_written"] + flows["rows_hole"])
                     if flows["rows_written"] + flows["rows_hole"] else None,
                     "gates": g, "passed": all(g.values())}
        data_s = res["minutes"] * 60.0                       # ring-stats spans more (set-up + tail)
        search[n]["gbps"] = 8.0 * (w1.get("bytes", 0) - w0.get("bytes", 0)) / data_s / 1e9
        search[n]["datagrams_per_s"] = recv_w / data_s
        summ = (eng_res.get(n) or {}).get("summary") or {}
        got = {int(r_["half"]): r_ for r_ in summ.get("results", []) if "half" in r_}
        for hh in (2 * j, 2 * j + 1):
            r_ = got.get(hh)
            if r_ is None:
                halves[hh] = {"node": n, "missing": True, "passed": False}
                continue
            hg = {"engine_status_ok": r_.get("status") == "ok",
                  "duty_le_0.75": r_.get("gpu_duty") is not None and r_["gpu_duty"] <= 0.75,
                  "wall_p99_le_0.9_cycle": r_.get("wall_ms_p99") is not None
                  and r_["wall_ms_p99"] <= 0.9 * CYCLE_S * 1e3,
                  "zero_seeks": r_.get("lag_recovery_seeks") == 0}
            halves[hh] = {"node": n, "gpu_duty": r_.get("gpu_duty"), "duty_aim_met": r_.get("duty_aim_met"),
                          "wall_ms_p99": r_.get("wall_ms_p99"), "lag_recovery_seeks": r_.get("lag_recovery_seeks"),
                          "lag_ms_max": r_.get("lag_ms_max"), "chain_us_per_image": r_.get("chain_us_per_image"),
                          "n_cycles": r_.get("n_cycles"), "stop": r_.get("stop"),
                          "startup_skipped_cycles": r_.get("startup_skipped_cycles"),
                          "fanin_check": r_.get("fanin_check"), "gates": hg, "passed": all(hg.values())}
    lossy_rep = {c: {n: search[n]["pairs"].get(c) for n in search if "pairs" in search[n]}
                 for c in lossy if c in res["tx_nodes"]}
    gates = {"corr": all(r["passed"] for r in corr.values()) and bool(corr),
             "corner_turn": all(r["passed"] for r in search.values()) and bool(search),
             "search": all(r["passed"] for r in halves.values()) and len(halves) == 2 * len(res["rx_nodes"])}
    return {"corr": corr, "search": search, "halves": halves, "lossy_flows": lossy_rep, "gates": gates,
            "passed": all(gates.values()) and not res.get("error")}


def _f(x, fmt="{:.1f}", none="-"):
    try:
        return none if x is None else fmt.format(x)
    except (ValueError, TypeError):
        return str(x)


def replay_markdown(res: dict) -> str:
    L = [f"# G1 fleet-replay {res['marker']}",
         "",
         f"Real 2x corr (`b8_corr2x.py live`, replay {res['replay']}, DEC {res['dec_deg']}) on "
         f"{len(res['tx_nodes'])} corr nodes -> C sender (frame v2, {res['burst_bytes']} B bursts) -> "
         f"`search_rx` v4 -> `b8_owner2x.py engine` on {len(res['rx_nodes'])} search nodes. "
         f"Format {res['format']}, plan {res['plan_id']}, {res['minutes']} min from "
         f"{res.get('start_utc', '-')} (block0 {res.get('block0', '-')}).",
         "",
         f"**Result: {'PASSED' if res.get('passed') else 'FAILED'}** -- gates {res.get('gates')}"
         + (f"; error `{res['error']}`" if res.get("error") else "")
         + f"; leftovers after teardown: {'none' if res.get('leftovers_clean') else res.get('leftovers')}",
         ""]
    if res.get("not_ready_at_start"):
        L += [f"Corr nodes not warm at the start: {res['not_ready_at_start']}", ""]
    L += ["## Corr (per node; block time from corr_fast's summary, over-budget blocks from the trace, which "
          "skips the first 32 blocks)", "",
          "| node | chg | mean ms | p99 ms | max ms | steady p99 / max ms | GPU block mean / p99 ms | "
          "blocks > 134.2 ms | blocks | overruns | late start s | TX Gb/s | TX drops | sent n01/n02/n09/n13 | gates |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in (res.get("corr") or {}).items():
        sent = "/".join(str(x) for x in (r.get("tx_sent_per_dest") or [])) or "-"
        bad = [k for k, v in (r.get("gates") or {}).items() if not v]
        L.append(f"| {n} | {r['chgroup']} | {_f(r['mean_ms'])} | {_f(r['p99_ms'])} | {_f(r['max_ms'])} | "
                 f"{_f(r['trace_p99_ms'])} / {_f(r['trace_max_ms'])} | "
                 f"{_f((r.get('gpu_block_ms') or {}).get('mean'))} / {_f((r.get('gpu_block_ms') or {}).get('p99'))} | "
                 f"{_f(r['over_budget_blocks'], '{}')} | {_f(r['n_blocks_in'], '{}')} | {_f(r['overruns'], '{}')} | "
                 f"{_f(r['writer_start_late_s'], '{:.2f}')} | {_f(r['tx_gbps'], '{:.2f}')} | "
                 f"{_f(r['tx_drops'], '{}')} | {sent} | {'ok' if not bad else 'FAIL ' + ','.join(bad)} |")
    L += ["", "## Corner turn (per search node; loss on the non-lossy flows vs max(1e-4, PHY CRC fraction "
          "over the same window))", "",
          "| node | sent | loss (excl. lossy) | loss (all) | PHY CRC frac | CRC per datagram | NIC discards | "
          "rejected | fan-in windows c/f/d | rows missing | Gb/s | gates |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in (res.get("search") or {}).items():
        if r.get("missing"):
            L.append(f"| {n} | missing | | | | | | | | | | FAIL |")
            continue
        w = r["fan_in_windows"]
        bad = [k for k, v in r["gates"].items() if not v]
        L.append(f"| {n} | {r['sent']} | {_f(r['loss_excl_lossy'], '{:.2e}')} | {_f(r['loss_all'], '{:.2e}')} | "
                 f"{_f(r['phy_crc_frac'], '{:.2e}')} | {_f(r.get('phy_crc_per_datagram'), '{:.2e}')} | "
                 f"{r['discards']} | {r['rejected'] or 0} | "
                 f"{w.get('complete', 0)}/{w.get('fan_in', 0)}/{w.get('deadline', 0)} | "
                 f"{_f(r['rows_missing_frac'], '{:.2e}')} | {_f(r.get('gbps'), '{:.2f}')} | "
                 f"{'ok' if not bad else 'FAIL ' + ','.join(bad)} |")
    if res.get("lossy_flows"):
        L += ["", "### Lossy corr nodes (reported separately, excluded from the loss gate)", "",
              "| corr node | search node | sent | received | loss |", "|---|---|---|---|---|"]
        for c, per in res["lossy_flows"].items():
            for n, p in per.items():
                if p:
                    L.append(f"| {c} | {n} | {p['sent']} | {p['received']} | {_f(p['loss'], '{:.2e}')} |")
    L += ["", "## Search (per half)", "",
          "| half | node | duty | wall p99 ms (gate 181.2) | seeks | chain us/image | cycles | start-up skips | "
          "stop | gates |", "|---|---|---|---|---|---|---|---|---|---|"]
    for hh, r in sorted((res.get("halves") or {}).items()):
        if r.get("missing"):
            L.append(f"| {hh} | {r['node']} | missing | | | | | | | FAIL |")
            continue
        bad = [k for k, v in r["gates"].items() if not v]
        L.append(f"| {hh} | {r['node']} | {_f(r['gpu_duty'], '{:.3f}')} | {_f(r['wall_ms_p99'])} | "
                 f"{r['lag_recovery_seeks']} | {_f(r['chain_us_per_image'], '{:.2f}')} | {r['n_cycles']} | "
                 f"{r['startup_skipped_cycles']} | {r['stop']} | {'ok' if not bad else 'FAIL ' + ','.join(bad)} |")
    L.append("")
    return "\n".join(L)


def cmd_replay_report(a) -> int:
    """Re-evaluate a fleet_replay.json from its raw corr / ring / engine records
    (rewrites fleet_replay.json + .md in --out)."""
    res = json.loads(Path(a.input).read_text())
    ring = res.get("ring_raw") or {}
    ev = evaluate_replay(res, res.get("corr_raw") or {}, ring, res.get("engines_raw") or {},
                         n_sub=_n_sub_of(res["format"]), lossy=res.get("lossy_tx_nodes") or [])
    res.update(ev)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(str(out / "fleet_replay.json"), res)
    (out / "fleet_replay.md").write_text(replay_markdown(res))
    print(json.dumps({"passed": res.get("passed"), "gates": res["gates"], "out": str(out)}))
    return 0


def cmd_teardown(a) -> int:
    drv = Driver(a)
    if getattr(a, "corr_replay", False):
        td = {"teardown": drv.teardown_replay(corr_nodes=drv.tx_nodes),
              "leftovers": drv.leftovers(corr_nodes=drv.tx_nodes)}
    else:
        td = drv.teardown()
    print(json.dumps(td))
    if a.out:
        _write_json(str(_out_path(a.out, "teardown")), td)
    return 0


# =============================================================================
# CLI
# =============================================================================


def _driver_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tx-nodes", required=True)
    p.add_argument("--rx-nodes", required=True)
    p.add_argument("--scratch", required=True, help="node scratch copy (workers run from it)")
    p.add_argument("--marker", required=True, help="DSART_X2_CP tag of every started process")
    p.add_argument("--out", required=False, default=None)
    p.add_argument("--py-node", default=PY_NODE)
    p.add_argument("--ip-map", default=None, help="n01=10.41.0.205,... overrides")
    p.add_argument("--port-base", type=int, default=PORT_BASE)
    p.add_argument("--rx-threads", type=int, default=4)
    p.add_argument("--lead-s", type=float, default=None,
                   help="start delay after every worker is ready (default 3 s sweep, 5 s ring loads)")
    p.add_argument("--ready-timeout-s", type=float, default=900.0,
                   help="max wait for workers to prepare (ring create / value maps)")
    p.add_argument("--baseline-loss", default=None,
                   help="per-node 1x loss: n01=3e-4,n02=7e-4 or a JSON file")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("sweep")
    _driver_args(p)
    p.add_argument("--frame-version", type=int, default=2)
    p.add_argument("--rates-gbps", required=True)
    p.add_argument("--dwell-s", type=float, default=120.0)
    p.add_argument("--gap-s", type=float, default=4.0)
    p.add_argument("--frame-bytes", type=int, default=MAX_DATAGRAM)
    p.add_argument("--record-pps", action="store_true", help="(always recorded)")
    p.add_argument("--burst-bytes", type=int, default=16 << 10,
                   help="sender token-bucket depth per destination (G1 2026-10-29: 512 KB -> 1-5.6%% NIC "
                        "port-buffer discards from incast microbursts; 16 KB -> none up to 17 Gb/s)")
    p.add_argument("--nic-counters", action="store_true")
    p.set_defaults(func=cmd_sweep)
    for name, func in (("fleet-load", cmd_fleet_load), ("snap-live", cmd_snap_live)):
        p = sub.add_parser(name)
        _driver_args(p)
        p.add_argument("--format", required=True, help="op-point id, e.g. 2x-gridded-ns2-cint4")
        p.add_argument("--plan", default=None, help="v3 plan npz path ON THE NODES")
        p.add_argument("--minutes", type=float, default=10.0)
        p.add_argument("--load-factor", type=float, default=1.1 if name == "fleet-load" else None)
        p.add_argument("--antpos-npz", default=None)
        p.add_argument("--dec-deg", type=float, default=71.63)
        p.add_argument("--retention-s", type=float, default=20.0)
        p.add_argument("--dm-subset", default=None, help="debug: comma list of coarse DMs to send")
        p.add_argument("--burst-bytes", type=int, default=16 << 10,
                       help="sender token-bucket depth per destination (see sweep --burst-bytes)")
        p.add_argument("--rx-no-prefault", action="store_true",
                       help="debug: search_rx --no-prefault (small test rings)")
        if name == "fleet-load":
            p.add_argument("--engine", default="", help="owner engine command (WS-S bench)")
        else:
            p.add_argument("--sender", default="c", choices=("c",))
        p.set_defaults(func=func)
    p = sub.add_parser("fleet-replay", help="G1 full chain: the real 2x corr on replayed voltages")
    _driver_args(p)
    p.add_argument("--format", required=True, help="op-point id, e.g. 2x-gridded-ns4-cint4")
    p.add_argument("--plan", required=True, help="v3 plan npz path ON THE SEARCH NODES")
    p.add_argument("--plan-id", required=True, help="the plan's id (corr_fast --plan-id, hex)")
    p.add_argument("--replay", default="250924mptq", help="voltage fixture run id (b8_corr2x --replay)")
    p.add_argument("--replay-file", default=None,
                   help="per-node voltage file overrides: n18=/path/x.out,... (b8_corr2x --replay-file)")
    p.add_argument("--dec-deg", type=float, default=None,
                   help="the corr's pointing DEC (default: the fixture's, 250924mptq 54.5737); the ring "
                        "spec and engines build their value maps at it")
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--antpos-npz", default=None, help="search side antpos (default: the x2 fixture, "
                   "== 250924mptq's cal blob antpos)")
    p.add_argument("--retention-s", type=float, default=20.0)
    p.add_argument("--burst-bytes", type=int, default=16 << 10, help="corr sender token bucket per destination")
    p.add_argument("--ab-mode", default="monitor", choices=("flag", "monitor", "off"),
                   help="corr_fast --rfi-array-burst-mode (monitor: the looped 2 s replay never zeroes blocks)")
    p.add_argument("--engine", default="bench/timeres/b8_owner2x.py engine --gpus 0,1 --input rx-v4",
                   help="owner engine command (scratch-relative)")
    p.add_argument("--absent-rx-host", default="127.0.0.1",
                   help="address for a search node not in --rx-nodes (subset runs): nothing reaches it")
    p.add_argument("--lossy-tx-nodes", default="n22",
                   help="corr nodes with a known PHY fault: their flows are reported separately and left "
                        "out of the corner-turn loss gate")
    p.add_argument("--nic-iface", default=None, help="search NIC for the counters (default: the data IP's)")
    p.add_argument("--tail-s", type=float, default=60.0, help="ring stats until start + minutes + tail")
    p.add_argument("--check-every-s", type=float, default=300.0, help="liveness probe period")
    p.add_argument("--collect-timeout-s", type=float, default=900.0,
                   help="max wait after the data end for every corr's live_2x.json")
    p.add_argument("--skip-preflight", action="store_true", help="run despite preflight findings")
    p.add_argument("--max-dead-corr", type=int, default=0,
                   help="abort (teardown) when more corr workers than this are dead at the start")
    p.set_defaults(func=cmd_fleet_replay, lead_s=180.0)
    p = sub.add_parser("replay-report", help="re-evaluate a fleet_replay.json (gates + markdown)")
    p.add_argument("--in", dest="input", required=True)
    p.add_argument("--out", required=True, help="directory")
    p.set_defaults(func=cmd_replay_report)
    p = sub.add_parser("teardown")
    _driver_args(p)
    p.add_argument("--corr-replay", action="store_true",
                   help="fleet-replay teardown: marker-scoped shm, the corr TEST DADA buffer + sentinel")
    p.set_defaults(func=cmd_teardown)
    # workers
    p = sub.add_parser("tx")
    p.add_argument("--mode", choices=("sweep", "format"), required=True)
    p.add_argument("--chgroup", type=int, required=True)
    p.add_argument("--n-tx", type=int, default=16)
    p.add_argument("--dests", required=True, help="ip:port per rx node, rx-node order")
    p.add_argument("--bind-ip", default="")
    p.add_argument("--rates", default="")
    p.add_argument("--dwell-s", type=float, default=10.0)
    p.add_argument("--gap-s", type=float, default=4.0)
    p.add_argument("--start-at", type=float, default=None)
    p.add_argument("--go-file", default=None)
    p.add_argument("--go-timeout-s", type=float, default=1800.0)
    p.add_argument("--frame-bytes", type=int, default=MAX_DATAGRAM)
    p.add_argument("--burst-bytes", type=int, default=16 << 10)
    p.add_argument("--format", default=None)
    p.add_argument("--plan", default=None)
    p.add_argument("--antpos-npz", default=None)
    p.add_argument("--dec-deg", type=float, default=71.63)
    p.add_argument("--load-factor", type=float, default=1.0)
    p.add_argument("--minutes", type=float, default=1.0)
    p.add_argument("--dm-subset", default=None)
    p.add_argument("--nic", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(func=worker_tx)
    p = sub.add_parser("rx")
    p.add_argument("--bind-ip", required=True)
    p.add_argument("--port-base", type=int, default=PORT_BASE)
    p.add_argument("--n-tx", type=int, default=16)
    p.add_argument("--dms", required=True)
    p.add_argument("--frame-bytes", type=int, default=MAX_DATAGRAM)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--t-buf", type=int, default=1024)
    p.add_argument("--ring", required=True)
    p.add_argument("--snap-at", default="")
    p.add_argument("--go-file", default=None)
    p.add_argument("--go-timeout-s", type=float, default=1800.0)
    p.add_argument("--start-at", type=float, default=None)
    p.add_argument("--n-steps", type=int, default=0)
    p.add_argument("--dwell-s", type=float, default=10.0)
    p.add_argument("--gap-s", type=float, default=4.0)
    p.add_argument("--nic", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(func=worker_rx)
    p = sub.add_parser("ring-stats")
    p.add_argument("--ring", required=True)
    p.add_argument("--end-at", type=float, default=None)
    p.add_argument("--go-file", default=None)
    p.add_argument("--go-timeout-s", type=float, default=1800.0)
    p.add_argument("--start-at", type=float, default=None)
    p.add_argument("--minutes", type=float, default=1.0)
    p.add_argument("--every-s", type=float, default=10.0)
    p.add_argument("--bind-ip", default="")
    p.add_argument("--nic", action="store_true")
    p.add_argument("--nic-iface", default=None, help="NIC for the counters (default: the one holding --bind-ip)")
    p.add_argument("--snap-before-go", action="store_true",
                   help="first snapshot before the ready/go barrier (fleet-replay)")
    p.add_argument("--tail-s", type=float, default=5.0, help="sample until go + minutes + tail")
    p.add_argument("--out", required=True)
    p.set_defaults(func=worker_ring_stats)
    return ap


SUBCOMMANDS = ("sweep", "fleet-load", "snap-live", "fleet-replay", "replay-report", "teardown", "tx", "rx",
               "ring-stats")


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in SUBCOMMANDS:
        return main_v1(argv)
    a = build_parser().parse_args(argv)
    if a.cmd in ("sweep", "fleet-load", "snap-live", "fleet-replay") and not a.out:
        print("--out is required", file=sys.stderr)
        return 2
    try:
        return int(a.func(a))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        if getattr(a, "out", None) and a.cmd in ("sweep", "fleet-load", "snap-live", "fleet-replay"):
            try:
                _write_json(str(_out_path(a.out, a.cmd.replace("-", "_"))), {"error": repr(exc)})
            except Exception:  # noqa: BLE001
                pass
        print(f"fake_corr_to_search {a.cmd}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
