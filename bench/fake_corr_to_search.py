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
    teardown   DRV                                      -> kill marked workers, rm test shm
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
    ``--every-s`` until ``--end-at``; used by fleet-load / snap-live."""
    from dsart.transport.recv_ring import RxRingV4Reader
    iface = detect_data_iface(a.bind_ip) if a.nic else None
    r = RxRingV4Reader.open(a.ring, access="pread", attach_timeout_s=300.0)
    out = {"role": "ring-stats", "host": socket.gethostname(), "iface": iface, "snaps": []}
    if a.end_at is None:
        a.end_at = _barrier(a) + a.minutes * 60.0 + 5.0
    try:
        while True:
            c = r.counters()
            out["snaps"].append({"t": time.time(), "writer": c["writer"],
                                 "per_dm": {str(d): v["writer"] for d, v in c["per_dm"].items()},
                                 "stall": r.stall_report(stall_s=2.0),
                                 "nic": read_nic(iface) if iface else {}})
            if time.time() >= a.end_at:
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

    def _is_local(self, node: str) -> bool:
        return node in ("localhost", "127.0.0.1")

    def _remote_cmd(self, args: list[str], log: str) -> str:
        env = (f"DSART_X2_CP={shlex.quote(self.marker)} PYTHONPATH={shlex.quote(self.scratch)}/src "
               f"DSART_CONFIG_DIR={shlex.quote(self.scratch)}/configs OMP_NUM_THREADS=1 "
               f"MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1")
        cmd = " ".join(shlex.quote(x) for x in args)
        # braces: only the worker is backgrounded (its fds redirected), so the
        # ssh / bash that launched it returns at once
        return (f"mkdir -p {shlex.quote(str(self.results))} && cd {shlex.quote(self.scratch)} && "
                f"{{ env {env} nohup {shlex.quote(self.py)} bench/fake_corr_to_search.py {cmd} "
                f"> {shlex.quote(log)} 2>&1 < /dev/null & }}")

    def launch(self, node: str, args: list[str], name: str) -> str:
        """Start a marked worker on ``node``; returns its result-JSON path
        (named after the marker: results of earlier runs are never read)."""
        name = f"{self.marker}_{name}"
        out = str(self.results / f"{name}.json")
        log = str(self.results / f"{name}.log")
        full = list(args) + ["--out", out]
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


def cmd_teardown(a) -> int:
    td = Driver(a).teardown()
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
    p = sub.add_parser("teardown")
    _driver_args(p)
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
    p.add_argument("--out", required=True)
    p.set_defaults(func=worker_ring_stats)
    return ap


SUBCOMMANDS = ("sweep", "fleet-load", "snap-live", "teardown", "tx", "rx", "ring-stats")


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in SUBCOMMANDS:
        return main_v1(argv)
    a = build_parser().parse_args(argv)
    if a.cmd in ("sweep", "fleet-load", "snap-live") and not a.out:
        print("--out is required", file=sys.stderr)
        return 2
    try:
        return int(a.func(a))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        if getattr(a, "out", None) and a.cmd in ("sweep", "fleet-load", "snap-live"):
            try:
                _write_json(str(_out_path(a.out, a.cmd.replace("-", "_"))), {"error": repr(exc)})
            except Exception:  # noqa: BLE001
                pass
        print(f"fake_corr_to_search {a.cmd}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
