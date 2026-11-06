#!/usr/bin/env python
"""SNAP sync + capture-drop check on the corr nodes, with production stopped (G2 pre-flight).

After a SNAP restart the operator checks the capture drop counters on Grafana: one SNAP armed on the
wrong PPS edge (or not sending) shows up as >= 1/16 of every capture's packets dropped on EVERY corr
node. This tool reproduces that check without starting production:

  node   (on a corr node, from a scratch copy, under tools/x2/prod_guard.sh STOPPED mode)
         two test DADA buffers (production block size) with dada_drain readers; the scratch build of
         dsart_capture_manythread on the real SNAP ports 4011/4012 (test control ports, so it can only be
         armed by this tool); a tcpdump header sample per port -> per-SNAP sequence offsets (direct sync
         check: synced SNAPs emit the same seq_no at the same time); then waits for a common arm,
         sends UTC_START-<seq>, samples the capture's mon shm and tears everything down.
  drive  (h23) launches `node` on every corr node, arms all of them at one future seq_no, collects,
         prints per node/port drop fractions next to a pre-outage Influx baseline window.

    python tools/x2/snap_capture_check.py drive --scratch /home/ubuntu/x2_scratch/x2_g2 --marker snapchk1 \
        --minutes 3 --out /path/to/results
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import struct
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

BLOCK_BYTES = 150_994_944          # 2048 packets x 16 SNAPs x 4608 B (configs/dsart_pipeline_rt.yaml dada/eada)
N_BUFS = 8
DATA_PORTS = (4011, 4012)
TEST_CTRL_PORTS = (11323, 11324)   # production uses 11223/11224
TEST_KEYS = ("c2a0", "c2a2")       # a DADA buffer with key K also uses K+1 (data block)
EXPECTED_PPS = 16 * 2048 / 0.134217728   # per capture (16 SNAPs) = 244,140.6
CORR_DATA_IP = {
    "n03": "10.41.0.224", "n04": "10.41.0.138", "n05": "10.41.0.206", "n06": "10.41.0.203",
    "n07": "10.41.0.208", "n08": "10.41.0.209", "n10": "10.41.0.211", "n11": "10.41.0.216",
    "n12": "10.41.0.218", "n14": "10.41.0.212", "n15": "10.41.0.223", "n16": "10.41.0.220",
    "n18": "10.41.0.228", "n19": "10.41.0.227", "n21": "10.41.0.250", "n22": "10.41.0.233",
}
CN_ID = {n: int(n[1:]) for n in CORR_DATA_IP}        # Influx cn_id tag = host number
NODE_PY = "/home/ubuntu/miniforge3/envs/dsa110-rt/bin/python"


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------- node role
def _iface_of(ip: str) -> str:
    out = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        f = line.split()
        if len(f) > 3 and f[3].split("/")[0] == ip:
            return f[1]
    raise SystemExit(f"no interface holds {ip}")


def decode_header(h: bytes) -> tuple[int, int, int]:
    """SNAP packet header (8 B) -> (seq_no, ch_id, ant_id); bit packing as dsart_capture_manythread.c."""
    seq = ((h[0] << 27) | (h[1] << 19) | (h[2] << 11) | (h[3] << 3) | ((h[4] & 224) >> 5))
    ch = ((h[4] & 31) << 8) | h[5]
    ant = (h[6] << 8) | h[7]
    return seq, ch, ant


def read_pcap_headers(path: str) -> list[tuple[float, int, int]]:
    """(t, seq_no, ant_id) of every UDP packet in a classic pcap (Ethernet, optional 802.1Q)."""
    out = []
    with open(path, "rb") as f:
        g = f.read(24)
        if len(g) < 24:
            return out
        magic = struct.unpack("<I", g[:4])[0]
        end = "<" if magic in (0xA1B2C3D4, 0xA1B23C4D) else ">"
        nano = magic in (0xA1B23C4D, 0x4D3CB2A1)
        while True:
            rh = f.read(16)
            if len(rh) < 16:
                break
            ts, tu, incl, _ = struct.unpack(end + "IIII", rh)
            pkt = f.read(incl)
            off = 14
            if len(pkt) >= 18 and pkt[12:14] == b"\x81\x00":
                off = 18
            if len(pkt) < off + 20:
                continue
            ihl = (pkt[off] & 15) * 4
            p = off + ihl + 8
            if len(pkt) < p + 8:
                continue
            seq, _, ant = decode_header(pkt[p:p + 8])
            out.append((ts + tu * (1e-9 if nano else 1e-6), seq, ant))
    return out


def sync_offsets(pk: list[tuple[float, int, int]]) -> dict:
    """Per SNAP slot (ant_id // 3): median of seq_no - rate * t; offsets vs the median slot, in seq units
    and microseconds. Synced SNAPs agree to network jitter; a SNAP armed on another PPS edge is off by
    ~rate x 1 s."""
    if len(pk) < 100:
        return {"error": f"only {len(pk)} packets sampled"}
    t0 = pk[0][0]
    raw: dict[int, list[tuple[float, int]]] = {}
    for t, s, ant in pk:
        raw.setdefault(ant // 3, []).append((t - t0, s))
    span_t = pk[-1][0] - t0
    # seq units per second: median of the per-slot slopes (a desynced slot must not bias it)
    slopes = sorted((max(s for _, s in v) - min(s for _, s in v)) / (max(t for t, _ in v) - min(t for t, _ in v))
                    for v in raw.values() if len(v) > 10 and max(t for t, _ in v) > min(t for t, _ in v))
    rate = slopes[len(slopes) // 2] if slopes else float("nan")
    by_slot = {k: [s - rate * t for t, s in v] for k, v in raw.items()}
    med = {k: sorted(v)[len(v) // 2] for k, v in by_slot.items()}
    ref = sorted(med.values())[len(med) // 2]
    us_per_seq = 1e6 / rate if rate == rate and rate > 0 else float("nan")
    slots = {str(k): {"n": len(by_slot[k]), "offset_seq": round(med[k] - ref, 2),
                      "offset_us": round((med[k] - ref) * us_per_seq, 1)} for k in sorted(med)}
    worst = max((abs(v["offset_us"]) for v in slots.values()), default=float("nan"))
    return {"n_packets": len(pk), "span_s": round(span_t, 4), "seq_rate_per_s": round(rate, 1),
            "n_slots": len(slots), "max_abs_offset_us": worst, "slots": slots}


def _mon(port: int) -> Optional[dict]:
    from dsart.capture.mon_shm import MonShm
    try:
        with MonShm.open(port) as m:
            return m.snapshot().to_dict()
    except Exception as exc:  # noqa: BLE001 -- not present yet / mid-startup
        return {"error": repr(exc)}


def _prod_capture_running(own: set[int]) -> bool:
    # exact comm (15 chars), not -f: -f also matches command lines that mention the binary
    out = subprocess.run(["pgrep", "-x", "dsart_capture_m"], capture_output=True, text=True).stdout
    return any(int(p) not in own for p in out.split())


def cmd_node(a) -> int:
    sc = os.getcwd()
    if not sc.startswith("/home/ubuntu/x2_scratch"):
        raise SystemExit("node: run from a /home/ubuntu/x2_scratch copy (prod_guard ignores only those)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def _term(signum, _frame):          # the guard / timeout stop us with TERM: still tear down
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _term)
    res: dict = {"node": socket.gethostname(), "data_ip": a.data_ip, "keys": TEST_KEYS, "snaps": [],
                 "t_launch": time.time()}
    iface = _iface_of(a.data_ip)
    res["iface"] = iface
    capbin = os.path.join(sc, "src/dsart/capture/dsart_capture_manythread")
    procs: dict[str, subprocess.Popen] = {}
    made: list[str] = []
    env = dict(os.environ, PYTHONPATH=os.path.join(sc, "src"))
    try:
        if _prod_capture_running(set()):
            raise RuntimeError("a capture binary is already running on this node")
        for k in TEST_KEYS:
            r = subprocess.run(["dada_db", "-k", k, "-b", str(BLOCK_BYTES), "-n", str(N_BUFS), "-l", "-p"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"dada_db -k {k}: {(r.stdout + r.stderr)[-300:]}")
            made.append(k)
        for k in TEST_KEYS:
            procs[f"drain_{k}"] = subprocess.Popen(
                [sys.executable, "-m", "dsart.services.dada_drain", "--key", k, "--log-every", "100000"],
                cwd=sc, env=env, stdout=open(out / f"drain_{k}.log", "w"), stderr=subprocess.STDOUT)
        for k, port, ctrl in zip(TEST_KEYS, DATA_PORTS, TEST_CTRL_PORTS):
            procs[f"cap_{port}"] = subprocess.Popen(
                [capbin, "-o", k, "-p", str(port), "-q", str(ctrl), "-j", a.data_ip],
                cwd=sc, stdout=open(out / f"cap_{port}.log", "w"), stderr=subprocess.STDOUT)
        t_end = time.time() + 60
        while time.time() < t_end:
            m = {p: _mon(p) for p in DATA_PORTS}
            if all(v and (v.get("last_seq_no") or 0) > 0 for v in m.values()):
                break
            for name, p in procs.items():
                if p.poll() is not None:
                    raise RuntimeError(f"{name} exited rc={p.returncode} before data flowed")
            time.sleep(1.0)
        else:
            raise RuntimeError(f"no SNAP packets seen by the captures in 60 s: {m}")
        # direct sync check: a header sample per port (tcpdump sniffs alongside the bound sockets)
        res["sync"] = {}
        for port in DATA_PORTS:
            pc = f"/tmp/x2_snapsync_{a.marker}_{port}.pcap"
            subprocess.run(["sudo", "-n", "timeout", "20", "tcpdump", "-i", iface, "-n", "-s", "64",
                            "-c", str(a.sample_packets), "-w", pc, f"udp dst port {port}"],
                           capture_output=True, text=True)
            try:
                res["sync"][str(port)] = sync_offsets(read_pcap_headers(pc))
            finally:
                subprocess.run(["sudo", "-n", "rm", "-f", pc])
        m1 = {p: _mon(p) for p in DATA_PORTS}
        t1 = time.time()
        time.sleep(2.0)
        m2 = {p: _mon(p) for p in DATA_PORTS}
        t2 = time.time()
        rate = max((m2[p]["last_seq_no"] - m1[p]["last_seq_no"]) / (t2 - t1) for p in DATA_PORTS)
        res["prearm"] = {"t": t2, "last_seq": {str(p): m2[p]["last_seq_no"] for p in DATA_PORTS},
                         "seq_rate_per_s": rate}
        _write_json(str(out / "prearm.json"), res["prearm"])
        _write_json(str(out / "capture_check.json"), res)
        arm_file = out / "arm.json"
        t_end = time.time() + a.arm_timeout_s
        while not arm_file.exists():
            if time.time() > t_end:
                raise RuntimeError("no arm.json from the driver")
            time.sleep(0.5)
        arm = json.loads(arm_file.read_text())["arm_seq"]
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for ctrl in TEST_CTRL_PORTS:
            s.sendto(f"UTC_START-{arm}".encode("ascii"), ("127.0.0.1", ctrl))
        s.close()
        now = {p: _mon(p)["last_seq_no"] for p in DATA_PORTS}
        t_arm = time.time() + (arm - min(now.values())) / rate
        res.update(arm_seq=arm, t_arm_est=t_arm, t_arm_sent=time.time())
        t_stop = t_arm + a.minutes * 60.0
        while time.time() < t_stop:
            res["snaps"].append({"t": time.time(), "mon": {str(p): _mon(p) for p in DATA_PORTS}})
            _write_json(str(out / "capture_check.json"), res)
            for name, p in procs.items():
                if p.poll() is not None:
                    raise RuntimeError(f"{name} exited rc={p.returncode} during the run")
            time.sleep(max(0.5, min(a.every_s, t_stop - time.time())))
        res["snaps"].append({"t": time.time(), "mon": {str(p): _mon(p) for p in DATA_PORTS}, "final": True})
    except BaseException as exc:  # noqa: BLE001 -- record and tear down on any failure (also the guard's TERM)
        res["error"] = repr(exc)
    finally:
        for name, p in procs.items():
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for name, p in procs.items():
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)
            res.setdefault("rc", {})[name] = p.returncode
        for k in made:
            r = subprocess.run(["dada_db", "-d", "-k", k], capture_output=True, text=True)
            res.setdefault("destroy", {})[k] = r.returncode
        own = {p.pid for p in procs.values()}
        if not _prod_capture_running(own):
            for port in DATA_PORTS:                       # a SIGKILLed capture leaves its mon shm behind
                try:
                    os.unlink(f"/dev/shm/dsart-capture-{port}")
                except FileNotFoundError:
                    pass
        res["t_done"] = time.time()
        res["done"] = True
        _write_json(str(out / "capture_check.json"), res)
    return 0 if "error" not in res else 1


# --------------------------------------------------------------------------------------- drive role
def _ssh(node: str, shell: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", f"{node}.pro.pvt", shell],
                          capture_output=True, text=True, timeout=timeout)


def _fetch(node: str, path: str) -> Optional[dict]:
    r = _ssh(node, f"cat {shlex.quote(path)} 2>/dev/null")
    try:
        return json.loads(r.stdout) if r.stdout.strip() else None
    except ValueError:
        return None


def summarise(node_res: dict) -> dict:
    """Per port over the armed run: received/dropped deltas between the first and last mon snapshot
    taken after the arm (blocks written > 0), plus the sync offsets."""
    out = {}
    snaps = [s for s in node_res.get("snaps", []) if s.get("mon")]
    for port in map(str, DATA_PORTS):
        armed = [s for s in snaps if (s["mon"].get(port) or {}).get("n_block_writes", 0) > 0]
        row: dict = {"sync": (node_res.get("sync") or {}).get(port)}
        if len(armed) >= 2:
            a0, a1 = armed[0]["mon"][port], armed[-1]["mon"][port]
            dt = armed[-1]["t"] - armed[0]["t"]
            d = {k: a1[k] - a0[k] for k in ("n_recv_packets", "n_dropped_payload", "n_dropped_kernel",
                                             "n_seq_skipped", "n_too_late", "n_block_writes")}
            tot = d["n_recv_packets"] + d["n_dropped_payload"]
            row.update(d, seconds=round(dt, 1), recv_pps=round(d["n_recv_packets"] / dt) if dt else None,
                       drop_frac=(d["n_dropped_payload"] / tot) if tot else None,
                       recv_vs_expected=round(d["n_recv_packets"] / dt / EXPECTED_PPS, 4) if dt else None,
                       blocks_per_s=round(d["n_block_writes"] / dt, 3) if dt else None)
        else:
            row["error"] = f"{len(armed)} armed snapshots"
        out[port] = row
    return out


def influx_baseline(start: str, end: str) -> dict:
    q = ("SELECT sum(n_recv_packets_delta) AS rcv, sum(n_dropped_payload_delta) AS drp FROM corr_rt_capture "
         f"WHERE time >= '{start}' AND time < '{end}' GROUP BY cn_id, udp_port")
    url = "http://10.42.0.228:8086/query?" + urllib.parse.urlencode({"db": "dsa110", "q": q})
    try:
        d = json.load(urllib.request.urlopen(url, timeout=60))
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}
    base = {}
    for s in d["results"][0].get("series", []):
        v = dict(zip(s["columns"], s["values"][0]))
        tot = (v["rcv"] or 0) + (v["drp"] or 0)
        base[f"{s['tags']['cn_id']}/{s['tags']['udp_port']}"] = (v["drp"] or 0) / tot if tot else None
    return base


def cmd_drive(a) -> int:
    nodes = [n for n in a.nodes.split(",") if n]
    bad = [n for n in nodes if n not in CORR_DATA_IP]
    if bad:
        raise SystemExit(f"not corr nodes: {bad}")
    sc = a.scratch
    rdir = f"{sc}/cp_results/{a.marker}"
    total = int(a.minutes * 60 + a.arm_timeout_s + 300)
    res: dict = {"marker": a.marker, "nodes": nodes, "minutes": a.minutes, "t0": time.time()}
    for n in nodes:
        cmd = (f"env GUARD_REQUIRE_STOPPED=1 bash tools/x2/prod_guard.sh timeout {total} {NODE_PY} "
               f"tools/x2/snap_capture_check.py node --marker {a.marker} --data-ip {CORR_DATA_IP[n]} "
               f"--minutes {a.minutes} --every-s {a.every_s} --arm-timeout-s {a.arm_timeout_s} "
               f"--sample-packets {a.sample_packets} --out {rdir}")
        r = _ssh(n, f"mkdir -p {rdir} && rm -f {rdir}/*.json && cd {sc} && "
                    f"{{ env PYTHONPATH={sc}/src nohup {cmd} > {rdir}/node.log 2>&1 < /dev/null & }}")
        if r.returncode != 0:
            print(f"{n}: launch failed {r.stderr[-300:]}", flush=True)
    print(f"launched on {len(nodes)} corr nodes; waiting for pre-arm", flush=True)
    pre: dict = {}
    t_end = time.time() + 180
    while time.time() < t_end and len(pre) < len(nodes):
        for n in nodes:
            if n not in pre:
                p = _fetch(n, f"{rdir}/prearm.json")
                if p:
                    pre[n] = p
                else:
                    c = _fetch(n, f"{rdir}/capture_check.json")
                    if c and c.get("error"):
                        pre[n] = {"error": c["error"]}
        time.sleep(2.0)
    res["prearm"] = pre
    ok = {n: p for n, p in pre.items() if "last_seq" in p}
    missing = [n for n in nodes if n not in ok]
    if missing:
        print(f"no pre-arm from {missing}: {[pre.get(n) for n in missing]}", flush=True)
    if not ok:
        _write_json(os.path.join(a.out, f"{a.marker}.json"), res)
        return 2
    rate = max(p["seq_rate_per_s"] for p in ok.values())
    now = time.time()
    est = max(max(p["last_seq"].values()) + rate * (now - p["t"]) for p in ok.values())
    arm = int(est + rate * a.arm_lead_s)
    arm += (-arm) % 32                        # on the 32-native grid (2x epoch rule)
    res.update(arm_seq=arm, seq_rate_per_s=rate)
    body = shlex.quote(json.dumps({"arm_seq": arm}))
    for n in ok:
        _ssh(n, f"printf %s {body} > {rdir}/arm.json.tmp && mv {rdir}/arm.json.tmp {rdir}/arm.json")
    print(f"armed {len(ok)} nodes at seq {arm} (~{a.arm_lead_s:.0f} s ahead); running {a.minutes} min",
          flush=True)
    time.sleep(a.arm_lead_s + a.minutes * 60 + 20)
    node_res: dict = {}
    t_end = time.time() + 120
    while time.time() < t_end and len(node_res) < len(ok):
        for n in ok:
            if n not in node_res:
                c = _fetch(n, f"{rdir}/capture_check.json")
                if c and c.get("done"):
                    node_res[n] = c
        time.sleep(3.0)
    res["raw"] = node_res
    res["summary"] = {n: summarise(c) for n, c in node_res.items()}
    res["errors"] = {n: c.get("error") for n, c in node_res.items() if c.get("error")}
    res["baseline_window"] = [a.baseline_start, a.baseline_end]
    base = influx_baseline(a.baseline_start, a.baseline_end)
    res["baseline"] = base
    os.makedirs(a.out, exist_ok=True)
    _write_json(os.path.join(a.out, f"{a.marker}.json"), res)
    print(f"{'node':4} {'port':>5} {'recv/exp':>8} {'blk/s':>6} {'drop_frac':>10} {'base_frac':>10} "
          f"{'kernel':>7} {'too_late':>8} {'sync_max_us':>11} {'slots':>5}")
    for n in nodes:
        for port in map(str, DATA_PORTS):
            r = (res["summary"].get(n) or {}).get(port) or {}
            sy = r.get("sync") or {}
            b = base.get(f"{CN_ID[n]}/{port}") if isinstance(base, dict) else None
            df = r.get("drop_frac")
            print(f"{n:4} {port:>5} {r.get('recv_vs_expected', float('nan')):>8} "
                  f"{r.get('blocks_per_s', float('nan')):>6} "
                  f"{(f'{df:.2e}' if df is not None else r.get('error', '-')):>10} "
                  f"{(f'{b:.2e}' if b is not None else '-'):>10} {r.get('n_dropped_kernel', '-'):>7} "
                  f"{r.get('n_too_late', '-'):>8} {sy.get('max_abs_offset_us', '-'):>11} {sy.get('n_slots', '-'):>5}")
    if res["errors"]:
        print(f"node errors: {res['errors']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("node", "drive"):
        p = sub.add_parser(name)
        p.add_argument("--marker", required=True)
        p.add_argument("--minutes", type=float, default=3.0, help="armed run length")
        p.add_argument("--every-s", type=float, default=5.0, help="mon-shm sample cadence")
        p.add_argument("--arm-timeout-s", type=float, default=300.0)
        p.add_argument("--sample-packets", type=int, default=40000, help="tcpdump header sample per port")
        p.add_argument("--out", required=True)
    p = sub.choices["node"]
    p.add_argument("--data-ip", required=True)
    p = sub.choices["drive"]
    p.add_argument("--nodes", default=",".join(CORR_DATA_IP))
    p.add_argument("--scratch", required=True)
    p.add_argument("--arm-lead-s", type=float, default=15.0)
    p.add_argument("--baseline-start", default="2026-10-21T00:00:00Z")
    p.add_argument("--baseline-end", default="2026-10-22T00:00:00Z")
    a = ap.parse_args(argv)
    return {"node": cmd_node, "drive": cmd_drive}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
