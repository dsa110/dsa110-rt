"""services/search_rx.py ring v4 mode: service lifecycle, status, /mon rx payload."""
from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

rr = pytest.importorskip("dsart.transport.recv_ring")
srx = pytest.importorskip("dsart.services.search_rx")

_spec = importlib.util.spec_from_file_location("wst_util", Path(__file__).with_name("wst_util.py"))
wu = importlib.util.module_from_spec(_spec)
sys.modules["wst_util"] = wu
_spec.loader.exec_module(wu)


class FakeStore:
    def __init__(self):
        self.puts = []

    def put_dict(self, key, payload):
        self.puts.append((key, dict(payload)))


def _free_port_base(n: int) -> int:
    rng = np.random.default_rng()
    for _ in range(50):
        base = int(rng.integers(30000, 60000))
        socks = []
        try:
            for i in range(n):
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                socks.append(s)
                s.bind(("127.0.0.1", base + i))
            return base
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    raise RuntimeError("no free port range")


def small_spec():
    return rr.RingV4Spec(dms=(3, 11), t_int_factor=(16, 32), t_buf=(64, 32), n_stream=4, n_sub=2,
                         bits_main=8, n_grid=256, s_main=float(np.float32(0.3534)),
                         s_heavy=float(np.float32(0.1)), stream_n_values=(20,) * 4,
                         stream_n_heavy=(2,) * 4, min_retention_s=0.0, reorder_rows=8)


def test_service_receives_reports_and_publishes_mon():
    base = _free_port_base(2)
    name = f"/dsart-x2-test-{uuid.uuid4().hex[:8]}"
    store = FakeStore()
    svc = srx.SearchRxV4Service(
        bind_host="127.0.0.1", port_base=base, n_ports=2, shm_name=name, spec=small_spec(),
        so_rcvbuf_bytes=8 << 20, n_threads=2, prefault=False, pattern_ids={0: 7, 1: 7, 2: 7, 3: 7},
        status_interval_s=0.3, mon_cn=13, mon_interval_s=0.2, mon_store=store)
    svc.setup()
    svc.start()
    th = threading.Thread(target=svc.run_status_loop)
    th.start()
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for rep in range(5):
            for s in range(4):
                for seq in range(rep * 8, rep * 8 + 8, 2):
                    if (s, seq) == (1, 18):
                        continue                                     # one lost datagram
                    tx.sendto(wu.synthetic_frame(stream=s, dm=3, seq_first=seq),
                              ("127.0.0.1", base + s // 2))
            time.sleep(0.15)
        time.sleep(0.6)
        st = svc.status_once()
        w = st["counters"]["writer"]
        assert w["ok"] == 4 * 20 - 1 and w["pattern_mismatch"] == 0
        assert st["counters"]["per_dm"][3]["writer"]["rows_hole"] == 2
        assert st["live_edge"][3] == 40
    finally:
        svc.request_stop()
        th.join()
        tx.close()
        svc.shutdown()
    assert not Path("/dev/shm/" + name.lstrip("/")).exists()
    keys = {k for k, _ in store.puts}
    assert keys == {"/mon/search_rt/13/rx"}
    pays = [p for _, p in store.puts]
    assert any(p["rx_gbps"] > 0 for p in pays)
    assert any(p["loss_frac"] > 0 for p in pays)                     # the 2-row hole
    assert all(set(p) == {"rx_gbps", "rx_datagrams_per_s", "loss_frac", "rejected_per_s",
                          "pattern_mismatch_per_s", "stalled_streams", "heartbeat_age_s",
                          "ring", "ts_wall_unix"} for p in pays)
    assert len(json.dumps(pays[-1])) < 400                           # small payload


def test_make_ring4_spec_from_plan_and_value_map(tmp_path):
    plan = wu.plan(2)
    plan.to_npz(tmp_path / "plan.npz")
    out = tmp_path / "spec.json"
    pid = tmp_path / "pids.json"
    rc = srx.main(["make-ring4-spec", "--plan", str(tmp_path / "plan.npz"),
                   "--opoint", "2x-gridded-ns2-cint4", "--node-index", "0",
                   "--antpos-npz", str(wu.FIX), "--dec-deg", str(wu.DEC),
                   "--out", str(out), "--pattern-ids-out", str(pid)])
    assert rc == 0
    spec = rr.RingV4Spec.from_dict(json.loads(out.read_text()))
    assert all(int(plan.owner_half[d]) in (0, 1) for d in spec.dms)
    assert spec.n_stream == 32 and spec.bits_main == 8 and spec.check_scales
    assert all(spec.retention_s(i) >= 20.0 for i in range(len(spec.dms)))
    vm = wu.value_map("gridded", 2, 8)
    assert json.loads(pid.read_text()) == [int(x) for x in vm.stream_pattern_ids]
