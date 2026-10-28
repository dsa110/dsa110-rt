"""bench/fake_corr_to_search.py v2 (CP1 contract): evaluation logic and local
end-to-end driver runs on 127.0.0.1 at low rate (< 50 Mb/s). Full-rate runs
are x2_ckpt (run at checkpoints with ``-m x2_ckpt``)."""
from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("fake_corr_to_search", REPO / "bench" / "fake_corr_to_search.py")
fc = importlib.util.module_from_spec(_spec)
sys.modules["fake_corr_to_search"] = fc
_spec.loader.exec_module(fc)

_spec2 = importlib.util.spec_from_file_location("wst_util", Path(__file__).with_name("wst_util.py"))
wu = importlib.util.module_from_spec(_spec2)
sys.modules["wst_util"] = wu
_spec2.loader.exec_module(wu)


def _drv_args(tmp_path, marker, **kw):
    a = ["--tx-nodes", kw.pop("tx", "localhost,localhost"), "--rx-nodes", kw.pop("rx", "localhost"),
         "--scratch", str(REPO), "--marker", marker, "--py-node", sys.executable,
         "--port-base", str(int(np.random.default_rng().integers(30000, 60000))),
         "--rx-threads", "1"]
    return a


def test_v1_stub_still_works():
    assert fc.main(["--self-test", "--rate", "fast", "--port-base",
                    str(int(np.random.default_rng().integers(30000, 60000)))]) == 0


def test_evaluate_sweep_clean_rule():
    tx = [{"steps": [{"sent_per_dest": [1000], "bytes_per_dest": [8972000], "t0": 0, "t1": 1}
                     for _ in range(3)]} for _ in range(2)]
    # node receives everything at step 0, loses 1e-3 at step 1, everything at step 2
    snaps = [{"datagrams_per_stream": [0], "nic": {"rx_packets_phy": 0, "rx_crc_errors_phy": 0,
                                                    "rx_discards_phy": 0, "rx_out_of_buffer": 0}}]
    tot = 0
    for recv in (2000, 1998, 2000):
        tot += recv
        snaps.append({"datagrams_per_stream": [tot], "nic": {"rx_packets_phy": tot,
                                                             "rx_crc_errors_phy": 0,
                                                             "rx_discards_phy": 0,
                                                             "rx_out_of_buffer": 0}})
    ev = fc.evaluate_sweep(tx, [{"snaps": snaps}], ["n09"], [9.0, 11.0, 13.0], {}, nic=True,
                           frame_bytes=8972)
    st = ev["per_node"]["n09"]
    assert [s["clean"] for s in st] == [True, False, True]
    assert ev["clean_gbps_per_node"] == {"n09": 9.0}                 # monotone: stops at 11
    assert abs(st[1]["loss"] - 1e-3) < 1e-12 and st[0]["datagrams_per_s"] == 2000
    assert st[0]["gbps_wire"] > st[0]["gbps_payload"]
    # a node with a known 1x loss of 2e-3 (PHY CRC) is clean at 1e-3
    ev = fc.evaluate_sweep(tx, [{"snaps": snaps}], ["n09"], [9.0, 11.0, 13.0], {"n09": 2e-3},
                           nic=True, frame_bytes=8972)
    assert ev["clean_gbps_per_node"] == {"n09": 13.0}
    # NIC discards make a step dirty even without datagram loss
    snaps[3]["nic"]["rx_discards_phy"] = 5
    ev = fc.evaluate_sweep(tx, [{"snaps": snaps}], ["n09"], [9.0, 11.0, 13.0], {"n09": 2e-3},
                           nic=True, frame_bytes=8972)
    assert [s["clean"] for s in ev["per_node"]["n09"]] == [True, True, False]


def test_flow_framer_block_parts_and_concat():
    vm = wu.value_map("gridded", 2, 8)
    sv = vm.streams[3]
    from dsart.transport import prod_frame2 as f2
    bps = f2.bytes_per_sample(sv.n_values, sv.n_heavy, 8, 16)
    pool = np.arange(16 * bps, dtype=np.uint64).astype(np.uint8).reshape(16, bps)
    fr = fc.FlowFramer(stream=3, dm=40, n_sub=2, bits_main=8, n_values=sv.n_values,
                       n_heavy=sv.n_heavy, t_int_factor=32, pattern_id=5, n_grid=256, value_kind=0,
                       s_main=0.3534, s_heavy=0.0773, step_main=0.2, step_heavy=0.05,
                       payload_pool=pool, k=f2.max_k(bps))
    full, tail = fr.block_parts(seq_first=128 * 7, rows=128)
    buf, offs, lens = fc.concat_parts([(full, tail)])
    hdrs = [f2.unpack_frame(bytes(buf[o:o + n]))[0] for o, n in zip(offs, lens)]
    assert sum(h.k for h in hdrs) == 128 and max(lens) <= 8972
    assert [h.seq_first for h in hdrs] == list(range(896, 896 + 128, fr.k))[: len(hdrs)]
    assert all(h.t0_native == h.seq_first * 32 for h in hdrs)
    assert [h.last_in_block for h in hdrs] == [False] * (len(hdrs) - 1) + [True]


def test_local_sweep_end_to_end(tmp_path):
    marker = f"t{uuid.uuid4().hex[:6]}"
    out = tmp_path / "capacity.json"
    rc = fc.main(["sweep", "--frame-version", "2"] + _drv_args(tmp_path, marker)
                 + ["--rates-gbps", "0.004,0.008", "--dwell-s", "3", "--gap-s", "2",
                    "--lead-s", "10", "--record-pps", "--out", str(out)])
    res = json.loads(out.read_text())
    if rc != 0:
        pytest.fail("SWEEP " + json.dumps(res.get("error") or {
            k: [(v["sent"], v["received"], v["timing_ok"], round(v["tx_late_s"], 2),
                 round(v["rx_late_s"], 2)) for v in vs] for k, vs in res["per_node"].items()}))
    st = res["per_node"]["localhost"]
    assert [s["clean"] for s in st] == [True, True]
    assert all(s["loss"] == 0 and s["sent"] > 0 for s in st)
    assert abs(st[1]["gbps_payload"] / 0.008 - 1) < 0.25          # paced to the target
    assert res["clean_gbps_per_node"] == {"localhost": 0.008}
    assert not list(Path("/dev/shm").glob(f"dsart-x2-{marker}*"))


def test_local_fleet_load_lite(tmp_path):
    """search_rx v4 (real service) + a format-mode corr TX on one coarse DM,
    WS-P plan and value map, at ~1/5 real cadence (~17 Mb/s)."""
    marker = f"t{uuid.uuid4().hex[:6]}"
    plan = tmp_path / "plan.npz"
    wu.plan(2).to_npz(plan)
    out = tmp_path / "full_load.json"
    rc = fc.main(["fleet-load"] + _drv_args(tmp_path, marker, tx="localhost")
                 + ["--format", "2x-gridded-ns2-cint4", "--plan", str(plan), "--load-factor", "0.2",
                    "--minutes", "0.1", "--dm-subset", "0", "--retention-s", "0.1", "--rx-no-prefault",
                    "--antpos-npz", str(wu.FIX), "--lead-s", "25", "--engine", "", "--out", str(out)])
    res = json.loads(out.read_text())
    if rc != 0:
        pytest.fail("FLEET " + json.dumps({k: res.get(k) for k in ("error", "gates", "per_node", "tx")})[:1500])
    n = res["per_node"]["localhost"]
    assert n["sent"] > 0 and n["loss"] == 0 and n["passed"] and not n["rejected"]
    assert res["tx"][0]["blocks"] >= 5 and res["tx"][0]["full_drops"] == 0
    assert not list(Path("/dev/shm").glob(f"dsart-x2-{marker}*"))


@pytest.mark.x2_ckpt
def test_ckpt_local_sweep_full_rate(request, tmp_path):
    """Checkpoint only (-m x2_ckpt): localhost sweep to a few Gb/s."""
    if "x2_ckpt" not in (request.config.option.markexpr or ""):
        pytest.skip("checkpoint test: run with -m x2_ckpt")
    marker = f"t{uuid.uuid4().hex[:6]}"
    out = tmp_path / "capacity.json"
    rc = fc.main(["sweep", "--frame-version", "2"] + _drv_args(tmp_path, marker, tx="localhost," * 3 + "localhost")
                 + ["--rates-gbps", "1,2,4", "--dwell-s", "10", "--gap-s", "3", "--lead-s", "8",
                    "--out", str(out)])
    assert rc in (0, 1)
    assert json.loads(out.read_text())["clean_gbps_per_node"]["localhost"] >= 1
