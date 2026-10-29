"""x2 WS-C G0 operational fixes in corr_fast:

* ``--plan-id``: the 2x service refuses a tiered plan that is not the set's ``${plan_id}``
  (SPEC §10); 1x refuses the option;
* the RFI FAR thresholds are solved before the ready sentinel (the bandpass test's first use
  otherwise blocks the hot path ~7.4 s, ~150 s into a run);
* ``DSART_HOST_PIN_PERSIST=1`` (GPU): a PSRDADA-like (non-owning) page is registered once and
  never unregistered when its per-block wrapper dies; numpy-owned buffers keep the finalizer;
  without the flag the wrapper churn re-registers the page (today's behaviour, unchanged).
"""
from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import dsart.services.corr_fast_integration as cfi


@pytest.mark.parametrize("layout", ["cal_solutions", "top_level"])
def test_cal_yaml_antenna_order_both_layouts(tmp_path, monkeypatch, layout) -> None:
    """Today's per-node yaml (``cal_solutions: {antenna_order}``) and the 2025-era
    beamformer_weights_<utc>.yaml (``antenna_order`` at the top level, the cal live for
    250924mptq) give the same station numbers to both readers. The order is permuted so a
    silent fall-back to the fleet order would be caught."""
    import numpy as np
    import yaml

    import dsart.cal.bf_weights as bw
    from dsart.grid.sparsity_pattern import core_baseline_mask_from_station_numbers
    fleet = cfi.load_fleet_antenna_order()
    assert fleet is not None
    order = [int(x) for x in np.asarray(fleet)[::-1]]
    body = {"antenna_order": order, "caltime": 60939.17, "flagged_antennas": [], "delays": [0] * len(order)}
    doc = {"cal_solutions": body} if layout == "cal_solutions" else body
    (tmp_path / "beamformer_weights_2025-09-21T03:58:36.yaml").write_text(yaml.safe_dump(doc))
    dat = tmp_path / "beamformer_weights_sb02.dat"
    dat.write_bytes(b"")
    z = np.zeros(len(order), np.float32)
    monkeypatch.setattr(bw, "load_bf_weights", lambda p: SimpleNamespace(antpos_e=z, antpos_n=z))
    assert cfi.cal_yaml_antenna_order(doc) == order
    _, _, mask = cfi.load_antpos_from_cal_blob(dat)
    want = core_baseline_mask_from_station_numbers(order)
    assert np.array_equal(mask, want)
    assert not np.array_equal(want, core_baseline_mask_from_station_numbers(list(fleet)))
    assert np.array_equal(cfi.load_station_numbers_for_cal(dat), np.asarray(order, np.int64))


def test_x2_epoch_from_the_capture_mon_shm(tmp_path, monkeypatch) -> None:
    """SPEC §5: t0_native = 2 x SNAP specnum. The epoch is the armed capture's
    utc_start_specnum from its MON shm (the PSRDADA header is a static template); live
    segments must agree, stale ones (dead pid / old update) are ignored; page n_in = 1 then
    starts at 2 x utc_start_specnum."""
    import struct
    import time

    from dsart.capture import mon_shm as ms
    monkeypatch.setattr(ms, "_SHM_DIR", str(tmp_path))

    def seg(port, utc, *, pid=None, age_s=0.0, state=ms.ArmState.WRITING):
        b = struct.pack(ms._FMT, ms._MON_MAGIC, ms._MON_VERSION, port, 11223, 0, int(state),
                        os.getpid() if pid is None else pid, 0, int((time.time() - age_s) * 1e9),
                        utc, 0, 0, *([0] * 12), 0, 0, 0, 0, 0)
        (tmp_path / f"dsart-capture-{port}").write_bytes(b)

    assert cfi.capture_epoch_specnum() is None                 # nothing armed
    U = 2048 * 123456
    seg(4001, U)
    seg(4002, U)
    seg(4003, U + 7, pid=2 ** 22 + 12345)                       # a dead capture: ignored
    seg(4004, U + 9, age_s=120.0)                               # a stale segment: ignored
    assert cfi.capture_epoch_specnum() == U
    n0 = cfi.x2_native0_from_specnum(U)
    assert n0 + 1 * 4096 == 2 * U and n0 % 32 == 0
    seg(4002, U + 2048)
    assert cfi.capture_epoch_specnum() is None                 # live captures disagree
    n1 = cfi.x2_native0_from_specnum(U + 5)                    # not a multiple of 16
    assert n1 % 32 == 0 and 0 < 2 * (U + 5) - (n1 + 4096) < 32


def test_plan_id_check() -> None:
    plan = SimpleNamespace(plan_id=0x1234ABCD)
    cfi.check_x2_plan_id(plan, None)
    cfi.check_x2_plan_id(plan, "")
    cfi.check_x2_plan_id(plan, "0x1234abcd")
    cfi.check_x2_plan_id(plan, "1234ABCD")
    with pytest.raises(ValueError, match="plan_id"):
        cfi.check_x2_plan_id(plan, "0x1234abce")
    cfg = cfi.FastIntegrationConfig(chgroup=1, obs_dec_rad=1.25)
    with pytest.raises(ValueError, match="plan-id"):
        cfi.run(0, "/tmp/dsart-x2-ws-corr-never", torch.device("cpu"), cfg, opoint="1x",
                plan_id="0x1")


def test_plan_id_reaches_run(monkeypatch) -> None:
    cap = {}
    monkeypatch.setattr(cfi, "run", lambda *a, **kw: cap.update(kw) or {})
    cfi.main(["--chgroup", "1", "--obs-dec-deg", "54.57", "--opoint", "2x-gridded-ns2-cint4",
              "--t-int-fast-native", "8", "--chan-sum-factor", "2", "--n-sub", "2",
              "--transport-tx-mode", "drain", "--plan-id", "0xdeadbeef", "--max-blocks", "1",
              "--output-dir", "/tmp/dsart-x2-ws-corr-never"])
    assert cap["plan_id"] == "0xdeadbeef" and cap["opoint"] == "2x-gridded-ns2-cint4"
    assert cap["x2_tx_burst_bytes"] == 16384 and cap["x2_tx_pace"] == 1.25
    cfi.main(["--chgroup", "1", "--obs-dec-deg", "54.57", "--opoint", "2x-gridded-ns4-cint4",
              "--t-int-fast-native", "8", "--chan-sum-factor", "2", "--n-sub", "4",
              "--transport-tx-mode", "drain", "--x2-tx-burst-bytes", "65536", "--max-blocks", "1",
              "--output-dir", "/tmp/dsart-x2-ws-corr-never"])
    assert cap["x2_tx_burst_bytes"] == 65536


def test_rfi_far_thresholds_solved_at_startup(monkeypatch) -> None:
    import dsart.rfi.far_calibration as fc
    calls = []
    monkeypatch.setattr(fc, "bandpass_threshold_k", lambda *a, **kw: calls.append(("bp", a, kw)) or 4.0)
    monkeypatch.setattr(fc, "group_threshold_k", lambda *a, **kw: calls.append(("grp", a, kw)) or 4.0)
    fl = SimpleNamespace(_m_values=(64, 256, 1024, 4096), _bandpass_k=None, _bandpass_far=1e-4,
                         _group_k=None, _group_far=1e-4)
    cfi._prewarm_rfi_far(SimpleNamespace(rfi_flagger=fl))
    assert [c[0] for c in calls] == ["bp", "grp"]
    assert calls[0][1] == (384, 1e-4) and calls[0][2]["m_acc"] == 4096
    assert calls[1][1] == (96, 1e-4) and calls[1][2] == {"n_chan": 384, "m_acc": 4096}
    calls.clear()
    cfi._prewarm_rfi_far(SimpleNamespace(rfi_flagger=SimpleNamespace(_inner=fl)))   # wrapped flagger
    assert len(calls) == 2
    cfi._prewarm_rfi_far(SimpleNamespace(rfi_flagger=None))


def _psrdada_like_page(m):
    """What psrdada's Reader.getNextPage returns: a READ-ONLY owner-less memoryview
    (PyMemoryView_FromMemory(ptr, size, PyBUF_READ)) over shared memory."""
    import ctypes
    f = ctypes.pythonapi.PyMemoryView_FromMemory
    f.restype, f.argtypes = ctypes.py_object, [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int]
    addr = ctypes.addressof((ctypes.c_char * len(m)).from_buffer(m))
    return f(addr, len(m), 0x100)                                # PyBUF_READ


def test_host_pin_eligibility(tmp_path) -> None:
    """Pinned: PSRDADA pages (read-only owner-less views of shm), writable buffers, read-only
    views of numpy memory. Never: file-backed np.memmap, or read-only memory owned by another
    Python object -- the I2 replay's np.frombuffer(f.read(...)) pages."""
    import mmap

    import numpy as np

    from dsart.services import host_pin as hp
    m = mmap.mmap(-1, 1 << 16)
    page = _psrdada_like_page(m)
    assert page.readonly and page.obj is None
    arr = np.asarray(page)
    assert not arr.flags["WRITEABLE"]
    assert hp._pin_eligible(arr) and hp._pin_eligible(page) and hp._pin_eligible(arr[:100])
    assert not hp._pin_eligible(np.frombuffer(bytes(4096), np.uint8))       # the I2 page
    assert not hp._pin_eligible(memoryview(bytes(64)))
    assert hp._pin_eligible(np.frombuffer(m, np.uint8))                     # writable shm ring
    own = np.zeros(64, np.uint8)
    assert hp._pin_eligible(own)
    v = own[:8]
    v.flags["WRITEABLE"] = False
    assert hp._pin_eligible(v)                                              # numpy-owned memory
    mm = np.memmap(tmp_path / "v.out", dtype=np.uint8, mode="w+", shape=(64,))
    assert not hp._pin_eligible(mm) and not hp._pin_eligible(mm[:8])
    ro = np.memmap(tmp_path / "v.out", dtype=np.uint8, mode="r", shape=(64,))
    assert not hp._pin_eligible(ro)


_PIN = r"""
import ctypes, gc, mmap, numpy as np, torch
from dsart.services import host_pin as hp
m = mmap.mmap(-1, 64 << 20)                              # shm-like memory numpy does not own
f = ctypes.pythonapi.PyMemoryView_FromMemory
f.restype, f.argtypes = ctypes.py_object, [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int]
addr = ctypes.addressof((ctypes.c_char * len(m)).from_buffer(m))
for _ in range(6):
    page = np.asarray(f(addr, len(m), 0x100))            # a fresh read-only wrapper per block, as
    assert hp.maybe_register_host_buffer(page)           # np.asarray(reader.getNextPage())
    del page
    gc.collect()
own = np.zeros(32 << 20, np.uint8)                      # numpy-owned: keeps the finalizer path
assert hp.maybe_register_host_buffer(own)
del own
gc.collect()
for _ in range(3):                                      # the I2 replay's bytes pages: never pinned
    assert not hp.maybe_register_host_buffer(np.frombuffer(bytes(16 << 20), np.uint8))
# a failed runtime call (unregistering memory that is not registered) must not leave its error
# for torch's next kernel-launch check
junk = np.zeros(1 << 20, np.uint8)
hp._make_unregister_finalizer(int(junk.ctypes.data), junk.nbytes, key=-1)()
x = torch.ones(1 << 10, device="cuda")
assert float((x * 2).sum()) == 2048.0
torch.cuda.synchronize()
s = hp.stats()
assert s["skipped_ineligible"] == 3 and s["cuda_errors_cleared"] >= 1, s
print("STATS", s["register_new"], s["finalizer_unregister"] - 1, s["force_clean"], len(hp._FINALIZERS))
"""


@pytest.mark.x2_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only (CP1)")
@pytest.mark.parametrize("persist", [False, True])
def test_host_pin_persist_flag(persist) -> None:
    env = dict(os.environ, DSART_HOST_PIN_PERSIST="1" if persist else "0")
    r = subprocess.run([sys.executable, "-c", _PIN], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    new, unreg, force, n_fin = (int(x) for x in r.stdout.split("STATS")[1].split())
    if persist:
        assert new == 2 and unreg == 1 and force == 0         # the page once; the owned buffer
        assert n_fin == 0                                      # its finalizer entry is removed
    else:
        assert new + force >= 3                                # today: the page is re-pinned
        assert n_fin >= 1                                      # today: entries are never removed


_PREPIN = r"""
import subprocess, numpy as np
from psrdada import Reader, Writer
from dsart.services import host_pin as hp
KEY = 0xd2f8                                            # x2 WS-C test key, removed below
B, N = 1 << 20, 6
subprocess.run(["dada_db", "-d", "-k", "d2f8"], capture_output=True)
assert subprocess.run(["dada_db", "-k", "d2f8", "-b", str(B), "-n", str(N), "-l", "-p", "-r", "1"],
                      capture_output=True).returncode == 0
try:
    w = Writer(KEY)
    w.setHeader({"X2_TEST": "1"})
    r = Reader(KEY)
    r.getHeader()
    pages, bufsz = hp.psrdada_page_addresses(r, KEY)
    info = hp.prepin_psrdada_reader(r, KEY, expected_bufsz=B)
    new0 = hp.stats()["register_new"]
    seen = []
    for i in range(2 * N):
        p = w.getNextPage(); np.asarray(p)[:] = i; w.markFilled()
        q = np.asarray(r.getNextPage())
        seen.append(int(q.ctypes.data))
        assert hp.maybe_register_host_buffer(q)
        r.markCleared()
    print("PREPIN", len(pages), bufsz, info.get("registered"), hp.stats()["register_new"] - new0,
          int(set(seen) <= set(pages)), int(hp.psrdada_page_addresses(r, KEY + 1) is None))
    r.disconnect(); w.disconnect()
finally:
    subprocess.run(["dada_db", "-d", "-k", "d2f8"], capture_output=True)
"""


@pytest.mark.x2_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only (CP1)")
def test_host_pin_prepins_every_psrdada_page() -> None:
    """DSART_HOST_PIN_PERSIST=1: every data-block page of a connected psrdada Reader is
    registered at once (libpsrdada's dada_hdu_db_addresses through the Reader's dada_hdu_t*,
    checked against its key first); the pages the reader then hands out are among them and
    need no further registration; a wrong key refuses to read the layout."""
    env = dict(os.environ, DSART_HOST_PIN_PERSIST="1")
    r = subprocess.run([sys.executable, "-c", _PREPIN], env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    n, bufsz, reg, new_later, subset, wrong_key = (int(x) for x in r.stdout.split("PREPIN")[1].split())
    assert (n, bufsz, reg, new_later, subset, wrong_key) == (6, 1 << 20, 6, 0, 1, 1)
