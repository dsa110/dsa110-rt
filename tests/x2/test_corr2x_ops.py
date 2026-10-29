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


_PIN = r"""
import gc, mmap, numpy as np
from dsart.services import host_pin as hp
m = mmap.mmap(-1, 64 << 20)                              # a shm-like mapping numpy does not own
for _ in range(6):
    page = np.frombuffer(m, dtype=np.uint8)              # a fresh wrapper per block, like np.asarray(page)
    assert hp.maybe_register_host_buffer(page)
    del page
    gc.collect()
own = np.zeros(32 << 20, np.uint8)                      # numpy-owned: keeps the finalizer path
assert hp.maybe_register_host_buffer(own)
del own
gc.collect()
s = hp.stats()
print("STATS", s["register_new"], s["finalizer_unregister"], s["force_clean"])
"""


@pytest.mark.x2_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only (CP1)")
@pytest.mark.parametrize("persist", [False, True])
def test_host_pin_persist_flag(persist) -> None:
    env = dict(os.environ, DSART_HOST_PIN_PERSIST="1" if persist else "0")
    r = subprocess.run([sys.executable, "-c", _PIN], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    new, unreg, force = (int(x) for x in r.stdout.split("STATS")[1].split())
    if persist:
        assert new == 2 and unreg == 1 and force == 0         # the page once; the owned buffer
    else:
        assert new + force >= 3                                # today: the page is re-pinned
