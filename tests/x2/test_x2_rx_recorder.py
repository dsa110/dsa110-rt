"""tools/x2/rx_recorder.py on a synthetic v3 ring (CPU only)."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

recv_ring = pytest.importorskip("dsart.transport.recv_ring")
RxRing, RxRingDims = recv_ring.RxRing, recv_ring.RxRingDims

_P = Path(__file__).resolve().parents[2] / "tools" / "x2" / "rx_recorder.py"
_spec = importlib.util.spec_from_file_location("rx_recorder", _P)
rr = importlib.util.module_from_spec(_spec)
sys.modules["rx_recorder"] = rr
_spec.loader.exec_module(rr)

N_CORR, N_DM, T_BUF, N_FILLED = 4, 3, 64, 10
ACTIVE = (0, 2)


def _payload(c: int, d: int, t: int) -> np.ndarray:
    return ((np.arange(N_FILLED * 2) * 7 + c * 31 + d * 17 + t * 3) % 251 - 125).astype(np.int8)


def _scale(c: int, d: int, t: int) -> float:
    return float(np.float32(1.0 + 0.01 * t + c + 0.5 * d))


def _vf(c: int, d: int, t: int) -> int:
    if (c, d, t) == (1, 2, 44):
        return recv_ring.VF_PATTERN_MISMATCH
    return recv_ring.VF_DATA_PRESENT


SKIP = {(3, 2, 45)}            # corr 3 never writes dm 2 at t=45 (a hole)


@pytest.fixture
def ring():
    name = f"/x2rec-{uuid.uuid4().hex[:10]}"
    r = RxRing.open_or_create(name, RxRingDims(N_CORR, N_DM, T_BUF, N_FILLED, 2))
    r.name = name
    yield r
    r.close()
    RxRing.unlink_name(name)


def _write(r, t0: int, t1: int) -> None:
    for t in range(t0, t1):
        for c in range(N_CORR):
            for d in ACTIVE:
                if (c, d, t) in SKIP:
                    r.write_slot(c, d, t, None, 0)          # hole stub, as the RX does
                    continue
                r.write_slot(c, d, t, _payload(c, d, t), _vf(c, d, t),
                             scale=_scale(c, d, t), offset=0.0)


def _cfg(r, out, **kw):
    base = dict(shm_name=r.name, out_dir=out, fan_in_min=N_CORR, lag_rows=2,
                chunk_rows=5, torn_margin_rows=1, poll_s=0.0, duration_s=5,
                max_disk_frac=1.01)
    base.update(kw)
    return rr.RecorderConfig(**base)


def _record(r, out, **kw):
    _write(r, 0, 21)
    rec = rr.Recorder(_cfg(r, out, **kw))
    assert rec.dms == list(ACTIVE) and rec.n_active == 2
    rec.poll_once()                                   # arms at the live edge
    for t0 in range(21, 61, 7):
        _write(r, t0, min(61, t0 + 7))
        rec.poll_once()
    rec.close()
    return rec


def test_round_trip_verbatim(ring, tmp_path):
    out = tmp_path / "rec"
    _record(ring, out)
    hdr0 = (out / "header_start.bin").read_bytes()
    assert len(hdr0) == 4096
    R = rr.Recording(out)
    assert R.manifest["final"] and R.manifest["layout"]["t_buf"] == T_BUF
    for d in ACTIVE:
        a, b = R.t_range(d)
        assert a == 19 and b == 59                    # live edge .. front (61 - lag 2)
        f = R.slots(d, a, b)
        for i, t in enumerate(range(a, b)):
            for c in range(N_CORR):
                if (c, d, t) in SKIP:
                    assert f["vf"][i, c] == 0
                    continue
                np.testing.assert_array_equal(f["payload"][i, c], _payload(c, d, t))
                assert f["scale"][i, c] == np.float32(_scale(c, d, t))
                assert f["vf"][i, c] == _vf(c, d, t)
        rows = R.rows(d)
        np.testing.assert_array_equal(rows["t"], np.arange(a, b))
        assert (rows["dm"] == d).all()
        assert (rows["corr_flags"] & rr.F_WRITTEN).all()
        assert not (rows["corr_flags"] & rr.F_TORN).any()
        assert (rows["wall_ns"] > 0).all()
    r2 = R.rows(2)
    i45 = 45 - R.t_range(2)[0]
    assert r2["n_present"][i45] == N_CORR - 1
    assert r2["n_present"][44 - R.t_range(2)[0]] == N_CORR - 1      # mismatch slot
    assert r2["corr_flags"][44 - R.t_range(2)[0], 1] & rr.F_VF_MISMATCH


def test_iter_cubes_equals_assemble_compact_block(ring, tmp_path):
    out = tmp_path / "rec"
    _record(ring, out)
    R = rr.Recording(out)
    for s in (24, 40):
        cube = next(R.iter_cubes(2, t_det=8, cube_cadence=8, start=s))
        cells, scale, ore, oim, valid, n_ovr, n_pat, n_nod = ring.assemble_compact_block(
            specnum_start=s, t_det=8, owned_dm=2,
            n_filled_per_corr=np.full(N_CORR, N_FILLED, np.int32),
            n_filled_max=N_FILLED, n_active_dms_per_corr=2)
        np.testing.assert_array_equal(cube["cells_packed"], cells)
        np.testing.assert_array_equal(cube["scale"], scale)
        np.testing.assert_array_equal(cube["offset_re"], ore)
        np.testing.assert_array_equal(cube["validity"], valid)
        assert (cube["n_overrun"], cube["n_pattern_mismatch"], cube["n_no_data_present"]) == (n_ovr, n_pat, n_nod)
    c40 = next(R.iter_cubes(2, t_det=8, cube_cadence=8, start=40))
    assert c40["n_pattern_mismatch"] == 1 and c40["n_no_data_present"] == 1
    assert not c40["validity"][44 - 40] and not c40["validity"][45 - 40]


def test_torn_rows_detected_when_writer_laps(ring, tmp_path):
    _write(ring, 0, 21)
    rec = rr.Recorder(_cfg(ring, tmp_path / "rec", dms=[0, 2]))
    rec.poll_once()
    _write(ring, 21, 30)
    # The writer laps the rows being copied (t + T_BUF) while the copy runs.
    rec._after_copy_hook = lambda: _write(ring, 30, 30 + T_BUF)
    rec.poll_once()
    rec.close()
    R = rr.Recording(tmp_path / "rec")
    rows = R.rows(0)
    assert rows.size and (rows["corr_flags"] & rr.F_TORN).all()
    assert (rows["n_torn"] == N_CORR).all() and (rows["n_present"] == 0).all()
    cube = next(R.iter_cubes(0, t_det=4, cube_cadence=4))
    assert not cube["validity"].any() and cube["n_overrun"] == 4 * N_CORR
    assert not cube["cells_packed"].any()


def test_recorder_is_passive(ring, tmp_path):
    _write(ring, 0, 40)
    seg = Path("/dev/shm") / ring.name.lstrip("/")
    before = hashlib.sha256(seg.read_bytes()).hexdigest()
    rec = rr.Recorder(_cfg(ring, tmp_path / "rec", lag_rows=1))
    rec.poll_once()
    rec.cfg.lag_rows = 0
    rec.poll_once()
    rec.close()
    rr.dry_run(ring.name, 0.05)
    assert hashlib.sha256(seg.read_bytes()).hexdigest() == before
    pr = rr.PassiveRing(ring.name)
    with pytest.raises(ValueError):
        pr.data[0, 0, 0, 0] = 1                      # PROT_READ mapping
    pr.close()


def test_disk_guard_stops(ring, tmp_path):
    _write(ring, 0, 21)
    rec = rr.Recorder(_cfg(ring, tmp_path / "rec", max_gib=1e-6))
    rec.poll_once()
    _write(ring, 21, 40)
    rec.poll_once()
    assert rec.stop_reason == "max_gib"
    rec.close()
    assert rr.Recording(tmp_path / "rec").manifest["stop_reason"] == "max_gib"


def test_run_stops_on_stall(ring, tmp_path):
    _write(ring, 0, 30)
    rec = rr.Recorder(_cfg(ring, tmp_path / "rec", stall_s=0.2, poll_s=0.01, duration_s=30))
    t = time.monotonic()
    reason = rec.run()
    assert "stalled" in reason and time.monotonic() - t < 5


def test_dry_run_reports_rates(ring):
    _write(ring, 0, 10)
    stop = threading.Event()

    def writer():
        t = 10
        while not stop.is_set():
            _write(ring, t, t + 1)
            t += 1
            time.sleep(0.005)

    th = threading.Thread(target=writer)
    th.start()
    try:
        rep = rr.dry_run(ring.name, 0.5)
    finally:
        stop.set()
        th.join()
    assert rep["active_dms"] == list(ACTIVE) and rep["n_active_dms_per_corr"] == 2
    assert rep["rows_per_s_median"] > 10 and not rep["corrs_stalled"]
    assert rep["layout"]["slot_stride"] == 32 and rep["recording_MB_per_s"] > 0


def test_replay_into_ring(ring, tmp_path):
    out = tmp_path / "rec"
    _record(ring, out)
    R = rr.Recording(out)
    name = f"/x2rep-{uuid.uuid4().hex[:10]}"
    r2 = RxRing.open_or_create(name, RxRingDims(N_CORR, N_DM, T_BUF, N_FILLED, 2))
    try:
        n = R.replay_into_ring(r2, rebase=True)
        a, b = R.t_range(0)
        assert n == (b - a) * N_CORR * len(ACTIVE)
        for t in (a + 1, b - 1):
            for c in range(N_CORR):
                # (read_slot's overrun test compares slot counts with t; read
                # the slot passively instead)
                pr = rr.PassiveRing(name)
                sl = pr.data[c, 0, (t - a) % T_BUF]
                assert sl[: 2 * N_FILLED].view(np.int8).tolist() == _payload(c, 0, t).tolist()
                assert sl[2 * N_FILLED + 8:2 * N_FILLED + 10].view("<u2")[0] == recv_ring.VF_DATA_PRESENT
                assert sl[2 * N_FILLED:2 * N_FILLED + 4].view("<f4")[0] == np.float32(_scale(c, 0, t))
                del sl
                pr.close()
    finally:
        r2.close()
        RxRing.unlink_name(name)


def test_flush_path_and_cli_dry_run(ring, tmp_path, monkeypatch, capsys):
    out = tmp_path / "rec"
    _record(ring, out, sync_every_mb=1e-4)               # fdatasync + DONTNEED every chunk
    assert rr.Recording(out).t_range(0) == (19, 59)
    monkeypatch.setattr(rr, "_lower_priority", lambda cpu: None)
    assert rr.main(["--shm-name", ring.name, "--dry-run", "--dry-run-s", "0.05"]) == 0
    assert '"active_dms"' in capsys.readouterr().out
