"""x2 D8: the four event panels rendered from cutouts."""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from dsart.coinc import plotter as PL  # noqa: E402
from dsart.coinc.archive import C1_WINDOW_CSV_FIELDS  # noqa: E402
from dsart.coinc.plotter_cutouts import assemble_dm_time  # noqa: E402
from dsart.coinc.wire import CUTOUT_FLAG_CUTOUT, CutoutRequest  # noqa: E402
from dsart.x2.cutout import products as PR  # noqa: E402
from dsart.x2.cutout import schema as S  # noqa: E402
from dsart.x2.opoint import opoint_2x  # noqa: E402
from dsart.x2.plan import build_tiered_plan_for_opoint  # noqa: E402

N = 24
NT = 64
NAME = "261022plot"
SIDS = [1, 2, 9, 13]


@lru_cache(maxsize=None)
def _plan():
    return build_tiered_plan_for_opoint(opoint_2x("gridded", 4, 8))


def _write_half(ev: Path, h: int, fine_global_peak: int, peak_dm: float) -> None:
    plan = _plan()
    ax = S.fine_axis_for_half(plan, h, t_ext_native=16)
    n_f = len(ax["fine_dm_pc_cc"])
    rng = np.random.default_rng(h)
    imgs = rng.normal(size=(NT, n_f, N, N)).astype(np.float32)
    hit = np.flatnonzero(ax["fine_global_idx"] == fine_global_peak)
    if hit.size:
        imgs[32:34, hit[0], 10, 14] += 30.0
    req = CutoutRequest(
        event_name=NAME, request_id=1, t_peak_ext=10_032, t0_ext=10_000,
        n_t=NT, width_ext=2, dm_peak_pc_cc=peak_dm, l_pix=10, m_pix=14,
        fine_global_peak=fine_global_peak,
        coarse_global_peak=int(plan.fine_to_coarse[fine_global_peak]),
        stamp_half_px=4, n_stamp_t=8, stamp_dm_halfwidth_pc_cc=20.0,
        flags=CUTOUT_FLAG_CUTOUT)
    fields = S.assemble_cutout(
        axis=ax, images=imgs, n_t=NT,
        sigma_applied=np.ones((NT, n_f), np.float32),
        row_valid=np.ones((NT, len(ax["coarse_owned"])), bool),
        input_valid_frac=np.ones(n_f, np.float32), request=req,
        identity=dict(search_node_id=SIDS[h // 2], gpu_half=h % 2,
                      half_index=h, plan_id=plan.plan_id),
        clock=dict(t_ext_us=524.288, native_per_ext=16),
        geometry=dict(n_grid=N))
    S.write_cutout(ev / "cutouts", fields)


def _event(tmp: Path, halves) -> Path:
    plan = _plan()
    fg = int(plan.fines_of_half(3)[7])
    dm = float(plan.fine_dm[fg])
    ev = tmp / NAME
    for h in halves:
        _write_half(ev, h, fg, dm)
    (ev / "Level2").mkdir(parents=True, exist_ok=True)
    row = {k: 0 for k in C1_WINDOW_CSV_FIELDS}
    row.update(dict(mjd=61335.0, event_specnum=10_032, snr=25.0, dm_pc_cc=dm,
                    dm_idx_global=fg, fine_dm_idx=3, l_pix=10, m_pix=14,
                    width_samples=2, kernel_id="unit:d1:b2", search_node_id=2,
                    gpu_half=1, trigger=NAME, inj_id=""))
    with open(ev / "Level2" / f"C1_window_{NAME}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(C1_WINDOW_CSV_FIELDS))
        w.writeheader()
        w.writerow(row)
    (ev / "Level3").mkdir(exist_ok=True)
    missing = {f"s{SIDS[h // 2]}g{h % 2}": "no_status"
               for h in range(8) if h not in halves}
    (ev / "Level3" / f"{NAME}.json").write_text(json.dumps(
        {"products": {"missing": missing}}))
    return ev


def test_assemble_dm_time_sorted_and_gapped(tmp_path: Path) -> None:
    ev = _event(tmp_path, [0, 1, 3, 4, 5, 6, 7])          # half 2 missing
    prod = PR.load_event_products(ev)
    img, t_ms, row_dm, row_step, src = assemble_dm_time(prod.planes, 10_032)
    finite = row_dm[np.isfinite(row_dm)]
    assert np.all(np.diff(finite) > 0)                     # sorted by DM
    assert np.isnan(row_dm).any()                          # gap band(s)
    assert t_ms[32] == pytest.approx(0.0)
    assert t_ms[1] - t_ms[0] == pytest.approx(0.524288)
    assert set(row_step[np.isfinite(row_dm)].tolist()) == {1, 2}
    n_real = sum(1 for s in src if s[0] >= 0)
    plan = _plan()
    assert n_real == sum(len(plan.fines_of_half(h)) for h in (0, 1, 3, 4, 5, 6, 7))


def test_render_from_cutouts(tmp_path: Path) -> None:
    _event(tmp_path, [0, 1, 3, 4, 5, 6, 7])
    paths = PL.render_event_plots(PL.PlotJob(event_name=NAME,
                                             archive_root=tmp_path))
    names = sorted(p.name for p in paths)
    assert names == sorted(f"{k}_{NAME}.png" for k in
                           ("dm_time", "image_peak", "lightcurve", "kernel_snrs"))
    for p in paths:
        assert p.is_file() and p.stat().st_size > 5000


def test_render_without_owner_half_uses_placeholders(tmp_path: Path) -> None:
    _event(tmp_path, [0, 1, 2])                            # owner half 3 missing
    paths = {p.name: p for p in PL.render_event_plots(
        PL.PlotJob(event_name=NAME, archive_root=tmp_path))}
    assert paths[f"dm_time_{NAME}.png"].is_file()
    assert paths[f"lightcurve_{NAME}.png"].is_file()


def test_cube_events_never_enter_the_cutout_path(tmp_path: Path, monkeypatch) -> None:
    """1x: no cutouts/ -> the cube renderer runs untouched."""
    import dsart.coinc.plotter_cutouts as PC

    def boom(*a, **k):
        raise AssertionError("cutout path used for a cube event")

    monkeypatch.setattr(PC, "render_cutout_plots", boom)
    (tmp_path / "260801cube" / "cubes").mkdir(parents=True)
    paths = PL.render_event_plots(PL.PlotJob(event_name="260801cube",
                                             archive_root=tmp_path))
    assert len(paths) == 4                                 # placeholders
