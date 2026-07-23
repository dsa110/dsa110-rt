"""Tests for the candidate-card compositor (dsart.services.candidate_card).

Uses tmp_path fixtures with tiny matplotlib-generated PNGs laid out like a
real candidate event directory (``Level2/plots/*.png``,
``filterbank/<name>.png``) to exercise both render modes
(``"cubes"`` default / ``"full"``), missing panels, missing/empty Level3
data, corrupted PNGs, and the sexagesimal RA/Dec + galactic (l, b) header
derivation. The compositor must never raise.
"""

from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

SRC_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import dsart.services.candidate_card as candidate_card  # noqa: E402
from dsart.services.candidate_card import (  # noqa: E402
    render_card,
    _compute_position,
    _position_header_str,
    _format_ra_hms_colon,
    _format_dec_dms_colon,
    _SIGMA_POS_ARCSEC,
    _line1_str,
    _line2_str,
    _resolve_detection_dm,
)


NAME = "260723test"

# mirrors a real event's c2 dict (260723zmtr) closely enough to exercise
# the RA/Dec/galactic transform with plausible numbers.
C2ROW: Dict[str, Any] = {
    "snr_max": 29.5,
    "dm_median": 169.6,
    "l_median": 0.0,
    "m_median": -0.0113,
    "width_median": 16.0,
    "t_peak_mjd": 61244.577173702724,
    "pointing_dec_deg": 16.27,
    "gal_dm_max_los_pc_cc": 245.3,
}

KEEP_REPORT: Dict[str, Any] = {
    "n_fragments_present": 16,
    "n_fragments_total": 16,
}

_CUBE_PANEL_NAMES = ["image_peak", "dm_time", "kernel_snrs", "lightcurve"]


def _make_tiny_png(path: Path) -> None:
    """A minimal 2x2 image saved via matplotlib, standing in for a real
    plot PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1, 1), dpi=50)
    ax.imshow([[0, 1], [1, 0]])
    ax.axis("off")
    fig.savefig(str(path))
    plt.close(fig)


def _make_full_event_dir(base: Path, name: str = NAME) -> Path:
    ev_dir = base / name
    _make_tiny_png(ev_dir / "filterbank" / f"{name}.png")
    plots = ev_dir / "Level2" / "plots"
    _make_tiny_png(plots / f"dm_time_{name}.png")
    _make_tiny_png(plots / f"image_peak_{name}.png")
    _make_tiny_png(plots / f"kernel_snrs_{name}.png")
    _make_tiny_png(plots / f"lightcurve_{name}.png")
    return ev_dir


def _make_cubes_only_event_dir(base: Path, name: str = NAME) -> Path:
    """No filterbank directory at all — the common case at C3 KEEP time,
    before/without bbproc voltage reprocessing."""
    ev_dir = base / name
    plots = ev_dir / "Level2" / "plots"
    _make_tiny_png(plots / f"dm_time_{name}.png")
    _make_tiny_png(plots / f"image_peak_{name}.png")
    _make_tiny_png(plots / f"kernel_snrs_{name}.png")
    _make_tiny_png(plots / f"lightcurve_{name}.png")
    return ev_dir


# ---------------------------------------------------------------------------
# happy path — default mode ("cubes")
# ---------------------------------------------------------------------------


def test_default_mode_is_cubes_only(tmp_path: Path) -> None:
    ev_dir = _make_full_event_dir(tmp_path)
    out_path = tmp_path / "card.png"

    result = render_card(ev_dir, NAME, C2ROW, out_path, KEEP_REPORT)

    assert result["ok"] is True
    assert result["error"] is None
    assert out_path.is_file()
    assert out_path.stat().st_size > 0
    # cubes mode never includes the filterbank/voltages panel, even
    # though one exists on disk in this fixture.
    assert sorted(result["panels"]) == sorted(_CUBE_PANEL_NAMES)


def test_output_file_reasonably_small(tmp_path: Path) -> None:
    ev_dir = _make_full_event_dir(tmp_path)
    out_path = tmp_path / "card.png"
    result = render_card(ev_dir, NAME, C2ROW, out_path, KEEP_REPORT)
    assert result["ok"] is True
    # spec target: well under ~1.5 MB for tiny stand-in source images
    assert out_path.stat().st_size < 1_500_000


def test_full_mode_output_also_reasonably_small(tmp_path: Path) -> None:
    ev_dir = _make_full_event_dir(tmp_path)
    out_path = tmp_path / "card_full.png"
    result = render_card(ev_dir, NAME, C2ROW, out_path, KEEP_REPORT,
                         mode="full")
    assert result["ok"] is True
    assert out_path.stat().st_size < 1_500_000


# ---------------------------------------------------------------------------
# mode="cubes" — filterbank ignored entirely, no placeholder for it
# ---------------------------------------------------------------------------


def test_cubes_mode_ignores_missing_filterbank_entirely(
    tmp_path: Path,
) -> None:
    """No filterbank/ dir exists at all. cubes mode must render cleanly
    with the four cube panels and no mention/placeholder of filterbank."""
    ev_dir = _make_cubes_only_event_dir(tmp_path)
    out_path = tmp_path / "card.png"

    result = render_card(ev_dir, NAME, C2ROW, out_path, KEEP_REPORT,
                         mode="cubes")

    assert result["ok"] is True
    assert out_path.is_file()
    assert "filterbank" not in result["panels"]
    assert sorted(result["panels"]) == sorted(_CUBE_PANEL_NAMES)


def test_cubes_mode_ignores_filterbank_even_when_present(
    tmp_path: Path,
) -> None:
    """A filterbank PNG exists on disk, but cubes mode must not load or
    report it (data-domain separation: cubes card never touches
    voltages)."""
    ev_dir = _make_full_event_dir(tmp_path)
    out_path = tmp_path / "card.png"

    result = render_card(ev_dir, NAME, C2ROW, out_path, KEEP_REPORT,
                         mode="cubes")

    assert result["ok"] is True
    assert "filterbank" not in result["panels"]


# ---------------------------------------------------------------------------
# mode="full" — cubes grid + voltages panel appended below
# ---------------------------------------------------------------------------


def test_full_mode_includes_all_five_panels(tmp_path: Path) -> None:
    ev_dir = _make_full_event_dir(tmp_path)
    out_path = tmp_path / "card_full.png"

    result = render_card(ev_dir, NAME, C2ROW, out_path, KEEP_REPORT,
                         mode="full")

    assert result["ok"] is True
    assert out_path.is_file()
    assert sorted(result["panels"]) == sorted(
        _CUBE_PANEL_NAMES + ["filterbank"])


def test_full_mode_missing_filterbank_yields_placeholder(
    tmp_path: Path,
) -> None:
    ev_dir = _make_cubes_only_event_dir(tmp_path)
    out_path = tmp_path / "card_full.png"

    result = render_card(ev_dir, NAME, C2ROW, out_path, KEEP_REPORT,
                         mode="full")

    assert result["ok"] is True
    assert out_path.is_file()
    assert "filterbank" not in result["panels"]
    assert sorted(result["panels"]) == sorted(_CUBE_PANEL_NAMES)


def test_unknown_mode_falls_back_to_cubes(tmp_path: Path) -> None:
    ev_dir = _make_full_event_dir(tmp_path)
    out_path = tmp_path / "card.png"

    result = render_card(ev_dir, NAME, C2ROW, out_path, KEEP_REPORT,
                         mode="bogus")

    assert result["ok"] is True
    assert "filterbank" not in result["panels"]
    assert sorted(result["panels"]) == sorted(_CUBE_PANEL_NAMES)


# ---------------------------------------------------------------------------
# missing panels / missing event dir
# ---------------------------------------------------------------------------


def test_entirely_missing_event_dir(tmp_path: Path) -> None:
    """No plots at all present anywhere — every cube panel is a
    placeholder, but the card still renders successfully."""
    ev_dir = tmp_path / "no_such_event"
    out_path = tmp_path / "card.png"

    result = render_card(ev_dir, NAME, C2ROW, out_path, KEEP_REPORT)

    assert result["ok"] is True
    assert out_path.is_file()
    assert result["panels"] == []


# ---------------------------------------------------------------------------
# missing / empty Level3 data
# ---------------------------------------------------------------------------


def test_empty_c2row_handled(tmp_path: Path) -> None:
    ev_dir = _make_full_event_dir(tmp_path)
    out_path = tmp_path / "card.png"

    result = render_card(ev_dir, NAME, {}, out_path, None)

    assert result["ok"] is True
    assert out_path.is_file()
    assert sorted(result["panels"]) == sorted(_CUBE_PANEL_NAMES)


def test_none_keep_report_handled(tmp_path: Path) -> None:
    ev_dir = _make_full_event_dir(tmp_path)
    out_path = tmp_path / "card.png"
    result = render_card(ev_dir, NAME, C2ROW, out_path, keep_report=None)
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# corrupted PNG
# ---------------------------------------------------------------------------


def test_corrupted_png_does_not_raise(tmp_path: Path) -> None:
    ev_dir = _make_full_event_dir(tmp_path)
    # clobber one of the plots with garbage bytes
    (ev_dir / "Level2" / "plots" / f"dm_time_{NAME}.png").write_bytes(
        b"not a real png at all")
    out_path = tmp_path / "card.png"

    result = render_card(ev_dir, NAME, C2ROW, out_path, KEEP_REPORT)

    # best-effort: either it renders with a placeholder for the corrupted
    # panel, or (in the worst case) reports failure cleanly — it must
    # never raise, which pytest would surface as an error rather than a
    # failed assertion.
    assert isinstance(result, dict)
    assert result["ok"] in (True, False)
    if result["ok"]:
        assert "dm_time" not in result["panels"]


def test_all_pngs_corrupted_still_no_raise(tmp_path: Path) -> None:
    ev_dir = tmp_path / NAME
    for rel in (
        f"filterbank/{NAME}.png",
        f"Level2/plots/dm_time_{NAME}.png",
        f"Level2/plots/image_peak_{NAME}.png",
        f"Level2/plots/kernel_snrs_{NAME}.png",
        f"Level2/plots/lightcurve_{NAME}.png",
    ):
        p = ev_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"garbage")
    out_path = tmp_path / "card_full.png"

    result = render_card(ev_dir, NAME, C2ROW, out_path, KEEP_REPORT,
                         mode="full")

    assert isinstance(result, dict)
    assert result["ok"] in (True, False)
    if result["ok"]:
        assert result["panels"] == []


# ---------------------------------------------------------------------------
# error path: unwritable out_path directory must not raise
# ---------------------------------------------------------------------------


def test_bad_out_path_returns_ok_false(tmp_path: Path) -> None:
    ev_dir = _make_full_event_dir(tmp_path)
    # a path with a null byte is guaranteed to fail on POSIX path ops
    bad_out = tmp_path / ("card\x00.png")

    result = render_card(ev_dir, NAME, C2ROW, bad_out, KEEP_REPORT)

    assert result["ok"] is False
    assert result["error"]
    assert result["path"] is None


# ---------------------------------------------------------------------------
# RA/Dec + galactic (l, b) header derivation
# ---------------------------------------------------------------------------


def test_compute_position_with_real_like_inputs() -> None:
    pos = _compute_position(C2ROW)
    assert pos is not None
    assert 0.0 <= pos.ra_deg < 360.0
    assert -90.0 <= pos.dec_deg <= 90.0
    # dec should track close to the pointing dec (m is small)
    assert abs(pos.dec_deg - C2ROW["pointing_dec_deg"]) < 2.0
    # galactic latitude/longitude in their standard ranges
    assert 0.0 <= pos.l_gal_deg < 360.0
    assert -90.0 <= pos.b_gal_deg <= 90.0


def test_position_header_str_is_sexagesimal_with_bracket_uncertainty() -> None:
    s = _position_header_str(C2ROW)
    assert s.startswith("RA ")
    assert "Dec" in s
    # NO decimal degrees anywhere in the RA/Dec portion — sexagesimal only
    assert "°" not in s.split("l=")[0]
    # sexagesimal hh:mm:ss.s / +dd:mm:ss.s
    assert re.search(r"\bRA \d{2}:\d{2}:\d{2}\.\d", s)
    assert re.search(r"Dec [+-]\d{2}:\d{2}:\d{2}\.\d", s)
    # bracket-style uncertainty, astronomy convention, matching the fixed
    # 60 arcsec the live dashboard shows for computed positions
    sigma = int(round(_SIGMA_POS_ARCSEC))
    assert f"(±{sigma}″)" in s
    # galactic l, b in decimal degrees on the same line
    assert re.search(r"l=\d+\.\d°", s)
    assert re.search(r"b=[+-]\d+\.\d°", s)


def test_position_header_str_missing_inputs_reports_unavailable() -> None:
    assert "unavailable" in _position_header_str({})
    assert "unavailable" in _position_header_str({"t_peak_mjd": 61244.5})


def test_format_ra_hms_and_dec_dms_colon_roundish() -> None:
    # 0 deg RA -> 00:00:00.0; 90 deg -> 06:00:00.0
    assert _format_ra_hms_colon(0.0) == "00:00:00.0"
    assert _format_ra_hms_colon(90.0) == "06:00:00.0"
    assert _format_dec_dms_colon(0.0) == "+00:00:00.0"
    assert _format_dec_dms_colon(-16.5) == "-16:30:00.0"


def test_render_card_renders_with_position_in_both_modes(
    tmp_path: Path,
) -> None:
    """Smoke-test that RA/Dec/galactic derivation doesn't break either
    mode's render (the header always includes the position line, even
    though we can't OCR the PNG here — covered textually by the tests
    above)."""
    ev_dir = _make_full_event_dir(tmp_path)
    for mode in ("cubes", "full"):
        out_path = tmp_path / f"card_{mode}.png"
        result = render_card(ev_dir, NAME, C2ROW, out_path, KEEP_REPORT,
                             mode=mode)
        assert result["ok"] is True
        assert out_path.is_file()


def test_render_card_handles_missing_position_inputs_gracefully(
    tmp_path: Path,
) -> None:
    ev_dir = _make_full_event_dir(tmp_path)
    out_path = tmp_path / "card.png"
    c2row_no_pos = {"snr_max": 10.0}
    result = render_card(ev_dir, NAME, c2row_no_pos, out_path, KEEP_REPORT)
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# v3.2 header additions: candidate MJD + galactic-DM discriminant
# ---------------------------------------------------------------------------


def test_line1_includes_raw_candidate_mjd_five_decimals() -> None:
    s = _line1_str(C2ROW)
    assert "MJD 61244.57717" in s
    # still carries the human-readable UTC alongside the raw MJD
    assert "UTC " in s


def test_line2_includes_galactic_dm_max_los() -> None:
    s = _line2_str(C2ROW, KEEP_REPORT)
    assert "gal DM(max LOS)=245.3 pc cm⁻³" in s


def test_line2_galactic_dm_missing_reports_na() -> None:
    s = _line2_str({}, {})
    assert "gal DM(max LOS)=n/a" in s


def test_line1_mjd_missing_reports_na() -> None:
    s = _line1_str({})
    assert "MJD n/a" in s


# ---------------------------------------------------------------------------
# v3.3: width in ms (samples in bracket), search-cube sample period
# 1048.576 us (src/dsart/coinc/plotter.py:649, configs/dsart_search_rt.yaml
# --t-int-search-us 1048.576) -> 16 samp * 1.048576 ms/samp = 16.777216 ms.
# ---------------------------------------------------------------------------


def test_line1_width_reports_ms_first_then_samples_in_bracket() -> None:
    s = _line1_str(C2ROW)
    assert "width=16.8 ms (16 samp)" in s


def test_line1_width_missing_reports_na() -> None:
    s = _line1_str({"width_median": None})
    assert "width=n/a" in s


# ---------------------------------------------------------------------------
# v3.6: header DM = the peak C1 detection's DM (from Level2/C1_window_
# <name>.csv, same resolution the plotter uses to place its reticles), not
# the c2 cluster median; falls back to the cluster median under an explicit
# "DM(cluster med)" label if the CSV is missing/unparsable. Also: DM units
# now rendered as "pc cm⁻³" everywhere in the header (matching the panel
# titles' notation), not the old "pc/cc".
# ---------------------------------------------------------------------------


_C1_WINDOW_FIELDS = [
    "search_node_id", "gpu_half", "fine_dm_idx", "l_pix", "m_pix",
    "dm_pc_cc", "snr", "width_samples", "kernel_id",
]


def _write_c1_window_csv(
    ev_dir: Path, name: str, rows: list,
) -> Path:
    """Synthetic ``Level2/C1_window_<name>.csv`` fixture — same columns
    ``dsart.coinc.plotter._peak_from_csv_rows`` reads (see plotter.py's
    ``_BurstPeak``/``_peak_from_csv_rows``)."""
    path = ev_dir / "Level2" / f"C1_window_{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_C1_WINDOW_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


_PEAK_ROW = {
    "search_node_id": 1, "gpu_half": 0, "fine_dm_idx": 3,
    "l_pix": 76, "m_pix": 128, "dm_pc_cc": 127.8, "snr": 29.5,
    "width_samples": 16, "kernel_id": "unit:d1:b16",
}
_NON_PEAK_ROW = {
    "search_node_id": 1, "gpu_half": 0, "fine_dm_idx": 5,
    "l_pix": 80, "m_pix": 130, "dm_pc_cc": 200.0, "snr": 10.0,
    "width_samples": 16, "kernel_id": "unit:d1:b17",
}


def test_line1_dm_unit_is_pc_cm3_not_pc_cc() -> None:
    s = _line1_str(C2ROW)
    assert "pc cm⁻³" in s
    assert "pc/cc" not in s


def test_resolve_detection_dm_reads_peak_row_from_c1_window_csv(
    tmp_path: Path,
) -> None:
    ev_dir = _make_cubes_only_event_dir(tmp_path)
    _write_c1_window_csv(ev_dir, NAME, [_NON_PEAK_ROW, _PEAK_ROW])
    dm = _resolve_detection_dm(ev_dir, NAME)
    assert dm == pytest.approx(127.8)


def test_resolve_detection_dm_none_when_csv_missing(tmp_path: Path) -> None:
    ev_dir = _make_cubes_only_event_dir(tmp_path)  # no C1_window csv written
    assert _resolve_detection_dm(ev_dir, NAME) is None


def test_line1_dm_value_and_label_can_be_overridden() -> None:
    s = _line1_str(C2ROW, dm_value=127.8, dm_label="DM")
    assert "DM=127.8 pc cm⁻³" in s
    assert "DM(cluster med)" not in s
    # c2's own dm_median (169.6) must NOT leak in when an override is given
    assert "169.6" not in s


def test_render_card_header_uses_detection_dm_when_csv_present(
    tmp_path: Path, monkeypatch,
) -> None:
    ev_dir = _make_cubes_only_event_dir(tmp_path)
    _write_c1_window_csv(ev_dir, NAME, [_NON_PEAK_ROW, _PEAK_ROW])

    captured: Dict[str, Any] = {}
    orig = candidate_card._draw_header_block

    def _spy(fig, subplot_spec, name, line1, line2, position_str):
        captured["line1"] = line1
        return orig(fig, subplot_spec, name, line1, line2, position_str)

    monkeypatch.setattr(candidate_card, "_draw_header_block", _spy)
    out_path = tmp_path / "card.png"
    result = render_card(ev_dir, NAME, C2ROW, out_path, KEEP_REPORT)
    assert result["ok"] is True
    assert "DM=127.8 pc cm⁻³" in captured["line1"]
    assert "DM(cluster med)" not in captured["line1"]


def test_render_card_header_falls_back_to_cluster_median_without_csv(
    tmp_path: Path, monkeypatch,
) -> None:
    ev_dir = _make_cubes_only_event_dir(tmp_path)  # no C1_window csv

    captured: Dict[str, Any] = {}
    orig = candidate_card._draw_header_block

    def _spy(fig, subplot_spec, name, line1, line2, position_str):
        captured["line1"] = line1
        return orig(fig, subplot_spec, name, line1, line2, position_str)

    monkeypatch.setattr(candidate_card, "_draw_header_block", _spy)
    out_path = tmp_path / "card.png"
    result = render_card(ev_dir, NAME, C2ROW, out_path, KEEP_REPORT)
    assert result["ok"] is True
    assert "DM(cluster med)=169.6 pc cm⁻³" in captured["line1"]
