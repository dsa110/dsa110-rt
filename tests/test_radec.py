"""Tests for the /bursts RA/Dec astrometry feature.

Covers event_astrometry (the ICRS recipe, the pointing-dec priority
chain, vectorization + cache) and its wiring into cands_panel_funcs and
the Flask app. The heavy end-to-end guards (worked example + a real
pulsar) exist to catch any LST / sign / precession regression.

Run:
    MKL_INTERFACE_LAYER=GNU,LP64 \
    /home/ubuntu/anaconda3/envs/dsart_h23/bin/python \
    -m pytest tests/test_radec.py -q
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DSA_MONITOR_DIR = REPO_ROOT / "tools" / "dashboard" / "dsa_monitor"
for _p in (str(DSA_MONITOR_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    import event_astrometry as ea
    import cands_panel_funcs as cpf
    import sky_astrometry as sa
except Exception as exc:  # pragma: no cover - env-dependent
    pytest.skip(f"astrometry deps unavailable: {exc!r}",
                allow_module_level=True)


# ---------------------------------------------------------------------------
# Known references
# ---------------------------------------------------------------------------

# Worked example (validated in the feature spec against a known pulsar).
WORKED = dict(
    mjd=61236.164976203705,
    l=0.00436332313,
    m=0.02094395102,
    pointing_dec_deg=16.273406015527343,
    exp_ra=234.0911,
    exp_dec=17.5598,
)

# PSR B1933+16 = J1935+1616 (psrcat), the ground-truth for 260709vmue.
PSR_J1935 = dict(ra=293.9493, dec=16.2778)

# 260709vmue archive inputs (read once from
# /dataz/dsa110/candidates/260709vmue/Level3/260709vmue.json and baked in
# so the test runs with no /dataz). That event carries NO archived
# pointing-declination (no filterbank.json, no calibration/ UVH5), so the
# pointing dec that was in effect is taken as the pulsar-field pointing
# ~16.2778 deg; m_median is tiny (0.0009 rad ~ 0.05 deg) so the dec is
# pointing-dominated. The end-to-end assertion is < 5 arcmin.
VMUE = dict(
    mjd=61230.34404863512,
    l=0.0235619449,
    m=0.000872664626,
    pointing_dec_deg=16.2778,
)


def _sep_arcmin(ra1, dec1, ra2, dec2) -> float:
    a = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg)
    b = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg)
    return float(a.separation(b).arcmin)


# ---------------------------------------------------------------------------
# 1. Worked-example test
# ---------------------------------------------------------------------------


def test_worked_example_icrs() -> None:
    ra, dec = ea.compute_icrs_scalar(
        WORKED["mjd"], WORKED["l"], WORKED["m"], WORKED["pointing_dec_deg"],
    )
    assert abs(ra - WORKED["exp_ra"]) < 0.01, ra
    assert abs(dec - WORKED["exp_dec"]) < 0.01, dec


# ---------------------------------------------------------------------------
# 2. Pulsar ground-truth (end-to-end guard)
# ---------------------------------------------------------------------------


def test_pulsar_hardcoded_within_5_arcmin() -> None:
    """Always-run variant: baked archive inputs -> < 5 arcmin of PSR."""
    ra, dec = ea.compute_icrs_scalar(
        VMUE["mjd"], VMUE["l"], VMUE["m"], VMUE["pointing_dec_deg"],
    )
    sep = _sep_arcmin(ra, dec, PSR_J1935["ra"], PSR_J1935["dec"])
    assert sep < 5.0, f"sep={sep:.2f} arcmin ra={ra:.4f} dec={dec:.4f}"


def test_pulsar_from_archive_within_5_arcmin() -> None:
    """Skip-if-missing variant: read the live archive c2 inputs and
    reproduce the pulsar position. The pointing dec is NOT archived for
    this event, so the pulsar-field pointing (~16.2778) is used — the
    test's job is to guard the LST/sign/precession chain, and the mjd/l/m
    come straight from disk."""
    p = Path("/dataz/dsa110/candidates/260709vmue/Level3/260709vmue.json")
    if not p.is_file():
        pytest.skip("archive event 260709vmue not present")
    doc = json.loads(p.read_text())
    c2 = doc.get("c2") or {}
    mjd = c2.get("t_peak_mjd")
    l = c2.get("l_median")
    m = c2.get("m_median")
    if None in (mjd, l, m):
        pytest.skip("archive event lacks c2 l/m/mjd")
    # Sanity: baked constants must still match the archive.
    assert abs(mjd - VMUE["mjd"]) < 1e-6
    ra, dec = ea.compute_icrs_scalar(mjd, l, m, VMUE["pointing_dec_deg"])
    sep = _sep_arcmin(ra, dec, PSR_J1935["ra"], PSR_J1935["dec"])
    assert sep < 5.0, f"sep={sep:.2f} arcmin"


# ---------------------------------------------------------------------------
# 3. LST regression (observatory longitude + UT1 handling)
# ---------------------------------------------------------------------------


def test_lst_regression() -> None:
    unix = (WORKED["mjd"] - 40587.0) * 86400.0
    lst = sa.lst_deg(unix)
    # Env-true apparent LST for astropy 7.2.0 + the pinned astropy-iers-data
    # in dsart_h23 (UT1-UTC = 0.053 s). The feature spec quotes 234.137;
    # that differs by 0.0013 deg (~4.7 arcsec) because the spec author's
    # environment carried a newer IERS UT1-UTC table — negligible vs our
    # ~1 arcmin accuracy, and within the +-0.9 s UT1 slack the module
    # explicitly accepts. The +-0.001 band still catches an apparent-vs-mean
    # swap (they differ by 0.0023 deg here) and any longitude error (degrees).
    assert abs(lst - 234.1357) < 0.001, lst


# ---------------------------------------------------------------------------
# 4. Sign-convention tests
# ---------------------------------------------------------------------------


def test_positive_l_moves_ra_east() -> None:
    base = ea.compute_icrs_scalar(WORKED["mjd"], 0.0, 0.0,
                                  WORKED["pointing_dec_deg"])
    east = ea.compute_icrs_scalar(WORKED["mjd"], 0.01, 0.0,
                                  WORKED["pointing_dec_deg"])
    # +l -> larger RA (east). Guard the seam at 0/360 with a wrap-safe delta.
    dra = (east[0] - base[0] + 180.0) % 360.0 - 180.0
    assert dra > 0.0, (base[0], east[0])


def test_positive_m_increases_dec() -> None:
    base = ea.compute_icrs_scalar(WORKED["mjd"], 0.0, 0.0,
                                  WORKED["pointing_dec_deg"])
    north = ea.compute_icrs_scalar(WORKED["mjd"], 0.0, 0.01,
                                   WORKED["pointing_dec_deg"])
    assert north[1] > base[1], (base[1], north[1])


def test_l_scales_by_inverse_cos_dec() -> None:
    """The apparent RA offset from +l is degrees(l)/cos(dec): larger at
    higher dec. Compare the RA excursion at dec ~16 vs dec ~60."""
    def ra_excursion(pdec: float) -> float:
        base = ea.compute_icrs_scalar(WORKED["mjd"], 0.0, 0.0, pdec)
        off = ea.compute_icrs_scalar(WORKED["mjd"], 0.02, 0.0, pdec)
        return (off[0] - base[0] + 180.0) % 360.0 - 180.0

    low = ra_excursion(16.0)
    high = ra_excursion(60.0)
    assert high > low > 0.0, (low, high)
    # cos(16)/cos(60) ~ 1.92; allow generous slop for projection/precession.
    assert high / low > 1.5


# ---------------------------------------------------------------------------
# 9. Vectorized == scalar
# ---------------------------------------------------------------------------


def test_vectorized_matches_scalar() -> None:
    mods = [
        ea._Modern(WORKED["mjd"], WORKED["l"], WORKED["m"],
                   WORKED["pointing_dec_deg"], "filterbank"),
        ea._Modern(VMUE["mjd"], VMUE["l"], VMUE["m"],
                   VMUE["pointing_dec_deg"], "filterbank"),
        ea._Modern(61236.5, -0.03, 0.015, 45.0, "filterbank"),
    ]
    ras, decs = ea._compute_icrs_vec(mods)
    for i, mod in enumerate(mods):
        rs, ds = ea.compute_icrs_scalar(mod.mjd, mod.l, mod.m,
                                        mod.pointing_dec_deg)
        assert abs(rs - ras[i]) < 1e-9, (rs, ras[i])
        assert abs(ds - decs[i]) < 1e-9, (ds, decs[i])


# ---------------------------------------------------------------------------
# Fixture builders (mirror tests/test_c2_cands_panel.py)
# ---------------------------------------------------------------------------


def _layout(root: Path, name: str, *, meta: dict | None,
            filterbank_dec: float | None = None) -> Path:
    ev = root / name
    (ev / "Level3").mkdir(parents=True, exist_ok=True)
    (ev / "Level2" / "plots").mkdir(parents=True, exist_ok=True)
    (ev / "cubes").mkdir(parents=True, exist_ok=True)
    if meta is not None:
        (ev / "Level3" / f"{name}.json").write_text(json.dumps(meta))
    if filterbank_dec is not None:
        (ev / "filterbank").mkdir(parents=True, exist_ok=True)
        (ev / "filterbank" / "filterbank.json").write_text(
            json.dumps({"ok": True, "n_fragments": 16,
                        "dec_deg": filterbank_dec}))
    return ev


def _modern_meta(name: str) -> dict:
    return {
        "event_name": name,
        "schema_version": 1,
        "trigger": {"class": "bright_frb"},
        "c2": {
            "n_events": 1,
            "snr_max": 18.4,
            "dm_median": 1702.6,
            "l_median": WORKED["l"],
            "m_median": WORKED["m"],
            "t_peak_mjd": WORKED["mjd"],
        },
    }


# ---------------------------------------------------------------------------
# 5. Legacy-schema pass-through
# ---------------------------------------------------------------------------


def test_legacy_flat_radec_passthrough(tmp_path) -> None:
    meta = {"event_name": "260101aaaa", "ra": 187.7059, "dec": 12.3911,
            "mjds": 60000.5}
    _layout(tmp_path, "260101aaaa", meta=meta)
    (s,) = cpf.ArchiveBrowser(tmp_path).list_events()
    assert s.radec_source == "legacy"
    assert s.ra_deg == pytest.approx(187.7059)
    assert s.dec_deg == pytest.approx(12.3911)


def test_legacy_implausible_radec_rejected(tmp_path) -> None:
    meta = {"event_name": "260101bbbb", "ra": 999.0, "dec": 12.0}
    _layout(tmp_path, "260101bbbb", meta=meta)
    (s,) = cpf.ArchiveBrowser(tmp_path).list_events()
    assert s.ra_deg is None and s.dec_deg is None
    assert s.radec_source is None


# ---------------------------------------------------------------------------
# 6. Missing-inputs event
# ---------------------------------------------------------------------------


def test_missing_inputs_no_position(tmp_path) -> None:
    # Modern schema but no filterbank dec and no legacy ra/dec -> None.
    _layout(tmp_path, "260101cccc", meta=_modern_meta("260101cccc"))
    (s,) = cpf.ArchiveBrowser(tmp_path).list_events()
    assert s.ra_deg is None and s.dec_deg is None
    assert s.radec_source is None


def test_empty_meta_no_position_no_exception(tmp_path) -> None:
    _layout(tmp_path, "260101dddd", meta={})
    (s,) = cpf.ArchiveBrowser(tmp_path).list_events()
    assert s.ra_deg is None and s.radec_source is None


# ---------------------------------------------------------------------------
# 7a. Level3 c2.pointing_dec_deg (coincidencer stamp) — top priority
# ---------------------------------------------------------------------------


def test_level3_pointing_dec_used_over_filterbank(tmp_path) -> None:
    """c2.pointing_dec_deg present -> used, provenance "level3", even
    when a filterbank.json with a DECOY dec also exists."""
    meta = _modern_meta("260101llll")
    meta["c2"]["pointing_dec_deg"] = WORKED["pointing_dec_deg"]
    meta["c2"]["pointing_dec_meta"] = {
        "etcd_key": "/mon/array/pointing_dec", "read_unix": 1784100000.0,
    }
    _layout(tmp_path, "260101llll", meta=meta,
            filterbank_dec=-40.0)  # decoy: must NOT be used
    (s,) = cpf.ArchiveBrowser(tmp_path).list_events()
    assert s.radec_source == "level3"
    assert s.ra_deg == pytest.approx(WORKED["exp_ra"], abs=0.01)
    assert s.dec_deg == pytest.approx(WORKED["exp_dec"], abs=0.01)


def test_level3_pointing_dec_null_falls_through_to_filterbank(
    tmp_path,
) -> None:
    """pointing_dec_deg present-but-null (etcd read failed at trigger
    time) must fall through to the filterbank source — never resolve to
    None-with-source-level3."""
    meta = _modern_meta("260101mmmm")
    meta["c2"]["pointing_dec_deg"] = None
    meta["c2"]["pointing_dec_meta"] = {
        "etcd_key": "/mon/array/pointing_dec", "read_unix": 1784100000.0,
    }
    _layout(tmp_path, "260101mmmm", meta=meta,
            filterbank_dec=WORKED["pointing_dec_deg"])
    (s,) = cpf.ArchiveBrowser(tmp_path).list_events()
    assert s.radec_source == "filterbank"
    assert s.ra_deg == pytest.approx(WORKED["exp_ra"], abs=0.01)


def test_level3_pointing_dec_null_no_other_source_is_none(tmp_path) -> None:
    meta = _modern_meta("260101nnnn")
    meta["c2"]["pointing_dec_deg"] = None
    _layout(tmp_path, "260101nnnn", meta=meta)
    (s,) = cpf.ArchiveBrowser(tmp_path).list_events()
    assert s.ra_deg is None and s.dec_deg is None
    assert s.radec_source is None


def test_level3_pointing_dec_out_of_range_falls_through(tmp_path) -> None:
    meta = _modern_meta("260101oooo")
    meta["c2"]["pointing_dec_deg"] = 123.4  # not a declination
    _layout(tmp_path, "260101oooo", meta=meta,
            filterbank_dec=WORKED["pointing_dec_deg"])
    (s,) = cpf.ArchiveBrowser(tmp_path).list_events()
    assert s.radec_source == "filterbank"


# ---------------------------------------------------------------------------
# 7b. Manual backfill provenance (tools/ops/backfill_pointing_dec.py)
# ---------------------------------------------------------------------------


def test_manual_backfill_meta_gives_manual_provenance(tmp_path) -> None:
    """pointing_dec_meta.source = "manual_backfill" -> provenance
    "manual", same computed position, sigma parenthetical present."""
    meta = _modern_meta("260101rrrr")
    meta["c2"]["pointing_dec_deg"] = WORKED["pointing_dec_deg"]
    meta["c2"]["pointing_dec_meta"] = {
        "etcd_key": None, "read_unix": 1784500000.0,
        "source": "manual_backfill",
        "note": "constant pointing 2026-07-09..15",
    }
    _layout(tmp_path, "260101rrrr", meta=meta)
    (s,) = cpf.ArchiveBrowser(tmp_path).list_events()
    assert s.radec_source == "manual"
    assert s.ra_deg == pytest.approx(WORKED["exp_ra"], abs=0.01)
    assert s.dec_deg == pytest.approx(WORKED["exp_dec"], abs=0.01)
    # Computed source -> the 60-arcsec parenthetical applies identically.
    assert s.ra_hms == "15:36:21.6(42)"
    assert s.dec_dms == "+17:33:35(60)"


def test_manual_in_computed_sources_sexagesimal() -> None:
    rd = ea.RaDec(234.089979, 17.559814, "manual")
    hms, dms = ea.sexagesimal_for(rd)
    assert hms == "15:36:21.6(42)"
    assert dms == "+17:33:35(60)"


def test_meta_absent_provenance_stays_level3(tmp_path) -> None:
    meta = _modern_meta("260101ssss")
    meta["c2"]["pointing_dec_deg"] = WORKED["pointing_dec_deg"]
    # No pointing_dec_meta at all.
    _layout(tmp_path, "260101ssss", meta=meta)
    (s,) = cpf.ArchiveBrowser(tmp_path).list_events()
    assert s.radec_source == "level3"


def test_meta_etcd_source_provenance_stays_level3(tmp_path) -> None:
    """A coincidencer-written meta (etcd provenance) is NOT manual."""
    meta = _modern_meta("260101tttt")
    meta["c2"]["pointing_dec_deg"] = WORKED["pointing_dec_deg"]
    meta["c2"]["pointing_dec_meta"] = {
        "etcd_key": "/mon/array/dec", "read_unix": 1784100000.0,
    }
    _layout(tmp_path, "260101tttt", meta=meta)
    (s,) = cpf.ArchiveBrowser(tmp_path).list_events()
    assert s.radec_source == "level3"


def test_meta_non_dict_provenance_stays_level3(tmp_path) -> None:
    meta = _modern_meta("260101uuuu")
    meta["c2"]["pointing_dec_deg"] = WORKED["pointing_dec_deg"]
    meta["c2"]["pointing_dec_meta"] = "manual_backfill"  # not a dict
    _layout(tmp_path, "260101uuuu", meta=meta)
    (s,) = cpf.ArchiveBrowser(tmp_path).list_events()
    assert s.radec_source == "level3"


# ---------------------------------------------------------------------------
# 7. filterbank present -> used, provenance filterbank, priority over legacy
# ---------------------------------------------------------------------------


def test_filterbank_dec_used(tmp_path) -> None:
    _layout(tmp_path, "260101eeee", meta=_modern_meta("260101eeee"),
            filterbank_dec=WORKED["pointing_dec_deg"])
    (s,) = cpf.ArchiveBrowser(tmp_path).list_events()
    assert s.radec_source == "filterbank"
    assert s.ra_deg == pytest.approx(WORKED["exp_ra"], abs=0.01)
    assert s.dec_deg == pytest.approx(WORKED["exp_dec"], abs=0.01)


def test_filterbank_priority_over_legacy(tmp_path) -> None:
    # Event has BOTH a modern c2 block + filterbank dec AND stray flat
    # ra/dec. Filterbank must win, and the position must be COMPUTED
    # (not the flat ra/dec, which we set to a decoy).
    meta = _modern_meta("260101ffff")
    meta["ra"] = 10.0
    meta["dec"] = -80.0
    _layout(tmp_path, "260101ffff", meta=meta,
            filterbank_dec=WORKED["pointing_dec_deg"])
    (s,) = cpf.ArchiveBrowser(tmp_path).list_events()
    assert s.radec_source == "filterbank"
    assert s.dec_deg == pytest.approx(WORKED["exp_dec"], abs=0.01)
    assert s.dec_deg != pytest.approx(-80.0, abs=1.0)


def test_detail_page_sexagesimal(tmp_path) -> None:
    _layout(tmp_path, "260101gggg", meta=_modern_meta("260101gggg"),
            filterbank_dec=WORKED["pointing_dec_deg"])
    d = cpf.ArchiveBrowser(tmp_path).event_detail("260101gggg")
    assert d.radec_source == "filterbank"
    assert d.ra_hms and ":" in d.ra_hms
    assert d.dec_dms and d.dec_dms.startswith("+")


# ---------------------------------------------------------------------------
# Sexagesimal display strings + catalog-style uncertainty parentheticals
# ---------------------------------------------------------------------------


def test_summary_sexagesimal_pinned_worked_example(tmp_path) -> None:
    """EventSummary for the 260715twmx fixture pins the exact display
    strings, incl. the (NN) uncertainty in units of the last digit:
    RA last digit 0.1 s -> round((60/(15*cos(dec)))/0.1) = 42;
    Dec last digit 1 arcsec -> round(60) = 60."""
    _layout(tmp_path, "260715twmx", meta=_modern_meta("260715twmx"),
            filterbank_dec=WORKED["pointing_dec_deg"])
    (s,) = cpf.ArchiveBrowser(tmp_path).list_events()
    assert s.radec_source == "filterbank"
    assert s.ra_hms == "15:36:21.6(42)"
    assert s.dec_dms == "+17:33:35(60)"


def test_legacy_sexagesimal_no_parenthetical(tmp_path) -> None:
    """Legacy T2 positions have unknown uncertainty -> no (NN)."""
    meta = {"event_name": "260101pppp", "ra": 234.089979, "dec": 17.559814}
    _layout(tmp_path, "260101pppp", meta=meta)
    (s,) = cpf.ArchiveBrowser(tmp_path).list_events()
    assert s.radec_source == "legacy"
    assert s.ra_hms == "15:36:21.6"
    assert s.dec_dms == "+17:33:35"
    assert "(" not in s.ra_hms and "(" not in s.dec_dms


def test_ra_parenthetical_cos_dec_scaling() -> None:
    """Higher |dec| -> larger RA parenthetical (1/cos scaling)."""
    lo = ea.format_ra_hms(100.0, sigma_arcsec=60.0, dec_deg=0.0)
    hi = ea.format_ra_hms(100.0, sigma_arcsec=60.0, dec_deg=60.0)
    lo_n = int(lo[lo.index("(") + 1:-1])
    hi_n = int(hi[hi.index("(") + 1:-1])
    assert lo_n == 40   # 60/15 = 4.0 s -> 40 tenths
    assert hi_n == 80   # 2x at dec 60
    assert hi_n > lo_n


def test_ra_parenthetical_near_pole_capped() -> None:
    s = ea.format_ra_hms(100.0, sigma_arcsec=60.0, dec_deg=89.0)
    assert s.endswith("(>999)")
    s2 = ea.format_ra_hms(100.0, sigma_arcsec=60.0, dec_deg=-89.9)
    assert s2.endswith("(>999)")


def test_format_fns_no_sigma_unchanged() -> None:
    assert ea.format_ra_hms(234.089979) == "15:36:21.6"
    assert ea.format_dec_dms(17.559814) == "+17:33:35"


# ---------------------------------------------------------------------------
# 10. Cache correctness
# ---------------------------------------------------------------------------


def test_cache_avoids_recompute(tmp_path, monkeypatch) -> None:
    _layout(tmp_path, "260101hhhh", meta=_modern_meta("260101hhhh"),
            filterbank_dec=WORKED["pointing_dec_deg"])
    ab = cpf.ArchiveBrowser(tmp_path)

    calls = {"n": 0}
    real = ea._compute_icrs_vec

    def counting(mods):
        if mods:
            calls["n"] += 1
        return real(mods)

    monkeypatch.setattr(ea, "_compute_icrs_vec", counting)

    (s1,) = ab.list_events()
    assert calls["n"] == 1
    (s2,) = ab.list_events()  # cache hit -> no recompute
    assert calls["n"] == 1
    assert (s1.ra_deg, s1.dec_deg) == (s2.ra_deg, s2.dec_deg)


def test_cache_invalidated_by_new_filterbank(tmp_path, monkeypatch) -> None:
    # Start with NO filterbank dec (result None), then add filterbank.json:
    # the key changes and the position recomputes.
    _layout(tmp_path, "260101iiii", meta=_modern_meta("260101iiii"))
    ab = cpf.ArchiveBrowser(tmp_path)

    calls = {"n": 0}
    real = ea._compute_icrs_vec

    def counting(mods):
        if mods:
            calls["n"] += 1
        return real(mods)

    monkeypatch.setattr(ea, "_compute_icrs_vec", counting)

    (s1,) = ab.list_events()
    assert s1.ra_deg is None and calls["n"] == 0
    # A filterbank.json appears (bbproc post-processing).
    fb = tmp_path / "260101iiii" / "filterbank"
    fb.mkdir(parents=True, exist_ok=True)
    (fb / "filterbank.json").write_text(
        json.dumps({"dec_deg": WORKED["pointing_dec_deg"]}))
    (s2,) = ab.list_events()
    assert calls["n"] == 1
    assert s2.radec_source == "filterbank"
    assert s2.ra_deg == pytest.approx(WORKED["exp_ra"], abs=0.01)


# ---------------------------------------------------------------------------
# 8. Flask-level: /bursts renders RA/Dec headers + a computed value
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app_module(tmp_path_factory):
    import sys as _sys
    src = REPO_ROOT / "src"
    if str(src) not in _sys.path:
        _sys.path.insert(0, str(src))
    dbfile = tmp_path_factory.mktemp("annot_radec") / "annot.db"
    os.environ["DSA_MONITOR_ANNOT_DB"] = str(dbfile)
    try:
        with mock.patch("rfi_store.RFIPoller.start", return_value=None):
            import app  # noqa: F401
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"app import needs live resources: {exc!r}")
    app.app.config.update(TESTING=True)
    return app


@pytest.fixture()
def client(app_module):
    return app_module.app.test_client()


def _summary_with_radec(name: str,
                        source: str = "filterbank") -> "cpf.EventSummary":
    rd = ea.RaDec(234.089979, 17.559814, source)
    hms, dms = ea.sexagesimal_for(rd)
    return cpf.EventSummary(
        name=name, mtime_unix=time.time(), mjd_peak=WORKED["mjd"],
        trigger_class="bright_frb", n_events=1, snr_max=18.4,
        dm_median=1702.6, l_median=WORKED["l"], m_median=WORKED["m"],
        n_cubes=0, n_plots=0, c3_action="KEEP",
        ra_deg=rd.ra_deg, dec_deg=rd.dec_deg, radec_source=source,
        ra_hms=hms, dec_dms=dms,
    )


def test_bursts_page_renders_radec(client, app_module) -> None:
    events = [_summary_with_radec("260101jjjj")]
    with mock.patch.object(app_module.cands_browser, "list_events",
                           return_value=events):
        html = client.get("/bursts").get_data(as_text=True)
    assert "RA (J2000)" in html
    assert "Dec (J2000)" in html
    # Sexagesimal cell values incl. the uncertainty parenthetical.
    assert "15:36:21.6(42)" in html
    assert "+17:33:35(60)" in html
    # Decimal degrees live in the per-cell tooltip, with provenance.
    assert "234.0900 deg" in html
    assert "17.5598 deg" in html
    assert "source: filterbank" in html


def test_bursts_page_legacy_tooltip_no_parenthetical(
    client, app_module,
) -> None:
    events = [_summary_with_radec("260101qqqq", source="legacy")]
    with mock.patch.object(app_module.cands_browser, "list_events",
                           return_value=events):
        html = client.get("/bursts").get_data(as_text=True)
    assert "15:36:21.6(" not in html          # no parenthetical for legacy
    assert "15:36:21.6" in html
    assert "uncertainty unknown (legacy value)" in html
