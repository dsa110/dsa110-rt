"""ICRS (RA, Dec) resolution for archived burst events.

Turns the per-event Level3 metadata into sky coordinates for the
``/bursts`` table and the ``/bursts/<name>`` detail page. All of the
astropy / sidereal-time / precession machinery lives here so that
``cands_panel_funcs`` and ``app`` stay free of astrometry logic.

Coordinate recipe (modern schema, validated twice against known
pulsars)
------------------------------------------------------------------
For an event whose Level3 JSON carries a ``c2`` block:

  * inputs: ``t_peak_mjd`` (UTC MJD), ``l_median`` (rad, east-positive),
    ``m_median`` (rad, north-positive) and a *pointing declination*
    (array meridian dec, degrees).
  * ``LST`` = apparent sidereal time at OVRO at ``t_peak``.
  * ``dec_apparent = pointing_dec_deg + degrees(m)``
  * ``ra_apparent  = LST_deg + degrees(l) / cos(radians(dec_apparent))``
  * these are *apparent* (epoch-of-date / TETE) coordinates; the
    ``tete -> icrs`` transform (obstime = t_peak) moves the position
    ~0.3 deg and is **not** optional.

Pointing-declination sourcing (priority chain, provenance recorded)
------------------------------------------------------------------
  1. Level3 ``c2.pointing_dec_deg`` — stamped by the coincidencer from
     the live etcd pointing key minutes after the trigger; the earliest
     contemporaneous record, present on every future event (may be null
     if the etcd read failed -> falls through). Provenance ``"level3"``
     — unless the sibling ``c2.pointing_dec_meta`` dict declares a
     ``"source"`` starting with ``"manual"`` (operator backfill via
     ``tools/ops/backfill_pointing_dec.py``), in which case provenance
     is ``"manual"``. Same 60-arcsec sigma applies (identical recipe;
     only how the pointing dec was recorded differs).
  2. ``filterbank/filterbank.json`` key ``dec_deg`` — present only for
     bbproc-processed (2026-07 transition-era) events.
     Provenance ``"filterbank"``.
  3. Legacy flat Level3 schema (pre-2026-07 events) with top-level
     numeric ``ra``/``dec`` — these are already ICRS; we pass them
     through unchanged (NO recompute). Provenance ``"legacy"``.
  4. UVH5 phase-center dec under ``<event>/calibration/`` — **skipped**:
     ``h5py`` is not installed in the ``dsart_h23`` dashboard env, and
     the deployment forbids new pip dependencies. See module note below.
  5. no source -> ``(None, None, None)``; the UI renders "—".

We never fall back to the live etcd dec for a historical event.

Performance
-----------
``EventAstrometry.compute`` batches every event that needs the
TETE->ICRS transform into a *single* array ``SkyCoord`` per page
render, and memoises the per-event result keyed on the exact input
tuple, so a page re-render (or a later-appearing ``filterbank.json``,
which changes the key) is cheap / correctly invalidated.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

# Reuse the single source of truth for the OVRO site (do NOT introduce
# new observatory constants — see sky_astrometry docstring).
from sky_astrometry import OVRO_LAT_DEG, OVRO_LON_DEG

LOG = logging.getLogger("dsa_monitor.event_astrometry")

#: OVRO site height (m); matches sky_astrometry.lst_deg.
OVRO_HEIGHT_M: float = 1222.0

__all__ = [
    "RaDec",
    "EventAstrometry",
    "SIGMA_POS_ARCSEC",
    "resolve_inputs",
    "read_filterbank_dec",
    "format_ra_hms",
    "format_dec_dms",
    "sexagesimal_for",
]


# ---------------------------------------------------------------------------
# Result + intermediate types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RaDec:
    """A resolved sky position (or the empty result)."""

    ra_deg: Optional[float]
    dec_deg: Optional[float]
    source: Optional[str]
    # "level3" | "manual" | "filterbank" | "legacy" | None


@dataclass(frozen=True)
class _Modern:
    """A modern-schema event that still needs the LST/precession compute."""

    mjd: float
    l: float
    m: float
    pointing_dec_deg: float
    source: str

    def key(self) -> Tuple:
        return ("modern", self.mjd, self.l, self.m,
                self.pointing_dec_deg, self.source)


_NONE_RESULT = RaDec(None, None, None)


# ---------------------------------------------------------------------------
# Input resolution (pure, no astropy)
# ---------------------------------------------------------------------------


def _as_float(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _plausible_radec(ra: float, dec: float) -> bool:
    return 0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0


def read_filterbank_dec(event_dir: Path) -> Optional[float]:
    """Pointing declination (deg) from ``filterbank/filterbank.json``.

    Returns None when the file is absent / unreadable / lacks a finite
    ``dec_deg`` — never raises."""
    path = Path(event_dir) / "filterbank" / "filterbank.json"
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        LOG.warning("read_filterbank_dec %s failed: %s", path, exc)
        return None
    if not isinstance(doc, dict):
        return None
    dec = _as_float(doc.get("dec_deg"))
    if dec is None or not (-90.0 <= dec <= 90.0):
        return None
    return dec


def resolve_inputs(
    event_dir: Path, meta: Optional[Mapping[str, Any]],
) -> Tuple[Optional[RaDec], Optional[_Modern], Tuple]:
    """Decide how an event's position is obtained.

    Returns ``(final, pending, cache_key)`` where exactly one of
    ``final`` (a ready :class:`RaDec`, incl. the empty result) or
    ``pending`` (a :class:`_Modern` needing the batched compute) is set.
    ``cache_key`` is a hashable tuple of the exact inputs — a change in
    it (e.g. a filterbank.json appearing) invalidates the cache.
    """
    meta = meta or {}
    c2 = meta.get("c2") if isinstance(meta.get("c2"), Mapping) else {}
    mjd = _as_float(c2.get("t_peak_mjd"))
    l = _as_float(c2.get("l_median"))
    m = _as_float(c2.get("m_median"))
    have_lm = mjd is not None and l is not None and m is not None

    # Priority 1: contemporaneous C2 stamp (c2.pointing_dec_deg), written
    # by the coincidencer minutes after the trigger from the live etcd
    # pointing key (sibling provenance dict: c2.pointing_dec_meta). It may
    # be null when the etcd read failed at trigger time, and is absent on
    # all historical events — both fall through to the next source.
    # Operator backfills (tools/ops/backfill_pointing_dec.py) write the
    # same field with pointing_dec_meta.source = "manual_*"; those get
    # provenance "manual" (the source is part of _Modern.key(), so the
    # cache distinguishes it).
    if have_lm:
        pdec_c2 = _as_float(c2.get("pointing_dec_deg"))
        if pdec_c2 is not None and -90.0 <= pdec_c2 <= 90.0:
            source = "level3"
            pm = c2.get("pointing_dec_meta")
            if isinstance(pm, Mapping) \
                    and str(pm.get("source", "")).startswith("manual"):
                source = "manual"
            pend = _Modern(mjd, l, m, pdec_c2, source)
            return None, pend, pend.key()

    # Priority 2: filterbank pointing dec + modern (l, m, mjd) -> compute.
    if have_lm:
        pdec = read_filterbank_dec(event_dir)
        if pdec is not None:
            pend = _Modern(mjd, l, m, pdec, "filterbank")
            return None, pend, pend.key()

    # Priority 3: legacy flat schema — stored ICRS ra/dec, pass through
    # UNCHANGED. Sanity-check the raw stored values (0<=ra<360,
    # -90<=dec<=90); a nonsensical stored value is dropped, not wrapped.
    ra_flat = _as_float(meta.get("ra"))
    dec_flat = _as_float(meta.get("dec"))
    if (ra_flat is not None and dec_flat is not None
            and _plausible_radec(ra_flat, dec_flat)):
        res = RaDec(ra_flat, dec_flat, "legacy")
        return res, None, ("legacy", ra_flat, dec_flat)

    # (UVH5 fallback intentionally skipped: no h5py in dsart_h23.)

    # No usable source.
    if have_lm:
        # Distinct key so that a later filterbank.json flips the result.
        return _NONE_RESULT, None, ("none-nodec", mjd, l, m)
    return _NONE_RESULT, None, ("none",)


# ---------------------------------------------------------------------------
# Astropy compute (scalar + vectorized) — the only heavy part
# ---------------------------------------------------------------------------


def _ensure_iers() -> None:
    """Pin IERS to offline/stale mode BEFORE any Time math.

    astropy 7.2.0 in this env raises offline without this; a stale
    predictive UT1-UTC (|UT1-UTC| <= 0.9 s ~ 13 arcsec) is well below
    our ~1 arcmin accuracy. Mirrors sky_astrometry.lst_deg."""
    from astropy.utils import iers

    iers.conf.auto_download = False
    iers.conf.auto_max_age = None


def _ovro_site():
    from astropy.coordinates import EarthLocation
    import astropy.units as u

    return EarthLocation(
        lat=OVRO_LAT_DEG * u.deg,
        lon=OVRO_LON_DEG * u.deg,
        height=OVRO_HEIGHT_M * u.m,
    )


def _apparent_to_icrs(
    lst_deg: np.ndarray, l: np.ndarray, m: np.ndarray,
    pointing_dec_deg: np.ndarray, mjd: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shared core: apparent (l, m, LST, pointing-dec) -> ICRS (deg).

    Vectorized over 1-D arrays; also used for the scalar path.
    """
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    import astropy.units as u

    dec_app = pointing_dec_deg + np.degrees(m)
    ra_app = lst_deg + np.degrees(l) / np.cos(np.radians(dec_app))
    obstime = Time(np.asarray(mjd, dtype=np.float64),
                   format="mjd", scale="utc")
    coord = SkyCoord(
        ra=ra_app * u.deg, dec=dec_app * u.deg,
        frame="tete", obstime=obstime,
    )
    icrs = coord.icrs
    return (np.asarray(icrs.ra.to_value(u.deg), dtype=np.float64) % 360.0,
            np.asarray(icrs.dec.to_value(u.deg), dtype=np.float64))


def compute_icrs_scalar(
    mjd: float, l: float, m: float, pointing_dec_deg: float,
) -> Tuple[float, float]:
    """Single-event ICRS (ra_deg, dec_deg). Kept as an independent
    (non-array) reference implementation for the vec==scalar test."""
    from astropy.time import Time
    import astropy.units as u

    _ensure_iers()
    site = _ovro_site()
    t = Time(float(mjd), format="mjd", scale="utc", location=site)
    lst = float(t.sidereal_time("apparent").to_value(u.deg))
    ra, dec = _apparent_to_icrs(
        np.array([lst]), np.array([l]), np.array([m]),
        np.array([float(pointing_dec_deg)]), np.array([mjd]),
    )
    return float(ra[0]), float(dec[0])


def _compute_icrs_vec(
    mods: List[_Modern],
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched ICRS for a list of modern events (ONE SkyCoord transform).

    This is the function the cache-correctness test monkeypatches to
    count how often the heavy compute actually runs."""
    if not mods:
        return np.empty(0), np.empty(0)
    _ensure_iers()
    from astropy.time import Time
    import astropy.units as u

    site = _ovro_site()
    mjd = np.array([x.mjd for x in mods], dtype=np.float64)
    l = np.array([x.l for x in mods], dtype=np.float64)
    m = np.array([x.m for x in mods], dtype=np.float64)
    pdec = np.array([x.pointing_dec_deg for x in mods], dtype=np.float64)
    t = Time(mjd, format="mjd", scale="utc", location=site)
    lst = np.atleast_1d(t.sidereal_time("apparent").to_value(u.deg))
    return _apparent_to_icrs(lst, l, m, pdec, mjd)


# ---------------------------------------------------------------------------
# Batching + memoisation
# ---------------------------------------------------------------------------


class EventAstrometry:
    """Per-render batching + in-process memo for event sky positions.

    One instance lives on the ArchiveBrowser; the cache persists across
    ``/bursts`` renders. Keyed on ``(event_name -> (input_key, RaDec))``
    so a re-render is free and a changed input (new filterbank.json)
    recomputes."""

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[Tuple, RaDec]] = {}

    def compute(
        self,
        requests: List[Tuple[str, Path, Optional[Mapping[str, Any]]]],
    ) -> Dict[str, RaDec]:
        """Resolve ``(name, event_dir, meta)`` triples to RaDecs, doing at
        most one array TETE->ICRS transform for the cache misses."""
        results: Dict[str, RaDec] = {}
        pending: List[Tuple[str, Tuple, _Modern]] = []
        for name, event_dir, meta in requests:
            try:
                final, pend, key = resolve_inputs(event_dir, meta)
            except Exception:                                  # noqa: BLE001
                LOG.exception("resolve_inputs failed for %s", name)
                results[name] = _NONE_RESULT
                continue
            cached = self._cache.get(name)
            if cached is not None and cached[0] == key:
                results[name] = cached[1]
                continue
            if final is not None:
                self._cache[name] = (key, final)
                results[name] = final
                continue
            pending.append((name, key, pend))  # type: ignore[arg-type]
        if pending:
            try:
                ras, decs = _compute_icrs_vec([p for _, _, p in pending])
            except Exception:                                  # noqa: BLE001
                LOG.exception("TETE->ICRS batch failed (%d events)",
                              len(pending))
                for name, key, _ in pending:
                    results[name] = _NONE_RESULT
                return results
            for (name, key, pend), ra, dec in zip(pending, ras, decs):
                res = RaDec(float(ra), float(dec), pend.source)
                self._cache[name] = (key, res)
                results[name] = res
        return results

    def compute_one(
        self, name: str, event_dir: Path,
        meta: Optional[Mapping[str, Any]],
    ) -> RaDec:
        return self.compute([(name, event_dir, meta)]).get(
            name, _NONE_RESULT)


# ---------------------------------------------------------------------------
# Sexagesimal formatting (pure-python, table + detail page)
# ---------------------------------------------------------------------------

#: Conservative 1-sigma position uncertainty (arcsec) quoted for COMPUTED
#: positions (sources "level3"/"manual"/"filterbank"): half-pixel image
#: quantization ~22.5 arcsec + end-to-end astrometric tie validated at
#: ~1.2 arcmin on pulsar crossings. This is a fixed display figure, NOT a
#: per-event statistical error. Legacy (stored T2) positions have unknown
#: uncertainty and get no parenthetical.
SIGMA_POS_ARCSEC: float = 60.0


def _ra_paren(sigma_arcsec: float, dec_deg: float) -> str:
    """Pulsar-catalog-style parenthetical for an RA string whose last
    displayed digit is 0.1 s of time: sigma in units of that digit.
    Capped at "(>999)" near the pole (cos blowup — unreachable for
    DSA-110 pointings, but never emit garbage)."""
    cosd = np.cos(np.radians(dec_deg))
    if abs(dec_deg) > 85.0 or cosd <= 0.0:
        return "(>999)"
    sigma_s = sigma_arcsec / (15.0 * cosd)
    units = int(round(sigma_s / 0.1))
    if units > 999:
        return "(>999)"
    return f"({units})"


def format_ra_hms(
    ra_deg: Optional[float],
    *,
    sigma_arcsec: Optional[float] = None,
    dec_deg: Optional[float] = None,
) -> Optional[str]:
    """ICRS RA (deg) -> "HH:MM:SS.s", optionally with a catalog-style
    "(NN)" uncertainty in units of the last digit (0.1 s of time). The
    parenthetical needs ``dec_deg`` for the 1/cos(dec) RA scaling; both
    must be given (else the plain string is returned). None-safe."""
    if ra_deg is None or not np.isfinite(ra_deg):
        return None
    hours = (float(ra_deg) % 360.0) / 15.0
    # Round to 0.1 s of time, then carry, so 23:59:59.95 never prints :60.
    total_tenths = int(round(hours * 3600.0 * 10.0))
    total_tenths %= 24 * 3600 * 10
    hh, rem = divmod(total_tenths, 3600 * 10)
    mm, rem = divmod(rem, 60 * 10)
    ss = rem / 10.0
    out = f"{hh:02d}:{mm:02d}:{ss:04.1f}"
    if sigma_arcsec is not None and dec_deg is not None \
            and np.isfinite(dec_deg):
        out += _ra_paren(float(sigma_arcsec), float(dec_deg))
    return out


def format_dec_dms(
    dec_deg: Optional[float],
    *,
    sigma_arcsec: Optional[float] = None,
) -> Optional[str]:
    """ICRS Dec (deg) -> "+DD:MM:SS", optionally with a catalog-style
    "(NN)" uncertainty in units of the last digit (1 arcsec). None-safe."""
    if dec_deg is None or not np.isfinite(dec_deg):
        return None
    d = float(dec_deg)
    sign = "-" if d < 0 else "+"
    total_arcsec = int(round(abs(d) * 3600.0))
    dd, rem = divmod(total_arcsec, 3600)
    mm, ss = divmod(rem, 60)
    out = f"{sign}{dd:02d}:{mm:02d}:{ss:02d}"
    if sigma_arcsec is not None:
        out += f"({int(round(float(sigma_arcsec)))})"
    return out


#: Provenances whose positions we computed ourselves (and therefore quote
#: the fixed SIGMA_POS_ARCSEC for). "manual" (operator-backfilled pointing
#: dec) uses the identical recipe, so the same sigma applies. Legacy/stored
#: values have unknown uncertainty -> no parenthetical.
_COMPUTED_SOURCES = frozenset({"level3", "manual", "filterbank"})


def sexagesimal_for(rd: RaDec) -> Tuple[Optional[str], Optional[str]]:
    """(ra_hms, dec_dms) display strings for a resolved position, with the
    catalog-style uncertainty parenthetical for computed sources only."""
    sigma = SIGMA_POS_ARCSEC if rd.source in _COMPUTED_SOURCES else None
    return (
        format_ra_hms(rd.ra_deg, sigma_arcsec=sigma, dec_deg=rd.dec_deg),
        format_dec_dms(rd.dec_deg, sigma_arcsec=sigma),
    )
