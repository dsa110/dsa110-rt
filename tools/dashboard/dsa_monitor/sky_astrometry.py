"""Astrometry + NVSS catalog support for the static-sky monitor.

The sky-monitor frames are dirty images of the UN-fringestopped fast-vis
stream, so the phase center is the meridian at the observing declination:
``(α₀, δ₀) = (LST(t_frame), obs_dec)``. The (l, m) ↔ (RA, Dec) math is
the SIN projection about that center, byte-matching the M2-validated
convention in ``bench/run_0319_pipeline._compute_expected_lm``:

    l = cos δ · sin(α − α₀)            (+l = east, toward increasing RA)
    m = sin δ · cos δ₀ − cos δ · sin δ₀ · cos(α − α₀)     (+m = north)

and the pixel mapping mirrors ``bench/_corr_fast_replay.pixel_to_lm_radians``:
pixel (row, col) = (m, l) / pixel_scale + n_pix//2, origin='lower'
(north up, east RIGHT — instrument frame, not the flipped-RA convention
of survey atlases).

NVSS catalog
============

``load_nvss`` streams the HEASARC ``.tdat`` dump (1.77M rows, ~260 MB;
pipe-separated: name|ra|dec|lii|bii|ra_err|dec_err|flux_20_cm|...),
keeps sources above a flux cut, and caches the result as an ``.npz``
keyed by (tdat mtime, size, flux cut) so dashboard restarts pay ~ms,
not ~10 s. The catalog is position-only metadata — we deliberately do
NOT import the dsa110-calib package (CASA-heavy) for this.

LST uses astropy (apparent sidereal time at OVRO); the dashboard env
ships astropy 7.x. Site coordinates follow casacore's OVRO_MMA entry
(the same one dsacalib resolves); the ~0.002° spread between the
OVRO_* values used across DSA repos is ~7 arcsec on the sky — well
under the ~13 arcsec/pixel scale of the 512² frames.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

LOG = logging.getLogger("dsa_monitor.sky_astrometry")

#: OVRO site (casacore OVRO_MMA, as used by dsacalib).
OVRO_LON_DEG: float = -118.283
OVRO_LAT_DEG: float = 37.2334

#: Sidereal rate (rad of RA per SI second). The drift-scan phase
#: center advances at this rate; used to re-reference per-chgroup
#: snapshots taken at different data times to a common frame time.
SIDEREAL_RATE_RAD_PER_S: float = 7.292115855e-5

#: DSA-110 dish diameter used by the canonical dsacalib beam model.
DISH_DIA_M: float = 4.7

#: Default HEASARC NVSS tdat dump on h23 (user-provided 2026-06-09).
NVSS_TDAT_DEFAULT: str = (
    "/home/ubuntu/proj/dsa110-shell/dsa110-calib/dsacalib/data/"
    "heasarc_nvss.tdat"
)

#: 0-based field indices in the tdat data rows (see file <HEADER>).
_F_NAME, _F_RA, _F_DEC, _F_FLUX = 0, 1, 2, 7


def lst_deg(unix_ts: float) -> float:
    """Apparent local sidereal time at OVRO (degrees) for a unix time."""
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    from astropy.utils import iers
    import astropy.units as u

    # Never let an IERS table HTTP fetch stall a frame build, and
    # accept stale predictive values: |UT1−UTC| ≤ 0.9 s by definition
    # (leap-second scheduling), i.e. ≤ ~13 arcsec on the sky — about
    # one pixel of the 512² frames. Fine for a display graticule.
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None

    site = EarthLocation(
        lat=OVRO_LAT_DEG * u.deg, lon=OVRO_LON_DEG * u.deg, height=1222 * u.m,
    )
    t = Time(unix_ts, format="unix", scale="utc", location=site)
    return float(t.sidereal_time("apparent").to_value(u.deg)) % 360.0


def phase_center_icrs(
    unix_ts: float, dec_apparent_deg: float,
) -> tuple[float, float]:
    """ICRS (J2000) coordinates of the drift-scan phase center.

    The instrument phase center is (HA=0, pointing dec) in the TRUE
    equator/equinox of DATE (LST is apparent sidereal time; the
    pointing dec is an apparent-frame elevation setting). Comparing
    against a J2000 catalog without transforming leaves ~26 yr of
    precession: ~18-19 arcmin of RA (≈ +45 px of l at 22.5"/px, the
    constant east offset measured 2026-07-19) and 8.85'·cos(α) of Dec
    (the drift of the m offset across that night, −22 → −10 px as α₀
    swept 14h → 17h). Same TETE→ICRS transform the burst event pages
    use (event_astrometry.py). Annual aberration (~20", <1 px) is not
    modelled.
    """
    from astropy.coordinates import SkyCoord, TETE
    from astropy.time import Time
    from astropy.utils import iers
    import astropy.units as u

    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    t = Time(unix_ts, format="unix", scale="utc")
    ra_app = lst_deg(unix_ts)
    c = SkyCoord(
        ra=ra_app * u.deg, dec=dec_apparent_deg * u.deg,
        frame=TETE(obstime=t),
    ).icrs
    return float(c.ra.deg) % 360.0, float(c.dec.deg)


def radec_to_lm(
    ra_deg: np.ndarray | float,
    dec_deg: np.ndarray | float,
    *,
    ra0_deg: float,
    dec0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """SIN-project (RA, Dec) about the phase center → (l, m) rad.

    Matches ``bench/run_0319_pipeline._compute_expected_lm`` with
    HA = α₀ − α (so +l is east / increasing RA).
    """
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    ra0 = np.deg2rad(float(ra0_deg))
    dec0 = np.deg2rad(float(dec0_deg))
    dra = ra - ra0
    l = np.cos(dec) * np.sin(dra)
    m = (np.sin(dec) * np.cos(dec0)
         - np.cos(dec) * np.sin(dec0) * np.cos(dra))
    return l, m


def lm_to_radec(
    l: np.ndarray | float,
    m: np.ndarray | float,
    *,
    ra0_deg: float,
    dec0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse SIN projection. Returns (ra_deg ∈ [0, 360), dec_deg)."""
    l = np.asarray(l, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    ra0 = np.deg2rad(float(ra0_deg))
    dec0 = np.deg2rad(float(dec0_deg))
    n = np.sqrt(np.clip(1.0 - l * l - m * m, 0.0, None))
    dec = np.arcsin(m * np.cos(dec0) + n * np.sin(dec0))
    ra = ra0 + np.arctan2(l, n * np.cos(dec0) - m * np.sin(dec0))
    return np.rad2deg(ra) % 360.0, np.rad2deg(dec)


def lm_to_pix(
    l: np.ndarray | float,
    m: np.ndarray | float,
    *,
    n_pix: int,
    fov_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(l, m) rad → fractional (row, col) of an ``n_pix``² frame whose
    full extent is ``fov_rad`` (= 1 / cell_lambda; oversampling changes
    ``n_pix`` but not the FOV). Mirrors
    ``bench/_corr_fast_replay.lm_to_pixel`` (row = m-axis, col = l-axis,
    center pixel ``n_pix // 2``)."""
    scale = fov_rad / float(n_pix)                  # rad per pixel
    row = np.asarray(m, dtype=np.float64) / scale + n_pix // 2
    col = np.asarray(l, dtype=np.float64) / scale + n_pix // 2
    return row, col


def sky_to_instrument_lm(
    l: np.ndarray | float,
    m: np.ndarray | float,
    *,
    dec0_deg: float,
    lat_deg: float = OVRO_LAT_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """True SIN-projected (l, m) → the fast-vis IMAGE frame.

    The corr gridder builds its (u, v) from RAW (ΔE, ΔN) antenna
    offsets (``grid/sparsity_pattern._per_baseline_uv_meters``) with
    no geometric projection, while the true meridian-pointing
    projected baselines are ``u = ΔE`` and
    ``v = ΔN·cos(lat − dec) [+ ΔU·sin(lat − dec)]``. The F21 DEC
    fringe-stop puts the phase center at (H=0, dec); the residual
    w-term is ``w = −ΔN·sin(lat − dec)``. Matching phase terms, a
    source at true (l, m) lands in the image at::

        l_img = l
        m_img = m·cos(lat − dec) + sin(lat − dec)·(l² + m²)/2

    — the m-axis is COMPRESSED by cos(lat − dec) (≈0.934 at
    dec = +16.3°, i.e. ~4′ of Dec error at the field edge if
    ignored) plus a small quadratic w-term warp (≲1′ in the field
    corners). Discovered empirically 2026-07-19 (Dec offsets of NVSS
    sources grew with |m|).

    NOTE: the search-side imager uses the SAME grid, so detector /
    C1 / C2 (l, m) are in this instrument frame too — sidereal-veto
    overlays plot directly, but any (l, m) → (RA, Dec) conversion
    (e.g. burst-event astrometry) must apply the inverse.
    """
    l = np.asarray(l, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    g = np.deg2rad(float(lat_deg) - float(dec0_deg))
    m_img = m * np.cos(g) + np.sin(g) * (l * l + m * m) / 2.0
    return l, m_img


def instrument_to_sky_lm(
    l_img: np.ndarray | float,
    m_img: np.ndarray | float,
    *,
    dec0_deg: float,
    lat_deg: float = OVRO_LAT_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`sky_to_instrument_lm` (one Newton refinement
    of the small quadratic term — sub-milliarcsec residual)."""
    l = np.asarray(l_img, dtype=np.float64)
    mi = np.asarray(m_img, dtype=np.float64)
    g = np.deg2rad(float(lat_deg) - float(dec0_deg))
    cg, sg = np.cos(g), np.sin(g)
    m = mi / cg
    for _ in range(3):
        m = (mi - sg * (l * l + m * m) / 2.0) / cg
    return l, m


def pb_resp_power(
    theta_rad: np.ndarray | float,
    *,
    freq_ghz: float = 1.405,
    dish_dia_m: float = DISH_DIA_M,
) -> np.ndarray:
    """DSA-110 interferometric primary-beam attenuation vs offset.

    Byte-matches ``dsacalib.fringestopping.pb_resp`` (tapered
    illumination): ``(cos(π·x)/(1 − 4x²))⁴`` with
    ``x = 1.2 · θ · D / λ``. The 4th power is the two-antenna
    *visibility-domain* response (voltage² per dish, two dishes) —
    the factor a source's flux is multiplied by in a dirty image made
    from uncorrected drift-scan visibilities. FWHM ≈ 1.8° at band
    center 1.405 GHz.
    """
    theta = np.abs(np.asarray(theta_rad, dtype=np.float64))
    lam = 0.299792458 / float(freq_ghz)
    x = 1.2 * theta * float(dish_dia_m) / lam
    # The (1 − 4x²) denominator has a removable singularity at x=0.5
    # (numerator cos(π/2)=0); nudge x off it.
    x = np.where(np.abs(np.abs(x) - 0.5) < 1e-9, x + 1e-8, x)
    resp = (np.cos(np.pi * x) / (1.0 - 4.0 * x * x)) ** 4
    return np.clip(resp, 0.0, 1.0)


# ---------------------------------------------------------------------------
# NVSS catalog
# ---------------------------------------------------------------------------


def _parse_nvss_tdat(
    tdat_path: Path, *, min_mjy: float,
) -> dict[str, np.ndarray]:
    """Stream-parse the HEASARC tdat dump, keeping rows with
    ``flux_20_cm >= min_mjy``. ~10 s for the 1.77M-row file."""
    names: list[str] = []
    ras: list[float] = []
    decs: list[float] = []
    fluxes: list[float] = []
    in_data = False
    with tdat_path.open("r", errors="replace") as f:
        for line in f:
            if not in_data:
                in_data = line.startswith("<DATA>")
                continue
            if line.startswith("<END>"):
                break
            parts = line.rstrip("\n").split("|")
            if len(parts) <= _F_FLUX:
                continue
            try:
                flux = float(parts[_F_FLUX])
            except ValueError:
                continue
            if flux < min_mjy:
                continue
            try:
                ra = float(parts[_F_RA])
                dec = float(parts[_F_DEC])
            except ValueError:
                continue
            names.append(parts[_F_NAME].strip())
            ras.append(ra)
            decs.append(dec)
            fluxes.append(flux)
    return {
        "name": np.asarray(names, dtype="U20"),
        "ra_deg": np.asarray(ras, dtype=np.float64),
        "dec_deg": np.asarray(decs, dtype=np.float64),
        "flux_mjy": np.asarray(fluxes, dtype=np.float64),
    }


def load_nvss(
    *,
    min_mjy: float = 100.0,
    tdat_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> Optional[dict[str, np.ndarray]]:
    """Load the NVSS subset above ``min_mjy``, via an npz cache.

    Returns None (with a WARN) when the tdat file is missing — the
    sky monitor then simply renders frames without the overlay.
    """
    path = Path(tdat_path or os.environ.get("DSA_NVSS_TDAT", NVSS_TDAT_DEFAULT))
    if not path.is_file():
        LOG.warning("NVSS tdat not found: %s (no overlay)", path)
        return None
    st = path.stat()
    cache: Optional[Path] = None
    if cache_dir is not None:
        cache = (
            Path(cache_dir)
            / f"nvss_min{min_mjy:g}mJy_{st.st_size}_{int(st.st_mtime)}.npz"
        )
        if cache.is_file():
            try:
                with np.load(cache, allow_pickle=False) as z:
                    return {k: np.asarray(z[k]) for k in
                            ("name", "ra_deg", "dec_deg", "flux_mjy")}
            except Exception:                                  # noqa: BLE001
                LOG.warning("NVSS cache unreadable, re-parsing: %s", cache)
    cat = _parse_nvss_tdat(path, min_mjy=min_mjy)
    LOG.info(
        "NVSS: %d sources >= %g mJy parsed from %s",
        cat["ra_deg"].size, min_mjy, path,
    )
    if cache is not None:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **cat)
            tmp.replace(cache)
        except OSError:
            LOG.warning("could not write NVSS cache %s", cache)
    return cat


class NvssCatalog:
    """Lazy, thread-safe NVSS holder for the dashboard.

    ``start_loading()`` kicks a daemon thread; ``get()`` returns the
    catalog dict or None while loading / on failure, so frame builds
    never block on the 10 s parse.
    """

    def __init__(
        self,
        *,
        min_mjy: float = 100.0,
        tdat_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.min_mjy = float(min_mjy)
        self._tdat_path = tdat_path
        self._cache_dir = cache_dir
        self._cat: Optional[dict[str, np.ndarray]] = None
        self._started = False
        self._lock = threading.Lock()

    def start_loading(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        threading.Thread(
            target=self._load, name="nvss-load", daemon=True,
        ).start()

    def _load(self) -> None:
        try:
            cat = load_nvss(
                min_mjy=self.min_mjy,
                tdat_path=self._tdat_path,
                cache_dir=self._cache_dir,
            )
        except Exception:                                      # noqa: BLE001
            LOG.exception("NVSS load failed (no overlay)")
            return
        with self._lock:
            self._cat = cat

    def get(self) -> Optional[dict[str, np.ndarray]]:
        with self._lock:
            return self._cat


def select_in_fov(
    cat: dict[str, np.ndarray],
    *,
    ra0_deg: float,
    dec0_deg: float,
    fov_rad: float,
    max_sources: int = 40,
) -> dict[str, np.ndarray]:
    """Subset of ``cat`` whose SIN-projected (l, m) fall inside the
    square FOV, capped at ``max_sources``, ranked by APPARENT flux
    (catalog flux × primary-beam attenuation at the source's offset —
    monotonic in expected S/N since the image noise is uniform).
    Ranking by raw catalog flux let bright far-off-beam sources
    displace detectable near-center ones (2026-07-19). Adds
    ``l_rad`` / ``m_rad`` columns."""
    # Cheap pre-cut in dec (±FOV) before the trig.
    half_deg = np.rad2deg(fov_rad) / 2.0
    pre = np.abs(cat["dec_deg"] - dec0_deg) <= half_deg * 1.05
    l, m = radec_to_lm(
        cat["ra_deg"][pre], cat["dec_deg"][pre],
        ra0_deg=ra0_deg, dec0_deg=dec0_deg,
    )
    half = fov_rad / 2.0
    inside = (np.abs(l) < half) & (np.abs(m) < half)
    idx = np.flatnonzero(pre)[inside]
    apparent = cat["flux_mjy"][idx] * pb_resp_power(
        np.hypot(l[inside], m[inside]),
    )
    order = np.argsort(-apparent)[:max_sources]
    idx = idx[order]
    return {
        "name": cat["name"][idx],
        "ra_deg": cat["ra_deg"][idx],
        "dec_deg": cat["dec_deg"][idx],
        "flux_mjy": cat["flux_mjy"][idx],
        "l_rad": l[inside][order],
        "m_rad": m[inside][order],
    }


__all__ = [
    "OVRO_LON_DEG",
    "OVRO_LAT_DEG",
    "SIDEREAL_RATE_RAD_PER_S",
    "DISH_DIA_M",
    "pb_resp_power",
    "NVSS_TDAT_DEFAULT",
    "NvssCatalog",
    "load_nvss",
    "lst_deg",
    "lm_to_pix",
    "lm_to_radec",
    "radec_to_lm",
    "select_in_fov",
]
