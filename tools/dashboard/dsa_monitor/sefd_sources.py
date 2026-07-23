"""DEC-dependent SEFD calibrator source resolution.

The SEFD page used to track a hard-coded 3-source catalog. When the
array moves in declination the useful calibrators change, so this
module resolves — at render time, from the current pointing dec — the
set of VLA-manual calibrators the calibration23 service actually
calibrates on at that dec, restricted to flux > 2 Jy for good SEFD
solutions.

Ground truth for "what the service calibrates on" is
``dsacalib.preprocess.generate_caltable(pt_dec)`` (the calibration23
container's own selection). That function needs pyuvdata (absent in
the dashboard env), so this module REPLICATES its selection exactly:

  * parse the same VLA manual (``read_vla_catalog`` ported verbatim),
    taking the 20cm (L-band) flux;
  * keep sources within ``radius`` (2.5 deg) of the pointing dec;
  * weight by the DSA primary beam (``pb_resp_power`` — byte-matches
    ``dsacalib.fringestopping.pb_resp``: (cos(pi x)/(1-4x^2))^4,
    x=1.2 theta D/lambda, D=4.7 m, 1.4 GHz) — at transit the source
    shares the pointing RA so the beam offset is |dec - pt_dec|;
  * keep weighted_flux > ``min_weighted_flux`` (1 Jy) and
    percent_flux > ``min_percent_flux`` (0.15), exactly as
    generate_caltable;
  * then apply the page's own flux > ``flux_min`` (2 Jy) cut.

Flux densities returned here come straight from the manual and are the
values fed to the SEFD calculation, per the operator request.

Keep the selection params in sync with
``dsacalib/preprocess.py::generate_caltable`` if the service changes.
"""
from __future__ import annotations

from collections import namedtuple
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from astropy.coordinates import Angle
import astropy.units as u

from sky_astrometry import pb_resp_power

#: Local copy of the VLA calibrator manual (same file dsacalib reads).
DEFAULT_VLA_MANUAL = (
    "/home/ubuntu/proj/dsa110-shell/dsa110-calib/dsacalib/data/"
    "vlacalibrators.txt"
)

# generate_caltable defaults (dsacalib/preprocess.py) — keep in sync.
SERVICE_RADIUS_DEG = 2.5
SERVICE_MIN_WEIGHTED_FLUX_JY = 1.0
SERVICE_MIN_PERCENT_FLUX = 0.15
#: The page's extra flux cut (operator request 2026-07-23).
PAGE_MIN_FLUX_JY = 2.0

_Calibrator = namedtuple("Calibrator", "source ra dec flux_20_cm code_20_cm")


def read_vla_catalog(manual_path: str = DEFAULT_VLA_MANUAL) -> pd.DataFrame:
    """Ported verbatim from ``dsacalib.preprocess.read_vla_catalog``.

    Returns a DataFrame indexed by source with columns ``ra`` (deg),
    ``dec`` (deg), ``flux_20_cm`` (mJy), ``code_20_cm``.
    """
    calsources = []
    with open(manual_path) as file:
        for _ in range(3):
            file.readline()
        while True:
            line = file.readline()
            if not line:
                break
            try:
                source, _, _, ra, dec, *_ = line.split()
            except ValueError:
                continue
            try:
                ra = Angle(ra).to_value(u.deg)
                dec = Angle(dec).to_value(u.deg)
            except Exception:                              # noqa: BLE001
                continue
            flux_20_cm = None
            code_20_cm = None
            for _ in range(4):
                file.readline()
            while True:
                line = file.readline()
                if line.isspace() or not line:
                    if flux_20_cm not in [None, "?"]:
                        try:
                            calsources.append(
                                _Calibrator(
                                    source, ra, dec,
                                    1000 * float(flux_20_cm), code_20_cm,
                                )
                            )
                        except ValueError:
                            pass
                    break
                if "20cm " in line:
                    parts = line.split()
                    if len(parts) >= 7:
                        (_, _, code_a, code_b, code_c, code_d,
                         flux_20_cm, *_) = parts
                        code_20_cm = code_a + code_b + code_c + code_d
    df = pd.DataFrame.from_records(calsources, columns=_Calibrator._fields)
    df.set_index("source", inplace=True)
    return df


def _pb(pt_dec_deg: float, src_dec_deg: float) -> float:
    """DSA primary-beam response for a meridian-transit source at
    dec offset |src_dec - pt_dec| (rad), 1.4 GHz, D=4.7 m — matches
    the service's ``pb_resp``."""
    theta = abs(np.deg2rad(src_dec_deg - pt_dec_deg))
    return float(pb_resp_power(theta, freq_ghz=1.4, dish_dia_m=4.7))


def resolve_sources_for_dec(
    pt_dec_deg: float,
    *,
    manual_path: str = DEFAULT_VLA_MANUAL,
    flux_min_jy: float = PAGE_MIN_FLUX_JY,
    radius_deg: float = SERVICE_RADIUS_DEG,
    min_weighted_flux_jy: float = SERVICE_MIN_WEIGHTED_FLUX_JY,
    min_percent_flux: float = SERVICE_MIN_PERCENT_FLUX,
) -> Dict[str, Dict[str, Any]]:
    """Return the DEC-appropriate calibrators for the SEFD page.

    Replicates ``generate_caltable(pt_dec)`` (dec window + weighted
    flux + percent flux), then applies the page's ``flux_min_jy`` cut.
    Keys are VLA source names; values carry ``flux_jy`` (from the
    manual, for the SEFD calc), ``dec_deg``, ``ra_deg``,
    ``weighted_flux_jy``, ``code``.
    """
    df = read_vla_catalog(manual_path)
    lo, hi = pt_dec_deg - radius_deg, pt_dec_deg + radius_deg
    # dec window + manual flux floor (service uses >1 Jy raw; we take
    # the operator's stronger >flux_min cut but keep the window sources
    # for the percent-flux field sum below).
    in_win = df[(df["dec"] > lo) & (df["dec"] < hi)].copy()
    if in_win.empty:
        return {}
    cosd = float(np.cos(np.deg2rad(pt_dec_deg))) or 1.0
    out: Dict[str, Dict[str, Any]] = {}
    for name, row in in_win.iterrows():
        flux_jy = float(row["flux_20_cm"]) / 1e3
        if flux_jy <= flux_min_jy:
            continue
        w = flux_jy * _pb(pt_dec_deg, float(row["dec"]))
        if w <= min_weighted_flux_jy:
            continue
        # percent_flux: this source's weighted flux over the summed
        # weighted flux of all window sources within +-radius in RA/dec
        # of it (matches generate_caltable's field definition).
        ra = float(row["ra"])
        field = in_win[
            (in_win["ra"] < ra + radius_deg / cosd)
            & (in_win["ra"] > ra - radius_deg / cosd)
        ]
        field_flux = float(
            (field["flux_20_cm"] / 1e3
             * field["dec"].map(lambda d: _pb(pt_dec_deg, float(d)))).sum()
        )
        pct = (w / field_flux) if field_flux > 0 else 0.0
        if pct <= min_percent_flux:
            continue
        out[str(name)] = {
            "flux_jy": round(flux_jy, 3),
            "dec_deg": round(float(row["dec"]), 4),
            "ra_deg": round(ra, 4),
            "weighted_flux_jy": round(w, 3),
            "code": row["code_20_cm"],
        }
    return out


def pointing_dec_deg(etcd_store: Any) -> Optional[float]:
    """Read the current pointing dec from etcd ``/mon/array/dec``."""
    try:
        doc = etcd_store.get_dict("/mon/array/dec")
        val = doc.get("dec_deg")
        return float(val) if val is not None else None
    except Exception:                                      # noqa: BLE001
        return None
