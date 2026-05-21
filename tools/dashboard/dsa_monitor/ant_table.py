"""Per-antenna monitor-points table assembly for the dashboard.

Pulls the dict at ``/mon/ant/<ant_num>`` from etcd (where ant_num is
1-based; ant_idx in the rest of the dashboard is 0-based) and joins
it with the dashboard's in-memory RFI ring buffer to produce the
table that the Antennas/RFI tab renders.

Schema (one dict; consumed by templates/antennas.html):

  {
    "ant_num": 12,
    "ant_idx": 11,
    "pointing": {
        "ant_el_deg":    {...},
        "ant_cmd_el_deg": {...},
        "ant_el_err_deg": {...},
    },
    "drive": {
        "drv_state":  {...},
        "drv_cmd":    {...},
        "drv_act":    {...},
        "at_north_lim": {...},
        "at_south_lim": {...},
        "brake_on":   {...},
        "fan_err":    {...},
        "emergency_off": {...},
    },
    "thermal": {
        "motor_temp_c": {...}, "focus_temp_c": {...},
        "feb_temp_a":   {...}, "feb_temp_b":   {...},
    },
    "rf": {
        "lna_current_a": {...}, "lna_current_b": {...},
        "noise_a_on":    {...}, "noise_b_on":    {...},
        "rf_pwr_a_dBm":  {...}, "rf_pwr_b_dBm":  {...},
        "feb_current_a": {...}, "feb_current_b": {...},
    },
    "power": {
        "laser_volts_a": {...}, "laser_volts_b": {...},
        "psu_volt":      {...},
    },
    "rfi": {                                # joined from the store
        "in_band_power_xx": {...},
        "in_band_power_yy": {...},
        "frac_flagged_total_xx": {...},
        "frac_flagged_total_yy": {...},
        "frac_flagged_total":   {...},
        "group_flagged_this_window": {...},  # bool per-pol
        "n_windows_observed":   {...},       # 0..N over 30 min
    },
    "etcd": {"age_s": ..., "stale": bool},  # freshness of /mon/ant/<n>
  }

Where each leaf {...} is ``{"value": X, "warn": bool}``, with
``warn`` reflecting simple operator-side thresholds (e.g. brake_on
True, drv_state != 0, fan_err True, motor_temp > 50, etc.).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

from rfi_store import StoreSnapshot

LOG = logging.getLogger("dsa_monitor.ant_table")

# Antenna numbering: ant_idx 0..95 is the corr's 0-based index into
# the (NANTS, NCHAN, NPOL) cubes; ant_num 1..96 is the operator-
# facing antenna ID (matches /mon/ant/<N>).
NANTS_DASH: int = 96

ETCD_MON_ANT_KEY = "/mon/ant/{ant_num}"
ETCD_STALENESS_S = 60.0                            # /mon/ant/<n> updates ~3 s


def ant_idx_to_ant_num(ant_idx: int) -> int:
    """Translate from cube 0-based ant_idx to operator-facing 1-based."""
    return int(ant_idx) + 1


def ant_num_to_ant_idx(ant_num: int) -> int:
    return int(ant_num) - 1


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------


def _cell(value: Any, *, warn: bool = False, fmt: Optional[str] = None) -> dict[str, Any]:
    if value is None:
        return {"value": None, "fmt": "—", "warn": warn}
    if fmt is not None:
        try:
            formatted = format(value, fmt)
        except (TypeError, ValueError):
            formatted = str(value)
    else:
        formatted = str(value)
    return {"value": value, "fmt": formatted, "warn": bool(warn)}


def _cell_bool(value: Any, *, warn_when_true: bool = False) -> dict[str, Any]:
    if value is None:
        return {"value": None, "fmt": "—", "warn": False}
    truthy = bool(value)
    return {
        "value": truthy, "fmt": "YES" if truthy else "no",
        "warn": warn_when_true and truthy,
    }


def _cell_float(
    value: Any, *, fmt: str = ".2f",
    warn_lo: float | None = None, warn_hi: float | None = None,
) -> dict[str, Any]:
    if value is None:
        return {"value": None, "fmt": "—", "warn": False}
    try:
        f = float(value)
    except (TypeError, ValueError):
        return {"value": value, "fmt": str(value), "warn": False}
    warn = False
    if warn_lo is not None and f < warn_lo:
        warn = True
    if warn_hi is not None and f > warn_hi:
        warn = True
    return {"value": f, "fmt": format(f, fmt), "warn": warn}


# ---------------------------------------------------------------------------
# Etcd fetch
# ---------------------------------------------------------------------------


def fetch_mon_ant(store, ant_num: int) -> Optional[dict[str, Any]]:
    """Try to fetch /mon/ant/<ant_num>. Returns None on failure."""
    key = ETCD_MON_ANT_KEY.format(ant_num=ant_num)
    try:
        v = store.get_dict(key)
    except Exception:
        LOG.exception("etcd get_dict %s failed", key)
        return None
    if v is None:
        return None
    return v


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _pointing(d: dict[str, Any]) -> dict[str, Any]:
    el = d.get("ant_el")
    cmd_el = d.get("ant_cmd_el")
    err = d.get("ant_el_err")
    return {
        "ant_el_deg":     _cell_float(el, fmt=".2f"),
        "ant_cmd_el_deg": _cell_float(cmd_el, fmt=".2f"),
        "ant_el_err_deg": _cell_float(
            err, fmt=".3f",
            warn_lo=-0.05, warn_hi=0.05,
        ),
    }


def _drive(d: dict[str, Any]) -> dict[str, Any]:
    drv_state = d.get("drv_state")
    return {
        "drv_state": _cell(
            drv_state, fmt="d",
            warn=(drv_state is not None and int(drv_state) not in (0, 2)),
        ),
        "drv_cmd": _cell(d.get("drv_cmd"), fmt="d"),
        "drv_act": _cell(d.get("drv_act"), fmt="d"),
        "at_north_lim": _cell_bool(d.get("at_north_lim"), warn_when_true=True),
        "at_south_lim": _cell_bool(d.get("at_south_lim"), warn_when_true=True),
        "brake_on": _cell_bool(d.get("brake_on")),
        "fan_err": _cell_bool(d.get("fan_err"), warn_when_true=True),
        "emergency_off": _cell_bool(
            d.get("emergency_off"), warn_when_true=True,
        ),
    }


def _thermal(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "motor_temp_c": _cell_float(
            d.get("motor_temp"), fmt=".1f", warn_hi=50.0,
        ),
        "focus_temp_c": _cell_float(
            d.get("focus_temp"), fmt=".1f", warn_hi=50.0,
        ),
        "feb_temp_a": _cell_float(
            d.get("feb_temp_a"), fmt=".1f", warn_hi=55.0,
        ),
        "feb_temp_b": _cell_float(
            d.get("feb_temp_b"), fmt=".1f", warn_hi=55.0,
        ),
    }


def _rf(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "lna_current_a_mA": _cell_float(
            d.get("lna_current_a"), fmt=".1f", warn_lo=10.0, warn_hi=120.0,
        ),
        "lna_current_b_mA": _cell_float(
            d.get("lna_current_b"), fmt=".1f", warn_lo=10.0, warn_hi=120.0,
        ),
        "noise_a_on": _cell_bool(d.get("noise_a_on"), warn_when_true=True),
        "noise_b_on": _cell_bool(d.get("noise_b_on"), warn_when_true=True),
        "rf_pwr_a_dBm": _cell_float(
            d.get("rf_pwr_a"), fmt=".2f",
            warn_lo=-100.0, warn_hi=-30.0,
        ),
        "rf_pwr_b_dBm": _cell_float(
            d.get("rf_pwr_b"), fmt=".2f",
            warn_lo=-100.0, warn_hi=-30.0,
        ),
        "feb_current_a_mA": _cell_float(d.get("feb_current_a"), fmt=".1f"),
        "feb_current_b_mA": _cell_float(d.get("feb_current_b"), fmt=".1f"),
    }


def _power(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "laser_volts_a": _cell_float(
            d.get("laser_volts_a"), fmt=".3f", warn_lo=2.0, warn_hi=4.0,
        ),
        "laser_volts_b": _cell_float(
            d.get("laser_volts_b"), fmt=".3f", warn_lo=2.0, warn_hi=4.0,
        ),
        "psu_volt": _cell_float(
            d.get("psu_volt"), fmt=".3f", warn_lo=4.5, warn_hi=5.5,
        ),
    }


def _rfi_aggregates(
    snap: StoreSnapshot, *, ant_idx: int,
) -> dict[str, Any]:
    """Build the per-ant RFI aggregates from the dashboard's in-mem ring.

    All metrics are computed against the LATEST 16-cube window per cn
    (concatenating across the 16 chgroups). When a cn has no records
    yet, that chgroup is skipped (incomplete chgroup coverage).
    """
    s1_pieces: list[np.ndarray] = []
    frac_pieces: list[np.ndarray] = []
    grp_xx_any = False
    grp_yy_any = False
    n_windows_observed = 0
    for cring in snap.per_chgroup:
        if not cring.records:
            continue
        rec = cring.records[-1]
        if not (0 <= ant_idx < rec.s1_full_mean.shape[0]):
            continue
        s1_pieces.append(rec.s1_full_mean[ant_idx])           # (NCHAN_DS, 2)
        n_cubes = max(1, rec.n_cubes)
        frac_pieces.append(
            rec.mask_count_final[ant_idx].astype(np.float32) / float(n_cubes)
        )
        # Per-pol "group-flagged this window"
        grp_arr = rec.mask_count_grp[ant_idx]                  # (NCHAN_DS, 2)
        grp_xx_any = grp_xx_any or bool((grp_arr[:, 0] > 0).any())
        grp_yy_any = grp_yy_any or bool((grp_arr[:, 1] > 0).any())
        n_windows_observed = max(n_windows_observed, len(cring.records))

    if not s1_pieces:
        return {
            "in_band_power_xx": _cell(None),
            "in_band_power_yy": _cell(None),
            "frac_flagged_total_xx": _cell(None),
            "frac_flagged_total_yy": _cell(None),
            "frac_flagged_total":   _cell(None),
            "group_flagged_xx_this_window": _cell_bool(None),
            "group_flagged_yy_this_window": _cell_bool(None),
            "n_windows_observed":   _cell(n_windows_observed, fmt="d"),
        }

    s1 = np.concatenate(s1_pieces, axis=0)                     # (N_TOT_CH, 2)
    frac = np.concatenate(frac_pieces, axis=0)                 # (N_TOT_CH, 2)
    in_band_xx = float(s1[:, 0].mean())
    in_band_yy = float(s1[:, 1].mean())
    fr_xx = float(frac[:, 0].mean())
    fr_yy = float(frac[:, 1].mean())
    fr_both = 0.5 * (fr_xx + fr_yy)
    return {
        "in_band_power_xx": _cell_float(in_band_xx, fmt=".2e"),
        "in_band_power_yy": _cell_float(in_band_yy, fmt=".2e"),
        "frac_flagged_total_xx": _cell_float(
            fr_xx, fmt=".3f", warn_hi=0.5,
        ),
        "frac_flagged_total_yy": _cell_float(
            fr_yy, fmt=".3f", warn_hi=0.5,
        ),
        "frac_flagged_total": _cell_float(
            fr_both, fmt=".3f", warn_hi=0.5,
        ),
        "group_flagged_xx_this_window": _cell_bool(
            grp_xx_any, warn_when_true=True,
        ),
        "group_flagged_yy_this_window": _cell_bool(
            grp_yy_any, warn_when_true=True,
        ),
        "n_windows_observed": _cell(n_windows_observed, fmt="d"),
    }


def _etcd_freshness(d: Optional[dict[str, Any]]) -> dict[str, Any]:
    if d is None:
        return {"age_s": _cell(None), "stale": _cell_bool(True, warn_when_true=True)}
    t = d.get("time")
    if t is None:
        return {"age_s": _cell(None), "stale": _cell_bool(True, warn_when_true=True)}
    # /mon/ant/<n>.time is MJD seconds-since-MJD0. Convert by treating
    # the integer day field as days since 1858-11-17 UT. We don't have
    # an MJD library guaranteed in the env; instead compute a relative
    # age using "wall-clock now in MJD" vs the stored MJD value.
    # MJD = (Unix epoch UTC - 40587.0) * 86400 seconds  ->
    # Unix = (MJD - 40587) * 86400.
    try:
        unix_t = (float(t) - 40587.0) * 86400.0
        age = max(0.0, time.time() - unix_t)
    except Exception:
        age = None
    if age is None:
        return {"age_s": _cell(None), "stale": _cell_bool(True, warn_when_true=True)}
    stale = age > ETCD_STALENESS_S
    return {
        "age_s": _cell_float(age, fmt=".1f", warn_hi=ETCD_STALENESS_S),
        "stale": _cell_bool(stale, warn_when_true=True),
    }


def build_ant_table(
    snap: StoreSnapshot, store, *, ant_idx: int,
) -> dict[str, Any]:
    """Assemble the per-antenna table for the Antennas/RFI tab."""
    ant_num = ant_idx_to_ant_num(ant_idx)
    raw = fetch_mon_ant(store, ant_num) or {}
    return {
        "ant_num": ant_num,
        "ant_idx": ant_idx,
        "pointing": _pointing(raw),
        "drive": _drive(raw),
        "thermal": _thermal(raw),
        "rf": _rf(raw),
        "power": _power(raw),
        "rfi": _rfi_aggregates(snap, ant_idx=ant_idx),
        "etcd": _etcd_freshness(raw or None),
        "raw_present": bool(raw),
    }
