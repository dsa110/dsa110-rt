"""Page data for the Array sums tab (M7.7).

The Antennas/RFI tab shows what one antenna sees. This one shows what
the *sums* see — the core, and each arm of the core, treated as extra
antennas.

Why they exist at all: every detector in the deployed flagger runs
inside a single antenna, and each needs a reference that the signal
does not also move. Bandpass-outlier compares a channel against the
other channels of the same antenna; group-outlier compares an antenna
against the other antennas; spectral kurtosis asks whether the
voltages are Gaussian within an accumulation. A broadband burst common
to the whole array defeats all three by construction. Measured on
260812imek sb02, such a burst is **0.05 sigma** in the cell the
flagger tests and **13.3 sigma** in the array-summed band power at
2.097 ms.

This module is pure data assembly: it turns the store snapshot into
plain dicts the Jinja template can render, and does no plotting (that
lives in :mod:`plot_render`).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from rfi_store import StoreSnapshot

LOG = logging.getLogger("dsa_monitor.array_sum_view")

#: Human labels for the summing groups.
GROUP_LABELS: dict[str, str] = {
    "all": "all antennas",
    "core": "core",
    "ew_arm": "core, E-W arm",
    "ns_arm": "core, N-S arm",
    "outriggers": "outriggers",
}

#: How the detector's mode should read on the page, and whether it
#: means data is actually being removed.
MODE_LABELS: dict[str, tuple[str, str]] = {
    "off": ("off", "The detector is not running."),
    "monitor": (
        "monitor only",
        "Computed and shown here, but NOTHING is excised. This is how "
        "it ships on the fast path.",
    ),
    "flag": (
        "ARMED",
        "Firing samples are zero-filled in the voltages before the "
        "correlator sees them.",
    ),
}


def _latest_with_groups(cring) -> Optional[Any]:
    """Newest record in a cn's ring that carries the group section."""
    for rec in reversed(cring.records):
        gz = getattr(rec, "group_z", None)
        if gz is not None and gz.size:
            return rec
    return None


def _fmt(v: Optional[float], spec: str = "%.3f") -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return spec % v


def build_array_sum_view(
    snap: StoreSnapshot, *, chgroup: int,
) -> dict[str, Any]:
    """Assemble everything the Array sums page needs.

    Args:
        snap: the dashboard's store snapshot.
        chgroup: which corr node's sub-band to show in detail. Each
            node decides independently on its own 11.72 MHz, so there
            is no fleet-wide time series to show — only a per-node
            one, plus the fleet summary table.

    Returns:
        A dict with ``detail`` (the selected chgroup, or ``None``),
        ``fleet`` (one row per chgroup), and ``running`` /
        ``mode`` / ``mode_label`` / ``mode_note`` describing the
        detector's state as last reported.
    """
    fleet: list[dict[str, Any]] = []
    detail: Optional[dict[str, Any]] = None
    mode = "off"
    flag_group = ""
    n_reporting = 0

    for cring in snap.per_chgroup:
        rec = _latest_with_groups(cring)
        row: dict[str, Any] = {
            "cn_id": cring.cn.cn_id,
            "host": cring.cn.host,
            "chgroup": cring.cn.chgroup,
            "reporting": rec is not None,
            "age_s": (
                round(max(0.0, snap.snapshot_unix - rec.publish_unix), 1)
                if rec is not None else None
            ),
        }
        if rec is not None:
            n_reporting += 1
            mode = rec.array_burst_mode or mode
            flag_group = rec.array_burst_flag_group or flag_group
            fi = rec.group_index(rec.array_burst_flag_group or "core")
            hot = (
                rec.group_fired[:, fi, :].astype(bool).any(axis=1)
                if (fi is not None and rec.group_fired is not None)
                else np.zeros(rec.group_z.shape[0], dtype=bool)
            )
            n_fired = int(hot.sum())
            n_samp = int(hot.size)
            row.update({
                "n_fired": n_fired,
                "n_samples": n_samp,
                "fired_pct": (100.0 * n_fired / n_samp) if n_samp else 0.0,
                "peak_sigma": float(
                    rec.group_z[:, fi, :].max()
                ) if fi is not None else None,
                "arm_ratio": _arm_ratio(rec, hot),
            })
        fleet.append(row)

        if cring.cn.chgroup == chgroup and rec is not None:
            detail = _detail_for(rec, cring)

    label, note = MODE_LABELS.get(mode, MODE_LABELS["off"])
    return {
        "chgroup": chgroup,
        "detail": detail,
        "fleet": fleet,
        "n_reporting": n_reporting,
        "n_chgroup": len(snap.per_chgroup),
        "running": n_reporting > 0,
        "mode": mode,
        "mode_label": label,
        "mode_note": note,
        "flag_group": flag_group,
        "snapshot_unix": snap.snapshot_unix,
    }


def _arm_ratio(rec, hot: np.ndarray) -> Optional[float]:
    """E-W / N-S fractional excess, averaged over the FIRED samples.

    Conditioning on the fired samples is not optional. Taking a peak
    over a whole window instead measures the largest noise excursion
    in ~1000 samples, which for the core is around 3 sigma of 0.1% —
    comfortably larger than the ~1.9% / 0.8% asymmetry it is meant to
    report, and completely insensitive to it.
    """
    if rec.group_band_frac is None or not hot.any():
        return None
    iew, ins = rec.group_index("ew_arm"), rec.group_index("ns_arm")
    if iew is None or ins is None:
        return None
    ew = float(rec.group_band_frac[hot, iew, :].mean())
    ns = float(rec.group_band_frac[hot, ins, :].mean())
    if abs(ns) < 1e-6:
        return None
    return ew / ns


def _detail_for(rec, cring) -> dict[str, Any]:
    """Per-group rows for the selected chgroup, plus window metadata."""
    fi = rec.group_index(rec.array_burst_flag_group or "core")
    hot = (
        rec.group_fired[:, fi, :].astype(bool).any(axis=1)
        if (fi is not None and rec.group_fired is not None)
        else np.zeros(rec.group_z.shape[0], dtype=bool)
    )
    rows: list[dict[str, Any]] = []
    for gi, name in enumerate(rec.group_names):
        z = rec.group_z[:, gi, :]
        bf = rec.group_band_frac[:, gi, :] if rec.group_band_frac is not None else None
        fired_g = (
            rec.group_fired[:, gi, :].astype(bool).any(axis=1)
            if rec.group_fired is not None else None
        )
        n_live = (
            float(rec.group_n_live[gi].mean())
            if rec.group_n_live is not None else None
        )
        rows.append({
            "name": name,
            "label": GROUP_LABELS.get(name, name),
            "is_flag_group": (gi == fi),
            "n_ant": (
                int(rec.group_sizes[gi]) if gi < len(rec.group_sizes) else 0
            ),
            "n_live": n_live,
            "n_live_s": _fmt(n_live, "%.0f"),
            "peak_sigma": float(z.max()),
            "peak_sigma_s": _fmt(float(z.max()), "%.1f"),
            "median_sigma_at_fire": (
                _fmt(float(np.median(z[hot])), "%.1f") if hot.any() else "—"
            ),
            "excess_at_fire_pct": (
                _fmt(100.0 * float(bf[hot].mean()), "%.2f")
                if (bf is not None and hot.any()) else "—"
            ),
            "fired_pct": (
                _fmt(100.0 * float(fired_g.mean()), "%.2f")
                if fired_g is not None else "—"
            ),
        })

    dt = rec.dt_s or 0.002097152
    return {
        "cn_id": cring.cn.cn_id,
        "host": cring.cn.host,
        "chgroup": cring.cn.chgroup,
        "rows": rows,
        "n_samples": int(rec.group_z.shape[0]),
        "n_fired": int(hot.sum()),
        "fired_pct": _fmt(100.0 * float(hot.mean()), "%.2f"),
        "dt_ms": round(dt * 1e3, 4),
        "window_s": round(rec.group_z.shape[0] * dt, 3),
        "block_n_start": rec.block_n_start,
        "block_n_end": rec.block_n_end,
        "arm_ratio": _arm_ratio(rec, hot),
        "arm_ratio_s": _fmt(_arm_ratio(rec, hot), "%.2f"),
        "publish_unix": rec.publish_unix,
    }
