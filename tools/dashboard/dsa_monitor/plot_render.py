"""matplotlib PNG renderers for the dashboard's Antennas/RFI tab.

All renderers take a :class:`StoreSnapshot` plus an ``ant_idx``
(0-based, 0..95) and return raw PNG bytes that the Flask route
serves with ``Content-Type: image/png``.

Renderer outputs:

  * :func:`render_bandpass_spectrum`   — latest 16-cube-mean
    pre-flag log10(S1) vs freq, with XX and YY overlaid. One line
    per pol, with cn boundaries marked.
  * :func:`render_bandpass_waterfall`  — 30-min (time, freq) heatmap
    of log10(S1) for the selected pol(s).
  * :func:`render_flag_spectrum`       — fraction of cubes flagged
    (final OR mask) in the latest window per channel.
  * :func:`render_flag_waterfall`      — 30-min flag-fraction
    waterfall (final OR), per (time, freq) bin.

All plots use the production 4× downsampled (1536 ch) freq axis
(:func:`freq_mapping.production_freq_axis_GHz`) — that matches the
arrays the corr-side window aggregator publishes.

Concurrency: matplotlib Agg backend is safe across threads because
each render creates its own ``Figure`` instance and never touches a
shared state machine (no ``plt.gcf()`` etc.).
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np

# Force Agg before any matplotlib import that might pick a GUI backend.
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402  (after matplotlib.use)
from matplotlib.figure import Figure  # noqa: E402

from freq_mapping import (
    production_chgroup_boundaries_GHz,
    production_freq_axis_GHz,
)
from rfi_store import StoreSnapshot

LOG = logging.getLogger("dsa_monitor.plot_render")


# Rest-frame neutral-hydrogen 21 cm spin-flip line. The DSA-110 L-band
# (1.311–1.499 GHz) brackets this; marking it on every plot makes
# RFI-vs-astrophysical separation obvious at a glance.
HI_REST_GHZ: float = 1.420405751768  # IAU 2009 best value


# ---------------------------------------------------------------------------
# Concatenation helpers
# ---------------------------------------------------------------------------


def _concat_per_ant_chgroups(
    snap: StoreSnapshot, *, ant_idx: int, accessor,
) -> Optional[np.ndarray]:
    """Concatenate per-chgroup arrays from the latest window of each cn.

    Returns ``None`` if any cn has no records (i.e. waiting on a
    cold-start). Otherwise returns a ``(N_CHGROUP * NCHAN_DS, NPOL)``
    fp32 array in chgroup order.

    ``accessor(rec)`` should return an ``(NANTS, NCHAN_DS, NPOL)``
    array. We then take ``[ant_idx, :, :]``.
    """
    pieces: list[np.ndarray] = []
    for cring in snap.per_chgroup:
        if not cring.records:
            return None
        rec = cring.records[-1]                    # latest in this cn's ring
        arr = accessor(rec)                        # (NANTS, NCHAN_DS, NPOL)
        if not (0 <= ant_idx < arr.shape[0]):
            return None
        pieces.append(arr[ant_idx])                # (NCHAN_DS, NPOL)
    return np.concatenate(pieces, axis=0)          # (N_CHGROUP*NCHAN_DS, NPOL)


def _concat_waterfall_per_ant(
    snap: StoreSnapshot, *, ant_idx: int, accessor,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Build a per-antenna 30-min waterfall.

    Returns ``(waterfall, time_axis_unix)``:
      waterfall shape ``(N_WIN, N_CHGROUP * NCHAN_DS, NPOL)`` (NPOL=2).
      time_axis_unix shape ``(N_WIN,)`` POSIX timestamps.

    Pairs windows from the 16 cn rings by ``block_n_start``: each row
    in the output corresponds to one unique block_n_start that has
    records from ALL 16 cn (we drop time bins where any cn is
    missing). The rows are ordered oldest-first.

    Returns ``None`` if there are no records on any cn.
    """
    rings = snap.per_chgroup
    if any(not cring.records for cring in rings):
        return None

    # Group records by block_n_start across all cn.
    # Map: block_n_start -> list of (chgroup, record).
    by_block: dict[int, dict[int, object]] = {}
    for g, cring in enumerate(rings):
        for r in cring.records:
            by_block.setdefault(r.block_n_start, {})[g] = r

    # Keep only blocks for which all 16 chgroups are present.
    aligned = sorted(
        b for b, gd in by_block.items() if len(gd) == 16
    )
    if not aligned:
        return None

    n_chan_total: int | None = None
    rows_xx: list[np.ndarray] = []
    rows_yy: list[np.ndarray] = []
    time_axis: list[float] = []
    for b in aligned:
        gd = by_block[b]
        pieces: list[np.ndarray] = []
        publish_unix_acc: list[float] = []
        for g in range(16):
            rec = gd[g]
            arr = accessor(rec)                    # (NANTS, NCHAN_DS, NPOL)
            if not (0 <= ant_idx < arr.shape[0]):
                return None
            pieces.append(arr[ant_idx])            # (NCHAN_DS, NPOL)
            publish_unix_acc.append(rec.publish_unix)
        full = np.concatenate(pieces, axis=0)      # (N_CHGROUP*NCHAN_DS, NPOL)
        if n_chan_total is None:
            n_chan_total = full.shape[0]
        rows_xx.append(full[:, 0])
        rows_yy.append(full[:, 1])
        time_axis.append(float(np.mean(publish_unix_acc)))

    waterfall = np.stack([
        np.stack(rows_xx, axis=0),
        np.stack(rows_yy, axis=0),
    ], axis=-1)                                    # (N_WIN, N_CH, NPOL)
    return waterfall, np.asarray(time_axis, dtype=np.float64)


# ---------------------------------------------------------------------------
# Figure → PNG
# ---------------------------------------------------------------------------


def _fig_to_png_bytes(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _placeholder_png(message: str) -> bytes:
    fig = plt.figure(figsize=(10, 3.5))
    ax = fig.add_subplot(111)
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center", fontsize=12,
        transform=ax.transAxes,
    )
    ax.set_axis_off()
    return _fig_to_png_bytes(fig)


def _add_chgroup_dividers(ax, *, axis: str = "x") -> None:
    bounds = production_chgroup_boundaries_GHz()
    if axis == "x":
        for b in bounds:
            ax.axvline(b, color="0.7", linestyle=":", linewidth=0.6,
                       alpha=0.7)
    else:
        for b in bounds:
            ax.axhline(b, color="0.7", linestyle=":", linewidth=0.6,
                       alpha=0.7)


def _add_hi_marker(
    ax, *, axis: str = "x", label: bool = True, color: str = "#00b894",
) -> None:
    """Draw the rest-frame HI (1.420 GHz) line + label on a freq axis."""
    if axis == "x":
        ax.axvline(
            HI_REST_GHZ, color=color, linestyle="--", linewidth=1.0,
            alpha=0.85, zorder=3,
        )
        if label:
            # Label sits at the top of the axes in axes-fraction y, but
            # at the HI frequency in data x. transform=ax.get_xaxis_transform()
            # gives that mixed coordinate system.
            ax.text(
                HI_REST_GHZ, 0.98, " HI",
                transform=ax.get_xaxis_transform(),
                color=color, fontsize=9, fontweight="bold",
                ha="left", va="top",
            )
    else:
        ax.axhline(
            HI_REST_GHZ, color=color, linestyle="--", linewidth=1.0,
            alpha=0.85, zorder=3,
        )
        if label:
            ax.text(
                0.02, HI_REST_GHZ, "HI ",
                transform=ax.get_yaxis_transform(),
                color=color, fontsize=9, fontweight="bold",
                ha="left", va="bottom",
            )


# ---------------------------------------------------------------------------
# Bandpass spectrum (latest 16-cube mean S1)
# ---------------------------------------------------------------------------


def render_bandpass_spectrum(
    snap: StoreSnapshot, *, ant_idx: int, ant_label: str,
) -> bytes:
    data = _concat_per_ant_chgroups(
        snap, ant_idx=ant_idx, accessor=lambda r: r.s1_full_mean,
    )
    if data is None:
        return _placeholder_png(
            f"ant {ant_label}: no latest-window bandpass yet "
            f"(waiting for all 16 corr nodes)"
        )

    freq_GHz = production_freq_axis_GHz()
    # Plot log10(S1+epsilon) per pol.
    eps = 1e-3
    xx = np.log10(np.maximum(data[:, 0], eps))
    yy = np.log10(np.maximum(data[:, 1], eps))

    fig = plt.figure(figsize=(11, 3.8))
    ax = fig.add_subplot(111)
    ax.plot(freq_GHz, xx, lw=0.8, color="#0984e3", label="XX (pol 0)")
    ax.plot(freq_GHz, yy, lw=0.8, color="#d35400", label="YY (pol 1)")
    _add_chgroup_dividers(ax, axis="x")
    _add_hi_marker(ax, axis="x")
    ax.invert_xaxis()                              # descending → ascending visually
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel(r"$\log_{10}\,S_1$ (pre-flag, 16-cube mean)")
    ax.set_title(f"Ant {ant_label} bandpass — latest window")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)
    return _fig_to_png_bytes(fig)


# ---------------------------------------------------------------------------
# Flag spectrum (latest 16-cube flag fraction)
# ---------------------------------------------------------------------------


def render_flag_spectrum(
    snap: StoreSnapshot, *, ant_idx: int, ant_label: str,
) -> bytes:
    data = _concat_per_ant_chgroups(
        snap, ant_idx=ant_idx, accessor=lambda r: r.mask_count_final,
    )
    if data is None:
        return _placeholder_png(
            f"ant {ant_label}: no latest-window flag spectrum yet"
        )
    # mask_count_final is in {0..n_cubes}; the latest record's n_cubes
    # is the same across all chgroups (16 cubes/window in production).
    # Use the first cn record's n_cubes as the normaliser; assume they
    # all match (which they do at the production cadence).
    n_cubes = snap.per_chgroup[0].records[-1].n_cubes if snap.per_chgroup else 16
    n_cubes = max(1, int(n_cubes))
    frac = data.astype(np.float32) / float(n_cubes)

    freq_GHz = production_freq_axis_GHz()
    fig = plt.figure(figsize=(11, 3.8))
    ax = fig.add_subplot(111)
    ax.plot(freq_GHz, frac[:, 0], lw=0.8, color="#0984e3",
            label="XX (pol 0)")
    ax.plot(freq_GHz, frac[:, 1], lw=0.8, color="#d35400",
            label="YY (pol 1)")
    _add_chgroup_dividers(ax, axis="x")
    _add_hi_marker(ax, axis="x")
    ax.invert_xaxis()
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel("flag fraction (latest 16-cube window)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Ant {ant_label} flag spectrum — latest window")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)
    return _fig_to_png_bytes(fig)


# ---------------------------------------------------------------------------
# Waterfall helpers
# ---------------------------------------------------------------------------


def _render_waterfall(
    *, waterfall: np.ndarray, time_axis_unix: np.ndarray,
    title: str, cbar_label: str, vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> bytes:
    """Render a two-panel waterfall (XX + YY) sharing an x axis.

    waterfall: (n_win, n_ch, 2 pol) fp32
    time_axis_unix: (n_win,) — most recent at top of plot
    """
    n_win, n_ch, _ = waterfall.shape
    freq_GHz = production_freq_axis_GHz()
    assert n_ch == freq_GHz.size, (
        f"waterfall n_ch={n_ch} != freq axis length={freq_GHz.size}"
    )
    # Time axis labels: minutes ago.
    now = time_axis_unix.max()
    minutes_ago = (now - time_axis_unix) / 60.0    # 0 = newest, +30 = oldest
    extent = (
        float(freq_GHz.max()), float(freq_GHz.min()),
        float(minutes_ago.max()), float(minutes_ago.min()),
    )

    fig = plt.figure(figsize=(11, 7.0))
    for i, (pol_name, pol_idx) in enumerate(
        (("XX", 0), ("YY", 1)),
    ):
        ax = fig.add_subplot(2, 1, i + 1)
        im = ax.imshow(
            waterfall[..., pol_idx][::-1, :],     # newest at top
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap="inferno",
            vmin=vmin, vmax=vmax,
        )
        ax.set_xlabel("frequency [GHz]" if i == 1 else "")
        ax.set_ylabel("minutes ago")
        ax.set_title(f"{title} — {pol_name}")
        # cn dividers (vertical lines)
        for b in production_chgroup_boundaries_GHz():
            ax.axvline(b, color="0.7", linestyle=":", linewidth=0.5,
                       alpha=0.5)
        _add_hi_marker(ax, axis="x")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.01,
                     label=cbar_label)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


# ---------------------------------------------------------------------------
# Bandpass waterfall (30 min)
# ---------------------------------------------------------------------------


def render_bandpass_waterfall(
    snap: StoreSnapshot, *, ant_idx: int, ant_label: str,
) -> bytes:
    wf = _concat_waterfall_per_ant(
        snap, ant_idx=ant_idx,
        accessor=lambda r: r.s1_full_mean,
    )
    if wf is None:
        return _placeholder_png(
            f"ant {ant_label}: no aligned waterfall windows yet "
            f"(waiting for blocks present on all 16 corr nodes)"
        )
    waterfall, time_axis = wf
    # log10 for dynamic range.
    waterfall = np.log10(np.maximum(waterfall, 1e-3)).astype(np.float32)
    return _render_waterfall(
        waterfall=waterfall, time_axis_unix=time_axis,
        title=f"Ant {ant_label} bandpass (log10 S1) — last 30 min",
        cbar_label=r"$\log_{10}\,S_1$",
    )


# ---------------------------------------------------------------------------
# Flag waterfall (30 min)
# ---------------------------------------------------------------------------


def render_flag_waterfall(
    snap: StoreSnapshot, *, ant_idx: int, ant_label: str,
) -> bytes:
    # We accumulate mask_count_final / n_cubes per record.
    def _frac_accessor(r):
        n = max(1, int(r.n_cubes))
        return r.mask_count_final.astype(np.float32) / float(n)

    wf = _concat_waterfall_per_ant(
        snap, ant_idx=ant_idx, accessor=_frac_accessor,
    )
    if wf is None:
        return _placeholder_png(
            f"ant {ant_label}: no aligned waterfall windows yet"
        )
    waterfall, time_axis = wf
    return _render_waterfall(
        waterfall=waterfall, time_axis_unix=time_axis,
        title=f"Ant {ant_label} flag fraction — last 30 min",
        cbar_label="flag fraction",
        vmin=0.0, vmax=1.0,
    )


# ---------------------------------------------------------------------------
# Fleet thumbnail grid (96 ants in a 12-col × 8-row arrangement)
# ---------------------------------------------------------------------------


_THUMB_GRID_COLS: int = 12
_THUMB_GRID_ROWS: int = 8


def render_thumb_grid(
    snap: StoreSnapshot, *, ant_nums: tuple[int, ...],
) -> bytes:
    """Render a 12 × 8 = 96 panel grid of latest pre-flag bandpasses.

    One mini-axes per antenna, with the real DSA-110 antenna number
    (from antenna_map.py) labelled in the top-left of each panel. XX +
    YY traces overlaid. No axis ticks or labels — these are at-a-glance
    spectra, not analysis plots; use the dropdown to drill into one.

    Antennas where the latest-window concat fails (e.g. one or more
    chgroups absent) are rendered as a greyed-out empty panel labelled
    with the ant number only, so the grid layout is preserved.

    Falls back to a single placeholder panel if NONE of the antennas
    have data (cold start case).
    """
    if len(ant_nums) != _THUMB_GRID_COLS * _THUMB_GRID_ROWS:
        raise ValueError(
            f"render_thumb_grid: expected "
            f"{_THUMB_GRID_COLS * _THUMB_GRID_ROWS} ant_nums, "
            f"got {len(ant_nums)}"
        )

    freq_GHz = production_freq_axis_GHz()
    eps = 1e-3

    # Pre-fetch all 96 antenna data once (avoids re-doing the per-ant
    # concat 96 times inside the matplotlib loop). Latest record per cn.
    per_ant_data: list[np.ndarray | None] = []
    for ant_idx in range(_THUMB_GRID_COLS * _THUMB_GRID_ROWS):
        d = _concat_per_ant_chgroups(
            snap, ant_idx=ant_idx, accessor=lambda r: r.s1_full_mean,
        )
        per_ant_data.append(d)

    have_any = any(d is not None for d in per_ant_data)
    if not have_any:
        return _placeholder_png(
            "fleet thumbnails: no latest-window data yet "
            "(waiting for all 16 corr nodes)"
        )

    # Compute a fleet-wide y-range so all 96 panels share a scale and
    # the operator can compare amplitudes by eye. Robust min/max
    # (5..99 percentile) to suppress single-channel RFI spikes.
    stacked = np.concatenate([
        np.log10(np.maximum(d, eps)).ravel()
        for d in per_ant_data if d is not None
    ])
    y_lo, y_hi = np.percentile(stacked, (5.0, 99.0))
    if y_hi - y_lo < 0.05:
        y_hi = y_lo + 0.05  # avoid degenerate range
    # Add headroom for the in-panel ant-num label.
    y_hi_padded = y_hi + 0.18 * (y_hi - y_lo)

    fig = plt.figure(figsize=(16.0, 8.6))
    fig.suptitle(
        "Fleet bandpass thumbnails — latest 16-cube window "
        "(XX = blue, YY = orange, HI dashed at 1.42 GHz)",
        fontsize=12, y=0.995,
    )
    # GridSpec with tight inter-panel spacing.
    gs = fig.add_gridspec(
        _THUMB_GRID_ROWS, _THUMB_GRID_COLS,
        wspace=0.06, hspace=0.18,
        left=0.025, right=0.995,
        bottom=0.015, top=0.955,
    )

    for ant_idx, ant_num in enumerate(ant_nums):
        row = ant_idx // _THUMB_GRID_COLS
        col = ant_idx % _THUMB_GRID_COLS
        ax = fig.add_subplot(gs[row, col])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("#b2bec3")

        d = per_ant_data[ant_idx]
        if d is None:
            ax.set_facecolor("#ecf0f1")
            ax.text(
                0.5, 0.5, "no data",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=7, color="#b2bec3",
            )
        else:
            xx = np.log10(np.maximum(d[:, 0], eps))
            yy = np.log10(np.maximum(d[:, 1], eps))
            ax.plot(freq_GHz, xx, lw=0.45, color="#0984e3")
            ax.plot(freq_GHz, yy, lw=0.45, color="#d35400")
            ax.set_ylim(y_lo, y_hi_padded)
            # HI marker (no text — too cramped at thumbnail size).
            ax.axvline(
                HI_REST_GHZ, color="#00b894",
                linestyle="--", linewidth=0.5, alpha=0.7,
            )
            ax.set_xlim(freq_GHz.max(), freq_GHz.min())  # descending

        # Ant-num label — top-left of every panel, ALWAYS rendered
        # (even for "no data" panels so the user can spot dead ants).
        ax.text(
            0.04, 0.94, f"ant {ant_num}",
            transform=ax.transAxes,
            fontsize=7.5, fontweight="bold",
            ha="left", va="top",
            color="#2d3436",
            bbox={
                "facecolor": "white",
                "alpha": 0.78,
                "edgecolor": "none",
                "pad": 1.2,
            },
        )

    return _fig_to_png_bytes(fig)
