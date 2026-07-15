"""Per-event archive layout writer.

For each triggered cluster, C2 lays out a directory tree under
``coinc.event_archive_root/<event_name>/`` (default
``/dataz/dsa110/candidates/<event_name>/``):

```
<event_name>/
├── Level2/
│   ├── C2_<name>.csv           # the cluster's per-candidate rows
│   ├── C1_window_<name>.csv    # all in-window C1 candidates around it
│   └── plots/                  # populated later by the plot worker
├── Level3/
│   └── <name>.json             # trigger metadata
├── cubes/                      # populated by cube_uploader rsyncs
├── voltages/                   # filled later by corr-side voltage dump
├── filterbank/                 # filled later
└── calibration/                # symlink to fixture cal/ for replays
```

See ``docs/c1c2/C1C2_DESIGN.md`` §3.5 for the layout contract.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from .stats import ClusterStats
from .window import WindowEntry

__all__ = [
    "EventArchiveWriter",
    "C1_WINDOW_CSV_FIELDS",
    "C2_CLUSTER_CSV_FIELDS",
    "EVENT_SUBDIRS",
]


_LOG = logging.getLogger("dsart.coinc.archive")


# Subdirectories created beneath /dataz/dsa110/candidates/<name>/.
# voltages/ and filterbank/ are filled later by other workers; we
# still create them so downstream tooling can assume they exist.
EVENT_SUBDIRS: tuple[str, ...] = (
    "Level2",
    "Level2/plots",
    "Level3",
    "cubes",
    "voltages",
    "filterbank",
    "calibration",
)


# C1 per-row CSV schema. Order matches docs/c1c2/C1C2_DESIGN.md §3.6.
# M7.4 Phase 6c.A appends ``inj_id`` at the end so existing hiplot /
# pandas readers that index by column-name still work; the new column
# is empty for non-injection rows.
C1_WINDOW_CSV_FIELDS: tuple[str, ...] = (
    "mjd",
    "event_specnum",
    "snr",
    "dm_pc_cc",
    "dm_idx_global",
    "fine_dm_idx",
    "l_rad",
    "m_rad",
    "l_pix",
    "m_pix",
    "width_samples",
    "kernel_id",
    "flags",
    "search_node_id",
    "gpu_half",
    "cube_id",
    "trigger",
    "inj_id",
)


# C2 per-cluster CSV schema. Order matches docs/c1c2/C1C2_DESIGN.md §3.6.
C2_CLUSTER_CSV_FIELDS: tuple[str, ...] = (
    "mjd_peak",
    "snr_max",
    "snr_sum",
    "snr_mean",
    "n_events",
    "n_search_nodes",
    "n_gpu_halves",
    "dm_median",
    "dm_iqr",
    "dm_min",
    "dm_max",
    "l_median",
    "m_median",
    "lm_diag_rad",
    "width_median",
    "width_min",
    "width_max",
    "t_span_s",
    "t_start_mjd",
    "t_end_mjd",
    "kernel_ids_distinct",
    "gal_dm_max_los_pc_cc",
    "dm_galactic_fraction",
    "trigger_class",
    "trigger",
)


def _atomic_write(path: Path, body: str) -> None:
    """Write ``body`` to ``path`` via tempfile + os.rename (atomic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        os.replace(tmp, path)
    except Exception:
        # Best effort to clean up the temp file on failure.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def entry_to_csv_row(
    entry: WindowEntry, *, trigger: str = "", inj_id: str = "",
) -> dict[str, object]:
    return {
        "mjd": f"{entry.mjd:.11f}",
        "event_specnum": entry.event_specnum,
        "snr": f"{entry.snr:.6e}",
        "dm_pc_cc": f"{entry.dm_pc_cc:.6f}",
        "dm_idx_global": entry.dm_idx_global,
        "fine_dm_idx": entry.fine_dm_idx,
        "l_rad": f"{entry.l_rad:.9e}",
        "m_rad": f"{entry.m_rad:.9e}",
        "l_pix": entry.l_pix,
        "m_pix": entry.m_pix,
        "width_samples": entry.width_samples,
        "kernel_id": entry.kernel_id,
        "flags": entry.flags,
        "search_node_id": entry.search_node_id,
        "gpu_half": entry.gpu_half,
        "cube_id": entry.cube_id,
        "trigger": trigger,
        "inj_id": inj_id,
    }


def stats_to_csv_row(
    stats: ClusterStats,
    *,
    trigger_class: str = "",
    trigger: str = "",
) -> dict[str, object]:
    t_span_s = (stats.t_end_mjd - stats.t_start_mjd) * 86400.0
    # NaN-safe stringification: write empty string instead of "nan" so
    # downstream consumers (pandas read_csv with default NaN sentinel,
    # hiplot) treat them as missing rather than the string literal.
    gal_dm = ""
    if math.isfinite(stats.gal_dm_max_los):
        gal_dm = f"{stats.gal_dm_max_los:.6f}"
    dm_frac = ""
    if math.isfinite(stats.dm_galactic_fraction):
        dm_frac = f"{stats.dm_galactic_fraction:.6f}"
    return {
        "mjd_peak": f"{stats.t_peak_mjd:.11f}",
        "snr_max": f"{stats.snr_max:.6e}",
        "snr_sum": f"{stats.snr_sum:.6e}",
        "snr_mean": f"{stats.snr_mean:.6e}",
        "n_events": stats.n_events,
        "n_search_nodes": stats.n_search_nodes,
        "n_gpu_halves": stats.n_gpu_halves,
        "dm_median": f"{stats.dm_median:.6f}",
        "dm_iqr": f"{stats.dm_iqr:.6f}",
        "dm_min": f"{stats.dm_min:.6f}",
        "dm_max": f"{stats.dm_max:.6f}",
        "l_median": f"{stats.l_median:.9e}",
        "m_median": f"{stats.m_median:.9e}",
        "lm_diag_rad": f"{stats.lm_diag_rad:.9e}",
        "width_median": f"{stats.width_median:.3f}",
        "width_min": stats.width_min,
        "width_max": stats.width_max,
        "t_span_s": f"{t_span_s:.6e}",
        "t_start_mjd": f"{stats.t_start_mjd:.11f}",
        "t_end_mjd": f"{stats.t_end_mjd:.11f}",
        "kernel_ids_distinct": ";".join(stats.kernel_ids_distinct),
        "gal_dm_max_los_pc_cc": gal_dm,
        "dm_galactic_fraction": dm_frac,
        "trigger_class": trigger_class,
        "trigger": trigger,
    }


class EventArchiveWriter:
    """Create + populate ``<archive_root>/<event_name>/``.

    Use:
        wr = EventArchiveWriter(archive_root, calibration_source=...)
        evdir = wr.create("260521abcd")
        wr.write_c2_cluster_csv(evdir, "260521abcd", stats,
                                trigger_class="bright_frb")
        wr.write_c1_window_csv(evdir, "260521abcd", members,
                               trigger="260521abcd")
        wr.write_l3_metadata(evdir, "260521abcd", metadata_dict)
    """

    def __init__(
        self,
        archive_root: Path,
        *,
        calibration_source: Optional[Path] = None,
    ) -> None:
        self._root = Path(archive_root)
        self._cal_src = (
            Path(calibration_source) if calibration_source else None
        )

    @property
    def root(self) -> Path:
        return self._root

    def event_dir(self, event_name: str) -> Path:
        return self._root / event_name

    def create(self, event_name: str) -> Path:
        """Create the per-event directory tree; idempotent."""
        if not event_name or "/" in event_name:
            raise ValueError(f"bad event_name: {event_name!r}")
        ev = self.event_dir(event_name)
        for sub in EVENT_SUBDIRS:
            (ev / sub).mkdir(parents=True, exist_ok=True)
        # Symlink calibration/ if a source is configured. Existing
        # entries (file or symlink) are left alone.
        if self._cal_src is not None:
            cal_dir = ev / "calibration"
            # We created cal_dir as a mkdir above; only replace it with
            # a symlink if empty (don't blow away anything the operator
            # may have dropped in there).
            if cal_dir.is_dir() and not any(cal_dir.iterdir()):
                try:
                    cal_dir.rmdir()
                    os.symlink(str(self._cal_src), str(cal_dir))
                except OSError as exc:
                    _LOG.warning(
                        "failed to symlink calibration/ -> %s for %s: %s",
                        self._cal_src, event_name, exc,
                    )
        return ev

    def write_c2_cluster_csv(
        self,
        event_dir: Path,
        event_name: str,
        stats: ClusterStats,
        *,
        trigger_class: str = "",
        trigger: str = "",
    ) -> Path:
        path = event_dir / "Level2" / f"C2_{event_name}.csv"
        row = stats_to_csv_row(
            stats, trigger_class=trigger_class, trigger=trigger,
        )
        body = self._render_csv(C2_CLUSTER_CSV_FIELDS, [row])
        _atomic_write(path, body)
        return path

    def write_c1_window_csv(
        self,
        event_dir: Path,
        event_name: str,
        members: Iterable[WindowEntry],
        *,
        trigger: str = "",
        inj_ids: Optional[Mapping[int, str]] = None,
    ) -> Path:
        """Write the per-event C1 window CSV.

        ``inj_ids`` (optional) maps a member's ``id(entry)`` to the
        injection id it matched (Phase 6c.A label-as-injection path).
        Members not in the map default to empty ``inj_id`` so the CSV
        is fully populated for non-injection rows.
        """
        path = event_dir / "Level2" / f"C1_window_{event_name}.csv"
        id_map: Mapping[int, str] = inj_ids or {}
        rows = [
            entry_to_csv_row(
                m, trigger=trigger, inj_id=id_map.get(id(m), ""),
            )
            for m in members
        ]
        body = self._render_csv(C1_WINDOW_CSV_FIELDS, rows)
        _atomic_write(path, body)
        return path

    def write_l3_metadata(
        self,
        event_dir: Path,
        event_name: str,
        metadata: Mapping[str, object],
    ) -> Path:
        path = event_dir / "Level3" / f"{event_name}.json"
        body = json.dumps(metadata, indent=2, sort_keys=True, default=str)
        _atomic_write(path, body + "\n")
        return path

    @staticmethod
    def _render_csv(
        fields: Sequence[str], rows: Sequence[Mapping[str, object]],
    ) -> str:
        import io
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf, fieldnames=list(fields), extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return buf.getvalue()


def stats_to_l3_metadata(
    *,
    event_name: str,
    stats: ClusterStats,
    trigger_class_name: str,
    trigger_action: str,
    holdoff_s: float,
    schema_version: int = 1,
    inj_ids: Optional[Iterable[str]] = None,
    pointing_dec_deg: Optional[float] = None,
    pointing_dec_meta: Optional[Mapping[str, object]] = None,
) -> dict[str, object]:
    """Build the L3 JSON payload the legacy archive consumer expects.

    Mirrors the legacy tasktrigger shape where it's compatible; new
    keys are nested under ``c2`` so we don't fight the old consumer.

    ``inj_ids`` (the distinct injection labels matched to this event's
    members at FIRE time, when the time-sensitive inject registry was
    still live) is recorded under a durable ``injection`` block so
    downstream consumers — notably C3, which runs minutes later, long
    after the registry's ~60 s TTL — can reliably exempt injections from
    the cube veto without re-querying the registry.

    ``pointing_dec_deg`` (degrees) is a contemporaneous snapshot of the
    array pointing declination (``/mon/array/dec``) taken at archive
    time. The ``c2.l_median``/``c2.m_median`` image offsets are RELATIVE
    to this pointing, so it is required to recover the event's absolute
    RA/Dec: the live etcd key carries no history and is overwritten on
    every re-point. ``None`` when the read failed (never fabricated);
    ``pointing_dec_meta`` records ``{"etcd_key", "read_unix"}`` for
    provenance. Both fields are additive under ``c2`` — consumers must
    tolerate their absence on pre-existing events — so ``schema_version``
    is intentionally NOT bumped (matches the ``gal_dm`` additive
    precedent above).
    """
    inj_list = sorted({str(i) for i in (inj_ids or ()) if str(i).strip()})
    return {
        "event_name": event_name,
        "schema_version": schema_version,
        "injection": {
            "is_injection": bool(inj_list),
            "inj_ids": inj_list,
        },
        "trigger": {
            "class": trigger_class_name,
            "action": trigger_action,
            "holdoff_s": holdoff_s,
        },
        "c2": {
            "n_events": stats.n_events,
            "n_search_nodes": stats.n_search_nodes,
            "n_gpu_halves": stats.n_gpu_halves,
            "snr_max": stats.snr_max,
            "snr_mean": stats.snr_mean,
            "snr_sum": stats.snr_sum,
            "dm_min": stats.dm_min,
            "dm_max": stats.dm_max,
            "dm_median": stats.dm_median,
            "dm_iqr": stats.dm_iqr,
            "l_median": stats.l_median,
            "m_median": stats.m_median,
            "lm_diag_rad": stats.lm_diag_rad,
            "width_min": stats.width_min,
            "width_max": stats.width_max,
            "width_median": stats.width_median,
            "t_start_mjd": stats.t_start_mjd,
            "t_end_mjd": stats.t_end_mjd,
            "t_peak_mjd": stats.t_peak_mjd,
            "peak_event_specnum": stats.peak_event_specnum,
            "kernel_ids_distinct": list(stats.kernel_ids_distinct),
            # Galactic-DM discriminant (None when /mon/array/gal_dm
            # was unavailable at the time of the trigger).
            "gal_dm_max_los_pc_cc": (
                stats.gal_dm_max_los
                if math.isfinite(stats.gal_dm_max_los) else None
            ),
            "dm_galactic_fraction": (
                stats.dm_galactic_fraction
                if math.isfinite(stats.dm_galactic_fraction) else None
            ),
            # Array pointing declination (deg) at archive time; the
            # l_median/m_median offsets above are relative to it. None
            # when /mon/array/dec could not be read (never invented).
            "pointing_dec_deg": pointing_dec_deg,
            "pointing_dec_meta": (
                dict(pointing_dec_meta)
                if pointing_dec_meta is not None else None
            ),
        },
    }
