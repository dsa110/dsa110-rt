"""C3 → dsa110-bbproc filterbank generation for KEEP candidates.

After C3 collects a KEEP event's voltage fragments into
``<cand>/Level2/voltages/``, this module (best-effort, never raises into
the C3 loop):

1. **snapshots the calibration**: copies the newest complete per-subband
   ``beamformer_weights_sb<NN>_<isot>.dat`` set (16 blobs) from the
   realtime "applied" directory into ``<cand>/filterbank/cal/`` so the
   filterbank is always reproducible with event-contemporaneous cal;
2. runs the dsa110-bbproc ``toolkit`` to coherently beamform the core
   antennas toward the cluster's ``(l_median, m_median)`` and write a
   full-band SIGPROC filterbank into ``<cand>/filterbank/`` — optionally
   twice (with and without the realtime-equivalent SK RFI flagging);
3. runs ``tools/plot_fil.py`` on each filterbank to produce the
   candidate inspection PNG the dashboard shows on the event page;
4. writes ``<cand>/filterbank/filterbank.json`` provenance (cal set,
   exact commands, return codes, durations).

Serialization: callers invoke :func:`run_for_event` synchronously from
the C3 scan loop, so filterbank jobs run strictly one after another —
no GPU or disk-bandwidth contention between events by construction.

The toolkit binary and plot script live in the dsa110-bbproc repo
(github.com/dsa110/dsa110-bbproc); paths come from the ``c3.filterbank``
config section.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("dsart.coinc.filterbank")

N_SUBBANDS = 16

#: .fil native sample = fada native sample (see dsa110-bbproc bbproc.h).
NATIVE_SAMPLE_US = 32.768

#: search sample (t_int_search) in us — width_median's unit in the C2 row.
T_INT_SEARCH_US = 1048.576


@dataclass(frozen=True)
class FilterbankConfig:
    enabled: bool = False
    toolkit_bin: str = "/home/ubuntu/proj/dsa110-shell/dsa110-bbproc/toolkit"
    plot_script: str = (
        "/home/ubuntu/proj/dsa110-shell/dsa110-bbproc/tools/plot_fil.py")
    core_antennas: str = (
        "/home/ubuntu/proj/dsa110-shell/dsa110-bbproc/config/"
        "core_antennas.txt")
    cal_applied_dir: str = (
        "/dataz/dsa110/operations/beamformer_weights/applied")
    gpu: int = 0
    tscrunch: int = 8
    #: "off" -> one unflagged .fil; "on" -> one SK-flagged .fil;
    #: "both" -> both (suffix _rfi on the flagged one).
    rfi_mode: str = "both"
    phase_only: bool = True
    timeout_s: float = 1800.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FilterbankConfig":
        d = d or {}
        return cls(
            enabled=bool(d.get("enabled", False)),
            toolkit_bin=str(d.get("toolkit_bin", cls.toolkit_bin)),
            plot_script=str(d.get("plot_script", cls.plot_script)),
            core_antennas=str(d.get("core_antennas", cls.core_antennas)),
            cal_applied_dir=str(d.get("cal_applied_dir",
                                      cls.cal_applied_dir)),
            gpu=int(d.get("gpu", 0)),
            tscrunch=int(d.get("tscrunch", 8)),
            rfi_mode=str(d.get("rfi_mode", "both")),
            phase_only=bool(d.get("phase_only", True)),
            timeout_s=float(d.get("timeout_s", 1800.0)),
        )


# ---------------------------------------------------------------------------
# cal snapshot
# ---------------------------------------------------------------------------

_CAL_RE = re.compile(
    r"beamformer_weights_sb(\d\d)_(?P<isot>[0-9T:\-]+)\.dat$")


def newest_complete_cal_set(applied_dir: str) -> Optional[Dict[str, str]]:
    """Newest ``isot`` for which all 16 per-subband blobs exist.

    Returns ``{sbNN: path}`` (16 entries) or None."""
    by_isot: Dict[str, Dict[str, str]] = {}
    for p in glob.glob(os.path.join(applied_dir,
                                    "beamformer_weights_sb??_*.dat")):
        m = _CAL_RE.search(os.path.basename(p))
        if not m:
            continue
        by_isot.setdefault(m.group("isot"), {})[m.group(1)] = p
    for isot in sorted(by_isot, reverse=True):
        if len(by_isot[isot]) == N_SUBBANDS:
            return by_isot[isot]
    return None


def snapshot_cal(applied_dir: str, dest_dir: Path) -> Optional[Path]:
    """Copy the newest complete cal set into ``dest_dir``.

    Idempotent: if ``dest_dir`` already holds a complete snapshot (e.g.
    a re-run after a crash), it is reused untouched — provenance beats
    freshness. Returns the snapshot's sb00 blob path, or None."""
    existing = sorted(dest_dir.glob("beamformer_weights_sb00_*.dat"))
    if existing:
        return existing[0]
    cal = newest_complete_cal_set(applied_dir)
    if cal is None:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    sb00: Optional[Path] = None
    for sb, src in sorted(cal.items()):
        dst = dest_dir / os.path.basename(src)
        shutil.copy2(src, dst)
        if sb == "00":
            sb00 = dst
    return sb00


# ---------------------------------------------------------------------------
# per-event pipeline
# ---------------------------------------------------------------------------


def _run(cmd: List[str], timeout_s: float) -> Dict[str, Any]:
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s)
        tail = ((proc.stdout or "") + (proc.stderr or "")).splitlines()[-12:]
        return {"cmd": cmd, "rc": proc.returncode,
                "elapsed_s": round(time.monotonic() - t0, 1),
                "tail": tail}
    except subprocess.TimeoutExpired:
        return {"cmd": cmd, "rc": None, "error": f"timeout {timeout_s}s",
                "elapsed_s": round(time.monotonic() - t0, 1)}
    except Exception as exc:  # noqa: BLE001
        return {"cmd": cmd, "rc": None,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(time.monotonic() - t0, 1)}


def run_for_event(cfg: FilterbankConfig, ev_dir: Path, name: str,
                  c2row: Dict[str, Any]) -> Dict[str, Any]:
    """Generate filterbank(s) + inspection plot(s) for one KEEP event.

    ``c2row`` is the ``c2`` dict from ``Level3/<name>.json`` (l_median,
    m_median, dm_median, width_median). Best-effort: returns a report
    dict; never raises."""
    report: Dict[str, Any] = {"ok": False}
    try:
        volt_dir = ev_dir / "Level2" / "voltages"
        n_frags = len(list(volt_dir.glob(f"{name}_sb??_data.out"))) \
            if volt_dir.is_dir() else 0
        if n_frags == 0:
            report["skipped"] = "no voltage fragments"
            report["ok"] = True  # nothing to do is not a failure
            return report
        fb_dir = ev_dir / "filterbank"
        fb_dir.mkdir(parents=True, exist_ok=True)

        l = float(c2row.get("l_median") or 0.0)
        m = float(c2row.get("m_median") or 0.0)
        dm = float(c2row.get("dm_median") or 0.0)
        width_search = float(c2row.get("width_median") or 1.0)
        fil_tsamp_us = NATIVE_SAMPLE_US * cfg.tscrunch
        width_fil = max(1, round(width_search * T_INT_SEARCH_US /
                                 fil_tsamp_us))
        report.update({"n_fragments": n_frags, "l": l, "m": m, "dm": dm,
                       "width_fil_samples": width_fil})

        cal_sb00 = snapshot_cal(cfg.cal_applied_dir, fb_dir / "cal")
        if cal_sb00 is not None:
            report["cal_sb00"] = str(cal_sb00)
        else:
            LOG.warning("filterbank %s: no complete cal set under %s — "
                        "beamforming without cal/(l,m) phasing", name,
                        cfg.cal_applied_dir)
            report["cal_sb00"] = None

        variants = {"both": [False, True], "on": [True],
                    "off": [False]}.get(cfg.rfi_mode, [False, True])
        runs: List[Dict[str, Any]] = []
        outputs: List[str] = []
        all_ok = True
        for rfi in variants:
            suffix = "_rfi" if rfi else ""
            fil = fb_dir / f"{name}{suffix}.fil"
            png = fb_dir / f"{name}{suffix}.png"
            cmd = [cfg.toolkit_bin, "-D", str(volt_dir), "-E", name,
                   "-P", str(fil), "--l", repr(l), "--m", repr(m),
                   "--core", cfg.core_antennas,
                   "--tscrunch", str(cfg.tscrunch),
                   "--gpu", str(cfg.gpu)]
            if cal_sb00 is not None:
                cmd += ["-w", str(cal_sb00)]
                if cfg.phase_only:
                    cmd += ["--phase-only"]
            if rfi:
                cmd += ["--rfi"]
            r = _run(cmd, cfg.timeout_s)
            runs.append(r)
            if r.get("rc") == 0 and fil.is_file():
                outputs.append(fil.name)
                pr = _run([sys.executable, cfg.plot_script, str(fil),
                           "--dm", repr(dm), "--width", str(width_fil),
                           "--event", name, "--lm", repr(l), repr(m),
                           "--out", str(png)], 600.0)
                runs.append(pr)
                if pr.get("rc") == 0 and png.is_file():
                    outputs.append(png.name)
                else:
                    all_ok = False
            else:
                all_ok = False

        report["outputs"] = outputs
        report["ok"] = all_ok and bool(outputs)
        report["runs"] = runs
        try:
            with (fb_dir / "filterbank.json").open("w") as fh:
                json.dump(report, fh, indent=2, default=str)
        except OSError as exc:
            LOG.warning("filterbank %s: provenance write failed: %s",
                        name, exc)
        LOG.info("filterbank %s: ok=%s outputs=%s", name, report["ok"],
                 outputs)
        return report
    except Exception as exc:  # noqa: BLE001 — never break the C3 loop
        LOG.exception("filterbank %s failed", name)
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report
