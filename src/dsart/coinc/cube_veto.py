"""Cube-morphology veto for C3 (productionised from the 2026-06-18
B1933/false-positive analysis).

Given a completed candidate directory (the dumped cubes + the C1-window
CSV C2 already wrote), decide whether the event is a high-confidence
false positive that C3 may conservatively reject, or whether it should be
kept (the default for anything ambiguous, and ALWAYS for injections).

The rules + thresholds are exactly the injection-safe scheme validated in
``reports/b1933_fp_analysis_20260618/REPORT.md`` §4:

  R1  image incoherence — apex image pixel > 40 px from the C1 (l, m)
  R2  time incoherence  — cube time-apex > 30 samples from the LC peak
  R3  DM incoherence    — cube DM-apex > 8 fine-DM trials from the trigger
  R4  no peak           — time prominence < 4σ AND image prominence < 4σ
  R5  DM-edge rail      — apex rails to a DM-trial edge AND > 4 trials off
  R10 cube doesn't confirm — global apex (over ALL cubes) lands in a
                             different cube than the trigger AND the
                             trigger feature is weak (tz_trig < 10)

Validated outcome: 74/89 true FPs removed, **0** injections and **0**
B1933 pulses lost. Injections are ALWAYS exempt (forced KEEP) — see
:func:`decide`.

Design: the rule logic (:func:`decide`) is pure over a :class:`CubeMetrics`
struct so it is trivially unit-testable; the file I/O that builds the
metrics lives in :func:`compute_metrics` / :func:`compute_metrics_from_grids`.
"""

from __future__ import annotations

import ast
import glob
import os
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "CubeVetoThresholds",
    "CubeMetrics",
    "VetoDecision",
    "decide",
    "compute_metrics",
    "compute_metrics_from_grids",
    "event_is_injection",
]


# ---------------------------------------------------------------------------
# Thresholds (report §4 — injection-safe scheme)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CubeVetoThresholds:
    r1_img_off_px: float = 40.0        # >0.5° at 45" pixels
    r2_t_shift_samples: int = 30       # >31 ms at the search cadence
    r3_dm_shift_trials: int = 8
    r4_time_prom_sigma: float = 4.0
    r4_image_prom_sigma: float = 4.0
    r5_dm_shift_trials: int = 4        # edge rail AND > this many trials off
    r10_tz_trig_sigma: float = 10.0    # trigger feature "weak" below this


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CubeMetrics:
    """Cube-derived discriminators for one event.

    Mirrors the columns produced by the analysis ``cube_analyze``/
    ``cube_analyze2`` scripts. ``ok`` is False when metrics could not be
    computed (missing trigger cube / unreadable npz) — :func:`decide`
    then KEEPs (fail-open) and records the reason.
    """

    ok: bool
    reason: str = ""

    # trigger (from the highest-SNR C1 row)
    snr_c1: float = 0.0
    dm_c1: float = 0.0
    fdm_trig: int = 0
    l_pix_trig: int = 0
    m_pix_trig: int = 0

    # trigger-cube discriminators
    t_img: int = 0                     # argmax of trigger-DM light curve
    t_apex: int = 0                    # time of trigger-cube (t,fdm) apex
    fdm_apex: int = 0                  # fine-DM of trigger-cube apex
    t_shift: int = 0                   # |t_apex - t_img|
    dm_shift_trials: int = 0           # |fdm_apex - fdm_trig|
    tz_trig: float = 0.0               # robust-z of trigger-DM light curve
    tz_apex: float = 0.0               # robust-z of apex-DM light curve
    dmz: float = 0.0                   # robust-z of DM profile
    dm_edge: int = 0                   # apex rails to first/last fine-DM bin
    imgz_apex: float = 0.0             # robust-z of apex image plane
    img_off_apex: float = 0.0         # apex image brightest pixel vs C1 (l,m)
    n_fdm: int = 0

    # global (all-cube) apex consistency
    g_apex_cube_same: int = 1          # 1 if global apex is in the trigger cube
    g_apex_val: float = 0.0
    n_cubes: int = 0


@dataclass(frozen=True)
class VetoDecision:
    keep: bool
    rules_fired: Tuple[str, ...] = field(default_factory=tuple)
    is_injection: bool = False
    notes: str = ""

    @property
    def action(self) -> str:
        return "KEEP" if self.keep else "REJECT"


# ---------------------------------------------------------------------------
# Pure rule logic
# ---------------------------------------------------------------------------


def decide(
    metrics: CubeMetrics,
    *,
    is_injection: bool,
    thresholds: Optional[CubeVetoThresholds] = None,
) -> VetoDecision:
    """Decide KEEP vs REJECT (pure).

    REJECT requires (a) metrics computed successfully, (b) the event is
    NOT an injection, and (c) at least one tier-1 (R1–R5) or R10 rule
    fires. Everything else KEEPs — the conservative default.
    """
    th = thresholds or CubeVetoThresholds()

    if is_injection:
        return VetoDecision(
            keep=True, rules_fired=(), is_injection=True,
            notes="injection — exempt from veto",
        )
    if not metrics.ok:
        return VetoDecision(
            keep=True, rules_fired=(), is_injection=False,
            notes=f"metrics unavailable ({metrics.reason}) — fail-open KEEP",
        )

    fired: List[str] = []

    # R1 image incoherence
    if metrics.img_off_apex > th.r1_img_off_px:
        fired.append("R1_image_offset")
    # R2 time incoherence
    if metrics.t_shift > th.r2_t_shift_samples:
        fired.append("R2_time_shift")
    # R3 DM incoherence
    if metrics.dm_shift_trials > th.r3_dm_shift_trials:
        fired.append("R3_dm_shift")
    # R4 no peak (both prominences weak)
    if (
        metrics.tz_apex < th.r4_time_prom_sigma
        and metrics.imgz_apex < th.r4_image_prom_sigma
    ):
        fired.append("R4_no_peak")
    # R5 DM-edge rail
    if (
        metrics.dm_edge == 1
        and metrics.dm_shift_trials > th.r5_dm_shift_trials
    ):
        fired.append("R5_dm_edge_rail")
    # R10 cube doesn't confirm the trigger
    if (
        metrics.g_apex_cube_same == 0
        and metrics.tz_trig < th.r10_tz_trig_sigma
    ):
        fired.append("R10_cube_unconfirmed")

    keep = len(fired) == 0
    return VetoDecision(
        keep=keep, rules_fired=tuple(fired), is_injection=False,
        notes="" if keep else "tier-1/R10 high-confidence false positive",
    )


# ---------------------------------------------------------------------------
# Metric computation — file I/O
# ---------------------------------------------------------------------------


def _robust_z(x: np.ndarray) -> float:
    """Prominence = (max - median) / (1.4826*MAD), MAD-robust."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sig = 1.4826 * mad if mad > 0 else (float(x.std()) if x.std() > 0 else 1.0)
    return (float(np.max(x)) - med) / sig


def _robust_rownorm(wf: np.ndarray) -> np.ndarray:
    """Per-DM-row robust z-score normalisation (matches the plotter)."""
    wf = np.asarray(wf, dtype=np.float64)
    med = np.median(wf, axis=0)
    mad = np.median(np.abs(wf - med), axis=0)
    sig = 1.4826 * mad
    pos = sig[sig > 0]
    fill = float(np.median(pos)) if pos.size else 1.0
    sig = np.where(sig == 0, fill, sig)
    return (wf - med) / sig


def _mmap_cube(npz_path: str) -> np.ndarray:
    """Memory-map the (uncompressed) ``cube`` array out of a .npz so we can
    pull a single 256×256 image plane without loading the whole 855 MiB."""
    with zipfile.ZipFile(npz_path) as zf:
        info = zf.getinfo("cube.npy")
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError("cube.npy is compressed; cannot mmap")
    with open(npz_path, "rb") as fh:
        fh.seek(info.header_offset)
        local = fh.read(30)
        name_len = struct.unpack("<H", local[26:28])[0]
        extra_len = struct.unpack("<H", local[28:30])[0]
        data_start = info.header_offset + 30 + name_len + extra_len
        fh.seek(data_start)
        magic = fh.read(10)
        major = magic[6]
        if major == 1:
            hl = struct.unpack("<H", magic[8:10])[0]
            npy_prefix = 10 + hl
        else:
            hl_ext = fh.read(2)
            hl = struct.unpack("<I", magic[8:10] + hl_ext)[0]
            npy_prefix = 12 + hl
        header_str = fh.read(hl).decode("latin1")
    hd = ast.literal_eval(header_str)
    dtype = np.dtype(hd["descr"])
    shape = tuple(hd["shape"])
    order = "F" if bool(hd.get("fortran_order", False)) else "C"
    return np.memmap(
        npz_path, dtype=dtype, shape=shape, order=order, mode="r",
        offset=data_start + npy_prefix,
    )


def compute_metrics_from_grids(
    *,
    grids: Dict[Tuple[int, int], np.ndarray],
    sid_trig: int,
    g_trig: int,
    fdm_trig: int,
    l_pix_trig: int,
    m_pix_trig: int,
    snr_c1: float = 0.0,
    dm_c1: float = 0.0,
    apex_image_fn: Optional[Callable[[int, int], np.ndarray]] = None,
) -> CubeMetrics:
    """Build :class:`CubeMetrics` from per-cube ``peak_grid`` arrays.

    ``grids`` maps ``(search_node_id, gpu_half) -> peak_grid`` with shape
    ``(T_det, n_fdm)``. ``apex_image_fn(t, fdm) -> image_plane`` returns
    the trigger-cube image plane at the apex (for the R1/R4 image
    metrics); when omitted the image metrics are left at 0 (R1 won't fire,
    R4 falls back to the time prominence alone — both fail-safe toward
    KEEP). Pure: no file access.
    """
    if (sid_trig, g_trig) not in grids:
        return CubeMetrics(ok=False, reason="trigger cube absent")
    wf = np.asarray(grids[(sid_trig, g_trig)], dtype=np.float64)
    if wf.ndim != 2 or wf.size == 0:
        return CubeMetrics(ok=False, reason="trigger peak_grid empty")
    T, F = wf.shape
    fdm_t = min(max(int(fdm_trig), 0), F - 1)

    # trigger-cube discriminators
    t_ap, fdm_ap = np.unravel_index(int(np.nanargmax(wf)), wf.shape)
    dm_prof = np.nanmax(wf, axis=0)
    lc_trig = wf[:, fdm_t]
    lc_apex = wf[:, int(fdm_ap)]
    tz_trig = _robust_z(lc_trig)
    tz_apex = _robust_z(lc_apex)
    dmz = _robust_z(dm_prof)
    dm_edge = int(int(fdm_ap) == 0 or int(fdm_ap) == F - 1)
    t_img = int(np.nanargmax(lc_trig))
    t_shift = int(abs(int(t_ap) - t_img))
    dm_shift = int(abs(int(fdm_ap) - fdm_t))

    imgz_apex = 0.0
    img_off_apex = 0.0
    if apex_image_fn is not None:
        try:
            img_ap = np.asarray(
                apex_image_fn(int(t_ap), int(fdm_ap)), dtype=np.float64,
            )
            if img_ap.ndim == 2 and img_ap.size:
                imgz_apex = _robust_z(img_ap)
                la, ma = np.unravel_index(
                    int(np.nanargmax(img_ap)), img_ap.shape,
                )
                img_off_apex = float(
                    np.hypot(int(la) - int(l_pix_trig),
                             int(ma) - int(m_pix_trig))
                )
        except Exception:  # noqa: BLE001 — image plane optional, fail-safe
            imgz_apex = 0.0
            img_off_apex = 0.0

    # global (all-cube) apex on robust row-normalised stacks (plotter parity)
    best_val = -np.inf
    best_cube: Optional[Tuple[int, int]] = None
    for key, g in grids.items():
        g = np.asarray(g, dtype=np.float64)
        if g.ndim != 2 or g.size == 0:
            continue
        z = _robust_rownorm(g)
        v = float(np.nanmax(z))
        if v > best_val:
            best_val = v
            best_cube = key
    g_same = int(best_cube == (sid_trig, g_trig))

    return CubeMetrics(
        ok=True,
        snr_c1=float(snr_c1),
        dm_c1=float(dm_c1),
        fdm_trig=fdm_t,
        l_pix_trig=int(l_pix_trig),
        m_pix_trig=int(m_pix_trig),
        t_img=t_img,
        t_apex=int(t_ap),
        fdm_apex=int(fdm_ap),
        t_shift=t_shift,
        dm_shift_trials=dm_shift,
        tz_trig=float(tz_trig),
        tz_apex=float(tz_apex),
        dmz=float(dmz),
        dm_edge=dm_edge,
        imgz_apex=float(imgz_apex),
        img_off_apex=float(img_off_apex),
        n_fdm=int(F),
        g_apex_cube_same=g_same,
        g_apex_val=float(best_val if np.isfinite(best_val) else 0.0),
        n_cubes=len(grids),
    )


def _load_peak_grid(npz_path: str) -> Tuple[np.ndarray, int, int]:
    with np.load(npz_path) as z:
        pg = np.asarray(z["peak_grid"], dtype=np.float32)
        tdet = int(z["t_det"])
        nf = int(z["n_fdm_in_cube"])
    if pg.shape == (nf, tdet):
        pg = pg.T
    return pg, tdet, nf


def _read_trigger_row(event_dir: Path, event_name: str) -> Optional[dict]:
    """Read the highest-SNR row of the C1-window CSV (pandas-free)."""
    import csv

    csv_path = event_dir / "Level2" / f"C1_window_{event_name}.csv"
    if not csv_path.is_file():
        return None
    best: Optional[dict] = None
    best_snr = -np.inf
    with csv_path.open("r", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                snr = float(row.get("snr", "nan"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(snr) and snr > best_snr:
                best_snr = snr
                best = row
    return best


def compute_metrics(event_dir: Path, event_name: str) -> CubeMetrics:
    """Build :class:`CubeMetrics` from a candidate directory on disk."""
    event_dir = Path(event_dir)
    trig = _read_trigger_row(event_dir, event_name)
    if trig is None:
        return CubeMetrics(ok=False, reason="no C1-window CSV / rows")
    try:
        sid_t = int(float(trig["search_node_id"]))
        g_t = int(float(trig["gpu_half"]))
        fdm_t = int(float(trig["fine_dm_idx"]))
        l_t = int(float(trig.get("l_pix", 0)))
        m_t = int(float(trig.get("m_pix", 0)))
        snr_c1 = float(trig.get("snr", 0.0))
        dm_c1 = float(trig.get("dm_pc_cc", 0.0))
    except (KeyError, TypeError, ValueError) as exc:
        return CubeMetrics(ok=False, reason=f"bad C1 row: {exc}")

    cubes = sorted(glob.glob(str(event_dir / "cubes" / "cube_s*_g*_*.npz")))
    if not cubes:
        return CubeMetrics(ok=False, reason="no cube npz files")

    grids: Dict[Tuple[int, int], np.ndarray] = {}
    trig_cube_path: Optional[str] = None
    for cf in cubes:
        base = os.path.basename(cf)
        parts = base.split("_")
        try:
            s = int(parts[1][1:])
            g = int(parts[2][1:])
        except (IndexError, ValueError):
            continue
        try:
            pg, _tdet, _nf = _load_peak_grid(cf)
        except Exception:  # noqa: BLE001
            continue
        grids[(s, g)] = pg
        if (s, g) == (sid_t, g_t):
            trig_cube_path = cf

    if not grids:
        return CubeMetrics(ok=False, reason="no readable peak_grids")

    apex_image_fn: Optional[Callable[[int, int], np.ndarray]] = None
    if trig_cube_path is not None:
        def apex_image_fn(t: int, fdm: int, _p=trig_cube_path) -> np.ndarray:
            cube = _mmap_cube(_p)
            try:
                return np.asarray(cube[t, fdm], dtype=np.float32)
            finally:
                del cube

    return compute_metrics_from_grids(
        grids=grids,
        sid_trig=sid_t,
        g_trig=g_t,
        fdm_trig=fdm_t,
        l_pix_trig=l_t,
        m_pix_trig=m_t,
        snr_c1=snr_c1,
        dm_c1=dm_c1,
        apex_image_fn=apex_image_fn,
    )


def event_is_injection(
    event_dir: Path,
    event_name: str,
    *,
    fired_log_path: Optional[Path] = None,
) -> bool:
    """Durable injection test for an archived event.

    Checks, in order:
      1. the Level3 JSON ``injection.is_injection`` marker C2 writes at
         fire time (authoritative, registry-independent), then
      2. any non-empty ``inj_id`` in the C1-window CSV (legacy fallback),
         then
      3. (when ``fired_log_path`` is given) DM+sky+time coincidence of
         any C1-window row with the durable fired-injection log
         (:mod:`dsart.coinc.inject_log`) — the registry-independent
         backstop for injections the live C2 matcher missed at fire time
         (the ``260612homx`` class of miss).

    Fails CLOSED toward "is injection" only on a positive signal; a
    missing marker means "treat as a real event" (the veto then applies).
    The coincidence fallback (3) only ever EXEMPTS (KEEP), so a loose
    match there is the safe direction.
    """
    import csv
    import json

    event_dir = Path(event_dir)
    l3 = event_dir / "Level3" / f"{event_name}.json"
    if l3.is_file():
        try:
            with l3.open("r") as fh:
                doc = json.load(fh)
            inj = doc.get("injection")
            if isinstance(inj, dict) and bool(inj.get("is_injection")):
                return True
        except (OSError, ValueError):
            pass
    csv_path = event_dir / "Level2" / f"C1_window_{event_name}.csv"
    rows: list = []
    if csv_path.is_file():
        try:
            with csv_path.open("r", newline="") as fh:
                for row in csv.DictReader(fh):
                    if str(row.get("inj_id", "")).strip():
                        return True
                    rows.append(row)
        except OSError:
            pass
    if fired_log_path is not None and rows:
        try:
            from .inject_log import (
                event_coincident_inj_id,
                load_fired_injections,
            )

            fired = load_fired_injections(fired_log_path)
            if fired:
                for row in rows:
                    try:
                        mjd = float(row.get("mjd", 0.0) or 0.0)
                        dm = float(row.get("dm_pc_cc", 0.0) or 0.0)
                        l_rad = float(row.get("l_rad", 0.0) or 0.0)
                        m_rad = float(row.get("m_rad", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        continue
                    hit = event_coincident_inj_id(
                        fired, mjd=mjd, dm_pc_cc=dm,
                        l_rad=l_rad, m_rad=m_rad,
                    )
                    if hit is not None:
                        return True
        except Exception:  # noqa: BLE001 - backstop must never raise
            pass
    return False
