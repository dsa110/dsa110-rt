"""Raw-stream window archive (x2 SPEC §9, docs/x2/downstream_inventory.md §3.8).

Layout per (event, half)::

    raw_s<sid>_g<g>/
        raw_d<DDD>.npz   one per owned coarse DM: ring-v4 bytes verbatim
        meta.json        request echo, op-point, plan_id, files + sizes + digests
        RAW_OK.json      written LAST: the window is complete on disk
        upload.log       uploader marker (per product subdir)

``raw_d<DDD>.npz`` keys (the ones WS-S's ``CutoutProducer.raw_snapshot``
writes, plus optional provenance):

=================  ======================  ==========================================
key                dtype / shape           meaning
=================  ======================  ==========================================
codes              u8 [n_stream, n, B]     ring rows verbatim (frame-v2 payload per sample)
valid              u8/bool [n_stream, n]   fan-in validity (0 = zero-filled)
row0               i64 ()                  tier sample index of row 0 (all streams)
t_native           i32 ()                  native samples per tier sample (16 | 32)
d                  i32 ()                  global coarse DM index
shifts             i32 [n_fdm, n_stream]   stage-2 shifts of d's fines (search reads t - shift)
fine_global        i32 [n_fdm]             plan-global fine indices of d
(optional) step_main / step_heavy f32 [n_stream, n], bits_main, bits_heavy,
n_values, n_heavy, value_kind, n_sub, n_grid, pattern_id (u64)
=================  ======================  ==========================================

The value maps are NOT archived per event: they are a deterministic
function of ``pattern_id`` (SPEC §3), uploaded once per pattern.

This module also defines the small ring interface the producer uses
(:class:`RingWindowSource`; WS-T's ring v4 ``wait_window``) and a
:class:`SyntheticRing` implementing it for CPU tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Tuple

import numpy as np

__all__ = [
    "RAW_SCHEMA",
    "RAW_SCHEMA_VERSION",
    "REQUIRED_KEYS",
    "RawSlab",
    "RawArchiveError",
    "RingWindowSource",
    "SyntheticRing",
    "raw_dir_name",
    "slab_filename",
    "write_raw_slab",
    "write_raw_meta",
    "mark_raw_ok",
    "is_raw_complete",
    "read_raw_dir",
    "stream_window",
]

RAW_SCHEMA = "dsart.rawwin"
RAW_SCHEMA_VERSION = 1
REQUIRED_KEYS: Tuple[str, ...] = (
    "codes", "valid", "row0", "t_native", "d", "shifts", "fine_global")
_OPTIONAL_KEYS: Tuple[str, ...] = (
    "step_main", "step_heavy", "bits_main", "bits_heavy", "n_values",
    "n_heavy", "value_kind", "n_sub", "n_grid", "pattern_id")


class RawArchiveError(ValueError):
    """A raw slab or raw dir violates the layout above."""


def raw_dir_name(sid: int, g: int) -> str:
    return f"raw_s{int(sid)}_g{int(g)}"


def slab_filename(d: int) -> str:
    return f"raw_d{int(d):03d}.npz"


# ---------------------------------------------------------------------------
# The ring interface (WS-T ring v4 / WS-S engine.ring)
# ---------------------------------------------------------------------------


class RingWindowSource(Protocol):
    """What a raw snapshot needs from the RX ring v4 (SPEC §6).

    ``wait_window(d, s0, n, timeout_s)`` blocks until tier samples
    ``[s0, s0+n)`` of every stream of owned coarse DM ``d`` are complete
    (or the deadline passes, then missing rows are zero-filled and flagged)
    and returns ``(codes u8[n_stream, n, B], valid u8[n_stream, n])``. It
    raises ``TimeoutError`` when ``s0`` has already rotated out.
    RE-WIRE: bind to WS-T's ring v4 snapshot API once merged.
    """

    def wait_window(self, d: int, s0: int, n: int, *,
                    timeout_s: float) -> Tuple[np.ndarray, np.ndarray]: ...


@dataclass
class SyntheticRing:
    """In-memory :class:`RingWindowSource` for tests: per coarse DM a
    ``codes[n_stream, n_total, B]`` array starting at tier sample ``base``;
    rows ``< oldest`` count as rotated out, rows ``>= frontier`` as not yet
    complete (zero-filled, valid 0)."""

    codes: Dict[int, np.ndarray]
    base: int = 0
    oldest: int = 0
    frontier: Optional[int] = None

    def wait_window(self, d: int, s0: int, n: int, *,
                    timeout_s: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        arr = self.codes[int(d)]
        if s0 < self.oldest:
            raise TimeoutError(f"rows from {s0} rotated out (oldest {self.oldest})")
        n_s, n_tot, b = arr.shape
        out = np.zeros((n_s, n, b), np.uint8)
        valid = np.zeros((n_s, n), np.uint8)
        hi = self.base + n_tot if self.frontier is None else min(
            self.base + n_tot, self.frontier)
        lo_r, hi_r = max(s0, self.base), min(s0 + n, hi)
        if hi_r > lo_r:
            out[:, lo_r - s0:hi_r - s0] = arr[:, lo_r - self.base:hi_r - self.base]
            valid[:, lo_r - s0:hi_r - s0] = 1
        return out, valid


def stream_window(t_peak_ext: int, raw_pre_ext: int, raw_post_ext: int, *,
                  ext_factor: int, max_pos: int, max_neg: int) -> Tuple[int, int]:
    """``(s0, n)`` tier samples to snapshot for one coarse DM.

    The window ``[t_peak - pre, t_peak + post)`` (T_ext) in tier samples
    (``ext_factor`` T_ext per tier sample: 1 tier A, 2 tier B), widened by
    the stage-2 extents of the DM's fines (the search reads
    ``stream[t - shift]``, shifts in ``[-max_neg, max_pos]``). Same rule as
    WS-S's producer."""
    r = int(ext_factor)
    lo_e = int(t_peak_ext) - int(raw_pre_ext)
    hi_e = int(t_peak_ext) + int(raw_post_ext)
    t_lo, t_hi = lo_e // r, -(-hi_e // r)
    s_lo, s_hi = t_lo - int(max_pos), t_hi + int(max_neg)
    return s_lo, s_hi - s_lo


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def _digest(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_raw_slab(
    out_dir: Path, *, codes: np.ndarray, valid: np.ndarray, row0: int,
    t_native: int, d: int, shifts: np.ndarray, fine_global: np.ndarray,
    **extra: Any,
) -> Path:
    """Atomically write one ``raw_d<DDD>.npz`` (uncompressed, fsynced)."""
    codes = np.asarray(codes)
    valid = np.asarray(valid)
    if codes.dtype != np.uint8 or codes.ndim != 3:
        raise RawArchiveError(f"codes must be u8 [n_stream, n, B], got {codes.dtype} {codes.shape}")
    if valid.shape != codes.shape[:2]:
        raise RawArchiveError(f"valid {valid.shape} != codes[:2] {codes.shape[:2]}")
    unknown = set(extra) - set(_OPTIONAL_KEYS)
    if unknown:
        raise RawArchiveError(f"unknown raw keys {sorted(unknown)}")
    arrays: Dict[str, Any] = dict(
        codes=codes, valid=valid.astype(np.uint8, copy=False),
        row0=np.int64(row0), t_native=np.int32(t_native), d=np.int32(d),
        shifts=np.asarray(shifts, np.int32),
        fine_global=np.asarray(fine_global, np.int32))
    for k, v in extra.items():
        arrays[k] = np.asarray(v, dtype=np.uint64) if k == "pattern_id" else np.asarray(v)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / slab_filename(d)
    tmp = out_dir / f".{path.name}.tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def write_raw_meta(out_dir: Path, request: Any, *, identity: Mapping[str, Any],
                   too_late: Iterable[int] = ()) -> Path:
    """``meta.json``: request echo, identity, every slab with size + digest."""
    out_dir = Path(out_dir)
    files = {}
    for p in sorted(out_dir.glob("raw_d*.npz")):
        files[p.name] = {"bytes": p.stat().st_size, "blake2b_128": _digest(p)}
    req = {}
    if request is not None:
        for f in ("event_name", "request_id", "t_peak_ext", "t0_ext", "n_t",
                  "raw_pre_ext", "raw_post_ext", "flags", "plan_id"):
            if hasattr(request, f):
                req[f] = int(getattr(request, f)) if f != "event_name" else str(getattr(request, f))
    meta = {"schema": RAW_SCHEMA, "schema_version": RAW_SCHEMA_VERSION,
            "request": req, "identity": dict(identity), "files": files,
            "too_late_coarse": sorted(int(d) for d in too_late),
            "bytes_total": int(sum(v["bytes"] for v in files.values())),
            "t_written_unix": time.time()}
    path = out_dir / "meta.json"
    tmp = out_dir / ".meta.json.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def mark_raw_ok(out_dir: Path) -> Path:
    """``RAW_OK.json`` -- written last; readers trust only marked dirs."""
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "meta.json").read_text())
    path = out_dir / "RAW_OK.json"
    tmp = out_dir / ".RAW_OK.json.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"files": sorted(meta["files"]), "bytes_total": meta["bytes_total"],
                   "t_unix": time.time()}, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def is_raw_complete(raw_dir: Path) -> bool:
    return (Path(raw_dir) / "RAW_OK.json").is_file()


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


@dataclass
class RawSlab:
    d: int
    codes: np.ndarray
    valid: np.ndarray
    row0: int
    t_native: int
    shifts: np.ndarray
    fine_global: np.ndarray
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return int(self.codes.shape[1])

    def rows(self, stream: int, s0: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Codes/validity of ``stream`` for tier samples [s0, s0+n) (zero /
        invalid outside the archived range)."""
        out = np.zeros((n, self.codes.shape[2]), np.uint8)
        v = np.zeros(n, np.uint8)
        lo, hi = max(s0, self.row0), min(s0 + n, self.row0 + self.n_rows)
        if hi > lo:
            out[lo - s0:hi - s0] = self.codes[stream, lo - self.row0:hi - self.row0]
            v[lo - s0:hi - s0] = self.valid[stream, lo - self.row0:hi - self.row0]
        return out, v


def _read_slab(path: Path) -> RawSlab:
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in REQUIRED_KEYS if k not in z.files]
        if missing:
            raise RawArchiveError(f"{path.name}: missing {missing}")
        arrs = {k: np.asarray(z[k]) for k in z.files}
    codes = arrs.pop("codes")
    valid = arrs.pop("valid")
    if codes.ndim != 3 or valid.shape != codes.shape[:2]:
        raise RawArchiveError(f"{path.name}: codes {codes.shape} / valid {valid.shape}")
    shifts = arrs.pop("shifts")
    fine_global = arrs.pop("fine_global")
    if shifts.ndim != 2 or shifts.shape != (fine_global.shape[0], codes.shape[0]):
        raise RawArchiveError(
            f"{path.name}: shifts {shifts.shape} vs fines {fine_global.shape} x streams {codes.shape[0]}")
    return RawSlab(
        d=int(arrs.pop("d")), codes=codes, valid=valid, row0=int(arrs.pop("row0")),
        t_native=int(arrs.pop("t_native")), shifts=shifts, fine_global=fine_global,
        extra={k: (v.item() if v.ndim == 0 else v) for k, v in arrs.items()})


def read_raw_dir(raw_dir: Path, *, require_ok: bool = True) -> Dict[int, RawSlab]:
    """All slabs of one half's raw dir, keyed by coarse DM."""
    raw_dir = Path(raw_dir)
    if require_ok and not is_raw_complete(raw_dir):
        raise RawArchiveError(f"{raw_dir}: no RAW_OK.json (incomplete window)")
    out: Dict[int, RawSlab] = {}
    for p in sorted(raw_dir.glob("raw_d*.npz")):
        s = _read_slab(p)
        out[s.d] = s
    return out
