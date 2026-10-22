"""``dsart.cutout`` v1: the per-half cutout file (docs/x2/SPEC.md §9).

One uncompressed NPZ per (event, search half): ``cutout_s<sid>_g<g>.npz``,
followed by its sidecar manifest ``cutout_s<sid>_g<g>.json`` (the C2
completion signal -- written last, atomically). Layout and semantics are
frozen in SPEC §9 / ``docs/x2/downstream_inventory.md`` §3.6; this module is
the single implementation of them:

* :data:`FIELDS` -- every key with its dtype and symbolic shape;
* :func:`fine_axis_for_half` -- the F / D axes of a half from the tiered plan
  (shared with the producer so both sides agree on order and t_step);
* :func:`reduce_images` / :func:`apex_per_block` -- numpy REFERENCE of the
  plane reductions (the GPU producer must match it; tests compare);
* :func:`validate`, :func:`write_cutout`, :func:`read_cutout`.

Units: rows are absolute external samples ``t_ext = native // native_per_ext``
(SPEC §8); row r of a cutout is ``t0_ext + r``. Tier-B fines repeat each
value on rows 2k, 2k+1 (``t0_ext`` even); their independent rows are
``r % t_step_ext[f] == 0``. Image values are Layer-1-normalised exactly
like today's retained ``cube`` (``sigma_applied`` records the divisor).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "SCHEMA",
    "SCHEMA_VERSION",
    "FIELDS",
    "REQUIRED",
    "CutoutError",
    "Cutout",
    "cutout_filename",
    "manifest_filename",
    "parse_cutout_filename",
    "fine_axis_for_half",
    "reduce_images",
    "apex_per_block",
    "independent_rows",
    "validate",
    "write_cutout",
    "read_cutout",
    "read_manifest",
    "assemble_cutout",
    "stamp_fines",
    "manifest_dict",
]

SCHEMA = "dsart.cutout"
SCHEMA_VERSION = 1

#: key -> (dtype, symbolic shape). Symbols: F fines of the half, D owned
#: coarse DMs, T = n_t rows, N = n_grid, Fs stamped fines, Ts stamp rows,
#: P = 2S+1 stamp side, K kernels. ``()`` = 0-d scalar. Shapes marked with a
#: leading "0|" may be zero-size (absent content, e.g. MANUAL requests).
FIELDS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    # -- A. scalars ----------------------------------------------------------
    "schema": ("<U32", ()),
    "schema_version": ("int32", ()),
    "event_name": ("<U16", ()),
    "request_id": ("int64", ()),
    "request_flags": ("int32", ()),
    "opoint": ("<U32", ()),
    "plan_id": ("uint64", ()),
    "pattern_id": ("uint64", ()),
    "code_version": ("<U64", ()),
    "search_node_id": ("int32", ()),
    "gpu_half": ("int32", ()),
    "half_index": ("int32", ()),
    "t_ext_us": ("float64", ()),
    "native_per_ext": ("int32", ()),
    "sample_period_specnum": ("int32", ()),
    "t0_ext": ("int64", ()),
    "n_t": ("int32", ()),
    "mjd_t0": ("float64", ()),
    "mjd_clock": ("<U32", ()),
    "t_peak_ext": ("int64", ()),
    "width_ext_req": ("int32", ()),
    "dm_req": ("float32", ()),
    "fine_global_req": ("int32", ()),
    "coarse_global_req": ("int32", ()),
    "l_pix_req": ("int32", ()),
    "m_pix_req": ("int32", ()),
    "n_grid": ("int32", ()),
    "cell_l_rad": ("float64", ()),
    "cell_m_rad": ("float64", ()),
    "l0_rad": ("float64", ()),
    "m0_rad": ("float64", ()),
    "status": ("<U16", ()),
    "status_json": ("<U8192", ()),
    # -- B. fine axis --------------------------------------------------------
    "fine_dm_pc_cc": ("float32", ("F",)),
    "fine_global_idx": ("int32", ("F",)),
    "coarse_global_idx": ("int16", ("F",)),
    "fine_local_idx": ("int16", ("F",)),
    "tier": ("uint8", ("F",)),
    "t_step_ext": ("uint8", ("F",)),
    "input_valid_frac": ("float32", ("F",)),
    # -- C. coarse axis ------------------------------------------------------
    "coarse_owned": ("int16", ("D",)),
    "fine_offset": ("int32", ("D+1",)),
    "row_valid": ("uint8", ("T", "D")),
    "apex_t": ("int32", ("D",)),
    "apex_f": ("int32", ("D",)),
    "apex_image": ("float16", ("D", "N", "N")),
    # -- D. planes -----------------------------------------------------------
    "peak_grid": ("float32", ("T", "F")),
    "peak_pix": ("int32", ("T", "F")),
    "pix_lc": ("float32", ("0|T", "F")),
    "sigma_applied": ("float32", ("T", "F")),
    # -- E. images near the candidate ----------------------------------------
    "has_peak_image": ("uint8", ()),
    "peak_image": ("float16", ("0|N", "0|N")),
    "peak_image_w": ("float16", ("0|N", "0|N")),
    "stamp_fine": ("int32", ("Fs",)),
    "stamp_t0_ext": ("int64", ()),
    "stamp": ("float16", ("Ts", "Fs", "P", "P")),
    # -- F. optional detector context -----------------------------------------
    "kernel_ids": ("<U32", ("K",)),
    "sigma_k": ("float32", ("D", "K")),
}

#: Keys every v1 file carries (F. is optional).
REQUIRED: Tuple[str, ...] = tuple(
    k for k in FIELDS if k not in ("kernel_ids", "sigma_k"))


class CutoutError(ValueError):
    """A cutout dict / file violates the v1 schema."""


# ---------------------------------------------------------------------------
# File names
# ---------------------------------------------------------------------------


def cutout_filename(search_node_id: int, gpu_half: int) -> str:
    return f"cutout_s{int(search_node_id)}_g{int(gpu_half)}.npz"


def manifest_filename(search_node_id: int, gpu_half: int) -> str:
    return f"cutout_s{int(search_node_id)}_g{int(gpu_half)}.json"


def parse_cutout_filename(name: str) -> Optional[Tuple[int, int]]:
    """``(sid, g)`` from ``cutout_s<sid>_g<g>.{npz,json}``, else None."""
    base = os.path.basename(name)
    stem, dot, ext = base.rpartition(".")
    if not dot or ext not in ("npz", "json") or not stem.startswith("cutout_s"):
        return None
    parts = stem.split("_")
    if len(parts) != 3 or not parts[1].startswith("s") or not parts[2].startswith("g"):
        return None
    try:
        return int(parts[1][1:]), int(parts[2][1:])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Axes from the tiered plan (shared with the producer)
# ---------------------------------------------------------------------------


def fine_axis_for_half(plan: Any, half: int, *, t_ext_native: int) -> Dict[str, np.ndarray]:
    """The B. fine axis + the D-axis CSR of half ``half`` from a tiered plan.

    ``plan`` is a :class:`dsart.x2.plan.TieredDmPlan` (duck-typed: needs
    ``fines_of_half``, ``dms_of_half``, ``fine_dm``, ``fine_to_coarse``,
    ``fine_local_idx``, ``fine_tier``, ``coarse_t_native``). Order is the
    plan's global (d, fine) order, so ``fine_global_idx`` is ascending.
    """
    fines = np.asarray(plan.fines_of_half(int(half)), dtype=np.int64)
    dms = np.asarray(plan.dms_of_half(int(half)), dtype=np.int64)
    f2c = np.asarray(plan.fine_to_coarse)[fines]
    t_nat = np.asarray(plan.coarse_t_native)[f2c]
    if np.any(t_nat % int(t_ext_native)):
        raise CutoutError(
            f"coarse sample {sorted(set(t_nat.tolist()))} not a multiple of "
            f"the external sample {t_ext_native}")
    counts = np.array([int(np.count_nonzero(f2c == d)) for d in dms], dtype=np.int64)
    offset = np.zeros(len(dms) + 1, dtype=np.int32)
    offset[1:] = np.cumsum(counts)
    if offset[-1] != len(fines):
        raise CutoutError("half's fines are not all in its owned coarse DMs")
    return {
        "fine_dm_pc_cc": np.asarray(plan.fine_dm)[fines].astype(np.float32),
        "fine_global_idx": fines.astype(np.int32),
        "coarse_global_idx": f2c.astype(np.int16),
        "fine_local_idx": np.asarray(plan.fine_local_idx)[fines].astype(np.int16),
        "tier": np.asarray(plan.fine_tier)[fines].astype(np.uint8),
        "t_step_ext": (t_nat // int(t_ext_native)).astype(np.uint8),
        "coarse_owned": dms.astype(np.int16),
        "fine_offset": offset,
    }


def independent_rows(n_t: int, t_step: int) -> np.ndarray:
    """Row indices carrying independent samples for a fine with ``t_step``
    (``t0_ext`` is even, so tier-B samples start on even rows)."""
    return np.arange(0, int(n_t), max(1, int(t_step)), dtype=np.int64)


# ---------------------------------------------------------------------------
# Reference reductions (numpy twin of the producer's GPU reductions)
# ---------------------------------------------------------------------------


def reduce_images(
    images: np.ndarray, l_pix: int = -1, m_pix: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(peak_grid, peak_pix, pix_lc)`` from width-1 images ``[T, F, N, N]``.

    ``peak_grid`` = max over all N^2 pixels -- exactly today's
    ``cube.max(axis=(2, 3))`` (``dump/cube_dump.py``); ``peak_pix`` = flat
    index ``l * N + m`` of the first maximum; ``pix_lc`` = value at
    ``(l_pix, m_pix)`` (shape ``(0, F)`` when no pixel was requested).
    """
    img = np.asarray(images)
    if img.ndim != 4 or img.shape[2] != img.shape[3]:
        raise CutoutError(f"images must be [T, F, N, N], got {img.shape}")
    t, f, n, _ = img.shape
    flat = img.reshape(t, f, n * n)
    peak_pix = np.argmax(flat, axis=2).astype(np.int32)
    peak_grid = np.take_along_axis(flat, peak_pix[..., None], axis=2)[..., 0]
    peak_grid = peak_grid.astype(np.float32)
    if l_pix >= 0 and m_pix >= 0:
        pix_lc = img[:, :, int(l_pix), int(m_pix)].astype(np.float32)
    else:
        pix_lc = np.zeros((0, f), dtype=np.float32)
    return peak_grid, peak_pix, pix_lc


def apex_per_block(peak_grid: np.ndarray, fine_offset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per coarse block k: ``np.unravel_index(np.nanargmax(block))`` over
    ALL rows -- C3's definition (``coinc/cube_veto.py`` apex). ``apex_f`` is
    local to the block. An all-NaN block gives (-1, -1)."""
    pg = np.asarray(peak_grid, dtype=np.float64)
    off = np.asarray(fine_offset, dtype=np.int64)
    d = len(off) - 1
    at = np.full(d, -1, dtype=np.int32)
    af = np.full(d, -1, dtype=np.int32)
    for k in range(d):
        blk = pg[:, off[k]:off[k + 1]]
        if blk.size == 0 or not np.any(np.isfinite(blk)):
            continue
        t_ap, f_ap = np.unravel_index(int(np.nanargmax(blk)), blk.shape)
        at[k], af[k] = int(t_ap), int(f_ap)
    return at, af


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _as_scalar(v: Any) -> Any:
    a = np.asarray(v)
    return a.item() if a.ndim == 0 else a


def validate(d: Mapping[str, Any]) -> Dict[str, int]:
    """Check keys, dtypes and shape consistency. Returns the resolved
    symbol sizes (F, D, T, N, ...). Raises :class:`CutoutError`."""
    missing = [k for k in REQUIRED if k not in d]
    if missing:
        raise CutoutError(f"missing keys: {missing}")
    unknown = [k for k in d if k not in FIELDS]
    if unknown:
        raise CutoutError(f"unknown keys: {unknown}")
    if str(_as_scalar(d["schema"])) != SCHEMA:
        raise CutoutError(f"schema={_as_scalar(d['schema'])!r}")
    if int(_as_scalar(d["schema_version"])) != SCHEMA_VERSION:
        raise CutoutError(f"schema_version={_as_scalar(d['schema_version'])}")
    sizes: Dict[str, int] = {}

    def bind(sym: str, n: int, key: str) -> None:
        zero_ok = sym.startswith("0|")
        s = sym[2:] if zero_ok else sym
        if zero_ok and n == 0:
            return
        if s == "D+1":
            s, n = "D", n - 1
        if s in sizes and sizes[s] != n:
            raise CutoutError(f"{key}: axis {s}={n}, expected {sizes[s]}")
        sizes[s] = n

    for key, (dtype, shape) in FIELDS.items():
        if key not in d:
            continue
        a = np.asarray(d[key])
        if np.dtype(dtype).kind == "U":
            if a.dtype.kind != "U":
                raise CutoutError(f"{key}: dtype {a.dtype}, expected unicode")
        elif a.dtype != np.dtype(dtype):
            raise CutoutError(f"{key}: dtype {a.dtype}, expected {dtype}")
        if a.ndim != len(shape):
            raise CutoutError(f"{key}: ndim {a.ndim}, expected {len(shape)} {shape}")
        for sym, n in zip(shape, a.shape):
            bind(sym, int(n), key)
    n_t = int(_as_scalar(d["n_t"]))
    if sizes.get("T", n_t) != n_t:
        raise CutoutError(f"n_t={n_t} but planes have {sizes['T']} rows")
    if n_t <= 0 or n_t % 2 or int(_as_scalar(d["t0_ext"])) % 2:
        raise CutoutError("n_t and t0_ext must be even (tier-B alignment)")
    n_grid = int(_as_scalar(d["n_grid"]))
    if sizes.get("N", n_grid) != n_grid:
        raise CutoutError(f"n_grid={n_grid} but images are {sizes['N']}")
    off = np.asarray(d["fine_offset"])
    if off[0] != 0 or np.any(np.diff(off) < 0) or off[-1] != sizes.get("F", 0):
        raise CutoutError("fine_offset is not a CSR over the fine axis")
    coarse = np.asarray(d["coarse_global_idx"]).astype(np.int64)
    owned = np.asarray(d["coarse_owned"]).astype(np.int64)
    for k in range(len(owned)):
        if np.any(coarse[off[k]:off[k + 1]] != owned[k]):
            raise CutoutError(f"block {k} fines do not all belong to coarse {owned[k]}")
    if np.any(np.diff(np.asarray(d["fine_global_idx"]).astype(np.int64)) <= 0):
        raise CutoutError("fine_global_idx must be strictly ascending")
    steps = np.asarray(d["t_step_ext"])
    if np.any((steps != 1) & (steps != 2)):
        raise CutoutError("t_step_ext must be 1 or 2")
    if int(_as_scalar(d["has_peak_image"])) and np.asarray(d["peak_image"]).size == 0:
        raise CutoutError("has_peak_image=1 but peak_image is empty")
    status = str(_as_scalar(d["status"]))
    if status not in ("complete", "partial", "failed"):
        raise CutoutError(f"status={status!r}")
    json.loads(str(_as_scalar(d["status_json"])) or "{}")
    return sizes


# ---------------------------------------------------------------------------
# Writer / reader
# ---------------------------------------------------------------------------


def _coerce(d: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Cast every value to its schema dtype (scalars become 0-d arrays)."""
    out: Dict[str, np.ndarray] = {}
    for key, value in d.items():
        if key not in FIELDS:
            raise CutoutError(f"unknown key {key!r}")
        dtype, shape = FIELDS[key]
        if np.dtype(dtype).kind == "U":
            arr = np.asarray(value, dtype="U")
        else:
            arr = np.asarray(value, dtype=dtype)
        if not shape:
            arr = arr.reshape(())
        out[key] = arr
    return out


def _file_digest(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_cutout(
    out_dir: Path, fields: Mapping[str, Any], *,
    extra_manifest: Optional[Mapping[str, Any]] = None,
) -> Tuple[Path, Path]:
    """Validate, then atomically write the NPZ and THEN its manifest.

    Returns ``(npz_path, manifest_path)``. The manifest is the completion
    signal C2 waits for, so it is only written after the NPZ is durable.
    """
    arrays = _coerce(fields)
    sizes = validate(arrays)
    sid = int(arrays["search_node_id"])
    g = int(arrays["gpu_half"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / cutout_filename(sid, g)
    tmp = npz.with_name(npz.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)          # uncompressed: members can be mmap'd
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, npz)
    manifest = manifest_dict(arrays, sizes)
    manifest.update({
        "file": npz.name,
        "bytes": npz.stat().st_size,
        "blake2b_128": _file_digest(npz),
        "t_written_unix": time.time(),
    })
    if extra_manifest:
        manifest.update(dict(extra_manifest))
    man = out_dir / manifest_filename(sid, g)
    mtmp = man.with_name(man.name + ".tmp")
    with open(mtmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True, default=_json_default)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(mtmp, man)
    return npz, man


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def manifest_dict(arrays: Mapping[str, np.ndarray], sizes: Mapping[str, int]) -> Dict[str, Any]:
    """Every A. scalar plus the axis sizes and per-coarse status."""
    out: Dict[str, Any] = {}
    for key, (_dtype, shape) in FIELDS.items():
        if not shape and key in arrays and key != "status_json":
            out[key] = _as_scalar(arrays[key])
    out["plan_id"] = f"0x{int(out['plan_id']):016x}"
    out["pattern_id"] = f"0x{int(out['pattern_id']):016x}"
    try:
        out["status_detail"] = json.loads(str(_as_scalar(arrays["status_json"])) or "{}")
    except ValueError:
        out["status_detail"] = {}
    out["sizes"] = dict(sizes)
    out["coarse_owned"] = [int(x) for x in np.asarray(arrays["coarse_owned"])]
    return out


@dataclass(frozen=True)
class Cutout:
    """A loaded cutout (all arrays in memory; files are a few MB)."""

    path: Path
    arrays: Dict[str, np.ndarray]

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    def scalar(self, key: str) -> Any:
        return _as_scalar(self.arrays[key])

    @property
    def sid(self) -> int:
        return int(self.scalar("search_node_id"))

    @property
    def gpu_half(self) -> int:
        return int(self.scalar("gpu_half"))

    @property
    def n_coarse(self) -> int:
        return int(self.arrays["coarse_owned"].shape[0])

    def block(self, k: int) -> slice:
        """Fine-axis slice of the k-th owned coarse DM."""
        off = self.arrays["fine_offset"]
        return slice(int(off[k]), int(off[k + 1]))

    def block_of_coarse(self, d: int) -> Optional[int]:
        hits = np.flatnonzero(self.arrays["coarse_owned"].astype(np.int64) == int(d))
        return int(hits[0]) if hits.size else None

    def fine_index_of_global(self, f_global: int) -> Optional[int]:
        hits = np.flatnonzero(self.arrays["fine_global_idx"].astype(np.int64) == int(f_global))
        return int(hits[0]) if hits.size else None


def read_cutout(path: Path, *, check: bool = True) -> Cutout:
    """Load a cutout NPZ (``allow_pickle=False``) and optionally validate."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    if check:
        validate(arrays)
    return Cutout(path=path, arrays=arrays)


def read_manifest(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def empty_status_json(per_coarse: Iterable[Tuple[int, str]] = ()) -> str:
    """Helper for producers: ``{"coarse": {d: state}}`` as JSON."""
    return json.dumps({"coarse": {str(int(d)): str(s) for d, s in per_coarse}})


ImageSource = Any   # ndarray [T, F, N, N] or callable f -> ndarray [T, N, N]


def _image_rows(images: ImageSource, f: int) -> np.ndarray:
    if callable(images):
        return np.asarray(images(int(f)))
    return np.asarray(images)[:, int(f)]


def assemble_cutout(
    *,
    axis: Mapping[str, np.ndarray],
    images: ImageSource,
    n_t: int,
    sigma_applied: np.ndarray,
    row_valid: np.ndarray,
    input_valid_frac: np.ndarray,
    request: Any,
    identity: Mapping[str, Any],
    clock: Mapping[str, Any],
    geometry: Mapping[str, Any],
    status: str = "complete",
    status_detail: Optional[Mapping[str, Any]] = None,
    kernel_ids: Optional[Sequence[str]] = None,
    sigma_k: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """CPU REFERENCE of a whole cutout from width-1 images.

    ``axis`` comes from :func:`fine_axis_for_half`; ``images`` yields the
    Layer-1-normalised images of plane-local fine ``f`` over the cutout rows
    (``[n_t, N, N]``), either as one ``[n_t, F, N, N]`` array or a callable
    (the production GPU producer must reproduce these values; the offline
    re-imager uses this directly). ``request`` is a
    :class:`dsart.coinc.wire.CutoutRequest` (duck-typed). ``identity`` holds
    search_node_id, gpu_half, half_index, opoint, plan_id, pattern_id,
    code_version; ``clock`` t_ext_us, native_per_ext, mjd_t0, mjd_clock;
    ``geometry`` n_grid, cell_l_rad, cell_m_rad, l0_rad, m0_rad.
    """
    fine_dm = np.asarray(axis["fine_dm_pc_cc"])
    off = np.asarray(axis["fine_offset"], dtype=np.int64)
    n_f = int(fine_dm.shape[0])
    n_d = len(off) - 1
    n_grid = int(geometry["n_grid"])
    l_req, m_req = int(request.l_pix), int(request.m_pix)
    has_pix = l_req >= 0 and m_req >= 0
    t0 = int(request.t0_ext)
    if int(n_t) != int(request.n_t):
        raise CutoutError(f"n_t={n_t} != request.n_t={request.n_t}")

    peak_grid = np.zeros((n_t, n_f), dtype=np.float32)
    peak_pix = np.zeros((n_t, n_f), dtype=np.int32)
    pix_lc = np.zeros((n_t if has_pix else 0, n_f), dtype=np.float32)
    for f in range(n_f):
        rows = _image_rows(images, f)
        if rows.shape != (n_t, n_grid, n_grid):
            raise CutoutError(f"fine {f}: images {rows.shape}, expected "
                              f"{(n_t, n_grid, n_grid)}")
        pg, pp, lc = reduce_images(rows[:, None], l_req, m_req)
        peak_grid[:, f], peak_pix[:, f] = pg[:, 0], pp[:, 0]
        if has_pix:
            pix_lc[:, f] = lc[:, 0]

    apex_t, apex_f = apex_per_block(peak_grid, off)
    apex_image = np.zeros((n_d, n_grid, n_grid), dtype=np.float16)
    for k in range(n_d):
        if apex_t[k] >= 0:
            apex_image[k] = _image_rows(images, int(off[k] + apex_f[k]))[apex_t[k]]

    # Peak image: the requested fine on its owner half, or (MANUAL / not
    # owned) nothing -- except MANUAL, which uses the half's strongest apex.
    fglob = np.asarray(axis["fine_global_idx"], dtype=np.int64)
    t_pk = int(request.t_peak_ext) - t0
    f_pk: Optional[int] = None
    hits = np.flatnonzero(fglob == int(request.fine_global_peak))
    if int(request.fine_global_peak) >= 0 and hits.size:
        f_pk = int(hits[0])
    elif bool(getattr(request, "is_manual", False)) and np.any(apex_t >= 0):
        vals = [peak_grid[apex_t[k], off[k] + apex_f[k]] if apex_t[k] >= 0
                else -np.inf for k in range(n_d)]
        kb = int(np.argmax(vals))
        f_pk, t_pk = int(off[kb] + apex_f[kb]), int(apex_t[kb])
    if f_pk is not None and 0 <= t_pk < n_t:
        rows = _image_rows(images, f_pk)
        peak_image = rows[t_pk].astype(np.float16)
        w = max(1, int(request.width_ext))
        lo = max(0, t_pk - w // 2)
        hi = min(n_t, t_pk - w // 2 + w)
        peak_image_w = rows[lo:hi].astype(np.float64).mean(axis=0).astype(np.float16)
        has_peak = 1
    else:
        peak_image = np.zeros((0, 0), dtype=np.float16)
        peak_image_w = np.zeros((0, 0), dtype=np.float16)
        has_peak = 0

    # Stamps around the requested pixel for fines near the requested DM.
    s_half = int(request.stamp_half_px)
    ts = int(request.n_stamp_t)
    sf = (stamp_fines(fine_dm, float(request.dm_peak_pc_cc),
                      float(request.stamp_dm_halfwidth_pc_cc))
          if (has_pix and s_half > 0 and ts > 0) else np.zeros(0, np.int32))
    p = 2 * s_half + 1 if sf.size else 0
    stamp_t0 = int(request.t_peak_ext) - ts // 2
    stamp = np.full((ts if sf.size else 0, sf.size, p, p), np.nan, dtype=np.float16)
    if sf.size:
        r_lo, r_hi = stamp_t0 - t0, stamp_t0 - t0 + ts
        src_r = slice(max(0, r_lo), min(n_t, r_hi))
        dst_r = slice(src_r.start - r_lo, src_r.stop - r_lo)
        l_lo, l_hi = max(0, l_req - s_half), min(n_grid, l_req + s_half + 1)
        m_lo, m_hi = max(0, m_req - s_half), min(n_grid, m_req + s_half + 1)
        dl = slice(l_lo - (l_req - s_half), l_hi - (l_req - s_half))
        dm_ = slice(m_lo - (m_req - s_half), m_hi - (m_req - s_half))
        for j, f in enumerate(sf):
            rows = _image_rows(images, int(f))
            if src_r.stop > src_r.start:
                stamp[dst_r, j, dl, dm_] = rows[src_r, l_lo:l_hi, m_lo:m_hi]

    out: Dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "event_name": request.event_name,
        "request_id": int(request.request_id),
        "request_flags": int(request.flags),
        "opoint": str(identity.get("opoint", "")),
        "plan_id": int(identity.get("plan_id", 0)),
        "pattern_id": int(identity.get("pattern_id", 0)),
        "code_version": str(identity.get("code_version", "")),
        "search_node_id": int(identity["search_node_id"]),
        "gpu_half": int(identity["gpu_half"]),
        "half_index": int(identity.get("half_index", -1)),
        "t_ext_us": float(clock["t_ext_us"]),
        "native_per_ext": int(clock["native_per_ext"]),
        "sample_period_specnum": int(clock["native_per_ext"]) // 2,
        "t0_ext": t0,
        "n_t": int(n_t),
        "mjd_t0": float(clock.get("mjd_t0", 0.0)),
        "mjd_clock": str(clock.get("mjd_clock", "")),
        "t_peak_ext": int(request.t_peak_ext),
        "width_ext_req": int(request.width_ext),
        "dm_req": float(request.dm_peak_pc_cc),
        "fine_global_req": int(request.fine_global_peak),
        "coarse_global_req": int(request.coarse_global_peak),
        "l_pix_req": l_req,
        "m_pix_req": m_req,
        "n_grid": n_grid,
        "cell_l_rad": float(geometry.get("cell_l_rad", 0.0)),
        "cell_m_rad": float(geometry.get("cell_m_rad", 0.0)),
        "l0_rad": float(geometry.get("l0_rad", 0.0)),
        "m0_rad": float(geometry.get("m0_rad", 0.0)),
        "status": status,
        "status_json": json.dumps(dict(status_detail or {}), sort_keys=True,
                                  default=_json_default),
        "input_valid_frac": np.asarray(input_valid_frac, dtype=np.float32),
        "row_valid": np.asarray(row_valid).astype(np.uint8),
        "apex_t": apex_t,
        "apex_f": apex_f,
        "apex_image": apex_image,
        "peak_grid": peak_grid,
        "peak_pix": peak_pix,
        "pix_lc": pix_lc,
        "sigma_applied": np.asarray(sigma_applied, dtype=np.float32),
        "has_peak_image": has_peak,
        "peak_image": peak_image,
        "peak_image_w": peak_image_w,
        "stamp_fine": sf.astype(np.int32),
        "stamp_t0_ext": stamp_t0,
        "stamp": stamp,
    }
    for key in ("fine_dm_pc_cc", "fine_global_idx", "coarse_global_idx",
                "fine_local_idx", "tier", "t_step_ext", "coarse_owned",
                "fine_offset"):
        out[key] = np.asarray(axis[key])
    if kernel_ids is not None and sigma_k is not None:
        out["kernel_ids"] = np.asarray(list(kernel_ids), dtype="U")
        out["sigma_k"] = np.asarray(sigma_k, dtype=np.float32)
    return out


def stamp_fines(fine_dm: Sequence[float], dm_req: float, halfwidth: float) -> np.ndarray:
    """Indices into the fine axis with ``|DM - dm_req| <= halfwidth``
    (empty for MANUAL requests, ``dm_req < 0``)."""
    dm = np.asarray(fine_dm, dtype=np.float64)
    if dm_req < 0 or halfwidth <= 0:
        return np.zeros(0, dtype=np.int32)
    return np.flatnonzero(np.abs(dm - float(dm_req)) <= float(halfwidth)).astype(np.int32)
