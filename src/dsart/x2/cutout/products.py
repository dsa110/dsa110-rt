"""Event products adapter: 2x cutouts OR 1x cubes behind one reader.

Every downstream consumer that used to glob ``<event>/cubes/cube_s*_g*_*.npz``
(C3 veto metrics, the C2 completion gate, Slack readiness, dashboard counts,
the plotter) reads through :func:`load_event_products` / :func:`summarize`
instead (docs/x2/downstream_inventory.md §3.9). The unit of the result is a
:class:`PeakPlane`: the DM-time plane (full-field width-1 image max per row
and fine DM) of ONE coarse DM of ONE half -- exactly what C3's rules consume.

* 2x (``<event>/cutouts/cutout_s<sid>_g<g>.npz``): one plane per owned
  coarse DM, with its fine DMs, global indices, tier steps, row validity
  and the apex image at the block apex.
* 1x (``<event>/cubes/``): one plane per cube file, built exactly as the
  pre-x2 readers did -- same glob, same sort, same filename parse, same
  ``peak_grid`` load/transpose, same mmap of ``cube[t, fdm]`` for the apex
  plane -- so C3 metrics on 1x events are bit-identical.

numpy-only (no torch); safe on h23 and in the dashboard.
"""

from __future__ import annotations

import ast
import glob
import os
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .schema import (
    Cutout,
    CutoutError,
    parse_cutout_filename,
    read_cutout,
)

__all__ = [
    "KIND_CUTOUTS",
    "KIND_CUBES",
    "KIND_NONE",
    "PeakPlane",
    "EventProducts",
    "load_event_products",
    "summarize",
    "halves_with_products",
    "mmap_npz_member",
    "load_cube_peak_grid",
    "cube_file_halves",
]

KIND_CUTOUTS = "cutouts"
KIND_CUBES = "cubes"
KIND_NONE = "none"

#: Glob the pre-x2 readers use for 1x cubes (C2 gate, C3, plotter).
CUBE_GLOB = "cube_s*_g*_*.npz"
CUTOUT_NPZ_GLOB = "cutout_s*_g*.npz"
CUTOUT_MANIFEST_GLOB = "cutout_s*_g*.json"


# ---------------------------------------------------------------------------
# 1x cube helpers (verbatim semantics of coinc/cube_veto.py)
# ---------------------------------------------------------------------------


def mmap_npz_member(npz_path: str, key: str = "cube") -> np.ndarray:
    """Memory-map an UNCOMPRESSED ``.npz`` member without loading it.

    Same algorithm as ``coinc/cube_veto.py::_mmap_cube`` (zip local header ->
    ``.npy`` header -> ``np.memmap``), generalised to any member name.
    """
    member = f"{key}.npy"
    with zipfile.ZipFile(npz_path) as zf:
        info = zf.getinfo(member)
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{member} is compressed; cannot mmap")
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


def load_cube_peak_grid(npz_path: str) -> Tuple[np.ndarray, int, int]:
    """``(peak_grid (T, F) float32, t_det, n_fdm)`` -- identical to
    ``coinc/cube_veto.py::_load_peak_grid`` (accepts either orientation)."""
    with np.load(npz_path) as z:
        pg = np.asarray(z["peak_grid"], dtype=np.float32)
        tdet = int(z["t_det"])
        nf = int(z["n_fdm_in_cube"])
    if pg.shape == (nf, tdet):
        pg = pg.T
    return pg, tdet, nf


def _cube_sid_g(path: str) -> Optional[Tuple[int, int]]:
    """``(sid, g)`` from ``cube_s<sid>_g<g>_<specnum>.npz`` exactly as C3
    parses it (``basename.split("_")``), else None."""
    parts = os.path.basename(path).split("_")
    try:
        return int(parts[1][1:]), int(parts[2][1:])
    except (IndexError, ValueError):
        return None


def cube_file_halves(cubes_dir: Path) -> Tuple[List[str], Set[Tuple[int, int]]]:
    """``(sorted cube files, distinct (sid, g))`` under ``cubes_dir``."""
    files = sorted(glob.glob(str(Path(cubes_dir) / CUBE_GLOB)))
    halves = {sg for sg in (_cube_sid_g(f) for f in files) if sg is not None}
    return files, halves


# ---------------------------------------------------------------------------
# Planes
# ---------------------------------------------------------------------------


@dataclass
class PeakPlane:
    """The DM-time plane of one coarse DM of one half.

    ``peak_grid`` is ``(T, F)`` float32 (rows x fines). For a 1x cube the
    plane is the whole cube (one coarse DM per half; ``coarse = -1``, fine
    metadata unknown without the plan -> ``None``).
    """

    sid: int
    gpu_half: int
    coarse: int
    peak_grid: np.ndarray
    source: str
    path: Path
    fine_dm: Optional[np.ndarray] = None
    fine_global_idx: Optional[np.ndarray] = None
    fine_local_idx: Optional[np.ndarray] = None
    t_step: int = 1
    row_valid: Optional[np.ndarray] = None
    t0_ext: Optional[int] = None
    t_ext_us: Optional[float] = None
    pix_lc: Optional[np.ndarray] = None
    apex_t: Optional[int] = None
    apex_f: Optional[int] = None
    _apex_image_fn: Optional[Callable[[int, int], np.ndarray]] = field(
        default=None, repr=False)

    @property
    def key(self) -> Tuple[int, int, int]:
        return (self.sid, self.gpu_half, self.coarse)

    @property
    def half(self) -> Tuple[int, int]:
        return (self.sid, self.gpu_half)

    def apex_image(self, t: int, f: int) -> np.ndarray:
        """Full image plane at row ``t``, plane-local fine ``f``.

        1x: any (t, f) via the cube mmap. 2x: only the stored block apex
        (``KeyError`` otherwise -- callers treat that as "image metrics
        unavailable", the fail-safe KEEP direction)."""
        if self._apex_image_fn is None:
            raise KeyError("no image source")
        return self._apex_image_fn(int(t), int(f))

    @property
    def has_image_source(self) -> bool:
        return self._apex_image_fn is not None


def _planes_from_cutout(c: Cutout) -> List[PeakPlane]:
    a = c.arrays
    planes: List[PeakPlane] = []
    n_t = int(c.scalar("n_t"))
    t0 = int(c.scalar("t0_ext"))
    t_ext_us = float(c.scalar("t_ext_us"))
    row_valid = np.asarray(a["row_valid"]).astype(bool)
    pix = np.asarray(a["pix_lc"])
    for k in range(c.n_coarse):
        sl = c.block(k)
        steps = np.asarray(a["t_step_ext"][sl]).astype(np.int64)
        t_step = int(steps.max()) if steps.size else 1
        at = int(a["apex_t"][k])
        af = int(a["apex_f"][k])

        def _img(t: int, f: int, _k=k, _at=at, _af=af) -> np.ndarray:
            if (t, f) != (_at, _af):
                raise KeyError(f"cutout holds only the apex image ({_at}, {_af})")
            return np.asarray(a["apex_image"][_k], dtype=np.float32)

        planes.append(PeakPlane(
            sid=c.sid, gpu_half=c.gpu_half,
            coarse=int(a["coarse_owned"][k]),
            peak_grid=np.asarray(a["peak_grid"][:, sl], dtype=np.float32),
            source=KIND_CUTOUTS, path=c.path,
            fine_dm=np.asarray(a["fine_dm_pc_cc"][sl], dtype=np.float64),
            fine_global_idx=np.asarray(a["fine_global_idx"][sl]).astype(np.int64),
            fine_local_idx=np.asarray(a["fine_local_idx"][sl]).astype(np.int64),
            t_step=t_step,
            row_valid=row_valid[:n_t, k],
            t0_ext=t0, t_ext_us=t_ext_us,
            pix_lc=(np.asarray(pix[:, sl], dtype=np.float32) if pix.size else None),
            apex_t=at if at >= 0 else None,
            apex_f=af if af >= 0 else None,
            _apex_image_fn=_img if at >= 0 else None,
        ))
    return planes


def _plane_from_cube(path: str, sid: int, g: int) -> PeakPlane:
    pg, _tdet, _nf = load_cube_peak_grid(path)

    def _img(t: int, f: int, _p=path) -> np.ndarray:
        cube = mmap_npz_member(_p, "cube")
        try:
            return np.asarray(cube[t, f], dtype=np.float32)
        finally:
            del cube

    return PeakPlane(
        sid=sid, gpu_half=g, coarse=-1, peak_grid=pg, source=KIND_CUBES,
        path=Path(path), _apex_image_fn=_img,
    )


# ---------------------------------------------------------------------------
# Event-level
# ---------------------------------------------------------------------------


@dataclass
class EventProducts:
    """All planes of one event plus where they came from."""

    kind: str
    event_dir: Path
    planes: List[PeakPlane]
    halves: Set[Tuple[int, int]]
    n_files: int
    errors: Dict[str, str] = field(default_factory=dict)
    cutouts: Dict[Tuple[int, int], Cutout] = field(default_factory=dict)

    def planes_of_half(self, sid: int, g: int) -> List[PeakPlane]:
        return [p for p in self.planes if p.half == (int(sid), int(g))]

    def plane_of_fine(self, fine_global: int) -> Optional[Tuple[PeakPlane, int]]:
        """(plane, plane-local fine index) holding a plan-global fine (2x)."""
        for p in self.planes:
            if p.fine_global_idx is None:
                continue
            hits = np.flatnonzero(p.fine_global_idx == int(fine_global))
            if hits.size:
                return p, int(hits[0])
        return None

    def plane_of_coarse(self, coarse: int) -> Optional[PeakPlane]:
        for p in self.planes:
            if p.coarse == int(coarse):
                return p
        return None


def load_event_products(event_dir: Path) -> EventProducts:
    """Read whatever products the event has: cutouts (2x) win over cubes.

    Unreadable files are skipped and recorded in ``errors`` (C3 fails open
    on missing data; it never raises here).
    """
    event_dir = Path(event_dir)
    cut_dir = event_dir / KIND_CUTOUTS
    files = sorted(glob.glob(str(cut_dir / CUTOUT_NPZ_GLOB)))
    if files:
        planes: List[PeakPlane] = []
        halves: Set[Tuple[int, int]] = set()
        errors: Dict[str, str] = {}
        cutouts: Dict[Tuple[int, int], Cutout] = {}
        for f in files:
            if parse_cutout_filename(f) is None:
                continue
            try:
                c = read_cutout(Path(f))
            except (OSError, ValueError, KeyError, CutoutError) as exc:
                errors[os.path.basename(f)] = repr(exc)
                continue
            halves.add((c.sid, c.gpu_half))
            cutouts[(c.sid, c.gpu_half)] = c
            planes.extend(_planes_from_cutout(c))
        return EventProducts(KIND_CUTOUTS, event_dir, planes, halves,
                             len(files), errors, cutouts)
    cube_files, _ = cube_file_halves(event_dir / KIND_CUBES)
    if cube_files:
        planes = []
        halves = set()
        errors = {}
        for f in cube_files:
            sg = _cube_sid_g(f)
            if sg is None:
                continue
            try:
                planes.append(_plane_from_cube(f, *sg))
            except Exception as exc:  # noqa: BLE001 - pre-x2 readers skip too
                errors[os.path.basename(f)] = repr(exc)
                continue
            halves.add(sg)
        return EventProducts(KIND_CUBES, event_dir, planes, halves,
                             len(cube_files), errors)
    return EventProducts(KIND_NONE, event_dir, [], set(), 0)


def halves_with_products(event_dir: Path) -> Tuple[str, Set[Tuple[int, int]], int]:
    """Cheap count (no NPZ reads): ``(kind, halves, n_files)``.

    2x: halves = distinct ``cutouts/cutout_s<sid>_g<g>.json`` MANIFESTS (the
    completion signal; written after the NPZ). 1x: halves = distinct (sid, g)
    of ``cubes/cube_s*_g*_*.npz`` and ``n_files`` = the raw file count the
    pre-x2 gate compared against 8.
    """
    event_dir = Path(event_dir)
    mans = sorted(glob.glob(str(event_dir / KIND_CUTOUTS / CUTOUT_MANIFEST_GLOB)))
    if mans:
        halves = {sg for sg in (parse_cutout_filename(m) for m in mans) if sg}
        return KIND_CUTOUTS, halves, len(mans)
    files, halves = cube_file_halves(event_dir / KIND_CUBES)
    if files:
        return KIND_CUBES, halves, len(files)
    return KIND_NONE, set(), 0


def summarize(
    event_dir: Path, expected: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict[str, object]:
    """Dashboard / Slack summary: kind, present / missing halves, counts."""
    kind, halves, n_files = halves_with_products(event_dir)
    exp = sorted({(int(s), int(g)) for s, g in (expected or ())})
    present = sorted(halves)
    missing = [sg for sg in exp if sg not in halves]
    return {
        "kind": kind,
        "n_files": n_files,
        "n_halves": len(present),
        "present": [f"s{s}g{g}" for s, g in present],
        "missing": [f"s{s}g{g}" for s, g in missing],
        "n_expected": len(exp),
        "complete": bool(exp) and not missing,
    }
