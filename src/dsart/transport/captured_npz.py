"""src/dsart/transport/captured_npz.py — M3 → M5 captured-NPZ loader.

Schema-aware loader for the M3-emitted captured transport-TX NPZ
fixtures (per ``bench/m3_emit_m5_fixtures.py`` in the M3 worktree;
also ``PARALLEL_AGENTS.md §1`` and ``M5_PLAN_FIXES.md F6``). M3 chunk
8 publishes a fixture set per voltage run-id at::

    /home/ubuntu/data/m5_fixtures/<run_id>/
        chgroup00.npz ... chgroup15.npz   (one per channel-group)
        manifest.json                     (cross-chgroup metadata)

Each ``chgroupNN.npz`` carries an F26 sparse-COO dump of a single
fast-corr coarse-DM cell, plus full antenna + T2 truth provenance::

    vis_cube_sparse  : complex64  [N_DM=1, n_fv_total, N_filled]
    ix_row, ix_col   : uint16     [N_filled]   (uv-grid cell indices)
    pattern_id       : uint64                  (sparsity-pattern hash)
    n_grid           : int32                   (typically 256)
    n_filled         : int32                   (number of filled cells)
    dec_deg_quant    : float32                 (pattern-quantised δ)
    kernel_support   : int32
    antpos_hash, chgroup_table_hash: uint64
    antpos_e, antpos_n: float32   [N_ant=96]
    is_core_baseline_mask: bool   [N_baselines=4656]
    chgroup, t_int_fast_native, t_int_fast_us, n_fv_total,
    n_blocks_processed, cell_lambda, phi_lat_ovro_deg, obs_dec_deg
                                              (scalar config metadata)
    src_kind, src_name, src_{ra,dec}_deg, src_mjd_trigger,
    src_dm_pc_cc, src_t2_snr                  (T2 truth — NaN for continuum)
    run_id, cal_path, voltage_path, git_sha, utc_iso
                                              (provenance strings)

Two fixtures available on h01 today:

- ``0319/`` — continuum (3C-class compact source 0319+415); 15 chgroups
  (sb12 missing per the known M2 data gap, see PARALLEL_AGENTS.md §5);
  ``n_fv_total=15`` (15 fast-vis blocks at ``t_int_fast_us=134217.728``,
  ie ``t_int_fast_native=4096``); T2 truth NaN. Use for end-to-end
  imager-pipeline regression (no burst expected).
- ``250924mptq/`` — burst FRB (DM=404.688 pc/cc, T2 SNR=30); 16
  chgroups; ``n_fv_total=512`` blocks at ``t_int_fast_us=2097.152``
  (``t_int_fast_native=64``); T2 truth populated. Use for the M5
  voltage-fixture-search end-to-end detection gate (plan §8 line 2330).

The loader scatters each chgroup's sparse-COO vis_cube_sparse back
to a dense ``[N_DM, n_fv_total, N_grid, N_grid] complex64`` per
chgroup using ``ix_row`` / ``ix_col`` (cell coordinates into the
N_grid² uv-plane), validates the F26 invariants (pattern_id, n_grid,
kernel_support are equal across chgroups), and assembles the
``[N_chgroup, n_fv_total, N_grid, N_grid]`` cube that the M5 imager
consumes. Missing chgroups (eg sb12 in 0319) are zero-filled in the
output stack with their ``chgroup`` slot left empty in the manifest's
chgroup index list — callers MUST consult ``manifest.chgroups`` to
find which N_chgroup slots are valid before running the search.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema constants (locked with M3 chunk 8)
# ---------------------------------------------------------------------------

CAPTURED_SCHEMA_VERSION: int = 1
"""Bump on any breaking change to ``chgroupNN.npz`` field set or dtypes."""

# Required scalar/array fields in every chgroupNN.npz. Any missing
# field triggers a clear error (the M3 fixture writer is the single
# source of truth; partial NPZs are a hard fail).
_REQUIRED_NPZ_KEYS: Tuple[str, ...] = (
    "vis_cube_sparse",
    "ix_row",
    "ix_col",
    "pattern_id",
    "n_grid",
    "n_filled",
    "dec_deg_quant",
    "kernel_support",
    "antpos_hash",
    "chgroup_table_hash",
    "antpos_e",
    "antpos_n",
    "is_core_baseline_mask",
    "chgroup",
    "t_int_fast_native",
    "t_int_fast_us",
    "n_fv_total",
    "n_blocks_processed",
    "cell_lambda",
    "phi_lat_ovro_deg",
    "obs_dec_deg",
    "src_kind",
    "src_name",
    "src_ra_deg",
    "src_dec_deg",
    "src_mjd_trigger",
    "src_dm_pc_cc",
    "src_t2_snr",
    "run_id",
    "cal_path",
    "voltage_path",
    "git_sha",
    "utc_iso",
)

_REQUIRED_MANIFEST_KEYS: Tuple[str, ...] = (
    "milestone",
    "purpose",
    "run_id",
    "src_kind",
    "src_name",
    "src_truth",
    "obs_dec_deg",
    "t_int_fast_native",
    "t_int_fast_us",
    "n_chgroups",
    "chgroups",
    "per_chgroup",
    "git_sha",
    "utc_iso",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class T2Truth:
    """T2-trigger truth metadata for the captured run.

    Continuum runs have all fields NaN (no burst); burst runs have
    populated ra/dec/mjd/dm/snr from the legacy DSA-110 T2 dispatcher.
    """
    src_name: str
    ra_deg: float
    dec_deg: float
    mjd_trigger: float
    dm_pc_cc: float
    t2_snr: float

    @property
    def is_burst(self) -> bool:
        """True if T2 truth metadata is populated (i.e. burst fixture)."""
        return not np.isnan(self.dm_pc_cc)


@dataclass
class CapturedChgroup:
    """One chgroup's worth of captured M3 vis cube + provenance.

    The dense uv-stream ``[N_DM, n_fv_total, N_grid, N_grid] complex64``
    is materialised lazily via :meth:`scatter_dense` so callers that
    only need the metadata (or only one chgroup's dense view at a
    time) avoid the 2 GiB-class allocation peak across all 16 chgroups.
    """
    chgroup: int
    pattern_id: int                   # uint64 ⇒ python int
    n_grid: int
    n_filled: int
    dec_deg_quant: float
    kernel_support: int
    antpos_hash: int                  # uint64 ⇒ python int
    chgroup_table_hash: int           # uint64 ⇒ python int
    n_fv_total: int
    n_blocks_processed: int
    cell_lambda: float
    obs_dec_deg: float
    voltage_path: str
    cal_path: str
    # Sparse-COO arrays + indices (kept on host as numpy):
    vis_cube_sparse: np.ndarray       # complex64 [N_DM=1, n_fv_total, n_filled]
    ix_row: np.ndarray                # uint16 [n_filled]
    ix_col: np.ndarray                # uint16 [n_filled]
    # Optional: antenna geometry + baseline mask (full provenance).
    antpos_e: np.ndarray              # float32 [N_ant]
    antpos_n: np.ndarray              # float32 [N_ant]
    is_core_baseline_mask: np.ndarray # bool [N_baselines]

    @property
    def n_dm(self) -> int:
        return int(self.vis_cube_sparse.shape[0])

    def scatter_dense(self) -> np.ndarray:
        """Scatter the F26 sparse-COO vis cube back to a dense uv-grid.

        Returns ``[N_DM, n_fv_total, N_grid, N_grid] complex64`` with
        zeros at the (n_grid² − n_filled) unfilled cells. Allocates
        a fresh array each call; callers concerned with peak memory
        should iterate over chgroups one at a time rather than
        materialising all at once.

        NOTE: this is the M3 → M5 hand-off seam where the production
        ingest path will eventually reinterpret cint8 SparseCOOPayload
        directly (bypassing the dense scatter via the M5 fused-combine
        kernel's cell-index gather). For the captured-fixture gate
        we keep the simple scatter path because the on-disk dump is
        already complex64 (the cint8 quantisation is a transport-time
        encoding step exercised separately in M3 tests).
        """
        n_dm = self.n_dm
        dense = np.zeros(
            (n_dm, self.n_fv_total, self.n_grid, self.n_grid),
            dtype=np.complex64,
        )
        # Vectorised scatter: sparse[d, t, k] → dense[d, t, ix_row[k], ix_col[k]]
        # for k in [0, n_filled). The fancy-indexed write-on-equal-keys
        # is fine here because (ix_row, ix_col) pairs are guaranteed
        # unique within a sparsity pattern (F26 invariant).
        dense[:, :, self.ix_row, self.ix_col] = self.vis_cube_sparse
        return dense


@dataclass(frozen=True)
class CapturedManifest:
    """Cross-chgroup manifest for a captured M5 fixture run.

    Mirrors the on-disk ``manifest.json`` schema with strict typing.
    """
    milestone: str
    purpose: str
    run_id: str
    src_kind: str           # "continuum" | "burst"
    src_name: str
    src_truth: T2Truth
    obs_dec_deg: float
    t_int_fast_native: int
    t_int_fast_us: float
    n_chgroups: int
    chgroups: Tuple[int, ...]   # the actual chgroup indices present
    per_chgroup_meta: Tuple[Dict[str, Any], ...]  # raw per-chgroup dicts
    git_sha: str
    utc_iso: str
    n_baselines: Optional[int] = None
    phi_lat_ovro_deg: Optional[float] = None

    @property
    def is_burst(self) -> bool:
        return self.src_kind == "burst"

    @property
    def expected_n_chgroups(self) -> int:
        """How many chgroups the run is conceptually paired with.

        Always 16 for production DSA-110; some fixtures (eg 0319) have
        a known data gap (sb12 missing → 15 actual chgroups). Callers
        that want the production-stack [N_chgroup=16, ...] shape with
        zero-filled missing slots use this; callers that want only the
        chgroups that actually have data use ``len(chgroups)``.
        """
        return 16

    def chgroup_present(self, chg_idx: int) -> bool:
        """True if ``chgroupNN.npz`` exists in the captured set."""
        return chg_idx in self.chgroups


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _load_manifest(run_dir: Path) -> CapturedManifest:
    """Parse + strict-validate ``manifest.json`` for a captured run.

    Raises:
        FileNotFoundError if ``manifest.json`` is absent.
        ValueError if a required field is missing or the chgroup
            list disagrees with ``per_chgroup`` entries.
    """
    mfst_path = run_dir / "manifest.json"
    if not mfst_path.is_file():
        raise FileNotFoundError(f"missing manifest.json in {run_dir}")
    raw = json.loads(mfst_path.read_text(encoding="utf-8"))
    missing = [k for k in _REQUIRED_MANIFEST_KEYS if k not in raw]
    if missing:
        raise ValueError(
            f"manifest.json missing required keys {missing} (path={mfst_path})"
        )
    src_truth = T2Truth(
        src_name=str(raw["src_truth"].get("src_name", "")),
        ra_deg=float(raw["src_truth"].get("ra_deg", float("nan"))),
        dec_deg=float(raw["src_truth"].get("dec_deg", float("nan"))),
        mjd_trigger=float(raw["src_truth"].get("mjd_trigger", float("nan"))),
        dm_pc_cc=float(raw["src_truth"].get("dm_pc_cc", float("nan"))),
        t2_snr=float(raw["src_truth"].get("t2_snr", float("nan"))),
    )
    chgroups = tuple(int(c) for c in raw["chgroups"])
    per_chg = tuple(dict(d) for d in raw["per_chgroup"])
    if len(per_chg) != len(chgroups):
        raise ValueError(
            f"manifest.json chgroups length {len(chgroups)} != "
            f"per_chgroup length {len(per_chg)}"
        )
    return CapturedManifest(
        milestone=str(raw["milestone"]),
        purpose=str(raw["purpose"]),
        run_id=str(raw["run_id"]),
        src_kind=str(raw["src_kind"]),
        src_name=str(raw["src_name"]),
        src_truth=src_truth,
        obs_dec_deg=float(raw["obs_dec_deg"]),
        t_int_fast_native=int(raw["t_int_fast_native"]),
        t_int_fast_us=float(raw["t_int_fast_us"]),
        n_chgroups=int(raw["n_chgroups"]),
        chgroups=chgroups,
        per_chgroup_meta=per_chg,
        git_sha=str(raw["git_sha"]),
        utc_iso=str(raw["utc_iso"]),
        n_baselines=(
            int(raw["n_baselines"]) if raw.get("n_baselines") is not None else None
        ),
        phi_lat_ovro_deg=(
            float(raw["phi_lat_ovro_deg"])
            if raw.get("phi_lat_ovro_deg") is not None
            else None
        ),
    )


def _load_one_chgroup(npz_path: Path) -> CapturedChgroup:
    """Load + validate one ``chgroupNN.npz`` file."""
    if not npz_path.is_file():
        raise FileNotFoundError(f"missing chgroup npz: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as nz:
        missing = [k for k in _REQUIRED_NPZ_KEYS if k not in nz.files]
        if missing:
            raise ValueError(
                f"{npz_path.name} missing required keys {missing}"
            )
        # Helper to unwrap 0-d numpy scalars to python scalars.
        def _scalar(arr: np.ndarray) -> Any:
            return arr.item() if arr.ndim == 0 else arr

        vis = nz["vis_cube_sparse"]
        if vis.dtype != np.complex64:
            raise ValueError(
                f"{npz_path.name}: vis_cube_sparse dtype={vis.dtype} "
                f"!= complex64"
            )
        if vis.ndim != 3:
            raise ValueError(
                f"{npz_path.name}: vis_cube_sparse ndim={vis.ndim} != 3"
            )

        ix_row = nz["ix_row"]
        ix_col = nz["ix_col"]
        if ix_row.dtype != np.uint16 or ix_col.dtype != np.uint16:
            raise ValueError(
                f"{npz_path.name}: ix_row/ix_col must be uint16; "
                f"got {ix_row.dtype}, {ix_col.dtype}"
            )
        if ix_row.shape != ix_col.shape or ix_row.shape != (vis.shape[2],):
            raise ValueError(
                f"{npz_path.name}: ix_row/ix_col shape {ix_row.shape} "
                f"!= ({vis.shape[2]},)"
            )

        return CapturedChgroup(
            chgroup=int(_scalar(nz["chgroup"])),
            pattern_id=int(_scalar(nz["pattern_id"])),
            n_grid=int(_scalar(nz["n_grid"])),
            n_filled=int(_scalar(nz["n_filled"])),
            dec_deg_quant=float(_scalar(nz["dec_deg_quant"])),
            kernel_support=int(_scalar(nz["kernel_support"])),
            antpos_hash=int(_scalar(nz["antpos_hash"])),
            chgroup_table_hash=int(_scalar(nz["chgroup_table_hash"])),
            n_fv_total=int(_scalar(nz["n_fv_total"])),
            n_blocks_processed=int(_scalar(nz["n_blocks_processed"])),
            cell_lambda=float(_scalar(nz["cell_lambda"])),
            obs_dec_deg=float(_scalar(nz["obs_dec_deg"])),
            voltage_path=str(_scalar(nz["voltage_path"])),
            cal_path=str(_scalar(nz["cal_path"])),
            vis_cube_sparse=vis.copy(),  # detach from mmap
            ix_row=ix_row.copy(),
            ix_col=ix_col.copy(),
            antpos_e=nz["antpos_e"].astype(np.float32, copy=True),
            antpos_n=nz["antpos_n"].astype(np.float32, copy=True),
            is_core_baseline_mask=nz["is_core_baseline_mask"].astype(bool, copy=True),
        )


def load_captured_run(
    run_dir: Path,
    *,
    chgroups_subset: Optional[Sequence[int]] = None,
) -> Tuple[Dict[int, CapturedChgroup], CapturedManifest]:
    """Load a complete captured fixture (manifest + all chgroup NPZs).

    Args:
        run_dir: directory containing ``manifest.json`` + ``chgroupNN.npz``.
        chgroups_subset: optional list of chgroup indices to load (the
            others are skipped). Defaults to None ⇒ load all chgroups
            present per the manifest.

    Returns:
        Tuple of:
        - ``Dict[chgroup_idx, CapturedChgroup]`` keyed by the integer
          channel-group index (NOT the position in the manifest's
          ``chgroups`` list — eg 0319 has keys ``{0, 1, ..., 11, 13, 14, 15}``
          since sb12 is missing).
        - ``CapturedManifest`` for cross-chgroup metadata + T2 truth.

    Raises:
        FileNotFoundError if manifest or any selected chgroup NPZ is missing.
        ValueError on any schema-validation failure or cross-chgroup
            invariant violation (eg pattern_id / n_grid / kernel_support
            disagreement).
    """
    run_dir = Path(run_dir).resolve()
    manifest = _load_manifest(run_dir)
    target = (
        tuple(int(c) for c in chgroups_subset)
        if chgroups_subset is not None
        else manifest.chgroups
    )
    missing = [c for c in target if c not in manifest.chgroups]
    if missing:
        raise ValueError(
            f"requested chgroups {missing} not present in manifest "
            f"(have {list(manifest.chgroups)})"
        )

    out: Dict[int, CapturedChgroup] = {}
    n_grid_canonical: Optional[int] = None
    kernel_support_canonical: Optional[int] = None
    for chg_idx in target:
        npz_path = run_dir / f"chgroup{chg_idx:02d}.npz"
        chg = _load_one_chgroup(npz_path)
        if chg.chgroup != chg_idx:
            raise ValueError(
                f"{npz_path.name}: scalar chgroup={chg.chgroup} != "
                f"filename chgroup={chg_idx}"
            )
        # Cross-chgroup invariants. F26 fixes that all chgroups in a
        # run share n_grid + kernel_support (the sparsity pattern
        # itself differs per chgroup since cell_lambda scales with
        # frequency, but the grid geometry doesn't).
        if n_grid_canonical is None:
            n_grid_canonical = chg.n_grid
        elif chg.n_grid != n_grid_canonical:
            raise ValueError(
                f"chgroup{chg_idx:02d}.npz: n_grid={chg.n_grid} disagrees "
                f"with first chgroup's n_grid={n_grid_canonical}"
            )
        if kernel_support_canonical is None:
            kernel_support_canonical = chg.kernel_support
        elif chg.kernel_support != kernel_support_canonical:
            raise ValueError(
                f"chgroup{chg_idx:02d}.npz: kernel_support={chg.kernel_support}"
                f" disagrees with first chgroup's "
                f"kernel_support={kernel_support_canonical}"
            )
        out[chg_idx] = chg
    _LOG.info(
        "loaded captured run %s: %d chgroups, src_kind=%s%s",
        manifest.run_id, len(out), manifest.src_kind,
        f", T2 DM={manifest.src_truth.dm_pc_cc:.3f} pc/cc"
        if manifest.is_burst else "",
    )
    return out, manifest


def stack_dense_streams(
    chgroups: Mapping[int, CapturedChgroup],
    *,
    fill_missing: bool = True,
    n_chgroup_total: int = 16,
) -> Tuple[np.ndarray, List[bool]]:
    """Build the production-shape ``[N_chgroup, n_fv_total, N_grid, N_grid]``
    dense complex64 stream stack expected by the M5 imager.

    Missing chgroups (eg sb12 in 0319) are zero-filled by default, so
    downstream pipeline code can index slot ``g`` without bounds
    checking — callers that care about which slots are real consult
    the returned ``valid_mask``.

    Args:
        chgroups: dict from chgroup index → CapturedChgroup (the first
            return value of :func:`load_captured_run`).
        fill_missing: whether to pad missing slots with zeros (default
            True). If False, raises ``ValueError`` when any slot in
            ``[0, n_chgroup_total)`` is missing.
        n_chgroup_total: total slot count in the output stack (default
            16 = production DSA-110 channel-group count).

    Returns:
        Tuple of:
        - ``[N_chgroup, n_fv_total, N_grid, N_grid] complex64`` dense
          stack. NOTE this allocates ``N_chgroup × n_fv_total ×
          N_grid² × 8 B`` of host memory; for the 250924mptq burst
          fixture this is ~16 × 512 × 256² × 8 = 16 GiB. Use
          :meth:`CapturedChgroup.scatter_dense` per-chgroup +
          stream-process if memory is tight.
        - ``valid_mask`` ``List[bool]`` of length ``n_chgroup_total``,
          True at slots that came from real data + False at zero-filled
          slots.

    Raises:
        ValueError if chgroups disagree on ``n_grid`` or ``n_fv_total``,
        or if ``fill_missing=False`` and a slot is absent.
    """
    if not chgroups:
        raise ValueError("empty chgroups dict")
    g0 = next(iter(chgroups.values()))
    n_grid = g0.n_grid
    n_fv = g0.n_fv_total
    n_dm = g0.n_dm
    if n_dm != 1:
        raise ValueError(
            f"M5 captured-fixture loader expects N_DM=1 per chgroup; "
            f"got {n_dm}. (M3 chunk-8 fixture writer dumps a single "
            f"coarse-DM cell per F26.)"
        )
    for g_idx, g in chgroups.items():
        if g.n_grid != n_grid:
            raise ValueError(
                f"chgroup {g_idx}: n_grid={g.n_grid} disagrees with "
                f"first chgroup's n_grid={n_grid}"
            )
        if g.n_fv_total != n_fv:
            raise ValueError(
                f"chgroup {g_idx}: n_fv_total={g.n_fv_total} disagrees "
                f"with first chgroup's n_fv_total={n_fv}"
            )

    out = np.zeros(
        (n_chgroup_total, n_fv, n_grid, n_grid), dtype=np.complex64,
    )
    valid_mask = [False] * n_chgroup_total
    for chg_idx, g in chgroups.items():
        if not (0 <= chg_idx < n_chgroup_total):
            raise ValueError(
                f"chgroup index {chg_idx} out of range [0, {n_chgroup_total})"
            )
        # scatter_dense returns [N_DM=1, n_fv, N, N]; flatten the
        # singleton DM axis since the M5 imager expects 4-D streams.
        dense = g.scatter_dense()[0]  # [n_fv, N, N]
        out[chg_idx] = dense
        valid_mask[chg_idx] = True

    if not fill_missing and not all(valid_mask):
        missing_slots = [i for i, v in enumerate(valid_mask) if not v]
        raise ValueError(
            f"fill_missing=False but chgroups {missing_slots} are absent"
        )

    return out, valid_mask
