#!/usr/bin/env python3
"""tools/viz/m6_cube_dump_verifier.py — operator-facing cube-dump NPZ
verifier (M6 chunk 8 — operator-approval gate).

Loads one or more NPZ cube dumps written by
``dsart.dump.cube_dump.CubeDumpWriter`` (M6 chunk 3) and produces, for
each:

  * ``${report_dir}/m6_cube_dump_${cube_id}.png`` — two-panel figure:

      Left:  per-frame max-pixel ``[T_det, n_fdm]`` heatmap, where
             each cell is ``max(l, m)`` of the cube tensor at that
             ``(t, f)`` slice (operator-readable view of where bright
             pixels concentrate in the cube).
      Right: peak ``(l, m)`` frame — the single ``(t*, f*)`` slice
             carrying the global max pixel.

  * ``${report_dir}/m6_cube_dump_verifier.md`` — markdown table
    aggregating one row per cube (cube_id, mjd_start, trigger_source,
    peak SNR, NPZ basename, PNG link).

NPZ schema (M6 D7 — see ``dsart.dump.cube_dump`` for the writer
contract)::

    cube                  -> [T_det, n_fdm, n_grid, n_grid] float16
    mjd_start             -> 0-d float64
    event_specnum_start   -> 0-d int64
    t_det                 -> 0-d int32
    n_fdm_in_cube         -> 0-d int32
    n_grid                -> 0-d int32
    cluster_record        -> 0-d 'U' (json-encoded asdict; "null" for udp)
    trigger_source        -> 0-d 'U'  ('auto' | 'udp')
    search_node_id        -> 0-d int32
    gpu_half              -> 0-d int32

CLI::

    python -m tools.viz.m6_cube_dump_verifier \\
        --dump-root bench/reports/M6/cube_dump \\
        --report-dir bench/reports/M6/viz \\
        --max-cubes 8

Per the M6 chunk-8 spec the report carries NO PASS/FAIL banner — the
operator inspects figures + the manifest table and signs off out-of-
band by editing ``bench/reports/M6/m_operator_approved.yaml``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Loaded-NPZ container
# ---------------------------------------------------------------------------


@dataclass
class CubeDumpInfo:
    """In-memory summary of one NPZ cube dump."""

    path: Path
    cube_id: int
    mjd_start: float
    event_specnum_start: int
    trigger_source: str
    search_node_id: int
    gpu_half: int
    t_det: int
    n_fdm_in_cube: int
    n_grid: int
    cluster_record: Optional[dict]
    cube_min: float
    cube_max: float
    cube_mean: float
    cube_std: float
    cube_nan_count: int
    peak_value: float
    peak_t_idx: int
    peak_f_idx: int
    peak_l_idx: int
    peak_m_idx: int
    cluster_peak_snr: float  # nan if no cluster_record
    png_filename: str


def _scalar(arr: Any) -> Any:
    """Unwrap a 0-d ``np.ndarray`` to a python scalar; pass through scalars."""
    a = np.asarray(arr)
    if a.shape == ():
        return a.item()
    if a.shape == (1,):
        return a.flat[0].item()
    return arr


def _parse_cluster_record(blob: Any) -> Optional[dict]:
    """Decode ``cluster_record`` from a 0-d 'U' ndarray to a dict-or-None.

    Per the writer contract, ``json.dumps(None)`` (``"null"``) is
    written for UDP triggers; ``json.dumps(asdict(record))`` for auto.
    Both round-trip via ``json.loads`` with no allow_pickle hazard.
    """
    text = _scalar(blob)
    if text is None:
        return None
    try:
        decoded = json.loads(str(text))
    except (TypeError, ValueError):
        return None
    if decoded is None or not isinstance(decoded, dict):
        return None
    return decoded


# ---------------------------------------------------------------------------
# Discovery + load
# ---------------------------------------------------------------------------


def discover_npz_files(dump_root: Path) -> List[Path]:
    """Return sorted ``cube_*.npz`` files under ``dump_root`` (depth=1).

    Sorted by filename so per-(sid, gpu_half, event_specnum_start)
    lexicographic order matches the writer's ``${event_specnum_start}``
    naming, which is monotonically increasing in production.
    """
    if not dump_root.exists():
        return []
    return sorted(dump_root.glob("cube_*.npz"))


def load_cube_dump(path: Path) -> Tuple[np.ndarray, CubeDumpInfo, str]:
    """Load one NPZ; return ``(cube, info, png_filename)``.

    ``allow_pickle=False`` per the writer contract (no field is an
    object array — ``cluster_record`` is a 0-d 'U' ndarray).
    """
    with np.load(path, allow_pickle=False) as data:
        cube = np.asarray(data["cube"])  # [T_det, n_fdm, n_grid, n_grid]
        mjd_start = float(_scalar(data["mjd_start"]))
        event_specnum_start = int(_scalar(data["event_specnum_start"]))
        t_det = int(_scalar(data["t_det"]))
        n_fdm_in_cube = int(_scalar(data["n_fdm_in_cube"]))
        n_grid = int(_scalar(data["n_grid"]))
        trigger_source = str(_scalar(data["trigger_source"]))
        sid = int(_scalar(data["search_node_id"]))
        g = int(_scalar(data["gpu_half"]))
        cluster_record = _parse_cluster_record(data["cluster_record"])

    if cube.ndim != 4:
        raise ValueError(
            f"{path}: cube has shape {cube.shape}; expected 4D "
            f"[T_det, n_fdm, n_grid, n_grid]"
        )
    if cube.shape != (t_det, n_fdm_in_cube, n_grid, n_grid):
        raise ValueError(
            f"{path}: cube shape {cube.shape} disagrees with manifest "
            f"({t_det}, {n_fdm_in_cube}, {n_grid}, {n_grid})"
        )

    cube_f32 = cube.astype(np.float32, copy=False)
    nan_mask = ~np.isfinite(cube_f32)
    n_nan = int(np.sum(nan_mask))
    if n_nan > 0:
        cube_f32 = np.nan_to_num(cube_f32, copy=False)

    flat_argmax = int(np.argmax(cube_f32))
    peak_t, peak_f, peak_l, peak_m = np.unravel_index(
        flat_argmax, cube_f32.shape
    )
    peak_val = float(cube_f32[peak_t, peak_f, peak_l, peak_m])

    # Cube id from filename: cube_s${sid}_g${g}_${event_specnum_start}.npz
    # Use event_specnum_start as a monotonic cube id surrogate when the
    # writer schema does not surface a cube_id key (it does not — see
    # ``cube_dump.CubeDumpWriter._write_one``). The cluster record
    # carries the cube_id when "auto"; "udp" dumps fall back to the
    # event-specnum surrogate, which is unique within a (sid, g)
    # process.
    if cluster_record is not None and "cube_id" in cluster_record:
        cube_id_eff = int(cluster_record["cube_id"])
    else:
        cube_id_eff = event_specnum_start
    png_filename = f"m6_cube_dump_{cube_id_eff}.png"

    cluster_snr = (
        float(cluster_record.get("snr", float("nan")))
        if cluster_record is not None
        else float("nan")
    )

    info = CubeDumpInfo(
        path=path,
        cube_id=cube_id_eff,
        mjd_start=mjd_start,
        event_specnum_start=event_specnum_start,
        trigger_source=trigger_source,
        search_node_id=sid,
        gpu_half=g,
        t_det=t_det,
        n_fdm_in_cube=n_fdm_in_cube,
        n_grid=n_grid,
        cluster_record=cluster_record,
        cube_min=float(np.min(cube_f32)),
        cube_max=float(np.max(cube_f32)),
        cube_mean=float(np.mean(cube_f32)),
        cube_std=float(np.std(cube_f32)),
        cube_nan_count=n_nan,
        peak_value=peak_val,
        peak_t_idx=int(peak_t),
        peak_f_idx=int(peak_f),
        peak_l_idx=int(peak_l),
        peak_m_idx=int(peak_m),
        cluster_peak_snr=cluster_snr,
        png_filename=png_filename,
    )
    return cube_f32, info, png_filename


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def render_cube_png(
    cube: np.ndarray, info: CubeDumpInfo, *, out_path: Path
) -> None:
    """Render the per-cube two-panel verification PNG.

    Args:
        cube: float32 cube tensor with shape
            ``[t_det, n_fdm, n_grid, n_grid]`` — the full output of
            ``np.load(...)["cube"]`` after upcast from float16.
        info: ``CubeDumpInfo`` returned from :func:`load_cube_dump`.
        out_path: file path to write the PNG to.
    """
    try:
        import matplotlib  # noqa: E402
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("matplotlib is required to render PNGs") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # (a) per-frame max-pixel heatmap [t, f] -> max over (l, m).
    tf_max = cube.reshape(info.t_det, info.n_fdm_in_cube, -1).max(axis=2)

    # (b) peak (l, m) frame at (t*, f*).
    peak_lm = cube[info.peak_t_idx, info.peak_f_idx, :, :]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

    title_pieces = [
        f"cube_id={info.cube_id}",
        f"trigger={info.trigger_source}",
        f"sid={info.search_node_id}",
        f"g={info.gpu_half}",
        f"mjd_start={info.mjd_start:.6f}",
    ]
    if info.cluster_record is not None and not np.isnan(info.cluster_peak_snr):
        title_pieces.append(f"cluster_snr={info.cluster_peak_snr:.2f}")
    fig.suptitle(" | ".join(title_pieces), fontsize=10)

    im_a = ax_a.imshow(
        tf_max.T, origin="lower", aspect="auto",
        extent=(-0.5, info.t_det - 0.5, -0.5, info.n_fdm_in_cube - 0.5),
        cmap="magma",
    )
    ax_a.set_title("(a) per-frame max-pixel(l,m)")
    ax_a.set_xlabel("t_in_cube (samples)")
    ax_a.set_ylabel("fine_dm_idx")
    fig.colorbar(im_a, ax=ax_a, label="max(l,m)")
    ax_a.scatter(
        [info.peak_t_idx], [info.peak_f_idx],
        marker="x", color="cyan", s=80, linewidth=2,
        label=f"peak (t*={info.peak_t_idx}, f*={info.peak_f_idx})",
    )
    ax_a.legend(fontsize=8, loc="upper right")

    im_b = ax_b.imshow(peak_lm, origin="lower", aspect="equal", cmap="viridis")
    ax_b.set_title(
        f"(b) peak (l,m) frame at t*={info.peak_t_idx}, f*={info.peak_f_idx}"
    )
    ax_b.set_xlabel("l_pix")
    ax_b.set_ylabel("m_pix")
    fig.colorbar(im_b, ax=ax_b, label="value")
    ax_b.scatter(
        [info.peak_l_idx], [info.peak_m_idx],
        marker="+", color="red", s=140, linewidth=2,
        label=f"peak ({info.peak_l_idx}, {info.peak_m_idx}) = {info.peak_value:.3g}",
    )
    ax_b.legend(fontsize=8, loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def render_markdown_report(
    *,
    infos: Sequence[CubeDumpInfo],
    dump_root: Path,
    report_dir: Path,
    out_path: Path,
    max_cubes: int,
    n_total: int,
) -> None:
    """Aggregate per-cube ``CubeDumpInfo`` into the operator markdown.

    Per the M6 chunk-8 spec — the report carries NO PASS/FAIL banner.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md: List[str] = []
    md.append("# M6 cube-dump NPZ verifier")
    md.append("")
    md.append(
        "Operator-facing summary of the M6 chunk-3 cube-dump NPZ files. "
        "No PASS/FAIL banner — sign off out-of-band by editing "
        "`bench/reports/M6/m_operator_approved.yaml`."
    )
    md.append("")
    md.append("## Run metadata")
    md.append("")
    md.append(f"- dump_root: `{dump_root}`")
    md.append(f"- report_dir: `{report_dir}`")
    md.append(f"- max_cubes: {max_cubes}")
    md.append(f"- npz files matched: {n_total}")
    md.append(f"- npz files inspected: {len(infos)}")
    md.append("")

    if not infos:
        md.append("(no NPZ files matched — empty report)")
        md.append("")
        out_path.write_text("\n".join(md))
        return

    md.append("## Manifest summary")
    md.append("")
    md.append(
        "| cube_id | npz | trigger | sid | g | mjd_start | "
        "event_specnum_start | T_det | n_fdm | n_grid | "
        "cluster_snr | peak_value | peak (t,f,l,m) | nan |"
    )
    md.append(
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | --- | ---: |"
    )
    for info in infos:
        snr_str = (
            f"{info.cluster_peak_snr:.3f}"
            if not np.isnan(info.cluster_peak_snr)
            else "-"
        )
        md.append(
            f"| {info.cube_id} | `{info.path.name}` | {info.trigger_source} | "
            f"{info.search_node_id} | {info.gpu_half} | "
            f"{info.mjd_start:.6f} | {info.event_specnum_start} | "
            f"{info.t_det} | {info.n_fdm_in_cube} | {info.n_grid} | "
            f"{snr_str} | {info.peak_value:.3g} | "
            f"({info.peak_t_idx},{info.peak_f_idx},{info.peak_l_idx},"
            f"{info.peak_m_idx}) | {info.cube_nan_count} |"
        )
    md.append("")
    md.append("## Cube tensor stats")
    md.append("")
    md.append("| cube_id | min | mean | std | max |")
    md.append("| ---: | ---: | ---: | ---: | ---: |")
    for info in infos:
        md.append(
            f"| {info.cube_id} | {info.cube_min:.3g} | "
            f"{info.cube_mean:.3g} | {info.cube_std:.3g} | "
            f"{info.cube_max:.3g} |"
        )
    md.append("")
    md.append("## Per-cube figures")
    md.append("")
    for info in infos:
        md.append(f"### cube_id = {info.cube_id}")
        md.append("")
        md.append(f"![cube {info.cube_id}]({info.png_filename})")
        md.append("")

    out_path.write_text("\n".join(md))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="m6_cube_dump_verifier",
        description=(
            "Operator-facing verifier for M6 chunk-3 cube-dump NPZ files."
        ),
    )
    p.add_argument(
        "--dump-root", required=True, type=Path,
        help="Directory holding cube_s*_g*_${event_specnum_start}.npz files.",
    )
    p.add_argument(
        "--report-dir", required=True, type=Path,
        help="Output directory for the per-cube PNGs + summary MD.",
    )
    p.add_argument(
        "--max-cubes", type=int, default=8,
        help="Cap on the number of NPZ files to inspect (sorted by name). "
             "Default 8.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    dump_root: Path = args.dump_root
    report_dir: Path = args.report_dir
    max_cubes: int = int(args.max_cubes)

    if max_cubes <= 0:
        raise SystemExit(
            f"--max-cubes={max_cubes} must be > 0"
        )

    npz_paths = discover_npz_files(dump_root)
    if not npz_paths:
        print(
            f"[m6_cube_dump_verifier] no NPZ files found under {dump_root!s}; "
            "still emitting an empty operator report."
        )

    selected = npz_paths[:max_cubes]

    report_dir.mkdir(parents=True, exist_ok=True)
    md_path = report_dir / "m6_cube_dump_verifier.md"

    infos: List[CubeDumpInfo] = []
    for path in selected:
        try:
            cube, info, png_filename = load_cube_dump(path)
        except (OSError, ValueError, KeyError) as exc:
            print(
                f"[m6_cube_dump_verifier] WARN: failed to load {path}: {exc}",
                file=sys.stderr,
            )
            continue
        png_path = report_dir / png_filename
        render_cube_png(cube, info, out_path=png_path)
        print(f"[m6_cube_dump_verifier] wrote {png_path}")
        infos.append(info)

    render_markdown_report(
        infos=infos,
        dump_root=dump_root,
        report_dir=report_dir,
        out_path=md_path,
        max_cubes=max_cubes,
        n_total=len(npz_paths),
    )
    print(f"[m6_cube_dump_verifier] wrote {md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
