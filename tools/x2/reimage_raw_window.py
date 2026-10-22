#!/usr/bin/env python3
"""Offline re-imaging of an archived 2x raw-stream window (x2 D4).

Reads one half's ``<event>/raw/raw_s<sid>_g<g>/`` (``raw_d<DDD>.npz`` +
``RAW_OK.json``, layout in :mod:`dsart.x2.cutout.raw_archive`), the tiered
DM plan, and re-images every archived coarse DM over a chosen window into
either

* ``--out-kind cutout``: a ``dsart.cutout`` v1 file (same content the live
  half returns; the CPU reference ``assemble_cutout``), or
* ``--out-kind cube``: a 1x-style cube NPZ per coarse DM,
  ``cube[t, fdm, l, m]`` float16 + ``fine_dm_pc_cc`` / ``t0_ext`` / ``t_ext_us``
  (for the notebooks and tools that want full image planes).

The imaging itself is WS-S's numpy reference ``dsart.x2.ref.search2x_ref``
(decode -> combine via the value map -> FFT -> Layer-1 normalisation).
RE-WIRE: :func:`default_imager` binds to ``search2x_ref.reimage_from_raw``;
until WS-S lands that entry point the tool exits with a clear message, and
tests inject their own imager. CPU only; never run against a live ring.

Example::

    python tools/x2/reimage_raw_window.py \\
        --raw /dataz/dsa110/candidates/261022abcd/raw/raw_s9_g1 \\
        --plan /home/ubuntu/data/dm_plans/x2_plan.npz \\
        --t-peak-ext 123456789 --n-t 512 --out /tmp/reimaged --out-kind cutout
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

#: imager(slab, fine_local, t0_ext, n_t, *, ext_factor) -> images [n_t, N, N]
Imager = Callable[..., np.ndarray]


def default_imager() -> Imager:
    try:
        from dsart.x2.ref import search2x_ref
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"dsart.x2.ref.search2x_ref is not available ({exc!r}); "
                         "the offline re-imager needs WS-S's reference") from exc
    fn = getattr(search2x_ref, "reimage_from_raw", None)
    if fn is None:
        raise SystemExit("dsart.x2.ref.search2x_ref has no reimage_from_raw(); "
                         "re-wire tools/x2/reimage_raw_window.default_imager")
    return fn


def _rows_for(slab: Any, t0_ext: int, n_t: int, ext_factor: int,
              imager: Imager, f_local: int) -> np.ndarray:
    imgs = np.asarray(imager(slab, f_local, t0_ext, n_t, ext_factor=ext_factor))
    if imgs.shape[0] != n_t:
        raise ValueError(f"imager returned {imgs.shape[0]} rows, expected {n_t}")
    return imgs


def reimage(
    raw_dir: Path, plan: Any, *, half: int, t0_ext: int, n_t: int,
    imager: Imager, t_ext_native: int = 16, n_grid: int = 256,
) -> Dict[str, Any]:
    """``{"axis", "images"(callable f -> [n_t, N, N]), "slabs"}`` for one half."""
    from dsart.x2.cutout.raw_archive import read_raw_dir
    from dsart.x2.cutout.schema import fine_axis_for_half

    slabs = read_raw_dir(raw_dir)
    axis = fine_axis_for_half(plan, half, t_ext_native=t_ext_native)
    fglob = axis["fine_global_idx"]
    off = axis["fine_offset"]
    owned = axis["coarse_owned"]
    missing = [int(d) for d in owned if int(d) not in slabs]
    cache: Dict[int, np.ndarray] = {}

    def images(f: int) -> np.ndarray:
        if f in cache:
            return cache[f]
        k = int(np.searchsorted(off, f, side="right") - 1)
        d = int(owned[k])
        slab = slabs.get(d)
        if slab is None:
            # not archived (too_late on the half): empty rows, and the
            # caller marks the block invalid.
            cache[f] = np.zeros((n_t, n_grid, n_grid), np.float32)
            return cache[f]
        f_loc = int(np.flatnonzero(slab.fine_global == fglob[f])[0])
        ext_factor = int(slab.t_native) // int(t_ext_native)
        rows = _rows_for(slab, t0_ext, n_t, ext_factor, imager, f_loc)
        cache[f] = rows
        return rows

    return {"axis": axis, "images": images, "slabs": slabs, "missing": missing}


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", required=True, type=Path,
                    help="<event>/raw/raw_s<sid>_g<g> directory")
    ap.add_argument("--plan", required=True, type=Path, help="tiered DM plan .npz")
    ap.add_argument("--t-peak-ext", required=True, type=int)
    ap.add_argument("--n-t", type=int, default=512)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--out-kind", choices=("cutout", "cube"), default="cutout")
    ap.add_argument("--n-grid", type=int, default=256)
    args = ap.parse_args(argv)

    from dsart.coinc.wire import CUTOUT_FLAG_CUTOUT, CutoutRequest, default_cutout_window
    from dsart.x2.cutout.schema import assemble_cutout, write_cutout
    from dsart.x2.plan import load_any_plan

    plan = load_any_plan(args.plan)
    meta = json.loads((args.raw / "meta.json").read_text())
    ident = meta.get("identity", {})
    sid, g = int(ident.get("search_node_id", 0)), int(ident.get("gpu_half", 0))
    half = int(ident.get("half_index", (1, 2, 9, 13).index(sid) * 2 + g))
    t0, n_t = default_cutout_window(args.t_peak_ext, args.n_t)
    imager = default_imager()
    r = reimage(args.raw, plan, half=half, t0_ext=t0, n_t=n_t, imager=imager,
                n_grid=args.n_grid)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.out_kind == "cube":
        ax = r["axis"]
        for k, d in enumerate(ax["coarse_owned"]):
            lo, hi = int(ax["fine_offset"][k]), int(ax["fine_offset"][k + 1])
            cube = np.stack([r["images"](f) for f in range(lo, hi)], axis=1)
            np.savez(args.out / f"cube_d{int(d):03d}.npz", cube=cube.astype(np.float16),
                     fine_dm_pc_cc=ax["fine_dm_pc_cc"][lo:hi], t0_ext=np.int64(t0),
                     t_ext_us=np.float64(16 * 32.768), coarse=np.int32(d))
        print(f"wrote {len(ax['coarse_owned'])} cubes to {args.out}")
        return 0
    req = CutoutRequest(event_name=str(meta.get("request", {}).get("event_name", "offline")),
                        request_id=0, t_peak_ext=args.t_peak_ext, t0_ext=t0, n_t=n_t,
                        flags=CUTOUT_FLAG_CUTOUT, stamp_half_px=0, n_stamp_t=0)
    ax = r["axis"]
    n_f = len(ax["fine_dm_pc_cc"])
    fields = assemble_cutout(
        axis=ax, images=r["images"], n_t=n_t,
        sigma_applied=np.ones((n_t, n_f), np.float32),
        row_valid=np.ones((n_t, len(ax["coarse_owned"])), bool),
        input_valid_frac=np.ones(n_f, np.float32), request=req,
        identity=dict(search_node_id=sid, gpu_half=g, half_index=half,
                      plan_id=int(getattr(plan, "plan_id", 0)), opoint="offline"),
        clock=dict(t_ext_us=16 * 32.768, native_per_ext=16),
        geometry=dict(n_grid=args.n_grid), status_detail={"source": "offline"})
    npz, _ = write_cutout(args.out, fields)
    print(f"wrote {npz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
