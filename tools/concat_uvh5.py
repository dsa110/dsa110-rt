"""tools/concat_uvh5.py — concatenate per-subband UVH5 files along the freq axis.

The slow-corr → meridian_fringestop pipeline emits one UVH5 file per
subband (each with `Nfreqs=48` after the 8x freq-int average and a
per-sb 11.7 MHz window). For inspection / imaging it is often
convenient to glue them into a single multi-spw UVH5 so that downstream
tools see one continuous frequency axis (e.g. 720 channels for the
full 0319+415 acceptance run minus sb12 → 14*48 = 672 channels; or
including sb12 if present, 15*48 = 720).

Implementation: thin wrapper around :class:`pyuvdata.UVData`'s
``fast_concat(axis="freq")``. We deliberately keep the heavy lifting
in pyuvdata so all the bookkeeping (uvw_array consistency,
antenna_positions, history, time_array, channel_width) is handled
correctly. The script runs in either ``dsa110-rt`` (pyuvdata 3.x) or
``casa38`` (pyuvdata 1.x) — both support ``fast_concat``.

Inputs may be:
  * a glob pattern (``--glob '/path/0319_sb*.uvh5'``), OR
  * an explicit list of paths (``--uvh5 a.uvh5 b.uvh5 c.uvh5``).

Output is a single UVH5 written to ``--out``. The sort order of input
files determines the resulting frequency-axis ordering — pyuvdata
auto-corrects but emits an explicit warning when channel ordering is
non-monotonic.
"""

from __future__ import annotations

import argparse
import glob as _glob
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--glob", help="glob pattern matching per-sb UVH5 files")
    src.add_argument("--uvh5", nargs="+",
                     help="explicit list of per-sb UVH5 files")
    ap.add_argument("--out", required=True, help="output UVH5 path")
    ap.add_argument("--axis", default="freq",
                    choices=("freq", "blt", "polarization"),
                    help="concat axis (default freq)")
    ap.add_argument("--clobber", action="store_true",
                    help="overwrite existing --out file")
    args = ap.parse_args(argv)

    if args.glob:
        paths = sorted(_glob.glob(args.glob))
    else:
        paths = sorted(args.uvh5)
    if not paths:
        raise SystemExit("no input UVH5 files found")

    out_path = Path(args.out)
    if out_path.exists() and not args.clobber:
        raise SystemExit(f"{out_path} already exists; pass --clobber to overwrite")

    print(f"[concat] inputs ({len(paths)}):", flush=True)
    for p in paths:
        print(f"  {p}", flush=True)
    print(f"[concat] output: {out_path}", flush=True)
    print(f"[concat] axis  : {args.axis}", flush=True)

    from pyuvdata import UVData  # heavy import; do it lazily

    t0 = time.monotonic()
    uv = UVData()
    print(f"[concat] reading {paths[0]} ...", flush=True)
    uv.read_uvh5(paths[0])
    others: list[UVData] = []
    for p in paths[1:]:
        print(f"[concat] reading {p} ...", flush=True)
        u = UVData()
        u.read_uvh5(p)
        others.append(u)

    if others:
        print(f"[concat] fast_concat({len(others)} others, axis={args.axis}) ...",
              flush=True)
        uv.fast_concat(others, axis=args.axis, inplace=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[concat] writing {out_path} ...", flush=True)
    uv.write_uvh5(str(out_path), clobber=args.clobber)
    elapsed = time.monotonic() - t0

    print(f"[concat] done in {elapsed:.1f}s; final shape: "
          f"Nblts={uv.Nblts} Nfreqs={uv.Nfreqs} Npols={uv.Npols}", flush=True)
    if args.axis == "freq":
        # Sanity: print first/last freq to confirm monotonic ordering.
        f0_GHz = float(uv.freq_array.flat[0]) / 1e9
        fN_GHz = float(uv.freq_array.flat[-1]) / 1e9
        df_MHz = float(uv.channel_width.flat[0] if hasattr(uv.channel_width, "flat")
                       else uv.channel_width) / 1e6
        print(f"[concat] freq axis: f[0]={f0_GHz:.4f} GHz  "
              f"f[-1]={fN_GHz:.4f} GHz  Δν={df_MHz:.3f} MHz", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
