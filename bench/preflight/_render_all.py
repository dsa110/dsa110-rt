"""Batch-render quicklooks for every injection produced by
inject_production_allgpu. For each <out_dir>/<tag>/ injection:
  - owning-owner 3-panel quicklook (the owner whose C1 has the top SNR)
  - all-owner stitched DM-time bowtie at the burst pixel

Burst pixel + true DM are read from each owning owner's C1 (fallback: a
fixed pixel / the tag's DM).

Usage: python _render_all.py <out_dir> <png_out_dir> [Lfallback Mfallback]
"""
from __future__ import annotations

import csv
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_top_c1(csv_path: str):
    try:
        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))
    except FileNotFoundError:
        return None
    if not rows:
        return None
    rows.sort(key=lambda r: -float(r["snr"]))
    return rows[0]


def tag_dm(tag: str) -> float:
    m = re.search(r"dm(\d+)", tag)
    return float(m.group(1)) if m else 0.0


def main() -> None:
    out_dir = Path(sys.argv[1])
    png_dir = Path(sys.argv[2])
    png_dir.mkdir(parents=True, exist_ok=True)
    Lfb = int(sys.argv[3]) if len(sys.argv) > 3 else 117
    Mfb = int(sys.argv[4]) if len(sys.argv) > 4 else 151

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = "src:bench:."

    tags = sorted(p.name for p in out_dir.iterdir()
                  if p.is_dir() and p.name.startswith("dm"))
    for tag in tags:
        tdir = out_dir / tag
        true_dm = tag_dm(tag)
        # find owning owner = owner whose C1 top SNR is largest
        best = (-1.0, None, None)
        for od in sorted(tdir.glob("owner*")):
            if not od.is_dir():
                continue
            top = read_top_c1(str(od / "candidates_c1.csv"))
            if top is None:
                continue
            snr = float(top["snr"])
            if snr > best[0]:
                best = (snr, od, top)
        L, M = Lfb, Mfb
        if best[1] is not None:
            top = best[2]
            L, M = int(top["l_pix"]), int(top["m_pix"])
            cube = glob.glob(str(best[1] / "cube_*.npz"))[0]
            ql = png_dir / f"ql_{tag}_owning.png"
            subprocess.run(
                [sys.executable, str(HERE / "_render_cube.py"),
                 "--cube", cube, "--c1", str(best[1] / "candidates_c1.csv"),
                 "--out", str(ql),
                 "--title", f"{tag} OWNING snr={best[0]:.1f} (fp16 production)"],
                env=env, check=False)
            print("owning quicklook:", ql, "snr", round(best[0], 1))
        else:
            print("WARN no candidate in any owner for", tag)

        bow = png_dir / f"bowtie_{tag}.png"
        subprocess.run(
            [sys.executable, str(HERE / "_render_bowtie.py"),
             str(tdir), str(L), str(M), str(true_dm), str(bow),
             f"{tag} all-8-GPU DM-time @ pixel ({L},{M})"],
            env=env, check=False)
        print("bowtie:", bow)


if __name__ == "__main__":
    main()
