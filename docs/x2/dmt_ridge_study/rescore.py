"""Re-score saved planes (planes_synth/, planes_truth/) and 2x cutouts with the CURRENT dmt_ridge -> JSONL."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))
from dsart.coinc.dmt_ridge import RidgeConfig, event_ridge, ridge_features  # noqa: E402

TMP = os.environ.get("DMT_STUDY_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
cfg = RidgeConfig()


def score_dir(d, out):
    with open(out, "w") as fh:
        for p in sorted(glob.glob(f"{d}/plane_*.npz")):
            with np.load(p) as z:
                meta = json.loads(str(z["meta"]))
                geo = {"width": int(z["width"]), "dm_trig": float(z["dm_trig"])}
                f, _ = ridge_features(z["z"].astype(np.float32), z["dms"], dm_trig=geo["dm_trig"],
                                      t_trig=int(z["t_trig"]), width=geo["width"], cfg=cfg)
            name = os.path.basename(p)[6:-4]
            fh.write(json.dumps({"name": name, **meta, **geo,
                                 **{f"r_{k}": v for k, v in f.to_dict().items()}}) + "\n")


if __name__ == "__main__":
    which = sys.argv[1:] or ["synth", "truth", "cut"]
    if "synth" in which:
        score_dir(os.environ.get("DMT_STUDY_PLANES", "/dataz/dsa110/operations/x2/dmt_ridge_study") + "/planes_synth", f"{TMP}/rs_synth.jsonl")
    if "truth" in which:
        score_dir(os.environ.get("DMT_STUDY_PLANES", "/dataz/dsa110/operations/x2/dmt_ridge_study") + "/planes_truth", f"{TMP}/rs_truth.jsonl")
    if "cut" in which:
        evs = sorted({os.path.basename(os.path.dirname(os.path.dirname(p))) for p in
                      glob.glob("/dataz/dsa110/candidates/26111*/cutouts/cutout_s*_g*.npz")})
        with open(f"{TMP}/rs_cut.jsonl", "w") as fh:
            for ev in evs:
                rec = {"name": ev}
                for kind in ("pix", "max"):
                    f, _, d = event_ridge(f"/dataz/dsa110/candidates/{ev}", kind=kind, cfg=cfg)
                    rec[kind] = f.to_dict()
                    if d is not None:
                        rec.update(dm=float(d["dm_trig"]), width=int(d["width"]), halves=d["n_halves"])
                fh.write(json.dumps(rec) + "\n")
