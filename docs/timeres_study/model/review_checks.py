"""Checks for the plan review: (a) quantisation efficiency of b-bit complex samples, (b) uv-column
occupancy behind the "skip ~half the FFT columns" claim, (c) redundancy multiplicities."""
import json

import numpy as np

from nfilled import BE, BN, C, COSZ
from physics import FREQS_MHZ

rng = np.random.default_rng(3)


def eta(bits, scale_err=1.0, n=2_000_000):
    """corr(x, Q(x)) for a Gaussian, symmetric mid-rise uniform quantiser with 2^bits levels whose
    step is optimal * scale_err."""
    x = rng.standard_normal(n)
    L = 2 ** bits
    best = None
    for step in np.linspace(0.05, 1.5, 300):
        q = np.clip(np.floor(x / step), -L // 2, L // 2 - 1) + 0.5
        c = np.corrcoef(x, q)[0, 1]
        if best is None or c > best[0]:
            best = (c, step)
    step = best[1] * scale_err
    q = np.clip(np.floor(x / step), -L // 2, L // 2 - 1) + 0.5
    return float(np.corrcoef(x, q)[0, 1]), float(best[1])


out = {"eta": {}}
for b in (4, 5, 6, 8):
    e, st = eta(b)
    lo, _ = eta(b, 0.7)
    hi, _ = eta(b, 1.4)
    out["eta"][b] = {"opt": e, "step_sigma": st, "scale_x0.7": lo, "scale_x1.4": hi}
    print(f"{b}-bit: eta {e:.4f} at step {st:.3f} sigma; step x0.7 -> {lo:.4f}, x1.4 -> {hi:.4f}")

# (b) column occupancy, rfft half-axis along u (EW) or along v (NS), all channels
cells = set()
for f in FREQS_MHZ[::4]:
    lam = C / (f * 1e6)
    u = np.rint(BE / lam / 17.9).astype(int)
    v = np.rint(BN * COSZ / lam / 17.9).astype(int)
    for uu, vv in zip(u, v):
        cells.add((uu, vv))
        cells.add((-uu, -vv))
cells = np.array(sorted(cells))
for axis, name in ((0, "u (EW)"), (1, "v (NS)")):
    half = cells[cells[:, axis] >= 0]
    other = 1 - axis
    cols = np.unique(half[:, axis])
    nonzero_other = np.unique(half[half[:, other] != 0][:, axis])
    print(f"half-axis along {name}: occupied columns {len(cols)} (max {cols.max()}); columns with any "
          f"entry off the {('v','u')[axis]}=0 line: {len(nonzero_other)} (max {nonzero_other.max()})")
    out[f"cols_{name}"] = {"occupied": int(len(cols)), "max": int(cols.max()),
                          "needing_fft": int(len(nonzero_other)), "max_needing_fft": int(nonzero_other.max())}

# (c) redundancy multiplicities (5 cm)
q = np.stack([np.rint(BE / 0.05), np.rint(BN / 0.05)], 1).astype(np.int64)
flip = (q[:, 1] < 0) | ((q[:, 1] == 0) & (q[:, 0] < 0))
q[flip] *= -1
_, mult = np.unique(q, axis=0, return_counts=True)
w_nat = mult.sum() / np.sqrt((mult ** 2).sum())        # S/N with equal weight per UNIQUE entry, rel. natural
print(f"unique {len(mult)}, multiplicity max {mult.max()}, entries with mult 1: {np.sum(mult == 1)}; "
      f"S/N if entries were AVERAGED and gridded with equal weight: x{np.sqrt(len(mult)) * mult.mean() / np.sqrt((mult**2).sum()) * np.sqrt(len(mult)) / np.sqrt(mult.sum()) * np.sqrt(mult.sum()) / np.sqrt(len(mult)):.4f}")
# point-source S/N with weights w_i per unique entry (sum of mult_i baselines, noise var mult_i):
for name, w in (("natural (w = mult)", mult.astype(float)), ("uniform per entry (w = 1, entries = averages)", np.ones(len(mult)))):
    # entry i = average over mult_i baselines -> signal 1, noise var 1/mult_i; weight w_i applied to the average
    s = np.sum(w)
    n = np.sqrt(np.sum(w ** 2 / mult))
    out[name] = float(s / n / np.sqrt(mult.sum()))
    print(f"  {name}: point-source S/N relative to natural = {s / n / np.sqrt(mult.sum()):.4f}")
json.dump(out, open("results/review_checks.json", "w"), indent=1)
