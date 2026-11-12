"""R12 study figure: (A) fitted vs true bandwidth, (B) decision plane f_fit vs fitted B."""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rule_eval as E  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, CONTEXT = "#2a78d6", "#eb6834", "#1baf7a", "#a8a7a1"
OUT = sys.argv[1]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, (a, b) = plt.subplots(1, 2, figsize=(12.0, 5.2))


def clip(v):
    return np.clip(np.asarray(v, float), 0.1, 300.0)


syn = [s for s in E.syn if s.get("r_ok") and (s.get("r_fit_sigma") or 0) >= 8]
a.scatter(clip([s["occ_true_mhz"] for s in syn]), clip([s["r_bw_fit_mhz"] for s in syn]), s=14, c=CONTEXT,
          alpha=0.55, linewidths=0, label=f"synthetic bursts with a line >= 8 sigma ({len(syn)})")
volt = [it for it in E.items if it[2] == "volt" and it[3].get("r_ok") and it[3].get("bw90_mhz")]
for cls, col, lab in (("real", BLUE, "real FRBs / pulsar (voltage recreation)"),
                      ("narrow", ORANGE, "narrow-band events (voltage-confirmed)")):
    xs = [it for it in volt if it[1].startswith(cls)]
    a.scatter(clip([it[3]["bw90_mhz"] for it in xs]), clip([it[3]["r_bw_fit_mhz"] for it in xs]), s=40, c=col,
              edgecolors=SURF, linewidths=1.5, label=f"{lab} ({len(xs)})", zorder=3)
a.axhline(6.0, color=INK2, lw=1.0, ls="--")
a.text(0.12, 6.6, "6 MHz cut", color=INK2, fontsize=9)
a.set_xscale("log")
a.set_yscale("log")
a.set_xlim(0.1, 300)
a.set_ylim(0.08, 400)
a.set_xlabel("true bandwidth (MHz): synthetic occupied / voltage 90% power")
a.set_ylabel("bandwidth fitted from the 2x DM-time plane (MHz)")
a.set_title("A. Bandwidth read from the DM-time plane", loc="left")
a.grid(True, color=GRID, lw=0.6)
a.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)

b.axvspan(0.08, 6.0, ymin=0, ymax=1, color="#fbe3d8", zorder=0)
for cls, src, col, lab in (("real", ("volt", "cut-pix"), BLUE, "real (FRBs, pulsar, injections)"),
                           ("narrow", ("volt", "cut-pix"), ORANGE, "narrow-band (voltage-confirmed)"),
                           ("other", ("cut-pix",), AQUA, "unlabelled 2x sky events")):
    xs = [it for it in E.items if it[1].startswith(cls) and it[2] in src and it[3].get("r_ok")
          and (it[3].get("r_fit_sigma") or 0) > 0]
    b.scatter(clip([it[3]["r_bw_fit_mhz"] for it in xs]), np.clip([it[3]["r_f_fit"] for it in xs], 0, 2.0),
              s=40, c=col, edgecolors=SURF, linewidths=1.5, label=f"{lab} ({len(xs)})", zorder=3)
b.plot([0.08, 6.0], [0.45, 0.45], color=INK2, lw=1.0, ls="--")
b.axvline(6.0, color=INK2, lw=1.0, ls="--")
b.text(0.1, 1.9, "flagged: B <= 6 MHz and f >= 0.45\n(f clipped at 2)", color=INK2, fontsize=9, va="top")
b.set_xscale("log")
b.set_xlim(0.08, 400)
b.set_ylim(0, 2.0)
b.set_xlabel("fitted bandwidth B (MHz)")
b.set_ylabel("f: line amplitude / trigger significance")
b.set_title("B. Decision plane (trigger-pixel plane)", loc="left")
b.grid(True, color=GRID, lw=0.6)
b.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK)
for s in ("top", "right"):
    a.spines[s].set_visible(False)
    b.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
