"""Build ../DSA110_snr_current_vs_2x.pdf (current, proposed 2x, single-shot).  Run from docs/timeres_study/snr2x/ after
run_snr2x.py, summarize.py and figs_snr2x.py."""
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from schemes import current, describe, proposed, single_shot

INK, INK2 = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e")
RULE, HEAD = colors.HexColor("#d9d8d3"), colors.HexColor("#f0efec")
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=14.5, leading=17.5,
                    spaceAfter=4, textColor=INK)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.2, leading=13.5,
                    spaceBefore=7, spaceAfter=3, textColor=INK)
B = ParagraphStyle("B", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.0, leading=11.8,
                   textColor=INK, spaceAfter=4)
BL = ParagraphStyle("BL", parent=B, leftIndent=11, bulletIndent=2, spaceAfter=2.5)
SM = ParagraphStyle("SM", parent=B, fontSize=7.8, leading=9.6, textColor=INK2)
CAP = ParagraphStyle("CAP", parent=SM, spaceBefore=2, spaceAfter=8)
TC = ParagraphStyle("TC", parent=B, fontSize=7.5, leading=9.1, spaceAfter=0)
TCB = ParagraphStyle("TCB", parent=TC, fontName="Helvetica-Bold")

Z = np.load("results/snr2x.npz")
W = Z["widths_ms"]


def P(t, st=B):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(t, BL, bulletText="•") for t in items]


def fig(name, width=6.9 * inch, caption=None):
    path = "../figs_snr2x/" + name
    w, h = ImageReader(path).getSize()
    out = [Image(path, width=width, height=width * h / w)]
    if caption:
        out.append(P(caption, CAP))
    return KeepTogether(out)


def table(rows, widths, extra=None):
    data = [[Paragraph(str(c), TCB if (i == 0 or j == 0) else TC) for j, c in enumerate(r)]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("BACKGROUND", (0, 0), (-1, 0), HEAD), ("TOPPADDING", (0, 0), (-1, -1), 1.8),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8)]
    t.setStyle(TableStyle(st + (extra or [])))
    return t


def band(key, lo, hi, stage=3):
    dm, ch = Z[f"{key}_dm"], Z[f"{key}_chain"]
    m = (dm >= lo) & (dm <= hi)
    return ch[m][:, stage]


def fac(key, lo=500, hi=1000):
    dm, ch = Z[f"{key}_dm"], Z[f"{key}_chain"]
    c = ch[(dm >= lo) & (dm <= hi)]
    return (c[:, 0].mean(0), (c[:, 1] / c[:, 0]).mean(0), (c[:, 2] / c[:, 1]).mean(0), (c[:, 3] / c[:, 2]).mean(0),
            c[:, 3].mean(0))


cur, pro, one = describe(current()), describe(proposed()), describe(single_shot())
ca, pa, pb, oa = cur["tiers"][0], pro["tiers"][0], pro["tiers"][1], one["tiers"][0]
c_m, p_m, o_m = (band(k, 100, 1500).mean(0) for k in ("current", "prop_q026", "single"))
c_10, p_10, o_10 = (np.percentile(band(k, 100, 1500), 10, 0) for k in ("current", "prop_q026", "single"))
p_hi = band("prop_q026", 1500, 3000).mean(0)
q_m = band("prop_q052", 100, 1500).mean(0)
J = {w: int(np.argmin(np.abs(W - w))) for w in W}
BANDS3 = ((100, 500), (500, 1000), (1000, 1500))


def bm(key, lo, hi, w):
    return band(key, lo, hi).mean(0)[J[w]]


def f3(x):
    return f"{x:.2f}"


story = []
A = story.append
A(P("DSA-110 search: S/N loss vs width and DM - current scheme, proposed 2x, and a single-shot "
    "0.26 ms / 244 kHz search", H1))
A(P("2026-10-21, branch timeres-study (docs/timeres_study/snr2x/). Recovered S/N is given as a fraction of an "
    "<b>ideal matched filter</b>: a Gaussian pulse of FWHM W, perfectly dedispersed, continuously sampled, no "
    "smearing, white noise. Every loss from channel smearing, time sampling, the boxcar ladder, DM-trial spacing "
    "and integer delays is included; losses common to all three (imaging, flagging, primary beam, sigma_k colour) "
    "are not.", SM))

A(P("Summary", H2))
story += bullets([
    f"<b>DM range.</b> Current: <b>{ca['dm'][0]:.0f}-{ca['dm'][1]:.0f}</b> pc cm<super>-3</super>. Proposed 2x: "
    f"<b>{pa['dm'][0]:.0f}-{pb['dm'][1]:.0f}</b> (0.52 ms up to DM 1600, 1.05 ms above). Single-shot: "
    f"<b>{oa['dm'][0]:.0f}-{oa['dm'][1]:.0f}</b>.",
    f"<b>2x vs current.</b> Mean over DM 100-1500: x{p_m[J[0.1]] / c_m[J[0.1]]:.2f} the S/N at 0.1 ms, "
    f"x{p_m[J[0.5]] / c_m[J[0.5]]:.2f} at 0.5 ms, <b>x{p_m[J[1.0]] / c_m[J[1.0]]:.2f} at 1 ms</b> "
    f"({f3(c_m[J[1.0]])} -> {f3(p_m[J[1.0]])} of ideal), x{p_m[J[2.0]] / c_m[J[2.0]]:.2f} at 2 ms, "
    f"x{p_m[J[4.0]] / c_m[J[4.0]]:.2f} at 4 ms; equal above ~8 ms, where every scheme sits at the 0.94 ceiling of a "
    f"power-of-two boxcar on a Gaussian. Over DM 1500-3000 the 2x scheme still recovers {f3(p_hi[J[1.0]])} at 1 ms.",
    f"<b>Single-shot vs current.</b> x{o_m[J[0.1]] / c_m[J[0.1]]:.2f} at 0.1 ms, x{o_m[J[0.5]] / c_m[J[0.5]]:.2f} "
    f"at 0.5 ms, <b>x{o_m[J[1.0]] / c_m[J[1.0]]:.2f} at 1 ms</b> ({f3(o_m[J[1.0]])} of ideal), "
    f"x{o_m[J[2.0]] / c_m[J[2.0]]:.2f} at 2 ms.",
    f"<b>Single-shot vs 2x: equal on average, opposite trends with DM</b> (DM-band table). Averaged over DM 100-1500 the "
    f"two agree to within 5% at every width. But single-shot is the best of the three below DM ~500 (0.2 ms: "
    f"{f3(bm('single', 100, 500, 0.2))} vs {f3(bm('prop_q026', 100, 500, 0.2))}; 1 ms: "
    f"{f3(bm('single', 100, 500, 1.0))} vs {f3(bm('prop_q026', 100, 500, 1.0))}), and falls behind the 2x above "
    f"DM ~1000 (0.2 ms: {f3(bm('single', 1000, 1500, 0.2))} vs {f3(bm('prop_q026', 1000, 1500, 0.2))}; 1 ms: "
    f"{f3(bm('single', 1000, 1500, 1.0))} vs {f3(bm('prop_q026', 1000, 1500, 1.0))}). Its worst DMs are worse "
    f"(10th percentile at 1 ms {f3(o_10[J[1.0]])} vs {f3(p_10[J[1.0]])}), and it does not search above 1500.",
    "<b>Why.</b> Single-shot pays almost nothing for its integer delays (one rounding at 0.26 ms: factor "
    "0.97-1.00) and samples 2x finer, but its 244 kHz channels smear 4x more than 61 kHz: 0.7-0.9 ms at "
    "DM 1000, which swamps the 0.26 ms sampling. The 2x scheme pays 4-15% (W &lt;= 1 ms) for its two-stage integer delays "
    "but keeps the smear small at every DM. The current scheme pays both: 244 kHz smear and coarse two-stage "
    "rounding at 1.05 ms.",
    f"<b>What 2x leaves on the table:</b> narrow bursts still lose about a third of their S/N to sampling at "
    f"0.52 ms ({f3(p_m[J[0.2]])} of ideal at 0.2 ms). The single-shot curve below DM 500 shows what finer "
    f"sampling buys; the 4x step (0.26 ms with 61 kHz channels) is what gets it at every DM.",
    f"<b>Stage-1 quantum (2x).</b> Rounding stage 1 to 0.52 ms instead of 0.26 ms costs "
    f"{100 * (1 - q_m[J[0.5]] / p_m[J[0.5]]):.0f}% at 0.5 ms and {100 * (1 - q_m[J[1.0]] / p_m[J[1.0]]):.0f}% at "
    f"1 ms. Keep 0.26 ms: it is also the corr-side code path for 4x.",
    "<b>Cost (context, not S/N).</b> The single-shot op-point is what the legacy beam search did. In dsart's "
    "image-plane search every imaging owner then needs every channel: two corner turns, ~41 Gb/s for the "
    "channels (244 kHz, 0.26 ms, 4-bit cells) plus ~90 Gb/s for 495 DM-trial planes, and ~1.5x the imaging load "
    "of the 2x plan. That is about three times today's 44 Gb/s of usable search ingress; the 2x plan needs 34.",
])

A(P("The three schemes", H2))
A(table([
    ["", "current (deployed 2026-09-23)", "proposed 2x", "single-shot"],
    ["DM searched", f"{ca['dm'][0]:.0f}-{ca['dm'][1]:.0f}",
     f"{pa['dm'][0]:.0f}-{pa['dm'][1]:.0f} at 0.52 ms; {pb['dm'][0]:.0f}-{pb['dm'][1]:.0f} at 1.05 ms",
     f"{oa['dm'][0]:.0f}-{oa['dm'][1]:.0f}"],
    ["search sample", "1.049 ms", "0.524 ms (DM < 1600), 1.049 ms above", "0.262 ms"],
    ["channel width at the shift", "244 kHz (8 native summed)", "61 kHz (2 native summed)",
     "244 kHz (8 native summed)"],
    ["dedispersion", "two-stage, 4 sub-bands per chgroup (2.93 MHz)", "two-stage, 2 sub-bands per chgroup (5.86 MHz)",
     "one stage (filterbank)"],
    ["coarse DMs (stage 1)", f"{ca['n_coarse']}, every ~{ca['coarse_spacing']:.0f}",
     f"{pa['n_coarse']} every {pa['coarse_spacing']:.0f} + {pb['n_coarse']} every {pb['coarse_spacing']:.0f}", "-"],
    ["DM trials", f"{ca['n_fine']} (step {ca['fine_step'][0]:.1f}-{ca['fine_step'][1]:.1f})",
     f"{pa['n_fine']} (step {pa['fine_step'][0]:.1f}-{pa['fine_step'][1]:.1f}) + {pb['n_fine']} "
     f"(step {pb['fine_step'][0]:.1f}-{pb['fine_step'][1]:.1f})",
     f"{oa['n_fine']}, Levin tol 1.25 (step {oa['fine_step'][0]:.1f}-{oa['fine_step'][1]:.1f})"],
    ["integer delays", "stage 1: rint native, then rint 1.049 ms; stage 2: one rint at 1.049 ms",
     "stage 1: rint native, then rint 0.262 ms; stage 2: one rint at the tier's sample",
     "one rint per channel at 0.262 ms"],
    ["boxcars (powers of 2)", "1-64 samples (1-67 ms)", "1-64 samples (0.5-34 ms below DM 1600)",
     "1-256 samples (0.26-67 ms)"],
    ["corner turn", "~6.4 Gb/s (16-bit cells)", "34 Gb/s (4-bit, redundancy-aware)", "~130 Gb/s (two corner turns)"],
], [1.25 * inch, 1.8 * inch, 2.0 * inch, 1.85 * inch]))
A(Spacer(1, 4))
A(P("The 2x DM grid is the Barsdell recursion at tol 1.25 for 61 kHz channels; its coarse DMs keep the residual "
    "sweep across one sub-band within one sample, the rule the current plan follows at 1.05 ms. The single-shot "
    "trials are the legacy Levin generator verbatim (tol 1.25, 1.405 GHz, 768 x 0.244 MHz, 0.262 ms) from DM 80. "
    "With 0.52 ms samples the C1->C2 width cap (16 samples) must become 32 to keep today's ~17 ms limit. The 2x "
    "floor of DM 50 comes from the network plan; raising it to 100 changes nothing above 100.", SM))

hdr = P("Recovered S/N, mean over DM 100-1500 (fraction of ideal)", H2)
cols = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
rows = [["W (ms)"] + [f"{w:g}" for w in cols],
        ["current"] + [f3(c_m[J[w]]) for w in cols],
        ["proposed 2x"] + [f3(p_m[J[w]]) for w in cols],
        ["single-shot"] + [f3(o_m[J[w]]) for w in cols],
        ["current, 10th pct"] + [f3(c_10[J[w]]) for w in cols],
        ["2x, 10th pct"] + [f3(p_10[J[w]]) for w in cols],
        ["single-shot, 10th pct"] + [f3(o_10[J[w]]) for w in cols],
        ["<b>2x / current</b>"] + [f"<b>x{p_m[J[w]] / c_m[J[w]]:.2f}</b>" for w in cols],
        ["<b>single-shot / current</b>"] + [f"<b>x{o_m[J[w]] / c_m[J[w]]:.2f}</b>" for w in cols],
        ["2x at DM 1500-3000"] + [f3(p_hi[J[w]]) for w in cols]]
A(KeepTogether([hdr, table(rows, [1.45 * inch] + [0.6 * inch] * len(cols))]))
A(Spacer(1, 6))
A(P("By DM band (mean recovered S/N)", TCB))
rows = [["DM band", "current 0.2 ms", "2x 0.2 ms", "single 0.2 ms", "current 1 ms", "2x 1 ms", "single 1 ms"]]
for lo, hi in BANDS3:
    rows.append([f"{lo}-{hi}"] + [f3(bm(k, lo, hi, w)) for w in (0.2, 1.0)
                                  for k in ("current", "prop_q026", "single")])
A(table(rows, [0.95 * inch] + [0.99 * inch] * 6))

A(fig("s1_vs_dm.png", caption="Figure 1. Recovered S/N vs true DM at three widths (each point: best of the 8 "
      "nearest trials, averaged over 16 arrival phases). The fine structure is real: it is how each DM lands "
      "relative to the integer delays and the trial grid. Current and single-shot stop at ~1500; the 2x scheme "
      "drops a step at 1600 where its sample doubles. The single-shot decline with DM is its 244 kHz smear."))
A(fig("s2_vs_width.png", caption="Figure 2. Left: mean recovered S/N vs width over DM 100-1500 (band: 10th "
      "percentile to maximum over DM), and the 2x scheme above DM 1500. Right: gain over the current scheme per "
      "DM band (blues: 2x; greens: single-shot); dotted, the 2x with stage 1 rounded to 0.52 ms."))
A(fig("s3_cascade.png", caption="Figure 3. The loss chain at DM 500-1000, applied one step at a time: native and "
      "summed-channel DM smear; sampling at T with the boxcar ladder; a DM-trial grid (exact shifts at the "
      "nearest trials); the real integer delays."))

A(P("Loss factors at DM 500-1000", H2))
fs = {"current": fac("current"), "2x": fac("prop_q026"), "single-shot": fac("single")}
rows = [["W (ms)", "scheme", "channel smear", "sampling + boxcars", "DM-trial spacing", "integer delays", "total"]]
for w in (0.1, 0.2, 0.5, 1.0, 2.0, 4.0):
    for i, (lab, f) in enumerate(fs.items()):
        rows.append([f"{w:g}" if i == 0 else "", lab] + [f3(f[k][J[w]]) for k in range(5)])
A(table(rows, [0.6 * inch, 0.8 * inch, 1.0 * inch, 1.2 * inch, 1.15 * inch, 1.0 * inch, 0.7 * inch]))
A(Spacer(1, 4))
A(P("Each factor is the ratio to the previous step, so the product is the total. Single-shot and current share "
    "the 244 kHz smear column; single-shot's integer delays cost 0-3%, the two-stage schemes' 4-16% at W &lt;= 1 ms. The 2x "
    "sampling factor at 0.1 ms is lower than the current one's because today's 244 kHz smear already spreads a "
    "narrow burst over ~0.5 ms, while at 61 kHz it stays narrow and the 0.52 ms sample is what dilutes it.", SM))

A(P("Model and validation", H2))
story += bullets([
    "One model for all three: 6144 native channels at the production frequencies, each with its own DM smear; "
    "channels summed before shifting (csf 8 / 2) at the production band-centre frequency. Two-stage: stage 1 to "
    "each sub-band's reference at the coarse DM with the production double rounding, stage 2 as one rounding at "
    "the fine DM. Single-shot: one rint per 244 kHz channel at 0.262 ms (the rounding error peaks at exactly "
    "T/2 = 131 us, checked). The 2x tiers are the network plan's (plan6.py).",
    "<b>Integer delays are the production ones.</b> For the deployed plan the model's shifts equal "
    "DMPlan.delay_bins_per_subband + compute_stage2_shifts + compute_time_shift_search(merge_coarse_rounding, "
    "n_sub 4) for all 200 trials x 6144 channels (validate.py).",
    "<b>Agrees with the 2026-09-23 study once its simplification is applied</b>: that study's current-scheme "
    "number at 1 ms (0.798 of perfect dedispersion at 1.05 ms) is reproduced exactly (0.798) when a pulse "
    "narrower than one sample is treated as zero-width, as its code did (convolution only for W &gt; T). With the "
    "width kept, the right value is <b>0.758</b>. The earlier study therefore overstated the current scheme's "
    "1 ms recovery by ~5%; its scheme-vs-scheme comparisons share the simplification.",
    "Arrival phase is averaged over 16 sub-sample offsets; DMs are sampled every ~4 pc cm<super>-3</super> with "
    "jitter; the detector keeps the best of the 8 nearest trials. Boxcars are searched at every start.",
])

doc = SimpleDocTemplate("../DSA110_snr_current_vs_2x.pdf", pagesize=letter, leftMargin=0.7 * inch,
                        rightMargin=0.7 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                        title="DSA-110 S/N: current vs proposed 2x vs single-shot", author="timeres-study")
doc.build(story)
print("wrote ../DSA110_snr_current_vs_2x.pdf")
