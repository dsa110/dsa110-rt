"""Build ../DSA110_snr_current_vs_2x.pdf.  Run from docs/timeres_study/snr2x/ after
run_snr2x.py, summarize.py and figs_snr2x.py."""
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from schemes import T2, current, describe, proposed

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


cur, pro = describe(current()), describe(proposed())
ca, pa, pb = cur["tiers"][0], pro["tiers"][0], pro["tiers"][1]
c_m, p_m = band("current", 100, 1500).mean(0), band("prop_q026", 100, 1500).mean(0)
c_10, p_10 = np.percentile(band("current", 100, 1500), 10, 0), np.percentile(band("prop_q026", 100, 1500), 10, 0)
p_hi = band("prop_q026", 1500, 3000).mean(0)
q_m = band("prop_q052", 100, 1500).mean(0)
J = {w: int(np.argmin(np.abs(W - w))) for w in W}


def f3(x):
    return f"{x:.2f}"


story = []
A = story.append
A(P("DSA-110 search: S/N loss vs width and DM, current scheme vs the proposed 2x", H1))
A(P("2026-10-21, branch timeres-study (docs/timeres_study/snr2x/). Recovered S/N is given as a fraction of an "
    "<b>ideal matched filter</b>: a Gaussian pulse of FWHM W, perfectly dedispersed, continuously sampled, no "
    "smearing, white noise. Every loss from channel smearing, time sampling, the boxcar ladder, DM-trial spacing "
    "and the integer delays of the two-stage chain is included; losses common to both schemes (imaging, flagging, "
    "primary beam, sigma_k colour) are not.", SM))

A(P("Summary", H2))
story += bullets([
    f"<b>DM range.</b> Current: <b>{ca['dm'][0]:.0f}-{ca['dm'][1]:.0f}</b> pc cm<super>-3</super> "
    f"(nothing above 1500 or below 100). Proposed 2x: <b>{pa['dm'][0]:.0f}-{pb['dm'][1]:.0f}</b>, at 0.52 ms up "
    f"to DM 1600 and 1.05 ms above.",
    f"<b>Narrow bursts gain most.</b> Mean over DM 100-1500, the 2x scheme recovers "
    f"x{p_m[J[0.1]] / c_m[J[0.1]]:.2f} the S/N at 0.1 ms, x{p_m[J[0.5]] / c_m[J[0.5]]:.2f} at 0.5 ms, "
    f"<b>x{p_m[J[1.0]] / c_m[J[1.0]]:.2f} at 1 ms</b>, x{p_m[J[2.0]] / c_m[J[2.0]]:.2f} at 2 ms and "
    f"x{p_m[J[4.0]] / c_m[J[4.0]]:.2f} at 4 ms. Above ~8 ms the two are equal: both sit at the 0.94 ceiling of a "
    f"power-of-two boxcar on a Gaussian.",
    f"<b>A 1 ms burst</b> goes from {f3(c_m[J[1.0]])} to {f3(p_m[J[1.0]])} of ideal (10th percentile over DM "
    f"{f3(c_10[J[1.0]])} -> {f3(p_10[J[1.0]])}). Over DM 1500-3000, mostly searched at 1.05 ms, the 2x scheme "
    f"still recovers {f3(p_hi[J[1.0]])}, more than the current scheme's average in any DM band.",
    "<b>Where the gain comes from</b> depends on width (figure 3). At 0.2 ms it is mostly the channel width "
    "(244 -> 61 kHz cuts the DM smear 4x). At 1 ms it is mostly better dedispersion: finer DM trials and finer "
    "integer delays (coarse DMs every 48 instead of every 175 pc cm<super>-3</super>; stage-1 rounded to 0.26 ms "
    "instead of 1.05 ms).",
    f"<b>What 2x leaves on the table:</b> narrow bursts still lose about a third of their S/N to sampling at "
    f"0.52 ms ({f3(p_m[J[0.2]])} of ideal at 0.2 ms). That is what the 4x step addresses.",
    f"<b>Stage-1 quantum.</b> Rounding stage 1 to 0.52 ms instead of 0.26 ms (cheaper fast GEMM, t_int 16) "
    f"costs {100 * (1 - q_m[J[0.5]] / p_m[J[0.5]]):.0f}% at 0.5 ms and {100 * (1 - q_m[J[1.0]] / p_m[J[1.0]]):.0f}% "
    f"at 1 ms. Keep 0.26 ms: it is also the corr-side code path for 4x.",
])

A(P("The two schemes", H2))
A(table([
    ["", "current (deployed 2026-09-23)", "proposed 2x"],
    ["DM searched", f"{ca['dm'][0]:.0f}-{ca['dm'][1]:.0f}",
     f"{pa['dm'][0]:.0f}-{pa['dm'][1]:.0f} at 0.52 ms; {pb['dm'][0]:.0f}-{pb['dm'][1]:.0f} at 1.05 ms"],
    ["search sample", "1.049 ms", "0.524 ms (DM < 1600), 1.049 ms above"],
    ["channel width at the shift", "244 kHz (8 native channels summed first)", "61 kHz (2 summed)"],
    ["sub-bands per chgroup", "4 (2.93 MHz)", "2 (5.86 MHz)"],
    ["coarse DMs (stage 1, corr)", f"{ca['n_coarse']}, every ~{ca['coarse_spacing']:.0f}",
     f"{pa['n_coarse']} every {pa['coarse_spacing']:.0f} + {pb['n_coarse']} every {pb['coarse_spacing']:.0f}"],
    ["fine DM trials", f"{ca['n_fine']} (step {ca['fine_step'][0]:.1f}-{ca['fine_step'][1]:.1f})",
     f"{pa['n_fine']} (step {pa['fine_step'][0]:.1f}-{pa['fine_step'][1]:.1f}) + {pb['n_fine']} "
     f"(step {pb['fine_step'][0]:.1f}-{pb['fine_step'][1]:.1f})"],
    ["stage-1 integer delay", "rint to native, then rint to 1.049 ms", "rint to native, then rint to 0.262 ms"],
    ["stage-2 integer delay", "one rint at 1.049 ms (merged rounding)", "one rint at the tier's sample"],
    ["boxcars", "1-64 samples (1-67 ms)", "1-64 samples (0.5-34 ms below DM 1600)"],
    ["corner turn", "~6.4 Gb/s (16-bit cells)", "34 Gb/s (4-bit, redundancy-aware streams)"],
], [1.55 * inch, 2.45 * inch, 2.9 * inch]))
A(Spacer(1, 4))
A(P("The 2x DM grid is the Barsdell/legacy recursion at tol 1.25 for 61 kHz channels; coarse DMs are spaced so the "
    "residual sweep across one sub-band stays within one sample, the same rule the current plan follows at "
    "1.05 ms. With 0.52 ms samples the C1->C2 width cap (16 samples) must become 32 to keep today's ~17 ms "
    "limit. The 2x floor of DM 50 comes from the network plan; raising it back to 100 changes nothing above 100 "
    "(and keeps near-zero-DM RFI out).", SM))

A(P("Recovered S/N, mean over DM 100-1500 (fraction of ideal)", H2))
cols = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
rows = [["W (ms)"] + [f"{w:g}" for w in cols],
        ["current"] + [f3(c_m[J[w]]) for w in cols],
        ["current, 10th pct"] + [f3(c_10[J[w]]) for w in cols],
        ["proposed 2x"] + [f3(p_m[J[w]]) for w in cols],
        ["proposed 2x, 10th pct"] + [f3(p_10[J[w]]) for w in cols],
        ["<b>gain</b>"] + [f"<b>x{p_m[J[w]] / c_m[J[w]]:.2f}</b>" for w in cols],
        ["2x at DM 1500-3000"] + [f3(p_hi[J[w]]) for w in cols]]
A(table(rows, [1.45 * inch] + [0.6 * inch] * len(cols)))

A(fig("s1_vs_dm.png", caption="Figure 1. Recovered S/N vs true DM at three widths (each point: best of the 8 "
      "nearest trials, averaged over 16 arrival phases). The fine structure is real: it is how each DM lands "
      "relative to the integer delays and the trial grid. The current scheme stops at 1496; the 2x scheme "
      "drops a step at 1600 where its sample doubles."))
A(fig("s2_vs_width.png", caption="Figure 2. Left: mean recovered S/N vs width over DM 100-1500 (band: 10th "
      "percentile to maximum over DM), and the 2x scheme above DM 1500. Right: the gain of the 2x scheme per DM "
      "band; dotted, the same with stage 1 rounded to 0.52 ms."))
A(fig("s3_cascade.png", caption="Figure 3. The loss chain at DM 500-1000, applied one step at a time: native and "
      "summed-channel DM smear; sampling at T with the boxcar ladder; a DM-trial grid (exact shifts at the "
      "nearest trials); the real integer delays of the two-stage chain."))

A(P("Loss factors at DM 500-1000", H2))
fc, fp = fac("current"), fac("prop_q026")
rows = [["W (ms)", "scheme", "channel smear", "sampling + boxcars", "DM-trial spacing", "integer delays", "total"]]
for w in (0.1, 0.2, 0.5, 1.0, 2.0, 4.0):
    for lab, f in (("current", fc), ("2x", fp)):
        rows.append([f"{w:g}" if lab == "current" else "", lab] + [f3(f[k][J[w]]) for k in range(5)])
A(table(rows, [0.6 * inch, 0.75 * inch, 1.0 * inch, 1.2 * inch, 1.15 * inch, 1.05 * inch, 0.7 * inch]))
A(Spacer(1, 4))
A(P("Each factor is the ratio to the previous step, so the product is the total. The sampling factor of the 2x "
    "scheme is lower than the current one's at 0.1 ms: today's 244 kHz smear already spreads a narrow burst over "
    "~0.5 ms, while at 61 kHz the burst stays narrow and the 0.52 ms sample is what dilutes it.", SM))

A(P("Model and validation", H2))
story += bullets([
    "One model for both schemes: 6144 native channels at the production frequencies, each with its own DM smear; "
    "channels summed before shifting (csf 8 / 2) at the production band-centre frequency; stage 1 to each "
    "sub-band's reference at the coarse DM with the production double rounding; stage 2 as one rounding at the "
    "fine DM. The 2x tiers are the network plan's (plan6.py).",
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
                        title="DSA-110 S/N: current vs proposed 2x", author="timeres-study")
doc.build(story)
print("wrote ../DSA110_snr_current_vs_2x.pdf")
