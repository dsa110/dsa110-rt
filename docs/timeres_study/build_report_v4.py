"""Build DSA110_timeres_study_v4.pdf: 4x is the goal; benchmarks MEASURED in the 2026-10-21 gap.
Run from docs/timeres_study/ after model/plan6.py <H_s> and model/figs5.py <H_s>."""
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

INK, INK2 = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e")
RULE, HEAD = colors.HexColor("#d9d8d3"), colors.HexColor("#f0efec")
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=18,
                    spaceAfter=4, textColor=INK)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14,
                    spaceBefore=8, spaceAfter=4, textColor=INK)
B = ParagraphStyle("B", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.1, leading=12,
                   textColor=INK, spaceAfter=4)
BL = ParagraphStyle("BL", parent=B, leftIndent=11, bulletIndent=2, spaceAfter=2.5)
SM = ParagraphStyle("SM", parent=B, fontSize=7.8, leading=9.6, textColor=INK2)
CAP = ParagraphStyle("CAP", parent=SM, spaceBefore=2, spaceAfter=8)
TC = ParagraphStyle("TC", parent=B, fontSize=7.4, leading=9.0, spaceAfter=0)
TCB = ParagraphStyle("TCB", parent=TC, fontName="Helvetica-Bold")
FIG = "figs_v4/"

P6 = json.load(open("model/results/plan6.json"))
H_S, N_S = P6["H_s"], P6["n_search"]
RATES = json.load(open("model/results/rates4.json"))["pkgs"]
RED = json.load(open("model/results/redundancy.json"))
CORR = json.load(open("bench_results/fabric/b5b_corr_nic/b5b_summary.json"))
try:
    SRCH = json.load(open("bench_results/fabric/b5c_search_knee/b5b_summary.json"))
except OSError:
    SRCH = {}
FABS = list(P6["fabric"])


def row(amb, fov="FoV/2", k=1.0):
    rs = [r for r in P6["rows"] if r["amb"] == amb and r["fov"] == fov and r["kernels"] == k]
    order = FABS + ["none"]
    return min(rs, key=lambda r: (order.index(r["cheapest"]), r["ct_gbps"]))


def ct(amb, n_sub=2):
    r = [r for r in P6["rows"] if r["amb"] == amb and r["fov"] == "FoV/2" and r["kernels"] == 1.0
         and r["n_sub"] == n_sub]
    return f"{r[0]['ct_gbps']:.0f}"


PHRASE = {"baseline": "fits with today's hardware", "eno2 on search nodes": "needs eno2 on the 4 search nodes",
          "eno2 on all nodes": "needs eno2 on all 20 nodes", "stretch: card in x16 slot": "needs the slot move",
          "none": "does not fit in any option"}


def cheapest(amb, fov="FoV/2", k=1.0):
    return PHRASE[row(amb, fov, k)["cheapest"]]


def g2(key):
    g = RATES[key]["gain"]
    return f"x{g['m1']:.2f} / x{g['m03']:.2f}"


def P(t, st=B):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(t, BL, bulletText="•") for t in items]


def fig(name, width=6.9 * inch, caption=None):
    path = FIG + name
    w, h = ImageReader(path).getSize()
    out = [Image(path, width=width, height=width * h / w)]
    if caption:
        out.append(P(caption, CAP))
    return KeepTogether(out)


def table(rows, widths, extra=None):
    data = [[Paragraph(str(c), TCB if i == 0 else TC) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("BACKGROUND", (0, 0), (-1, 0), HEAD), ("TOPPADDING", (0, 0), (-1, -1), 2),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    t.setStyle(TableStyle(st + (extra or [])))
    return t


def footer(c, d):
    c.saveState()
    c.setFont("Helvetica", 7.5)
    c.setFillColor(INK2)
    c.drawString(0.7 * inch, 0.45 * inch, "DSA-110 real-time search: 4x time resolution, measured (study v4), "
                 "2026-10-21 - branch timeres-study")
    c.drawRightString(letter[0] - 0.7 * inch, 0.45 * inch, f"{d.page}")
    c.restoreState()


def deliv(S, node, key):
    v = S[node][key]
    return v["gbps_phy"], v["gbps_phy"] * (1 - v["discard_frac"]), v["discard_frac"]


def corr_line():
    out = []
    for n in ("n04", "n11", "n22"):
        pts = [deliv(CORR, n, k) for k in ("baseline", "g2", "g4", "g8")]
        out.append(f"{n}: offered " + " / ".join(f"{p[0]:.1f}" for p in pts) + " -> delivered "
                   + " / ".join(f"{p[1]:.1f}" for p in pts))
    return "; ".join(out)


def srch_line():
    if not SRCH:
        return "not run"
    out = []
    for n in sorted(SRCH):
        ks = [k for k in SRCH[n] if k != "baseline"]
        pts = [deliv(SRCH, n, k) for k in ks]
        out.append(f"{n}: " + ", ".join(f"{p[0]:.1f} -> {p[1]:.1f}" for p in pts))
    return ("offered -> delivered, Gb/s: " + "; ".join(out) + ". No NIC discards; median frame loss 1-2e-4 "
            "at 6/9/11 Gb/s, 3.7% at 13 (b5)")


def snap_disc():
    return ", ".join(f"{n} {100 * CORR[n]['baseline']['discard_frac']:.1f}% at "
                     f"{CORR[n]['baseline']['gbps_phy']:.1f} Gb/s" for n in ("n04", "n11", "n22"))


story = []
A = story.append

# ================================================================ page 1: summary
A(P("DSA-110 real-time FRB search: 4x time resolution - what the benchmarks say", H1))
A(P("Study v4, 2026-10-21. Supersedes v3. Goal (fixed): <b>0.26 ms time resolution through dedispersion, "
    "DM at least 1500</b>, RFI excision kept, a 2x FoV loss acceptable, <b>no reduction in detection threshold "
    "assumed</b>. Every number below that was an estimate in v3 is now measured on the nodes "
    "(GPU suite on n01 + n04, both GPUs; fabric on n04/n11/n22 and n01/n02/n09/n13) or derived from the "
    "real antenna positions.", SM))
A(P("Conclusions", H2))
story += bullets([
    "<b>Compute is not the limit.</b> An imaging owner costs <b>3.3 us per 256x128 dirty image</b> "
    "(combine + irfft2 + boxcar, measured kernels rescaled to the n_sub 2 stream shape) - up to 79 fine DMs "
    "per GPU at 0.26 ms. The corr-side 4x fast path (t_int 8 GEMM 23.4 ms + 61 kHz stage-1 for 31 coarse DMs "
    "35.7 ms per 134 ms block) fits on corr GPU 1, so today's slow/fast GPU split can stay.",
    f"<b>The network is the limit, and it is tighter than v3 assumed.</b> Every ConnectX-4 sits in an x4 slot. "
    f"On a corr node it <b>delivers at most ~17 Gb/s to the host whatever is offered</b>, and SNAP already "
    f"supplies 16-18 Gb/s, so <b>corr nodes have zero ingress headroom</b> (v3 hoped for 2-4): corner-turn "
    f"traffic only displaces SNAP packets. The search nodes took <b>{H_S:.0f} Gb/s each with no loss</b> "
    f"(b5c) and lost 3.7% of frames at 13. So the whole corner turn must fit in {N_S} x {H_S:.0f} = "
    f"{N_S * H_S:.0f} Gb/s.",
    f"<b>The two-stage corner turn grows as 1/T<super>2</super></b> (coarse DMs scale as 1/T, and so does the "
    f"stream rate). At FoV/2 with 4-bit cells: 2x {ct('2x')} Gb/s, 4x below DM 500 {ct('4x below DM 500')}, "
    f"below DM 800 {ct('4x below DM 800')}, to DM 1600 {ct('4x to DM 1600')} Gb/s (n_sub 2; n_sub 4 is "
    f"~10% less network for ~10% more owner GPU).",
    "<b>Redundant-baseline averaging</b> (your suggestion) is already done implicitly: gridding merges the "
    f"{RED['n_baselines']} core baselines into {RED['cells_per_channel_full']} cells per channel, which equals "
    f"the {RED['unique_tol0.05m']} exactly-unique baselines. What gridding does not remove is the union of "
    "cells over a chgroup's channels. Shipping one redundancy-averaged visibility per unique baseline per "
    "sub-band (or halving the FoV along u) removes it and <b>cuts the corner turn 17-29%</b>. This is folded "
    "into every number here. It helps, but it does not close the gap on its own.",
    f"<b>Verdict.</b> With today's hardware, <b>2x to DM 1600 fits</b> ({ct('2x')} of {N_S * H_S:.0f} Gb/s) "
    f"and no 4x variant does. The cheapest way to 4x is <b>not</b> moving the NIC but <b>cabling the free "
    f"on-board 10G port (eno2, Intel X550T on its own PCIe link)</b>. On the 4 search nodes alone it gives "
    f"<b>4x below DM 500</b>, or 4x below DM 800 once the owner kernels are fused. On all 20 nodes it gives "
    f"<b>4x to DM 1600 at FoV/2</b>, the goal as stated, with the corr GPU 0s as extra imaging owners. "
    f"Re-seating the ConnectX in an x16 slot remains the stretch: it buys margin and full FoV at the "
    f"lower DMs, but full FoV at 4x to DM 1600 is GPU-bound even then. If bigger datagrams lift the search "
    f"ingress to the ~16 Gb/s seen in September, 4x below DM 500 fits without any hardware change.",
    "<b>4x is a true 4x.</b> 0.26 ms sampling with 61 kHz channels improves the effective resolution after "
    "dedispersion by <b>exactly 4.0x at every DM</b> versus today's 1.05 ms / 244 kHz, because sampling and "
    "channel smear shrink together: 1110 -> 278 us at DM 500, 1516 -> 379 us at DM 1500.",
    f"<b>Rate gain from resolution alone</b> (Euclidean, fixed 9.55 sigma, DM log-normal median 500, widths "
    f"median 1 ms / 0.3 ms): 2x {g2('2x: 0.52 ms below DM 1600')}, 4x below DM 500 {g2('4x below DM 500')}, "
    f"4x below DM 800 {g2('4x below DM 800, tol 1.4')}, 4x to DM 1600 {g2('4x to DM 1600, tol 1.4')}.",
    "<b>RFI: a detector-side spectral test should ship with 4x.</b> 4x raises the S/N of impulsive RFI as much as "
    "that of FRBs. Owners already hold 32 sub-band streams per coarse DM, and a candidate pixel is the sum of "
    "their contributions, so every candidate gets a free 32-point on-pulse spectrum. In the model, ~1 MHz "
    "impulses put 99-100% of S/N<super>2</super> in two adjacent sub-bands; no broadband, 60 MHz band-limited or "
    "fully scintillated burst exceeds 0.9, and only 0.3% of 20 MHz-wide ones do. This matches the archive (the 1335 MHz fakes put 85-92% of S/N<super>2</super> in 0.8 MHz). It rejects them "
    "before C2 and the voltage dumps, and can go in today (n_sub 4 = 64 sub-bands).",
    "<b>Voltage zoom is dropped</b>, per your concern that RFI storms would swamp dumps, transfer and disks. "
    "Measured too: a naive coherent re-beamform costs ~5 s of GPU per candidate.",
])
A(P("Needs a decision / action", H2))
story += bullets([
    "<b>eno2 cabling</b>: is a 10GBASE-T path into the data network possible? Two options: a 24-port 10GBASE-T "
    "switch uplinked to the Mellanox, or RJ45 SFP+ modules on QSFP breakout ports of the existing switch. "
    "This decides 4x without opening any node.",
    "<b>Search ingress with 8900 B datagrams</b> (single fragment; the cap is committed on a side branch, "
    "f751bf5): re-run b5c in the next gap. 16 Gb/s per search node would put 4x below DM 500 on today's "
    "hardware.",
    f"<b>n01 corner-turn ingress</b>: n01 delivered only 1.6 Gb/s of 13 offered (88% loss) in b5, a separate "
    f"fault (no other node shows it). All plans here assume {N_S} working search nodes.",
    f"<b>SNAP loss on corr NICs (check in production)</b>: with the pipeline stopped, n04/n11 discarded "
    f"{snap_disc()} of SNAP packets at the NIC. Lifetime counters on n04 average 0.12%, so check the capture "
    f"loss counters once observing resumes. If it is real, the x4 slot is already costing data today.",
    "<b>4-bit corner turn</b>: every volume here is cint4 (production sends 16 bit per cell). Modelled cost ~1% "
    "S/N; it halves the network and is required for any 4x.",
])
A(Spacer(1, 6))

# ================================================================ measurements
A(P("1. What was measured in the gap", H2))
A(table([
    ["bench", "measured (2080 Ti, prototype kernels)", "what it decides"],
    ["b1 irfft2", "256x256 fp16 2.5 us/image; 256x128 1.4 us", "imaging is cheap"],
    ["b2 detector", "chunk-max boxcar 1.8 / 0.94 us/image (K7, 256x256 / 256x128) vs 3.7 / 1.8 for a "
     "production-style kernel", "the detector rewrite halves its cost"],
    ["b3 owner chain", "256x128: 4.78 us/image with 128 streams x 1900 cells = 54 fine DMs/GPU; "
     "3.3 us rescaled to n_sub 2 streams; 256x256: 7.6-8.3 us", "owner capacity (fig. 5)"],
    ["b4 X-engine", "slow GEMM K4096 8.9 ms/block; fast t_int 8 23.4 ms; int8 = fp16 (no gain); PCIe H2D 12.3, "
     "D2H 13.2 GB/s", "4x fast path fits on GPU 1"],
    ["b7 stage-1", "31 coarse DMs x 2 sub-bands at 61 kHz 35.7 ms/block (122 kHz: 17.6)", "corr-side 4x fits"],
    ["b6 zoom", "~5 s per candidate (naive beamform + dedisperse)", "dropped"],
    ["b5 search fabric", "13 Gb/s offered -> 12.3 delivered, 3.7% frame loss (n02/n09/n13); 20 -> 4.7; "
     "n01 1.6 at any rate", "search ingress"],
    ["b5b corr NIC", corr_line(), "corr headroom = 0"],
    ["b5c search knee", srch_line(), f"search ingress ~{H_S:.0f} Gb/s"],
], [0.95 * inch, 4.3 * inch, 1.65 * inch]))
A(Spacer(1, 6))
A(P("GPU-1 rows from the first suite run were invalid (no torch.cuda.set_device; fixed, rerun spot-checks "
    "match GPU 0). Corr nodes' _recv_epoll extension is a stale py3.8 build, so corr receive was measured at "
    "the NIC (rx_bytes_phy, rx_discards_phy), not by the receiver. rx_out_of_buffer and UDP receive-buffer "
    "errors stayed at zero throughout, so the loss is PCIe back-pressure at the card, not the host.", SM))
A(fig("f1_fabric.png", caption="Figure 1. Delivered vs offered ingress, measured. Corr nodes start at their "
      "SNAP load (16-18 Gb/s) and cannot go higher: the extra is discarded at the card. Search nodes deliver "
      "everything up to 11 Gb/s with no NIC discards; at 13 Gb/s (b5) 3.7% of frames are lost, above the "
      "card, with this harness's 5 kB two-fragment frames."))
A(PageBreak())

# ================================================================ network
A(P("2. Why 4x is a network problem", H2))
A(P("Stage 1 on the corr nodes dedisperses each chgroup to a set of coarse DMs; the owners combine the 16 "
    "chgroups at fine DMs. The coarse DMs must be close enough that the residual sweep across one sub-band "
    "stays under one sample, so their number scales as 1/T, and each stream's rate scales as 1/T too. "
    "Going from 1.05 to 0.26 ms therefore multiplies the corner turn by up to 16 over the 4x range. "
    "One-shot dedispersion (two corner turns) avoids the 1/T coarse-DM growth but ships every channel's cells "
    "(164 Gb/s at 61 kHz), which is worse here.", B))
A(fig("f3_network.png", caption=f"Figure 2. Corner-turn volume per ambition (FoV/2, cint4, redundancy-aware "
      f"streams, DM grid tol 1.4) against the ingress each fabric option provides ({N_S} search nodes at "
      f"{H_S:.0f} Gb/s; eno2 adds 9.4 Gb/s per node; the stretch doubles the x4 ceiling)."))
A(fig("f4_matrix.png", caption="Figure 3. Best configuration that fits each ambition in each fabric option "
      "(network AND GPU: 8 search GPUs at 0.8 load, 16 corr GPUs at 0.4 beside corr_slow). 'Optimised "
      "kernels' = 0.7x the measured owner cost (fused combine + irfft2 + detector; not yet written)."))
A(PageBreak())

# ================================================================ redundancy
A(P("3. Redundant-baseline averaging", H2))
A(P(f"The core array has {RED['n_baselines']} baselines but only {RED['unique_tol0.05m']} distinct ones "
    f"(positions agreeing to 5 cm), because the arms are regularly spaced. Gridding at the full-FoV cell "
    f"(17.9 lambda) lands them in {RED['cells_per_channel_full']} cells per channel, so <b>today's sparse "
    f"gridded corner turn already averages redundant baselines</b>. The remaining inefficiency is "
    f"frequency: a stream sums a chgroup's channels, and a baseline's cell moves with frequency, so a "
    f"stream carries the union of cells over its channels: {RED['full nsub2']:.0f} at n_sub 2 and "
    f"{RED['full nsub4']:.0f} at n_sub 4 (full FoV).", B))
story += bullets([
    "<b>Per-unique-baseline streams</b>: send one redundancy-averaged visibility per distinct baseline per "
    f"sub-band ({RED['unique_tol0.05m']} entries at any n_sub) and grid at the owner, where the combine kernel "
    "already scatters by a per-stream cell table. The price is bandwidth smearing within a sub-band: the "
    "longest core baseline (483 m) moves 9.4 lambda across a 5.9 MHz sub-band (4.7 at n_sub 4), against the "
    "+/-9 lambda that nearest-cell gridding already accepts. Saves 29% (n_sub 2) / 16% (n_sub 4) at full FoV.",
    "<b>Quasi-redundant averaging = the FoV/2 we are allowed</b>: making the 256x128 image by doubling the uv "
    f"cell along u merges neighbouring EW spacings: {RED['FoV/2 halve u nsub2']:.0f} cells per stream at "
    f"n_sub 2 (vs {RED['FoV/2 sym nsub2']:.0f} for the symmetric FoV/2 assumed in v3), "
    f"{RED['FoV/2 halve u nsub4']:.0f} at n_sub 4.",
    "Not useful: averaging beyond one cell (that is a further FoV cut), or the low rank of a single-channel "
    "0.26 ms visibility matrix (16 voltage snapshots x 82 antennas): the channel sum inside a sub-band "
    "raises the rank to 16 x channels, so nothing survives to the stream.",
])
A(fig("f2_redundancy.png", width=6.6 * inch, caption="Figure 4. Complex entries per (corr node, sub-band) "
      "stream per sample. Grey/light: what gridding already achieves; orange: today's stream shapes; blue: "
      "the redundancy-aware streams used in this study."))
A(PageBreak())

# ================================================================ compute
A(P("4. Compute: both sides fit", H2))
A(fig("m2_corr_budget.png", caption="Figure 5. Corr GPUs at 4x (per 134 ms block). GPU 1 runs the fast path at "
      "t_int 8 with 61 kHz stage-1 for 31 coarse DMs x 2 sub-bands; GPU 0 keeps corr_slow and has ~0.4 spare "
      "for imaging - usable only if the data can reach it (eno2)."))
A(fig("f5_owner.png", caption="Figure 6. Owner cost per dirty image. The combine scales with streams x cells, "
      "so the n_sub 2 redundancy-aware streams cost 4.3x less to combine than the benchmark shape."))
A(PageBreak())

# ================================================================ rates + RFI
A(P("5. What 4x buys (no threshold change)", H2))
A(fig("m4_rates.png", caption="Figure 7. Detection-rate gain vs today from time resolution alone, at the fixed "
      "9.55 sigma floor. More trials do not force a higher threshold: the false-alarm probability above 9.55 "
      "sigma stays negligible for 16x the trials (the near-threshold population is RFI, not noise)."))
A(P("6. Detector-side narrow-band rejection", H2))
A(P("Higher time resolution boosts short RFI impulses as much as FRBs, so 4x will raise the candidate rate "
    "unless the detector can recognise narrow-band signals. The DM-time plane alone is not enough: a narrow-band "
    "impulse dedisperses to a clean line with a peaked DM profile (seen on the 1335 MHz events). The spectrum "
    "is what separates them, and the owners already have it.", B))
A(fig("m5_spectral_test.png", caption="Figure 8. Model (Monte Carlo, 32 sub-bands of 5.86 MHz): fraction of "
      "S/N<super>2</super> in the best two adjacent sub-bands vs candidate S/N, median and 1-99% band. A 17 MHz "
      "interferer and a 20 MHz-wide burst look alike to this test, so the 1484/1494 MHz storms still need the "
      "flagger. Validate on injections, the 16 archived narrow-band events and the 9 DEC +16 real ones before "
      "arming."))
story += bullets([
    "<b>Per-candidate sub-band spectrum (recommended, ~free).</b> For each candidate (pixel, fine DM, t, width), "
    "DFT each of the 32 (n_sub 2) or 64 (n_sub 4, today) streams to that pixel over the boxcar window. The "
    "values sum exactly to the detected pixel. Statistic: the fraction of S/N<super>2</super> in the best two "
    "adjacent sub-bands, robust to an interferer on a chgroup edge (1335.02 MHz is 0.33 MHz from one). "
    "Reject above 0.9 and tag 0.5-0.9 (figure 8): a 0.5 cut would also lose 5-6% of fully scintillated "
    "bursts and every burst 20 MHz wide or narrower. Cost is 32 x ~1800 complex MACs per sample per candidate, on data already "
    "in GPU memory.",
    "<b>Dispersion check from the same spectrum</b>: for bright candidates, per-sub-band arrival times must "
    "follow the cold-plasma sweep. A narrow-band signal has no lever arm, so it fails by construction.",
    "<b>DM extent at C1/C2 (cheap, to validate)</b>: a signal of bandwidth B stays above threshold over "
    "dDM ~ (S/thr)<super>2</super> x w f<super>3</super> / (8.3 us B) - ~290 pc/cc per ms of width for 1 MHz vs "
    "~2 pc/cc for a 187 MHz burst. Generalises the existing DM-comb veto, but replay it on archived clusters "
    "first: the fine-trial count alone misled an earlier analysis.",
    "Keep: per-DM-row normalisation (persistent stripes), the conditional flagger for the 1484/1494 MHz "
    "storms, and the WideEscapeGovernor. Together with the spectral test, they protect dumps, transfer and disks "
    "during storms without lowering any threshold.",
])
A(Spacer(1, 6))

# ================================================================ roadmap
A(P("7. Roadmap to 4x", H2))
A(table([
    ["step", "change", "needs", "gives"],
    ["0 (now)", "owner-side spectral test; fix n01 ingress; max_frag_payload 8900; check SNAP NIC discards",
     "software", "storm-proof candidates today"],
    ["1", "two-stage done right: native-precision stage-1, 61 kHz, streams at tier cadence; cint4 + "
     "redundancy-aware streams; chunk-max detector", "software", f"2x to DM 1600 ({g2('2x: 0.52 ms below DM 1600')})"],
    ["2", "eno2 cabled on the 4 search nodes; 256x128 imaging", "4 copper 10G links",
     f"4x below DM 500 ({g2('4x below DM 500')})"],
    ["3", "fused owner kernels (combine + irfft2 + detector, ~0.7x)", "software",
     f"4x below DM 800 ({g2('4x below DM 800, tol 1.4')})"],
    ["4", "eno2 on all 20 nodes; corr GPU 0 becomes an imaging owner", "10GBASE-T path",
     f"<b>4x to DM 1600</b> ({g2('4x to DM 1600, tol 1.4')})"],
    ["stretch", "ConnectX re-seated in an x16 slot", "open every node",
     "margin only: same reach as step 4 in this model"],
], [0.6 * inch, 3.2 * inch, 1.55 * inch, 1.55 * inch]))
A(Spacer(1, 6))
A(P("Every step is useful on its own. Steps 0-1 use only the hardware we have; steps 2-3 need four 10G "
    "links. Step 4 meets the goal as stated (0.26 ms to DM 1600, 4.0x effective resolution at every DM) "
    "with 20 copper 10G links and no slot move.", B))
A(P("Caveats", H2))
story += bullets([
    "Owner and stage-1 costs are prototype kernels on isolated GPUs; a production owner also streams, "
    "decodes and publishes. The 0.8 / 0.4 GPU load caps are the margin for that.",
    "The 11 Gb/s search ingress was measured with the corner-turn harness's 5 kB two-fragment frames over 90 "
    "flows; the loss at 13 Gb/s was above the card. Single-fragment 9 kB datagrams ran clean at 16 Gb/s in "
    "September, and RX ring sizes (1024 of 8192) are another untested lever. The corr-node ~17 Gb/s ceiling "
    "is at the card and will not move without the slot change.",
    "eno2 throughput (9.4 Gb/s) is the line rate minus overhead, not measured: the ports are not cabled.",
    "Rate gains assume Euclidean counts and the v3 width/DM populations; they do not include any change in "
    "RFI false positives.",
])

doc = SimpleDocTemplate("DSA110_timeres_study_v4.pdf", pagesize=letter, leftMargin=0.7 * inch,
                        rightMargin=0.7 * inch, topMargin=0.6 * inch, bottomMargin=0.7 * inch,
                        title="DSA-110 4x time resolution study v4", author="timeres-study")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote DSA110_timeres_study_v4.pdf")
