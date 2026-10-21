"""Build DSA110_timeres_study_v2.pdf (reportlab).  Run from docs/timeres_study/."""
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
TC = ParagraphStyle("TC", parent=B, fontSize=7.6, leading=9.2, spaceAfter=0)
TCB = ParagraphStyle("TCB", parent=TC, fontName="Helvetica-Bold")

R = json.load(open("model/results/rates.json"))
RG = R["rate_gain"]
F = R["f_at_100us"]
DMS = R["dms"]
i5, i15, i12 = DMS.index(500.0), DMS.index(1500.0), DMS.index(1200.0)


def P(t, st=B):
    return Paragraph(t, st)


def bullets(items, st=BL):
    return [Paragraph(t, st, bulletText="•") for t in items]


def fig(path, width=6.9 * inch, caption=None):
    w, h = ImageReader(path).getSize()
    out = [Image(path, width=width, height=width * h / w)]
    if caption:
        out.append(P(caption, CAP))
    return KeepTogether(out)


def table(rows, widths):
    data = [[Paragraph(str(c), TCB if i == 0 else TC) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("BACKGROUND", (0, 0), (-1, 0), HEAD), ("TOPPADDING", (0, 0), (-1, -1), 2),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def footer(c, d):
    c.saveState()
    c.setFont("Helvetica", 7.5)
    c.setFillColor(INK2)
    c.drawString(0.7 * inch, 0.45 * inch, "DSA-110 real-time search: time-resolution study v2 (revised after "
                 "review), 2026-10-21 - branch timeres-study")
    c.drawRightString(letter[0] - 0.7 * inch, 0.45 * inch, f"{d.page}")
    c.restoreState()


def g(name, pop):
    return f"x{RG[name][pop]:.2f}"


m1, m3 = "median 1 ms", "median 0.3 ms"
S = []
# ============================================================== summary
S += [P("Improving the time resolution of the DSA-110 real-time FRB search - v2", H1),
      P("Revised after review (2026-10-21). Everything is modelled on h23 (numpy emulator of the full "
        "dedispersion chain on 6144 native channels, cost and network models built on measured numbers); "
        "no new GPU code was run. Benchmarks for the next observing gap are in "
        "<font face='Courier'>bench/timeres/</font>; models in <font face='Courier'>docs/timeres_study/model/</font>.",
        SM),
      P("What changed from v1", H2)]
S += bullets([
    "<b>Channel smear at typical DMs is small.</b> At DM 500 one native 30.5 kHz channel smears by 38-56 us "
    "(46 us at band centre) - your estimate is right. v1 over-stated the need for native channels: 61-122 kHz "
    "is enough below DM ~1000; channel width matters mainly at high DM.",
    "<b>One-shot vs two-stage, and your two-corner-turn scheme, re-examined.</b> At matched shift precision "
    "and channel width they recover the same S/N. The two-corner-turn scheme is network-feasible (it is the "
    "single-copy fan-out I wrongly dismissed) and has real advantages; it costs ~8 Gb/s more per node.",
    "<b>NIC facts corrected.</b> Each node has ONE dual-port ConnectX-4 in an electrically x4 slot (both 40G "
    "ports share ~25 Gb/s), plus a dual-port 10G X550T. Four of the six CPU x16 root ports have no device, so "
    "the ConnectX-4 can probably be moved to an x16 slot (x8 card: ~50 Gb/s, second port useful) - to be "
    "confirmed physically, since the double-width GPUs may block neighbouring slots.",
    "<b>New options:</b> DM-tiered cadence (halves 4x imaging: ~14x today's images instead of v1's ~20-36x) "
    "and a voltage-ring 'zoom' on candidates. v1's rate numbers were not DM-weighted; v2's are.",
])
S += [P("Summary", H2)]
S += bullets([
    "<b>Today</b> a 0.1 ms burst delivers %.2f of an ideal matched filter at DM 500 and %.2f at DM 1200. At "
    "DM 500 the effective resolution (~1.2 ms) is set by the 1.05 ms sample plus ~0.5 ms of residual "
    "two-stage smear; the 244 kHz channels (0.37 ms) matter less there, but dominate by DM 1500 (1.1 ms)."
    % (F["today"][i5], F["today"][i12]),
    "<b>Target cadence, DM-tiered:</b> 0.26 ms below DM 800, 0.52 ms to DM 1600, 1.05 ms above, with 61 kHz "
    "channels (122 kHz loses &lt;=7%% up to DM ~500, ~10%% at DM 800, ~20%% at DM 1500). This keeps 95-97%% of uniform-4x's rate gain at about half "
    "its imaging. A 0.1 ms burst at DM 500 then delivers %.2f (2x: %.2f; 1x fixed: %.2f)."
    % (F["ts4_c2"][i5], F["ts2_c2"][i5], F["tc1_c2"][i5]),
    "<b>Rate gains</b> (Euclidean, DM median 500, DM &lt;=1500; width populations with median 1 ms / 0.3 ms): "
    "1x fixed %s/%s, 2x %s/%s, 4x tiers %s/%s. 2x captures most of the 4x gain for the median-1 ms "
    "population; 4x adds ~16%% for the narrow one."
    % (g("1x fixed (two CTs)", m1), g("1x fixed (two CTs)", m3), g("2x tiers (2-stage)", m1),
       g("2x tiers (2-stage)", m3), g("4x tiers (2-stage, 61 kHz)", m1), g("4x tiers (2-stage, 61 kHz)", m3)),
    "<b>Dedispersion:</b> what matters is (a) shift precision at least as fine as the search sample, (b) "
    "channels no wider than the sample at that DM, (c) for two-stage, enough coarse-DM x sub-band products. "
    "Today's chain fails all three. One-shot and a properly sized two-stage then agree to ~2%.",
    "<b>Your two-corner-turn architecture is the recommended end state</b> if the NIC can be re-seated (or "
    "eno2 10G carries the first hop): the corr side is fixed at final resolution from day one (grid every "
    "61 kHz channel at 0.26 ms, 4-bit, send); the DM plan and cadence live entirely downstream; shift "
    "precision comes from the first hop, so even at today's 1.05 ms it is worth %s/%s; losing a node costs a "
    "little sensitivity everywhere instead of a DM gap. A two-stage design done right reaches the same S/N "
    "within today's network if the NIC cannot move." % (g("1x fixed (two CTs)", m1), g("1x fixed (two CTs)", m3)),
    "<b>Compute:</b> with rewritten owner kernels on 24 owners (16 corr second GPUs + 8 search GPUs), 4x tiers "
    "peaks at ~0.4 (FoV/2) - 0.7 (full FoV) of a GPU per owner. 2x fits on the 8 search GPUs alone at FoV/2.",
    "<b>Voltage zoom:</b> coherent beams from the 25 s voltage ring around each candidate give 0.92 of ideal at "
    "any DM and width (native 32.8 us, no channel smear) for ~ms of GPU per candidate. Detection is still set by "
    "the search, but cheap confirmation makes a lower first-stage threshold affordable: 9.55 -&gt; 8 sigma is "
    "worth x1.30 at any cadence (4x tiers + zoom: %s/%s) - if the candidate rate at 8 sigma allows; measure it "
    "first." % (g("4x tiers + zoom", m1), g("4x tiers + zoom", m3)),
])
S += [PageBreak()]

# ============================================================== 1 limits / resolution
S += [P("1. What sets the time resolution", H2),
      P("Effective resolution is roughly the quadrature sum of the search sample, the channel smear "
        "8.3 us DM dnu/nu^3, and any residual of the dedispersion chain. Fig. 1 shows it against the FRB DM "
        "distribution. Today's chain carries three avoidable terms: 1.05 ms samples; 244 kHz channels summed "
        "inside the fast GEMM before any shift; and an 8 coarse-DM x 4 sub-band chain whose stage-1 shifts are "
        "rounded to 1.05 ms (rms residual ~0.5 ms). Native channel smear (dashed) only rivals a 0.26 ms sample "
        "above DM ~2000."),
      fig("figs_v2/r1_resolution.png", 6.3 * inch,
          "Fig. 1 - Effective time resolution vs DM (rms channel smear across the band). Shaded: 90% of the "
          "assumed log-normal FRB DM distribution (median 500)."),
      table([["channel width", "smear at DM 500 (top / centre / bottom)", "at DM 1000", "at DM 1500"],
             ["30.5 kHz (native)", "38 / 46 / 56 us", "91 us (centre)", "137 us"],
             ["61 kHz", "75 / 91 / 112 us", "183 us", "274 us"],
             ["122 kHz", "150 / 183 / 225 us", "365 us", "548 us"],
             ["244 kHz (today)", "301 / 365 / 449 us", "730 us", "1096 us"]],
            [1.3 * inch, 2.4 * inch, 1.4 * inch, 1.4 * inch]),
      P("Table 1 - Intra-channel dispersion smear.", CAP),
      PageBreak()]

# ============================================================== 2 what it buys
S += [P("2. What finer resolution buys", H2),
      fig("figs_v2/r2_snr_vs_width.png", 6.5 * inch,
          "Fig. 2 - Recovered S/N (fraction of an ideal matched filter) vs intrinsic width at DM 500 and 1500. "
          "Emulated end to end: native-channel smear, channel summing, shift rounding, the real DM grids, random "
          "sample phase and DM, power-of-two boxcars."),
      table([["package", "rate gain, median 1 ms", "median 0.3 ms", "comment"],
             ["1x fixed, two-stage done right", g("1x fixed (2-stage)", m1), g("1x fixed (2-stage)", m3),
              "native stage-1, 32x8, shifts at 1.05 ms"],
             ["1x fixed, two corner turns", g("1x fixed (two CTs)", m1), g("1x fixed (two CTs)", m3),
              "shifts at 0.26 ms from the first hop"],
             ["2x (0.52 ms below DM 1600)", g("2x tiers (2-stage)", m1), g("2x tiers (2-stage)", m3),
              "two-stage and two corner turns identical"],
             ["4x tiers (0.26 ms below DM 800)", g("4x tiers (2-stage, 61 kHz)", m1),
              g("4x tiers (2-stage, 61 kHz)", m3), "122 kHz in the low tier: same within 1%"],
             ["4x uniform to DM 3000", g("4x uniform (2-stage, 61 kHz)", m1),
              g("4x uniform (2-stage, 61 kHz)", m3), "~2x the imaging of 4x tiers"],
             ["+ voltage zoom, threshold 9.55 -&gt; 8", "x1.30 on top", "x1.30 on top", "if the 8-sigma "
              "candidate rate is manageable"]],
            [2.1 * inch, 1.2 * inch, 1.0 * inch, 2.5 * inch]),
      P("Table 2 - Detection-rate gain over today (Euclidean counts, rate ~ S/N^1.5 at fixed threshold), "
        "averaged over a log-normal DM distribution (median 500, sigma_ln 0.6) for DM &lt;=1500. Extending the "
        "search to DM 3000 adds the ~3.4% of FRBs above DM 1500. The ~20x larger search volume at 4x needs ~+0.3 "
        "sigma of threshold (~4% of rate) for the same false-alarm rate.", CAP),
      PageBreak()]

# ============================================================== 3 one-shot vs two-stage
S += [P("3. One-shot vs two-stage dedispersion, and two corner turns", H2),
      P("For the same input data a brute-force one-shot dedisperser is the reference; a sub-band (two-stage) "
        "tree can only match it. Your intuition that one-shot beats today's chain is right: today stage-1 "
        "shifts are rounded to 1.05 ms after summing 244 kHz channels, and 8 x 4 coarse-DM x sub-band products "
        "leave ~0.5 ms of residual smear. When the two-stage chain is sized properly (stage-1 at native 32.8 us "
        "precision, coarse x sub-band product &gt;= ~120 for DM &lt;=1500 at 0.26 ms), the two agree to ~2% "
        "(Fig. 3). The S/N is set by the shift precision and the channel width, not by the number of stages."),
      fig("figs_v2/r3_oneshot_vs_twostage.png", 6.7 * inch,
          "Fig. 3 - Recovered S/N for a 0.1 ms burst. Within each cadence group, schemes with the same shift "
          "precision agree; finer shift precision (0.26 ms at a 1.05 ms cadence, 0.13 ms at 0.26 ms) is worth "
          "5-12%."),
      P("<b>Where to do the one-shot.</b> A DM-partitioned one-shot needs every DM owner to ingest the whole "
        "band: 20 Gb/s at today's cadence and 244 kHz (multicast), 80-160 Gb/s at 4x - impossible on these NICs. "
        "Your two corner turns avoid this: the first hop sends each 61 kHz channel's ~1750 uv cells ONCE, "
        "partitioned by uv cell (~8 Gb/s per receiving node at 0.26 ms, 4-bit); each cell owner dedisperses "
        "its cells for all DMs (FDMT or brute force over the ~30% of channels that hit each cell; cost ~ today's "
        "combine per image); the second hop gathers the ~6000-cell dedispersed planes by DM group to the imaging "
        "owners - per owner ~5 Gb/s at 4x tiers, comparable to the two-stage corner turn (~3.7 Gb/s). The "
        "extra cost is the first hop, ~8 Gb/s per node."),
      PageBreak()]

# ============================================================== 4 architectures + network
S += [P("4. Architectures and network", H2),
      fig("figs_v2/r4_architectures.png", 6.0 * inch,
          "Fig. 4 - (A) two-stage done right, (B) two corner turns, (C) voltage zoom. The X-engine on corr GPU A "
          "produces the raw all-antenna slow visibilities and the fast product from one voltage copy, freeing "
          "corr GPU B (sec. 5)."),
      PageBreak(),
      fig("figs_v2/r5_network.png", 6.6 * inch,
          "Fig. 5 - Corr-node NIC ingress (corr nodes are the binding case: they also take ~18 Gb/s of SNAP "
          "data). Search nodes: two-stage 1.9-7.4 Gb/s; two corner turns 10-19 Gb/s."),
      table([["fact (checked on all 20 nodes, read-only sysfs)", "consequence"],
             ["one ConnectX-4 (MT27700) per node: PCI 81:00.0 and 81:00.1 are the two ports of ONE card",
              "the 'second NIC' on the 40G fabric is the second port of the same card"],
             ["link x4 of x8 at 8 GT/s; its root port 80:00.0 is itself x4 (an electrically x4 slot)",
              "~25 Gb/s usable for BOTH ports together; enabling f1 adds ports, not bandwidth"],
             ["chassis SYS-7049GP-TRT; GPUs on two of the six CPU x16 root ports; the other four have no "
              "device (3a/5d/ae/d7 on n04, 17/5d/85/d7 on n01)", "moving the card to a free x16 slot should give "
              "x8 (~50 Gb/s) and make f1 useful - verify physically (double-width GPUs may block slots)"],
             ["f1 down everywhere (no carrier on search nodes; admin-down on corr; n14 runs on f1)",
              "cabling and switch ports to be checked (Arista 7050QX-32 free ports undocumented)"],
             ["Intel X550T 10GBASE-T: eno1 up (10G), eno2 no carrier; separate PCIe x4 device",
              "eno2 can carry the first hop of two corner turns (~8 Gb/s) given a 10G switch"]],
            [3.6 * inch, 3.2 * inch]),
      P("Table 3 - Network hardware.", CAP),
      PageBreak()]

# ============================================================== 5 compute
S += [P("5. Compute", H2),
      fig("figs_v2/r6_gpu.png", 6.2 * inch,
          "Fig. 6 - Imaging + detection load (one dirty image per fine DM trial per sample) at 3.5 us (256x128, "
          "FoV/2) or 5.9 us (256x256) per image - assumed for rewritten owner kernels; today's in-situ cost is "
          "41.9 us (13.7 us of kernels). Trial counts: Levin recursion, tol 1.25, 61 kHz."),
      P("4x tiers needs 1003 trials (506 below DM 800 at 0.26 ms, 264 to DM 1600 at 0.52 ms, 233 above at "
        "1.05 ms) = 2.7e6 images/s, 14x today's rate; uniform 0.26 ms to DM 3000 would need 1476 trials and "
        "5.6e6 images/s. The corr side: one X-engine on GPU A at 0.26 ms (slow product as today's K=4096 GEMM, "
        "fast core Stokes-I per 61 kHz channel), with either per-channel gridding (two corner turns: simplest) "
        "or a fused multi-coarse-DM stage-1 (two-stage). The v1 budget (GPU A ~75 ms of compute per 134 ms "
        "block at 4x, ~24 ms of overlapped DMA) still holds and is what benchmark b4 tests; GPU B becomes an "
        "imaging owner and also hosts the zoom."),
      fig("figs/f8_corr_budget.png", 6.3 * inch,
          "Fig. 7 - Corr GPU time per block (v1 estimate for the two-stage front end; two corner turns replaces "
          "the stage-1 segment with per-channel gridding and is lighter)."),
      PageBreak()]

# ============================================================== 6 zoom
S += [P("6. Voltage zoom: native resolution where it matters", H2),
      P("The corr nodes already keep 25 s of voltages (52.6 GiB per node), and the bbproc toolkit already forms "
        "coherent (l, m) beams from them (it recovered the first FRB at 21.6 sigma). A zoom service turns that "
        "into a real-time step: each search candidate above a first-stage threshold theta1 (with a governor, "
        "e.g. top-N per second) sends a request straight to the 16 corr nodes (not through C2's 10-17 s path); "
        "each node forms 3-9 coherent beams around (l, m) from ~1.5 s of its chgroup, dedisperses coherently "
        "within channels at the candidate DM and incoherently over a local DM range, and ships native-resolution "
        "series (~50 MB per candidate in total) to an aggregator that sums the 16 chgroups."),
      table([["property", "value"],
             ["S/N recovered", "0.92 of ideal for widths &gt;= 0.1 ms at any DM (emulated, coherent, 32.8 us)"],
             ["latency", "search detection ~2-5 s after the burst + ~1 s zoom &lt;&lt; 25 s ring"],
             ["GPU cost", "beamforming 3 beams over 12 blocks ~ms per corr node (b6 tests it)"],
             ["network", "~50 MB per candidate: 0.4 Gb/s at one candidate per second"],
             ["what it changes", "detection threshold unchanged by itself; but cheap confirmation makes a lower "
              "theta1 affordable (9.55 -&gt; 8 sigma = x1.30 rate at any cadence); native-resolution widths, "
              "scattering and spectra for every candidate; a far better RFI/real discriminator than C3's R10"],
             ["risk", "candidate rate at theta1: clean sky at 8.55 sigma was ~free on 10-05, but the twice-daily "
              "1484/1494 MHz storms and satellite passes flood - the governor must hold; measure with a log-only "
              "counter first"]],
            [1.4 * inch, 5.4 * inch]),
      P("Table 4 - Zoom service.", CAP),
      Spacer(1, 6),
      P("7. Packages and recommended path", H2),
      fig("figs_v2/r7_rates.png", 6.4 * inch,
          "Fig. 8 - Rate gain per package. Zoom rows assume the first-stage threshold can drop from 9.55 to 8 "
          "sigma."),
      PageBreak()]
S += [table([["step", "what", "needs", "gain (median 1 ms / 0.3 ms)"],
             ["0", "measure: candidate rate in [7.5, 9.55) sigma (log-only counter); run bench/timeres in a gap; "
              "inspect a chassis for a free x16 slot", "a 30-min gap, a site visit", "-"],
             ["1", "re-seat the ConnectX-4s in x16 slots; unified X-engine on GPU A (slow + fast from one copy), "
              "freeing GPU B", "site visit; corr-side code", "enables everything below"],
             ["2", "two corner turns with the corr side at final resolution (0.26 ms, 61 kHz, 4-bit); imaging "
              "owners on 16 corr GPU B + 8 search GPUs at 1.05 ms", "owner service, FDMT cell owners",
              "%s / %s" % (g("1x fixed (two CTs)", m1), g("1x fixed (two CTs)", m3))],
             ["3", "voltage zoom; lower theta1 as the measured rate allows", "zoom service (bbproc kernels)",
              "x1.30 on top (theta1 = 8)"],
             ["4", "imaging at 0.52 ms below DM 1600 (downstream only)", "rewritten owner kernels",
              "%s / %s" % (g("2x tiers (2-stage)", m1), g("2x tiers (2-stage)", m3))],
             ["5", "0.26 ms below DM 800 (downstream only)", "owner kernels at ~3.5-5.9 us/image",
              "%s / %s; with zoom %s / %s" % (g("4x tiers (2-stage, 61 kHz)", m1), g("4x tiers (2-stage, 61 kHz)", m3),
                                             g("4x tiers + zoom", m1), g("4x tiers + zoom", m3))],
             ["alt", "if the NIC cannot move: two-stage done right (native stage-1, n_coarse decoupled from the "
              "search GPUs, 4 sub-bands per coarse DM, DM tiers) - fits today's x4 NICs",
              "corr-side stage-1 rewrite", "same S/N as steps 2-5 within ~2%"]],
            [0.4 * inch, 3.3 * inch, 1.5 * inch, 1.6 * inch]),
      P("Table 5 - Recommended path. Steps 4-5 change only the imaging owners: the corr nodes and the first hop "
        "already run at the final resolution after step 2.", CAP),
      P("Risks and what to measure", H2)]
S += bullets([
    "<b>Owner kernel efficiency</b> (3.5-5.9 us/image assumed vs 13.7 us of kernels today): b3 measures it with "
    "simple prototypes. 4x tiers fits even at ~2x worse (24 owners at full FoV peak at ~0.7).",
    "<b>Corr GPU A at 0.26 ms:</b> fast GEMM at K=16 and per-channel gridding or stage-1 are estimates (b4).",
    "<b>First-stage threshold with zoom:</b> must be set from a measured candidate rate in clean sky and storms.",
    "<b>Network:</b> NIC slot move to be confirmed physically; b5 measures owner ingress with SNAP traffic present.",
    "<b>Models:</b> white noise; imaging loss (x0.89) unchanged and excluded; Euclidean rates; the emulator "
    "reproduces the September measurements of the deployed chain (0.81-0.84 of perfect dedispersion at 1 ms).",
])
S += [P("Benchmarks for the next gap (bench/timeres)", H2),
      table([["script", "node", "firms up"],
             ["b0_selftest", "node GPU", "prototype Triton kernels vs torch references"],
             ["b1_fft / b2_detector / b3_owner_e2e", "search + corr", "us/image for FFT, detector and the whole "
              "owner chain; fine DMs one GPU sustains (need &gt;= ~45 per owner for 4x tiers)"],
             ["b4_xengine", "corr", "slow + fast GEMM per block at t_int 32/16/8, int8 vs fp16, PCIe"],
             ["b6_zoom", "corr", "beamform + local dedispersion per candidate (expect ~ms)"],
             ["b5_fabric.sh", "h23 -&gt; nodes", "owner ingress 7-20 Gb/s with SNAP traffic"]],
            [2.0 * inch, 1.1 * inch, 3.7 * inch]),
      Spacer(1, 4),
      P("Appendix: method. <font face='Courier'>model/emulate.py</font> (chain emulator; coherent mode for the "
        "zoom), <font face='Courier'>run_broad.py</font> (all families, DM 300-2500), "
        "<font face='Courier'>rates.py</font> (package rate gains), <font face='Courier'>packages2.py</font> "
        "(trials, imaging load, owner assignment, per-node ingress), <font face='Courier'>nfilled.py</font> "
        "(uv cells from antennas.out: 1748 per channel, ~1900 per sub-band stream, ~6000 per dedispersed "
        "plane). v1 report and figures are kept alongside for comparison.", SM)]

doc = SimpleDocTemplate("DSA110_timeres_study_v2.pdf", pagesize=letter, leftMargin=0.7 * inch,
                        rightMargin=0.7 * inch, topMargin=0.6 * inch, bottomMargin=0.7 * inch,
                        title="DSA-110 real-time search: time-resolution study v2")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote DSA110_timeres_study_v2.pdf")
