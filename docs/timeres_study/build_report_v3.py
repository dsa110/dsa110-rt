"""Build DSA110_timeres_study_v3.pdf: baseline = NICs stay in their x4 slots; NIC move = stretch.
Run from docs/timeres_study/."""
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
OKC, NOC = colors.HexColor("#dff3e4"), colors.HexColor("#fbe3e1")
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

P3 = json.load(open("model/results/plan3.json"))
GA = P3["gains"]
FE = P3["feas"]


def gx(key, pop="m1", zoom=False):
    return f"x{GA[key]['gain_zoom8' if zoom else 'gain'][pop]:.2f}"


def g2(key, zoom=False):
    return f"{gx(key, 'm1', zoom)} / {gx(key, 'm03', zoom)}"


def P(t, st=B):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(t, BL, bulletText="•") for t in items]


def fig(path, width=6.9 * inch, caption=None):
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
    c.drawString(0.7 * inch, 0.45 * inch, "DSA-110 real-time search: time-resolution study v3 (baseline: NICs stay in "
                 "their slots), 2026-10-21 - branch timeres-study")
    c.drawRightString(letter[0] - 0.7 * inch, 0.45 * inch, f"{d.page}")
    c.restoreState()


# ---------------------------------------------------------------- feasibility matrix
SCN = [("baseline (corr +0, search 17)", "corr +0"), ("baseline (corr +2, search 17)", "corr +2 (central)"),
       ("baseline (corr +4, search 22)", "corr +4, RX tuned"), ("baseline + eno2 10G", "+ eno2 10G"),
       ("stretch: NIC in x16 slot", "STRETCH")]
AMB = [("1x", "1x"), ("2x", "2x"), ("4x-lite", "4x &lt;DM 500"), ("4x", "4x &lt;DM 800")]


def cell(scn, amb, fov, two_ct):
    rows = [r for r in FE if r["scenario"] == scn and r["ambition"] == amb and r["fov"] == fov]
    rows = [r for r in rows if (r["scheme"] != "two-stage") == two_ct]
    ok = [r for r in rows if r["fits"]]
    if not ok:
        return "no", False
    if two_ct:          # prefer the finest first hop that fits
        pref = sorted(ok, key=lambda r: ("61 kHz" not in r["scheme"], "0.52" in r["scheme"]))
        r = pref[0]
        tag = "61k" if "61 kHz" in r["scheme"] else ("122k" if "0.26" in r["scheme"] else ".52/122k")
        return f"{r['max_gpu']:.2f} ({tag})", True
    return f"{ok[0]['max_gpu']:.2f}", True


hdr = ["package", "scheme"] + [f"{lab}<br/>FoV/2 | full" for _, lab in SCN]
mrows, style = [hdr], []
for ai, (amb, alab) in enumerate(AMB):
    for si, (two_ct, slab) in enumerate(((False, "two-stage"), (True, "two corner turns"))):
        row = [alab if si == 0 else "", slab]
        for ci, (scn, _) in enumerate(SCN):
            a, oka = cell(scn, amb, "FoV/2", two_ct)
            b, okb = cell(scn, amb, "full FoV", two_ct)
            row.append(f"{a} | {b}")
            r = len(mrows)
            style.append(("BACKGROUND", (2 + ci, r), (2 + ci, r), OKC if (oka and okb) else (NOC if not oka else
                                                                                               colors.HexColor("#fff3d6"))))
        mrows.append(row)
MATRIX = table(mrows, [0.75 * inch, 0.95 * inch] + [1.03 * inch] * 5, style)

S = []
# ================================================================ summary
S += [P("Improving the time resolution of the DSA-110 real-time FRB search - v3", H1),
      P("Baseline assumption: the ConnectX-4 NICs stay in their current (electrically x4) PCIe slots; moving them "
        "is a stretch goal, assessed separately (sec. 6). All results are modelled on h23 (end-to-end dedispersion "
        "emulator on 6144 native channels; cost and network models on measured numbers); no new GPU code was "
        "run. Benchmarks for the next observing gap: <font face='Courier'>bench/timeres/</font>.", SM),
      P("Summary", H2)]
S += bullets([
    "<b>The baseline network is the binding constraint.</b> Each node's two 40G ports share one PCIe Gen3 x4 link "
    "(~20-25 Gb/s). Corr nodes already spend ~18 Gb/s of it on SNAP voltages, so they can take only ~2-4 Gb/s "
    "of corner-turn data (to be measured); search nodes saturate near 17 Gb/s today. Egress from corr nodes is "
    "free. So in the baseline, data must flow corr -&gt; search and the corr second GPUs are <i>network-starved</i>, "
    "not compute-starved: they suit the X-engine, the voltage zoom and small high-DM buckets.",
    "<b>Recommended baseline design: two-stage dedispersion done right.</b> Native-precision stage-1 on the corr "
    "node; 61 kHz channels (122 kHz is enough below DM ~500); 4-bit sub-band streams at the search cadence; "
    "DM-tiered cadence (0.26 ms below DM 800, 0.52 ms to 1600, 1.05 ms above); imaging on the 8 search GPUs plus "
    "corr second GPUs for low-data buckets. At matched precision it recovers the same S/N as one-shot "
    "dedispersion (sec. 2).",
    "<b>Baseline reach</b> (rate gain, width median 1 ms / 0.3 ms; DM median 500): 1x fixed %s; "
    "<b>2x %s</b> - robust: fits even with zero corr headroom (FoV/2), and at full FoV with ~2 Gb/s; "
    "<b>4x below DM 800 %s</b> - fits only at FoV/2, needs ~2 Gb/s of corr headroom and owner kernels at "
    "~3.5 us/image (peak owner load 0.75: tight)." % (g2("base-ts|1x"), g2("base-ts|2x"), g2("base-ts|4x")),
    "<b>Voltage zoom fits the baseline</b> (it uses corr egress and ~0.4 Gb/s): coherent beams from the 25 s "
    "voltage ring give 0.92 of ideal at native 32.8 us for every candidate, and cheap confirmation makes a lower "
    "first-stage threshold affordable: 2x + zoom %s, 4x + zoom %s if 9.55 -&gt; 8 sigma holds."
    % (g2("base-ts|2x", True), g2("base-ts|4x", True)),
    "<b>Two corner turns (one-shot) does not fit the baseline</b>: its first hop puts 2-8 Gb/s on every node, "
    "including corr nodes. Cabling the free 10G eno2 port (separate PCIe device; needs a 10G switch) makes it "
    "feasible at 1x and 2x, where its finer shift precision is worth a few per cent.",
    "<b>Stretch - NICs re-seated in x16 slots (~50 Gb/s):</b> removes the dependence on corr headroom, lets 4x run "
    "at full FoV, enables two corner turns with its operational advantages, and adds 3-8%% in rate "
    "(4x %s; + zoom %s). It does not make uniform 0.26 ms to DM 3000 affordable (GPU-bound)."
    % (g2("str-tc|4x"), g2("str-tc|4x", True)),
    "<b>Measure first:</b> corr-node ingress headroom beside SNAP (b5 S1/S2), owner-kernel cost per image (b3), "
    "and the candidate rate between 7.5 and 9.55 sigma (log-only counter) before committing to 4x or a lower "
    "threshold.",
])
S += [PageBreak()]

# ================================================================ 1 network budget
S += [P("1. The baseline network budget", H2),
      fig("figs_v3/n1_network_budget.png", 6.9 * inch,
          "Fig. 1 - Per-node ingress for each package in the central baseline (corr +2 Gb/s, search 17). Two-stage "
          "data lands mostly on search nodes; the first hop of two corner turns lands on every node."),
      table([["fact (all 20 nodes, read-only sysfs)", "baseline consequence"],
             ["one dual-port ConnectX-4 per node behind root port 80:00.0, itself x4 (8 GT/s)",
              "~20-25 Gb/s for BOTH 40G ports; enabling the second port adds nothing"],
             ["corr nodes take ~18 Gb/s of SNAP voltages on the same link",
              "~2-4 Gb/s of corner-turn ingress at most (unmeasured - b5 S1/S2)"],
             ["search nodes: measured knee 16-20 Gb/s (plateau 17.4, single RX thread, rings 1024 of 8192)",
              "~17 Gb/s; RX tuning (bigger rings, RSS, more threads) might reach ~22"],
             ["egress (TX) shares nothing with SNAP ingress", "corr nodes can send freely: favour corr -&gt; search flows"],
             ["Intel X550T 10GBASE-T: eno2 has no carrier (separate PCIe x4 device)",
              "+~9 Gb/s per node if cabled to a 10G switch - not a slot move"]],
            [3.6 * inch, 3.2 * inch]),
      P("Table 1 - Network facts behind the baseline.", CAP),
      P("Imaging compute per Gb/s of corner-turn data is roughly constant across DM tiers (~0.12 GPU-equivalent "
        "per Gb/s at FoV/2, ~0.2 at full FoV), so a corr node with 2 Gb/s of headroom can host only ~0.25 GPU of "
        "imaging; total baseline imaging capacity is ~10 GPU-equivalents at FoV/2."),
      PageBreak()]

# ================================================================ 2 physics recap
S += [P("2. What sets the time resolution (unchanged from v2)", H2),
      fig("figs_v2/r1_resolution.png", 6.0 * inch,
          "Fig. 2 - Effective time resolution vs DM. Shaded: 90% of a log-normal FRB DM distribution (median 500)."),
      table([["channel width", "smear at DM 500 (top / centre / bottom)", "DM 1000", "DM 1500"],
             ["30.5 kHz (native)", "38 / 46 / 56 us", "91 us", "137 us"],
             ["61 kHz", "75 / 91 / 112 us", "183 us", "274 us"],
             ["122 kHz", "150 / 183 / 225 us", "365 us", "548 us"],
             ["244 kHz (today)", "301 / 365 / 449 us", "730 us", "1096 us"]],
            [1.3 * inch, 2.4 * inch, 1.4 * inch, 1.4 * inch]),
      P("Table 2 - Intra-channel smear (band centre unless stated).", CAP),
      P("At matched shift precision and channel width, one-shot and a properly sized two-stage dedispersion "
        "recover the same S/N to ~2% (v2 Fig. 3). Today's chain loses because stage-1 shifts are rounded to "
        "1.05 ms after a 244 kHz channel sum and 8 x 4 coarse-DM x sub-band products leave ~0.5 ms of residual "
        "smear. In the baseline the sub-band streams travel at the search cadence (not finer), which costs ~3-5% "
        "of S/N against streams at half the cadence - the price of the x4 NICs."),
      PageBreak()]

# ================================================================ 3 baseline reach
S += [P("3. What the baseline can reach", H2),
      fig("figs_v3/n3_snr.png", 6.5 * inch,
          "Fig. 3 - Recovered S/N (fraction of an ideal matched filter) for the baseline configurations."),
      MATRIX,
      P("Table 3 - Feasibility: peak GPU load of the busiest owner (cap 0.8) at FoV/2 | full FoV, or 'no'. Green: "
        "fits at both; amber: FoV/2 only; red: not even at FoV/2. Two-stage owner kernels at 3.5 us (FoV/2) / 5.9 us "
        "(full) per image; two corner turns shows the finest first hop that fits (61k = 0.26 ms / 61 kHz).", CAP),
      PageBreak(),
      fig("figs_v2/r6_gpu.png", 5.8 * inch,
          "Fig. 4 - Imaging load: DM tiers halve the 4x load (uniform cadence to DM 3000 shown for comparison)."),
      fig("figs/f8_corr_budget.png", 6.0 * inch,
          "Fig. 5 - Corr GPU time per block (v1 estimate): one corr GPU can carry the X-engine (slow + fast from one "
          "voltage copy) at 4x; the other is free for the zoom and small imaging buckets."),
      PageBreak()]

# ================================================================ 4 rates + zoom
S += [P("4. Rate gains", H2),
      fig("figs_v3/n2_rates.png", 6.6 * inch,
          "Fig. 6 - Detection-rate gain over today (Euclidean counts; DM log-normal, median 500; DM &lt;=1500)."),
      table([["package (baseline unless stated)", "median 1 ms / 0.3 ms", "+ zoom (theta1 = 8)"],
             ["1x: two-stage done right", g2("base-ts|1x"), g2("base-ts|1x", True)],
             ["2x: 0.52 ms below DM 1600", g2("base-ts|2x"), g2("base-ts|2x", True)],
             ["4x below DM 500 (FoV/2)", g2("base-ts|4x-lite"), g2("base-ts|4x-lite", True)],
             ["4x below DM 800 (FoV/2)", g2("base-ts|4x"), g2("base-ts|4x", True)],
             ["+ eno2: two corner turns at 2x", g2("str-tc|2x"), g2("str-tc|2x", True)],
             ["stretch: two corner turns, 4x", g2("str-tc|4x"), g2("str-tc|4x", True)],
             ["stretch: uniform 4x to DM 3000 (does not fit)", g2("str-ts|4x-uniform"), g2("str-ts|4x-uniform", True)]],
            [3.0 * inch, 1.8 * inch, 1.8 * inch]),
      P("Table 4 - Rate gains. Extending to DM 3000 adds the ~3.4% of FRBs above DM 1500; the larger search volume "
        "needs ~+0.3 sigma of threshold (~4% of rate) for the same false-alarm rate.", CAP),
      P("5. Voltage zoom", H2),
      P("Each search candidate above a first-stage threshold (with a governor, e.g. top-N per second) sends a "
        "request straight to the 16 corr nodes; each forms 3-9 coherent beams around (l, m) from ~1.5 s of its "
        "25 s voltage ring (bbproc toolkit kernels), dedisperses coherently within channels and incoherently over "
        "a local DM range, and ships native-resolution series (~50 MB per candidate) to an aggregator. It recovers "
        "0.92 of ideal at any DM and width, costs milliseconds of GPU (b6), and adds no corner-turn ingress to corr "
        "nodes - so it is the cheapest way to native time resolution in the baseline. Its rate gain comes from "
        "the lower first-stage threshold it makes affordable; the candidate rate at ~8 sigma (clean sky was "
        "nearly free at 8.55 on 10-05; storms and satellite passes flood) must be measured first."),
      PageBreak()]

# ================================================================ 6 stretch
S += [P("6. Stretch goal: what moving the NICs adds", H2),
      P("Four of the six CPU x16 root ports have no device on the nodes checked, so re-seating each ConnectX-4 in an "
        "x16 slot (x8 card: ~50 Gb/s, second 40G port useful) is probably possible - to be confirmed physically, "
        "since the double-width GPUs may block slots. What it buys over the baseline:"),
      table([["", "baseline (x4 slot)", "stretch (x16 slot)"],
             ["corr-node corner-turn ingress", "~2-4 Gb/s beside SNAP (unmeasured)", "~30 Gb/s"],
             ["2x (0.52 ms)", "fits; full FoV needs ~2 Gb/s corr headroom", "fits at full FoV, any scheme"],
             ["4x below DM 800", "FoV/2 only; peak load 0.75; needs corr headroom", "full FoV (peak ~0.73-0.76)"],
             ["dedispersion scheme", "two-stage; streams at the search cadence",
              "two corner turns (first hop 0.26 ms / 61 kHz) or two-stage with half-cadence streams"],
             ["sensitivity at 4x", g2("base-ts|4x"), g2("str-tc|4x")],
             ["+ zoom", g2("base-ts|4x", True), g2("str-tc|4x", True)],
             ["operations", "DM plan lives on the corr nodes (buckets); an owner loss is a DM gap",
              "corr side fixed at final resolution; DM plan / cadence only downstream; owner loss = small loss "
              "everywhere"],
             ["uniform 0.26 ms to DM 3000", "no", "no - GPU-bound (~20 GPU-equivalents at FoV/2)"]],
            [1.6 * inch, 2.5 * inch, 2.7 * inch]),
      P("Table 5 - Baseline vs stretch. The sensitivity gain of the stretch is modest (3-8%); its value is margin "
        "(no reliance on unmeasured corr headroom), full FoV at 4x, and the simpler, more robust two-corner-turn "
        "architecture.", CAP),
      fig("figs_v2/r4_architectures.png", 5.4 * inch,
          "Fig. 7 - (A) two-stage done right: the baseline design; (B) two corner turns: needs eno2 or the stretch; "
          "(C) voltage zoom: fits either."),
      PageBreak()]

# ================================================================ 7 roadmap
S += [P("7. Recommended path (baseline first)", H2),
      table([["step", "what", "needs", "gain (median 1 ms / 0.3 ms)"],
             ["0", "measure: corr ingress beside SNAP (b5 S1/S2); owner kernels (b3); X-engine (b4); zoom cost (b6); "
              "candidate rate in [7.5, 9.55) sigma", "a 30-min gap; a log-only counter", "decides steps 3-5"],
             ["1", "unified X-engine on corr GPU A (slow + fast from one copy); native-precision stage-1; n_coarse "
              "decoupled from the search GPUs; 4-bit streams; C transmitter", "corr-side code",
              g2("base-ts|1x")],
             ["2", "voltage zoom on corr GPU B; lower the first-stage threshold as the measured rate allows",
              "zoom service (bbproc kernels)", "x1.30 at theta1 = 8"],
             ["3", "rewritten owner kernels on the 8 search GPUs (+ corr GPU B for low-data buckets); 0.52 ms below "
              "DM 1600", "owner service", g2("base-ts|2x")],
             ["4", "0.26 ms below DM 500, then 800, at FoV/2 - if step 0 shows &gt;= 2 Gb/s corr headroom and "
              "&lt;= 3.5 us/image", "kernels at target cost", f"{g2('base-ts|4x-lite')} -&gt; {g2('base-ts|4x')}"],
             ["opt", "cable eno2 10G: two corner turns at 1x/2x", "10G switch ports", g2("str-tc|2x")],
             ["stretch", "re-seat NICs in x16 slots: full-FoV 4x, two corner turns", "site visit",
              f"{g2('str-tc|4x')} (+ zoom {g2('str-tc|4x', True)})"]],
            [0.55 * inch, 3.25 * inch, 1.45 * inch, 1.55 * inch]),
      P("Table 6 - Baseline-first roadmap.", CAP),
      P("Risks and measurements", H2)]
S += bullets([
    "<b>Corr-node ingress headroom beside SNAP</b> is the one number that decides baseline 4x: b5 S1 (2 Gb/s) and "
    "S2 (4 Gb/s) on three corr receivers, with the SNAPs streaming and capture drop counters watched.",
    "<b>Owner kernel cost</b> (3.5 us/image at FoV/2 assumed; 13.7 us of kernels today): b3 measures simple "
    "prototypes; at ~5 us baseline 4x needs 0.26 ms only below DM ~500.",
    "<b>Search RX</b>: 13 Gb/s per search node at baseline 4x is below the 17 Gb/s knee; b5 S4 tests 20 Gb/s.",
    "<b>First-stage threshold with zoom</b>: set from the measured candidate rate in clean sky and storms.",
    "<b>Models</b>: white noise, Euclidean counts, imaging loss (x0.89) excluded; the emulator reproduces the "
    "September measurements of the deployed chain.",
])
S += [Spacer(1, 4),
      P("Changes from v2: baseline now assumes the NICs stay put; two-stage streams travel at the search cadence "
        "(v2's two-stage S/N assumed half-cadence streams, which the x4 network cannot carry); feasibility is "
        "checked per node against SNAP-aware ingress ceilings; two corner turns moves to the eno2 / stretch "
        "options. Models: <font face='Courier'>model/run_baseline.py</font>, <font face='Courier'>plan3.py</font>, "
        "<font face='Courier'>figs3.py</font>; v2 and v1 kept alongside.", SM)]

doc = SimpleDocTemplate("DSA110_timeres_study_v3.pdf", pagesize=letter, leftMargin=0.7 * inch,
                        rightMargin=0.7 * inch, topMargin=0.6 * inch, bottomMargin=0.7 * inch,
                        title="DSA-110 real-time search: time-resolution study v3")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote DSA110_timeres_study_v3.pdf")
