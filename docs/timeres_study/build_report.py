"""Build DSA110_timeres_study.pdf (reportlab).  Run from docs/timeres_study/."""
import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from reportlab.lib.utils import ImageReader

INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
RULE = colors.HexColor("#d9d8d3")
HEAD = colors.HexColor("#f0efec")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=18,
                    spaceAfter=6, textColor=INK)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14,
                    spaceBefore=8, spaceAfter=4, textColor=INK)
B = ParagraphStyle("B", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.2, leading=12.2,
                   textColor=INK, alignment=TA_LEFT, spaceAfter=4)
BL = ParagraphStyle("BL", parent=B, leftIndent=11, bulletIndent=2, spaceAfter=2.5)
SM = ParagraphStyle("SM", parent=B, fontSize=7.8, leading=9.6, textColor=INK2)
CAP = ParagraphStyle("CAP", parent=SM, spaceBefore=2, spaceAfter=8)
TC = ParagraphStyle("TC", parent=B, fontSize=7.6, leading=9.2, spaceAfter=0)
TCB = ParagraphStyle("TCB", parent=TC, fontName="Helvetica-Bold")

cost = {p["name"]: p for p in json.load(open("model/results/cost_points.json"))}
derived = json.load(open("model/results/derived.json"))
rg1 = derived["rate_gain"]["median 1 ms (DSA-like)"]
rg3 = derived["rate_gain"]["median 0.3 ms (narrow)"]


def P(t, st=B):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(t, BL, bulletText="•") for t in items]


def fig(path, width=6.9 * inch, caption=None):
    ir = ImageReader(path)
    w, h = ir.getSize()
    im = Image(path, width=width, height=width * h / w)
    out = [im]
    if caption:
        out.append(P(caption, CAP))
    return KeepTogether(out)


def table(rows, widths, header=True, fs=None):
    data = [[Paragraph(str(c), TCB if (header and i == 0) else TC) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), HEAD))
    t.setStyle(TableStyle(st))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(INK2)
    canvas.drawString(0.7 * inch, 0.45 * inch,
                      "DSA-110 real-time search: time-resolution study - initial conclusions, 2026-10-21 "
                      "(branch timeres-study)")
    canvas.drawRightString(letter[0] - 0.7 * inch, 0.45 * inch, f"{doc.page}")
    canvas.restoreState()


S = []
# ------------------------------------------------------------------ page 1: summary
S += [P("Improving the time resolution of the DSA-110 real-time FRB search", H1),
      P("Initial conclusions - 2026-10-21. Analysis on h23 only (no GPU code run); benchmarks to firm "
        "up the extrapolated numbers are in <font face='Courier'>bench/timeres/</font> for the next "
        "observing gap. Models, figures and this report: <font face='Courier'>docs/timeres_study/</font> "
        "on branch <font face='Courier'>timeres-study</font>.", SM),
      Spacer(1, 4),
      P("Summary", H2)]
S += bullets([
    "<b>Today's effective time resolution is ~1.2 ms at low DM and ~1.6-1.9 ms at DM 1500</b>, not the "
    "1.05 ms sample. A 0.1 ms burst at DM 1500 delivers 0.20 of an ideal matched filter (legacy dsaX: 0.28). "
    "Four things set this (sec. 1): the fast GEMM sums 8 channels (244 kHz) <i>before</i> any "
    "dedispersion shift; the 1.05 ms sample; an under-provisioned two-stage chain (8 coarse DMs x 4 "
    "sub-bands, coarse count pinned by the DM-plan schema to the 8 search GPUs); and both ends running "
    "at their real-time limit with every post-GEMM step scaling as 1/t_int.",
    "<b>4x sampling alone does not deliver 4x after dedispersion.</b> With 244 kHz channels a 0.26 ms "
    "search only matches legacy dsaX: its gain over today shrinks with DM and is gone by DM ~1500, where it "
    "equals simply fixing the channels at 1.05 ms. The &gt;=4x at DM &gt;=1000 needs the native 30.5 kHz "
    "channels (61 kHz is an acceptable compromise). Going past 4x (0.13 ms) is limited by native-channel smear above DM ~1000 "
    "and buys +5-20% more at the narrowest widths; it needs up-channelisation and more GPUs.",
    "<b>Recommended operating point: 262 us sampling, native channels, two-stage dedispersion done "
    "right</b> - stage-1 shifts at the native 32.8 us inside the corr node, 16-32 coarse DMs x 8 "
    "sub-bands, a single rounding at the fine DM. Emulated gain over today: <b>x2.9 S/N</b> for 0.1 ms and "
    "<b>x2.5</b> for 0.25 ms bursts at DM 1500, x1.6 at 1 ms; Euclidean rate x%.1f (population with "
    "median width 1 ms) to x%.1f (median 0.3 ms)." % (rg1["x4_c1"], rg3["x4_c1"]),
    "<b>Single-stage brute force is not needed.</b> A correctly sized two-stage chain matches or beats "
    "brute force at the same sampling and stays within 2-3% of an ideal single stage up to DM 2500 "
    "(Fig. 4). True single-stage across 16 distributed channel groups would need the full per-channel "
    "visibility stream (~0.7 Tb/s) in one place, or a time-multiplexed corner turn with tens of seconds "
    "of latency.",
    "<b>The cost is ~20x today's imaging work</b> (fine-DM trials x5, samples x4): 3.75e6 dirty images/s "
    "for DM &lt;=1500. It fits in the existing 40 GPUs only if <b>(a) the corr nodes' second GPUs become "
    "imaging owners</b> (your proposal) and <b>(b) the owner kernels are rewritten</b> to 3.5-6 us/image "
    "(today 13.7 us of kernels, 41.9 us wall). With the allowed FoV/2 trade it needs ~13 of the 24 "
    "available GPU-equivalents; DM &lt;=3000 needs ~24 (or ~17 with 0.52 ms sampling above DM 1500).",
    "<b>Your proposal works, with three amendments.</b> (i) Images never cross the network - a bucket's "
    "image cube is ~6-12x the uv streams that make it (44-88 Gb/s per owner) - so detection stays on the "
    "imaging GPU; search nodes run the same "
    "owner code plus C1, not 'pure filtering'. (ii) The dedispersion on corr GPU B is the second stage of "
    "a sub-band scheme, not single-stage. (iii) Slow visibilities from one unified X-engine is a clear "
    "win - it removes the duplicated copy+unpack+correlation and frees a whole GPU per corr node - but it "
    "must keep the raw, uncalibrated, all-antenna product and process isolation.",
    "<b>New hardware finding: every node's ConnectX-4 trains at PCIe Gen3 x4</b> (card is x8-capable): "
    "~25 Gb/s usable per node across both ports, of which corr nodes already spend ~18 Gb/s on SNAP data. "
    "The spare 40G ports add nothing until the NICs sit in x8 slots. With 4-bit uv streams (7.4 Gb/s per "
    "owner) the design fits marginally; re-seating the NICs is the cheap enabler.",
    "<b>RFI excision is unchanged in kind</b> (same voltage-domain moments and detectors; weights applied "
    "to the fast product only). FoV/2 is used only as margin; DM 1500 is met with margin, DM 3000 with "
    "FoV/2.",
])
S += [Spacer(1, 4), P("Contents: 1 limits today - 2 what finer resolution buys - 3 dedispersion: one stage "
                      "vs two - 4 cost - 5 architectures - 6 corr-node and network budgets - 7 recommended "
                      "path and risks - 8 benchmarks - A method, sources.", SM),
      PageBreak()]

# ------------------------------------------------------------------ 1 limits
S += [P("1. What restricts the time resolution today", H2),
      P("The deployed chain (corr_fast <font face='Courier'>--t-int-fast-native 32 --chan-sum-factor 8 "
        "--n-sub 4</font>, 8 coarse x 25 fine DMs over 103-1496, search T = 1048.576 us, n_grid 256) was "
        "traced stage by stage; Fig. 1 shows the resulting effective resolution and Fig. 2 what reaches the "
        "detector. In order of impact:"),
      table([
          ["limit", "where", "number", "consequence"],
          ["8 channels summed inside the fast GEMM, before any shift",
           "triton_corr_fused.py:21-27; corr_fast_kernel.py:640-664",
           "244 kHz: 0.90 (top) - 1.35 (bottom) ms smear at DM 1500",
           "floor independent of sampling above DM ~400; legacy had the same"],
           ["fast-vis sample t_int = 32 native", "dsart_pipeline_rt.yaml:629",
           "1.049 ms", "x8 (262 us) measured 169 ms per 134 ms block in May -> abandoned (0f42f7f)"],
          ["8 coarse DMs, pinned to the 8 search GPU halves", "build_dm_plan.py:403-418 (schema)",
           "8 x n_sub 4: ~0.8 ms intra-sub-band smear", "no fine trial can remove it; product n_c*n_s must "
           "grow 4x at 0.26 ms"],
          ["stage-1 shift quantised at T, per-chgroup fine correction", "coarse_dm/, fine_dm/combiner.py",
           "x1.10 S/N lost (Sept study); merged rounding live", "matters more as T shrinks"],
          ["corr_fast real-time", "live 123.5-125.6 of 134.2 ms",
           "post-GEMM steps all ~1/t_int", "permute 9->36 ms, stage-1 0.75->3 ms per coarse DM, D2H "
           "70->280 MB, Python TX ~4x; both GPUs copy+unpack the same block"],
          ["search real-time", "4.96-5.15 vs 4.967 cubes/s",
           "13.7 us GPU / 41.9 us wall per image", "single-thread RX assembly 42 ms/cube; ring 15 GiB -> 61 "
           "GiB at 262 us (93 GB nodes); v4 combine shared memory fails at 4x shift spread; 64-stream caps"],
          ["network", "NIC PCIe Gen3 x4 (all nodes)", "~25 Gb/s usable; knee ~17 Gb/s",
           "x8 sampling at n_sub 4 already reaches the search-node knee"],
          ["sample-denominated knobs", "C1/C2 width caps, NMS 128, merger x16, Layer-2 cadence",
           "16-sample width caps", "all x4 at 0.26 ms (simple, many)"],
      ], [1.55 * inch, 1.55 * inch, 1.6 * inch, 2.2 * inch]),
      Spacer(1, 6),
      fig("figs/f1_effective_resolution.png", 6.4 * inch,
          "Fig. 1 - Effective time resolution (quadrature sum of sample, rms channel smear across the band "
          "and, for today, the rms intra-sub-band residual of the 8x4 chain). The dashed floor is the best "
          "any incoherent up-channelisation of the native 30.5 kHz x 32.8 us grid can do."),
      PageBreak()]

# ------------------------------------------------------------------ 2 what it buys
S += [P("2. What finer resolution buys", H2),
      P("An operation-by-operation emulator (6144 native channels, native-channel smear, channel summing "
        "before the shift, stage-1/stage-2 shifts and their quanta, the real fine-DM grids, random sample "
        "phase and DM, power-of-two boxcars) gives the fraction of an ideal matched filter that reaches the "
        "detector. It reproduces the September study (deployed chain 0.81-0.84 of perfect dedispersion at "
        "1 ms) and the Levin trial counts (218 over 100-1500 at 1.05 ms)."),
      fig("figs/f2_recovered_snr.png", 6.1 * inch,
          "Fig. 2 - Recovered S/N vs intrinsic width at DM 300 and DM 1500. Shaded: the range holding half of "
          "the DSA voltage-measured widths. At DM 1500 the 244 kHz curves (today, legacy, '4x same channels') "
          "are channel-smear limited; native channels recover most of the loss."),
      fig("figs/f3_gain_vs_today.png", 5.4 * inch,
          "Fig. 3 - S/N gain at fixed fluence over the deployed chain. Solid: proposed 4x native. Dashed: "
          "the dedispersion fix alone at today's 1.05 ms (native channels, native-precision stage-1, 16x8) - "
          "worth x1.45 at DM 1500 by itself."),
      table([["population (log-normal widths)", "dedisp. fix 1.05 ms", "2x 30.5 kHz", "4x 244 kHz",
              "4x 61 kHz", "4x 30.5 kHz", "8x 30.5 kHz"],
             ["median 1 ms (DSA-like), sigma_ln 1.0"] + [f"x{rg1[k]:.2f}" for k in
                                                         ("x1_c1", "x2_c1", "x4_c8", "x4_c2", "x4_c1", "x8_c1")],
             ["median 0.3 ms (narrow), sigma_ln 0.8"] + [f"x{rg3[k]:.2f}" for k in
                                                         ("x1_c1", "x2_c1", "x4_c8", "x4_c2", "x4_c1", "x8_c1")]],
            [1.9 * inch] + [0.83 * inch] * 6),
      P("Table 1 - Euclidean detection-rate gain over today (S/N^1.5 at fixed threshold, averaged over DM "
        "300/800/1500). Excludes the ~3% threshold rise needed for the ~20x larger search volume.", CAP),
      PageBreak()]

# ------------------------------------------------------------------ 3 dedispersion
S += [P("3. Dedispersion: does it need to be single-stage?", H2),
      P("Brute-force single-stage dedispersion needs every channel of every baseline (or uv cell) for a "
        "given DM trial in one place. Here the 16 channel groups live on 16 corr nodes, so a single stage "
        "requires either the whole per-channel stream at each DM owner (~0.7 Tb/s of gridded visibilities "
        "at native channels and 0.26 ms) or a time-multiplexed corner turn, in which each owner takes the "
        "full band for every 24th time chunk; that latency (tens of seconds) exceeds the 20 s voltage ring, "
        "and the uv-domain sum per image grows 6-48x. The sub-band (two-stage) scheme is the efficient "
        "factorisation; what made today's version lossy is under-provisioning, not the idea:"),
      fig("figs/f4_dedisp_fidelity.png", 6.7 * inch,
          "Fig. 4 - Recovered S/N for a 0.25 ms burst at T = 0.26 ms with native channels. Grey dashed: "
          "brute-force single stage (shifts rounded to whole samples, as legacy dsaX did). Stage-2 at T/2 "
          "means the sub-band streams are produced at 131 us and summed to 262 us after the fine shift "
          "(2x stream rate - affordable with 4-bit values)."),
      P("<b>Rule for the redesign:</b> stage-1 at native 32.8 us precision (inside the corr node); n_coarse x "
        "n_sub &gt;= ~120 for DM &lt;=1500 and ~250 for DM &lt;=3000 at 0.26 ms; one rounding at the fine DM "
        "(already live); stage-2 at T/2 where the network allows. n_coarse must be decoupled from the number "
        "of search GPUs in the DM-plan schema."),
      PageBreak()]

# ------------------------------------------------------------------ 4 cost
pts = [cost[k] for k in cost]
rows = [["design point", "fine DMs", "images/s", "x today", "GPUs: today's in-situ", "today's kernels",
         "rewritten", "rewritten + FoV/2", "corr out Gb/s", "owner in Gb/s"]]
for p in pts:
    rows.append([p["name"], p["n_fine"], f"{p['images_per_s']:.2g}".replace("e+0", "e"), f"{p['x_today']:.1f}",
                 f"{p['gpu_today_wall']:.0f}", f"{p['gpu_today_kernels']:.0f}", f"{p['gpu_proj_full']:.1f}",
                 f"{p['gpu_proj_half']:.1f}", f"{p['corr_out_gbps']:.1f}", f"{p['owner_in_gbps']:.1f}"])
S += [P("4. What 4x costs", H2),
      P("Imaging and detection scale with (fine-DM trials) x (samples/s) x (pixels): one dirty image per "
        "trial per sample. Trials follow the legacy Levin recursion (tol 1.25, DM 50-1500 unless noted); "
        "per-image costs are today's measured in-situ cost (41.9 us = 201.3 ms cube / 4800 images), today's "
        "isolated kernels (combine 4.1 + irfft2 2.1 + mask 1.1 + Layer-1 0.6 + detector 5.8 us), and "
        "assumed costs for rewritten owner kernels (5.9 us; 3.5 us on a 256x128 FoV/2 grid) that "
        "benchmark b3 will test."),
      fig("figs/f5_compute.png", 6.6 * inch,
          "Fig. 5 - GPUs needed for imaging + detection. Lines: today's 8 search GPUs; 24 = the 16 corr "
          "second GPUs + 8 search GPUs; all 40."),
      table(rows, [1.75 * inch, 0.45 * inch, 0.55 * inch, 0.45 * inch, 0.62 * inch, 0.55 * inch, 0.55 * inch,
                   0.62 * inch, 0.5 * inch, 0.5 * inch]),
      P("Table 2 - Design points. Network columns: 1900 uv cells per (sub-band stream, coarse DM), cint8 "
        "unless noted; 'owner in' is one coarse-DM bucket. Using 0.52 ms sampling above DM 1500 cuts the "
        "DM-3000 case from 6.9e6 to 4.7e6 images/s. tol 1.4 instead of 1.25 removes 23% of trials.", CAP),
      PageBreak()]

# ------------------------------------------------------------------ 5 architectures
S += [P("5. Architectures", H2),
      fig("figs/f7_architecture.png", 5.9 * inch,
          "Fig. 6 - Today vs the proposed layout (your proposal, amended)."),
      table([
          ["option", "what", "verdict"],
          ["A. Scale today's topology", "keep corr -> 4 search nodes; 0.26 ms; 244 kHz; FoV/2; rewritten "
           "search kernels on 8 GPUs", "feasible only at 244 kHz (6.9 GPU-equiv. on 8) - gain fades with DM, "
           "none left by DM 1500; rate x%.2f-%.2f. Search NICs need 4-bit streams." % (rg1["x4_c8"], rg3["x4_c8"])],
          ["<b>B. Distributed imaging owners (your proposal, amended) - recommended</b>",
           "corr GPU A: one X-engine (slow + fast from one copy/unpack) + native-channel stage-1 + grid + "
           "4-bit TX; 24 owners (16 corr GPU B + 8 search GPUs) each image + detect one coarse bucket; search "
           "nodes also aggregate C1",
           "meets 4x at native channels to DM 1500 (DM 3000 with FoV/2); rate x%.2f-%.2f. Needs new owner "
           "kernels, schema change for n_coarse, NIC x8 slots advisable." % (rg1["x4_c1"], rg3["x4_c1"])],
          ["C. Single stage via time-multiplexed corner turn", "each owner gets the whole band for every "
           "24th time chunk", "rejected: latency tens of s (> 20 s voltage ring); 6-48x more uv-domain sums "
           "per image"],
          ["D. Ship sub-band or fine-DM images to search nodes for 'pure filtering'", "corr GPU B images; "
           "search nodes only boxcar", "rejected: a bucket's images are ~6-12x its uv streams "
           "(44-88 Gb/s out of each corr node) - far beyond the x4 NICs"],
          ["E. Dedispersing correlator", "stage-1 inside the GEMM: K spans (channel, sample, pol) with "
           "per-channel time offsets per coarse DM", "an option for GPU A: native precision, no 2.6 GB/block "
           "per-channel intermediate; cost = n_coarse full correlations (~1.3 TFLOP/s each): ~16 coarse DMs "
           "per GPU in fp16, several x more with int8 IMMA (b4)"],
          ["F. Triggered refinement (stop-gap)", "re-process the voltage ring at 0.26 ms around 1 ms "
           "candidates above a lowered threshold (~7 sigma) on the half-idle corr GPU0s",
           "cheap, uses existing pieces; narrow-burst threshold ~x1.36 better (rate ~x1.6) vs x2-3 for a full "
           "search; vulnerable to RFI floods"],
          ["G. 8x and beyond", "0.13 ms, or 2x up-channelisation (15 kHz / 65.5 us)", "+5-20% S/N at the "
           "narrowest widths for ~3.5x more imaging (Fig. 5) - needs more GPUs"],
          ["H. More GPUs", "e.g. 2 more modern GPUs per search node", "complements B; the corner-turn "
           "ingress then lands on 4 nodes and requires the NIC x8 fix"],
      ], [1.55 * inch, 2.5 * inch, 2.85 * inch]),
      P("Table 3 - Options considered.", CAP),
      P("<b>Slow visibilities from the fast correlator.</b> The two corr GPUs today each copy the same 288 "
        "MiB block (24 ms), unpack it and correlate it. One X-engine on GPU A can produce both products from "
        "one copy: the slow product as the large-K GEMM (96 antennas, XX/YY, 384 channels, integrated over the "
        "block - 8.9 ms measured), the fast product as core Stokes-I at 0.26 ms with calibration and RFI "
        "weights applied to the fast operand only. The slow product stays raw, uncalibrated, all-antenna and "
        "byte-identical at the bada contract, so the September objections to removing corr_slow (circular "
        "antennas.out cold start, un-flaggable antennas, the MS archive as RFI baseline, the HI guard, SPL's "
        "384 channels) do not apply. Process isolation is kept by running the X-engine as a small, stable "
        "process that writes bada and a best-effort GPU ring shared by CUDA IPC with the fast front-end "
        "process, so a crash in the complex part cannot stop the slow product."),
      P("<b>Why not pure filtering on the search nodes.</b> The data volume is smallest in the uv domain "
        "(~1900 cells per sub-band stream, measured from antennas.out; production 2134 at n_sub 4) and "
        "largest after imaging (32-65k pixels per image per fine DM). Imaging must therefore happen after "
        "the corner turn, and detection where the images are. Search GPUs are best used as 8 more owners.")]

# ------------------------------------------------------------------ 6 budgets
S += [P("6. Corr-node and network budgets", H2),
      fig("figs/f8_corr_budget.png", 6.6 * inch,
          "Fig. 7 - GPU time per 134.2 ms block: today (measured) vs the proposed GPU A (X-engine + front end "
          "at 0.26 ms, native channels) and GPU B (imaging owner, ~44 fine DMs, 256x128). Hatched segments are "
          "estimates that benchmarks b3/b4 test."),
      P("GPU A carries ~74 ms of compute plus 24 ms of DMA (overlappable) at 4x - feasible but only with a "
        "fast-path rewrite: the GEMM writes 384 native channels (no channel sum) at t_int 8, a fused "
        "multi-coarse-DM stage-1 + gridding kernel reads that 2.6 GB/block intermediate once, quantisation "
        "to 4 bits happens on the GPU, and TX is C (sendmmsg) instead of Python workers. The 11 GB card "
        "holds ~5-7 GB of working set. GPU B as owner needs ~80 ms per block (59%) at the assumed 3.5 "
        "us/image."),
      fig("figs/f6_network.png", 6.4 * inch,
          "Fig. 8 - Per-node NIC ingress. Every ConnectX-4 reads 'x4/8 8GT/s' in sysfs (n01, n02, n04, n09, "
          "n11, n13 checked; GPUs are x16). Corr nodes already take ~18 Gb/s of SNAP data."),
      P("<b>Network design:</b> 4-bit uv streams (efficiency ~0.99 for noise-dominated cells), 16x8 or 24x6 "
        "coarse x sub-bands, owners weighted so that search nodes take two buckets (14.8 Gb/s) and corr nodes "
        "one (25.4 Gb/s with SNAP - marginal on x4). Re-seating the NICs in x8 slots (a site visit; check "
        "the Supermicro slot map and n14's f0/f1 swap) makes every option comfortable and lets the spare 40G "
        "port carry the corner turn. The switch (Arista 7050QX-32) has the capacity (~0.5 Tb/s total).")]

# ------------------------------------------------------------------ 7 path + risks
S += [P("7. Recommended path", H2)]
S += bullets([
    "<b>P0 - next observing gap (30 min):</b> run bench/timeres (sec. 8). Check NIC slot options on one "
    "corr and one search chassis. Decide FoV/2 axis (256x128 grid halves one axis; 192^2 fp32 is the "
    "symmetric alternative).",
    "<b>P1 - software, at today's 1.05 ms (gains now):</b> unified X-engine on GPU A (slow + fast from one "
    "copy, frees GPU B on every corr node); native-channel stage-1 at native precision with n_coarse "
    "decoupled from the search GPUs (16x8); GPU-side 4-bit quantisation and a C transmitter. Alone worth "
    "x1.45 S/N at DM 1500 for narrow bursts (Fig. 3) and removes the corr real-time pressure.",
    "<b>P2 - imaging owners:</b> one owner service (GPU-side RX assembly, combine with large shift spreads, "
    "irfft2 on 256x128, chunk-max detector, Layer-1/2 fused) deployed on the 16 corr GPU Bs and 8 search "
    "GPUs at 1.05 ms first, then 0.52 ms (2x, ~5 GPU-equiv.), then 0.26 ms (4x). Scale C1/C2/C3 "
    "sample-denominated knobs x2 then x4.",
    "<b>P3 - reach:</b> DM to 3000 (0.52 ms above DM 1500), 0.13 ms only with more GPUs.",
])
S += [P("Risks and open questions", H2)]
S += bullets([
    "<b>Owner kernel efficiency</b> (3.5-6 us/image assumed vs 13.7 today) is the crux; b3 measures it with "
    "simple prototype kernels. If it lands at ~8 us, use tol 1.4 (-23% trials) or 0.52 ms sampling above "
    "DM ~800.",
    "<b>GPU A budget</b>: fast GEMM at K=16 and the 384-channel stage-1 are estimates (b4); the "
    "dedispersing correlator (option E) is the fallback.",
    "<b>Search volume x20</b> raises the false-alarm rate at fixed sigma: +0.3 sigma (~3% S/N) restores it. "
    "RFI behaviour at 0.26 ms (impulsive, satellites) is unknown; the flagger's finest window (M=64, 2.1 ms) "
    "is coarser than the sample and may need an M=8 stage.",
    "<b>Process isolation</b> for the unified X-engine is a design requirement (plan sec. 0), met by the "
    "X-engine / front-end process split.",
    "<b>Memory</b>: 11 GB GPUs and 93 GB search hosts; the RX ring must shrink (2 DM slabs, not 8) and "
    "the cube-retention ring move to compact visibilities with re-imaging on demand.",
    "<b>Models</b>: the emulator excludes imaging loss (x0.89, unchanged) and assumes white noise; the "
    "corr/owner budgets are extrapolations until b3-b5 run.",
])
S += [Spacer(1, 4), P("8. Benchmarks for the next observing gap", H2),
      table([["script", "node", "time", "firms up"],
             ["b0_selftest", "node GPU", "10 s", "prototype Triton kernels (combine, chunk-max) vs torch"],
             ["b1_fft", "search", "1 min", "irfft2 us/image: 256^2, 256x128, 192^2, fp16/fp32"],
             ["b2_detector", "search", "2 min", "detector bracket: read floor / production kernel / chunk-max, "
              "b64 and b256"],
             ["b3_owner_e2e", "search + corr", "3 min", "<b>fine DMs one GPU sustains in real time</b> "
              "(need &gt;=44 for DM 1500, &gt;=78 for 3000)"],
             ["b4_xengine", "corr", "2 min", "slow + fast GEMM per block at t_int 32/16/8/4, dedispersing-"
              "correlator shape, int8 vs fp16, PCIe"],
             ["b5_fabric.sh", "h23 -> 15 corr + 7 owners", "6 min", "owner ingress 7.4-20 Gb/s incl. corr "
              "owners with SNAP traffic"]],
            [1.0 * inch, 1.3 * inch, 0.55 * inch, 4.05 * inch]),
      P("rsync <font face='Courier'>bench/timeres/</font> to a search and a corr node and run "
        "<font face='Courier'>bash run_suite.sh 0 1</font> on each, then "
        "<font face='Courier'>bash bench/timeres/b5_fabric.sh</font> from h23. Each GPU script refuses to "
        "start while any process holds a GPU. Thresholds are in bench/timeres/README.md.", SM),
      PageBreak()]

# ------------------------------------------------------------------ appendix
S += [P("A. Method and sources", H2)]
S += bullets([
    "Emulator (<font face='Courier'>model/emulate.py</font>): 6144 native channels at 1498.75 - i x 30.518 "
    "kHz; per-channel arrival K DM nu^-2; native-channel smear 8.3 us DM dnu/nu^3; csf channels share one "
    "shift; stage-1 = q1 x rint(K DM_c (nu^-2 - nu_sb_top^-2)/q1); stage-2 = q2 x rint(K DM_f (nu_sb_top^-2 "
    "- nu_top^-2)/q2) (merged rounding); Gaussian burst; random DM within +/-3% and sample phase; best of "
    "the three nearest fine trials; boxcars 2^0..2^8. Validated against the September study and the live "
    "plan (<font face='Courier'>model/results/*.log</font>).",
    "DM trials: Barsdell / legacy <font face='Courier'>gen_dmtrials</font> recursion (ti 40 us), "
    "reproduces 218 trials over 100-1500 at 1.05 ms.",
    "uv cells: <font face='Courier'>model/nfilled.py</font> on _inspect/sensitivity/antennas.out (82 core "
    "antennas, 3321 baselines, 17.9 lambda cells, dec 71.6): 2000-2180 per stream at n_sub 4 (production "
    "2134), -> 1800 at n_sub 16; FoV/2 cells reduce this only ~5%.",
    "Measured numbers: corr_fast block 123.5-125.6 ms (09-23), corr_phase 99.5 ms (564ecc3), fused GEMM "
    "44 ms (c71dd4d), corr_slow GEMM 8.9 ms and 80.3 / 48 ms totals (09-22; unresolved discrepancy), x8 "
    "corr_fast 169 ms (05-10), stage-1 0.966 + 0.753 n_dm ms + 9 ms permute (n04), D2H 13.1 GB/s pinned "
    "(f751bf5); search v4 combine 19.6 ms (0ebac08), irfft2 13.78 ms / 6528 images (n02), detector 37.7 "
    "ms (n02), RX assembly 41.9 ms (9d88f5e), live 4.96-5.15 cubes/s; fabric 1x/4x/8x 6.4/25.6/51.3 Gb/s, "
    "knee 16-20 Gb/s per search node (09-22). Live: corr GPU0 ~47% / GPU1 ~87% SM-busy, search GPUs 75-90% "
    "(nvidia-smi dmon, 2026-10-21).",
    "Reproduce: <font face='Courier'>cd docs/timeres_study/model; python run_final.py; python "
    "run_dedisp_sweep.py; python cost.py; python figs.py; cd ..; python build_report.py</font> "
    "(numpy/matplotlib/reportlab; ~6 min on h23).",
])

doc = SimpleDocTemplate("DSA110_timeres_study.pdf", pagesize=letter, leftMargin=0.7 * inch,
                        rightMargin=0.7 * inch, topMargin=0.6 * inch, bottomMargin=0.7 * inch,
                        title="DSA-110 real-time search: time-resolution study",
                        author="dsa110-rt timeres-study")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote DSA110_timeres_study.pdf")
