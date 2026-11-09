"""Build DSA110_2x_development_report.pdf -- the human-readable account of the 2x time-resolution search:
what was built, why it is faster and more sensitive, sensitivity vs DM against 1x and a legacy-style search,
measured results (real FRB, injection campaign, production real-time margins), the compromises, and how to
operate it. Numbers that come from measurements are read from report_data.json (filled from the campaign and
production monitoring); the model figures from figs/ (make_figs.py).

    python build_report.py      (run from docs/x2/report/, dsart_h23 env: reportlab + matplotlib)
"""
import json
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2 = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e")
RULE, HEAD = colors.HexColor("#d9d8d3"), colors.HexColor("#f0efec")
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=19,
                    spaceAfter=4, textColor=INK)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15,
                    spaceBefore=10, spaceAfter=4, textColor=INK)
H3 = ParagraphStyle("H3", parent=H2, fontSize=10, leading=13, spaceBefore=6, spaceAfter=2)
B = ParagraphStyle("B", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.3, leading=12.4,
                   textColor=INK, spaceAfter=4)
BL = ParagraphStyle("BL", parent=B, leftIndent=12, bulletIndent=2, spaceAfter=2.5)
SM = ParagraphStyle("SM", parent=B, fontSize=7.9, leading=9.8, textColor=INK2)
CAP = ParagraphStyle("CAP", parent=SM, spaceBefore=2, spaceAfter=8)
TC = ParagraphStyle("TC", parent=B, fontSize=7.8, leading=9.4, spaceAfter=0)
TCB = ParagraphStyle("TCB", parent=TC, fontName="Helvetica-Bold")

D = json.load(open(os.path.join(HERE, "report_data.json")))
FJ = json.load(open(os.path.join(HERE, "figs", "figs.json")))


def P(t, st=B):
    return Paragraph(t, st)


def bullets(items, st=BL):
    return [Paragraph(t, st, bulletText="•") for t in items]


def numbered(items):
    return [Paragraph(t, BL, bulletText=f"{i}.") for i, t in enumerate(items, 1)]


def fig(path, width=6.9 * inch, caption=None):
    ir = ImageReader(path)
    w, h = ir.getSize()
    out = [Image(path, width=width, height=width * h / w)]
    if caption:
        out.append(P(caption, CAP))
    return KeepTogether(out)


def table(rows, widths, header=True):
    data = [[P(str(c), TCB if (header and i == 0) else TC) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), HEAD))
    t.setStyle(TableStyle(st))
    return t


def f2(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def build():
    s = []
    s += [P("DSA-110 real-time FRB search at twice the time resolution", H1),
          P(f"Development report, {D['date']}. The 2x search has been the production search since "
            f"{D['production_since']}.", SM), Spacer(1, 6)]

    # ---------------------------------------------------------------- summary
    s += [P("Summary", H2)]
    s += bullets([
        "<b>What changed.</b> The search now runs at <b>0.52 ms</b> resolution for DM 50-1600 "
        "and 1.05 ms for DM 1600-3000. The old (1x) search ran at 1.05 ms for DM 100-1500. "
        "Dedispersion now uses 61 kHz channels instead of 244 kHz, and the DM range is twice as wide.",
        "<b>What it buys.</b> In the loss-chain model, a 1 ms burst recovers <b>0.83</b> of the ideal "
        "matched-filter S/N at DM 100-1500, against 0.70 for 1x (<b>x1.19</b>). Narrower bursts gain more "
        "(x1.5 at 0.13 ms). At DM 1500 the gain is x1.7, because 1x ran out of trials there. DM 1500-3000 "
        "was not searched at all before.",
        f"<b>Measured.</b> {D['measured_summary']}",
        "<b>Real time, with margin.</b> The correlator path now uses about half its 134 ms block budget. "
        "1x used about 97% of it. Each GPU half of the search is busy 55-59% of the time against a 75% "
        "gate, and every search node carries about 11 Gb/s of corner-turn traffic.",
        f"<b>Status.</b> {D['status_summary']}",
    ])

    s += [P("1x and 2x side by side", H3),
          table([["", "1x dsart (until 2026-11-08)", "2x dsart (production)"],
                 ["Search sampling", "1.05 ms", "0.52 ms (DM 50-1600), 1.05 ms (1600-3000)"],
                 ["Dedispersion channels", "244 kHz (8 native channels summed)",
                  "61 kHz (2 summed); stage 1 at 0.26 ms"],
                 ["DM range / trials", "103-1496, 8 coarse x 25 fine = 200",
                  f"50-2996, {D['n_coarse']} coarse, {D['n_fine']} fine (two tiers)"],
                 ["Corner-turn format", "cint8, per-row max-abs scale (lossy above ~50 sigma)",
                  "4-bit gridded cells + 8-bit heavy cells, robust scale, sqrt(n) weighting"],
                 ["Network", "~5 Gb/s per search node, 2-fragment frames",
                  "~11 Gb/s per search node, single-datagram frames, paced C sender"],
                 ["Search ownership", "8 halves, contiguous DM ranges",
                  "8 halves, interleaved DMs (3 coarse DMs each)"],
                 ["Raw data kept for cutouts", "15 GB ring + cube files",
                  "20 s of every DM (29 GB per node): cutouts on demand"],
                 ["Correlator block time", "~129 ms of 134 ms (97%)", D["corr_block_summary"]]],
                [1.35 * inch, 2.45 * inch, 3.1 * inch])]

    # ---------------------------------------------------------------- sensitivity
    s += [PageBreak(), P("Sensitivity vs DM", H2),
          P("The figure compares three searches using one loss model: dedispersion trial spacing, "
            "sampling, and intra-channel smearing, as in the time-resolution study. The 2x curve includes "
            "the deployed corner-turn format's 0.55% quantisation cost. The legacy-style curve is a "
            "single-stage search at 0.26 ms with 244 kHz channels, which is what the legacy dsaX "
            "beam search did."),
          fig(FJ["sensitivity"], caption=(
              "Recovered fraction of the ideal matched-filter S/N vs DM at four burst widths (running "
              "median over DM). Blue: 2x (deployed). Orange dashed: 1x. Green dotted: legacy-style "
              "single-stage search. The vertical line marks the 2x tier boundary at DM 1600; above it "
              "2x samples at 1.05 ms."))]
    s += [P("How to read it", H3)]
    s += bullets([
        "<b>At 1 ms and wider, 2x is the best of the three at every DM.</b> At low DM it matches the "
        "legacy-style search to within a few percent. Above DM ~800 it beats it, because two-stage "
        "dedispersion with 61 kHz channels smears less than 244 kHz single-stage.",
        "<b>Below 0.5 ms, at low DM, the legacy-style search is still ahead.</b> At 0.2 ms it recovers "
        "0.7 against 0.52 for 2x. Its 0.26 ms sampling beats 2x's 0.52 ms. Matching it needs the 4x "
        "option, which is network-bound (see compromises). 2x still recovers 1.5x more than 1x there.",
        "<b>DM 1500-3000 is new sky.</b> 1x and the legacy-style search fall off at DM 1500. 2x keeps "
        "0.72-0.75 at 1 ms up to DM 3000.",
    ])
    tab = [["DM band", "2x 0.5 ms", "1x 0.5 ms", "legacy 0.5 ms", "2x 1 ms", "1x 1 ms", "legacy 1 ms",
            "2x 4 ms", "1x 4 ms"]]
    for r in FJ["table"]:
        def g(key, w):
            v = r.get(key)
            return "-" if not v else f"{v[w]:.2f}"
        tab.append([r["band"], g("prop_ns4", "0.5"), g("current", "0.5"), g("single", "0.5"),
                    g("prop_ns4", "1"), g("current", "1"), g("single", "1"), g("prop_ns4", "4"),
                    g("current", "4")])
    s += [table(tab, [0.75 * inch] + [0.68 * inch] * 8),
          P("Mean recovered fraction per DM band (model). A dash marks a band that search does not cover.",
            CAP)]

    # ---------------------------------------------------------------- measured
    s += [PageBreak(), P("What was measured", H2)]
    s += [P("A real FRB through both searches", H3),
          P("FRB 250924mptq (DM 404.7) was run from recorded voltages through both the old and the new "
            "searches. 2x put it at exactly the same pixel as 1x, at the predicted DM trial, with "
            "<b>S/N 266 against 145 (x1.84)</b>. This is the model's x1.25 at 1 ms, multiplied by the "
            "removal of 1x's lossy wire compression for bright bursts (0.68 at ~150 sigma per sample).")]
    s += [P("Injection calibration campaign", H3), P(D["campaign_text"])]
    if FJ.get("k_shape") and os.path.exists(FJ["k_shape"]):
        s += [fig(FJ["k_shape"], caption=D.get("k_shape_caption", ""))]
    if D.get("gain_text"):
        s += [P("Against 1x", H3), P(D["gain_text"])]
    if os.path.exists(FJ["gain"]):
        s += [fig(FJ["gain"], caption=D["gain_caption"])]
    if D.get("campaign_table"):
        s += [table(D["campaign_table"], D.get("campaign_table_widths") or None)]
        s += [P(D.get("campaign_table_caption", ""), CAP)]
    s += [P("Real time in production", H3), P(D["realtime_text"])]
    if D.get("realtime_table"):
        s += [table(D["realtime_table"], [2.2 * inch, 2.2 * inch, 2.5 * inch])]

    # ---------------------------------------------------------------- how
    s += [PageBreak(), P("What made it possible", H2),
          P("A twice-as-fine search is about 4x the work per second in every stage: twice the samples, "
            "four times the DM trials, and a corner turn carrying twice the data. Every stage had to get "
            "faster, not just the search.")]
    s += [P("Correlator (corr_fast, one GPU per node)", H3)]
    s += bullets([
        "<b>A new GEMM</b> computes only the core baselines the search images. It loads and calibrates "
        "voltages directly in the multiply, and writes fp16 output in time-minor order. This replaced "
        "separate cal, mask and copy kernels.",
        "<b>A new stage-1 dedispersion kernel</b> runs at 0.26 ms. It is driven by a cell lookup table, "
        "stages each data window in shared memory once, and computes every DM of a tier in registers. "
        "Using four sub-bands (n_sub 4) halves its cost for under 0.5% of S/N.",
        "<b>A GPU packer</b> codes each value in 4 bits against a robust running noise scale, weighted by "
        "sqrt(n) so every value carries equal noise. The heaviest cells, about 1% of values but 20% of "
        "the weight, travel at 8 bits so bright bursts never clip.",
        "<b>A C sendmmsg sender</b> runs outside Python's lock, paced in 16 KB bursts. With 512 KB bursts, "
        "the incast microbursts overflowed the search NICs' port buffers and lost 1-5.6% of the data; "
        "16 KB pacing gives zero NIC discards up to 17 Gb/s per search node.",
        "<b>Asynchronous host-to-device copies, deferred host reads and pre-allocation</b> take the "
        "per-block GPU time from about 129 ms to about 50 ms.",
    ])
    s += [P("Search (8 GPU halves on 4 nodes)", H3)]
    s += bullets([
        "<b>A per-DM raw ring</b> holds 20 s of every owned DM. It replaces the 1x cube files, and cutouts "
        "for C2 events are re-imaged on demand from it.",
        "<b>The owner engine</b> is a chain of custom kernels: unpack, combine through a lookup table, a "
        "fused FFT imager (column then row inverse FFT, mask and scale in one pass), and a register-tiled "
        "detector with a B-spline filter bank of 8 widths (0.52-33 ms). It runs at 3.5-3.6 us per "
        "256x256 image, all-in.",
        "<b>Interleaved DM ownership:</b> half h owns coarse DMs d with d mod 8 = h. Every half sees "
        "both tiers, so load is balanced.",
        "<b>Fan-in on data time, not wall-clock time,</b> releases a cycle when 60 of the 64 streams are "
        "complete. The known fabric loss on n01/n02 therefore never stalls the search.",
        "<b>Priming:</b> each engine runs two throwaway cycles on in-memory data before attaching. This "
        "fixed a start-up lag-recovery seek found in the first live fleet run.",
    ])
    s += [P("Downstream, monitoring and operations", H3)]
    s += bullets([
        "C1, C2 and C3 work in external-clock units (0.524 ms samples). C2 never discards an event with "
        "missing cubes, and cutouts come from the raw ring.",
        "Injections use a calibration factor K per (DM, width) bucket, so the bot can inject at a "
        "chosen S/N across the whole width range.",
        "One code base supports both op-points; configs are pushed per op-point to etcd. The web UI "
        "Start offers only 2x, and an x2 health page shows block times, duty, loss and fan-in.",
        "A test-capture tool replaces the Grafana drop-counter check after SNAP restarts. It also "
        "measures each SNAP's sequence offset directly.",
    ])

    # ---------------------------------------------------------------- compromises
    s += [PageBreak(), P("Compromises", H2),
          P("Everything below was either decided with you at a checkpoint or found during "
            "deployment. Costs are relative to an ideal implementation of the same design.")]
    s += numbered(D["compromises"])

    s += [P("Found and fixed while deploying", H2)]
    s += bullets(D["fixed_in_deploy"])
    if D.get("incidents"):
        s += [P("Incidents on the first day of 2x observing", H2)]
        s += bullets(D["incidents"])

    # ---------------------------------------------------------------- operating
    s += [PageBreak(), P("Operating it", H2)]
    s += bullets(D["operating"])
    s += [P("Open items and recommendations", H2)]
    s += numbered(D["open_items"])
    s += [Spacer(1, 8), P(D["provenance"], SM)]

    out = os.path.join(HERE, "DSA110_2x_development_report.pdf")
    SimpleDocTemplate(out, pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                      topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                      title="DSA-110 2x time-resolution search: development report").build(s)
    print(out)


if __name__ == "__main__":
    build()
