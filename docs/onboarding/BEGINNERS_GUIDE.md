# A Beginner's Guide to the DSA-110 Real-Time FRB Search Pipeline (`dsart`)

Welcome to the project. This guide is written for someone who knows what a
Fast Radio Burst (FRB) and a dispersion measure are, roughly, but has never
worked with a radio correlator, PSRDADA, etcd, or this codebase. The goal is to
get you from "I have no idea what any of these words mean" to "I can read the
code, run an observing session, and understand what the pipeline is doing and
why." Read it like lecture notes, not a spec sheet.

A note on trust: every concrete number, path, port, and etcd key below is cited
to a source file (repo-relative, under `dsa110-rt/` unless noted) or to one of
the four research packets in `research/`. Where the truth is genuinely unknown
or undocumented, the guide says so out loud — those honest gaps are things you
should ask the team about, not things to guess at.

---

## Table of contents

1. [What is DSA-110 and what are we looking for?](#1-what-is-dsa-110-and-what-are-we-looking-for)
2. [Image-plane FRB searching, explained from scratch](#2-image-plane-frb-searching-explained-from-scratch)
3. [The machines and where to find things](#3-the-machines-and-where-to-find-things)
4. [The journey of a voltage sample](#4-the-journey-of-a-voltage-sample)
5. [Operating the system](#5-operating-the-system)
6. [Injections — how we know the search works](#6-injections--how-we-know-the-search-works)
7. [Voltage dumps and the trigger cascade](#7-voltage-dumps-and-the-trigger-cascade)
8. [What works, what's partial, what's missing](#8-what-works-whats-partial-whats-missing)
9. [Glossary](#9-glossary)
10. [Where to read next](#10-where-to-read-next)

---

## 1. What is DSA-110 and what are we looking for?

The **Deep Synoptic Array (DSA-110)** is a radio interferometer at the Owens
Valley Radio Observatory (OVRO), described in the project overview as "a
110-element radio interferometer at OVRO operating in the 1.31–1.50 GHz band"
(`docs/overview/dsa110-rt-overview.tex:193`). An *interferometer* is an array of
many small dishes that are combined electronically to act, collectively, like
one enormous telescope. Instead of one big mirror, you have 110 antennas whose
signals are cross-multiplied against each other; the resolution you get is set
by the largest *baseline* (the distance between the two most widely separated
dishes), not by any single dish.

**An important, honest caveat up front.** The docs say 110 elements, but the
real-time pipeline actually processes **96 "online" antennas**
(`configs/corr_setup_96.yaml:222`; research packet 01 §5). Why the gap between
110 and 96? *It is nowhere reconciled in the documentation.* There is an
inference — the config carries "outrigger" antennas 110, 113, 114, 115 with
large cable delays (`REALTIME_FRB_SEARCH.md:708`) — but nobody has written down
the definitive explanation. Treat "110 vs 96" as an open question to ask the
team, not something to paper over.

**How it points.** DSA-110 is a **drift-scan** (also called *meridian-transit*)
instrument. It does not slew to track a source across the sky. The dishes are
fixed in the east–west sense and can only be tilted in **declination** (the
north–south sky coordinate); the sky then *drifts* through the beam as the Earth
rotates. So the array observes whatever strip of sky is currently crossing the
local meridian at the chosen declination. The single live pointing parameter is
therefore just one number — the declination — published at the etcd key
`/mon/array/dec` (`docs/overview/dsa110-rt-overview.tex:4173`). This drift-scan,
declination-only geometry is a recurring theme: it is *why* the imaging math can
use a static, pre-computed projection valid only at meridian transit (more in §2).

**What's an FRB, and why real-time?** A Fast Radio Burst is a millisecond-scale
flash of radio emission, most of them extragalactic, of still-debated origin.
Because they travel through the ionized plasma between us and the source, lower
radio frequencies arrive *later* than higher ones — the burst is *dispersed*.
The total delay is proportional to the **dispersion measure (DM)**, the
integrated column of free electrons along the line of sight.

Detecting an FRB in your data hours later is scientifically nice but limited.
The real prize — localizing the burst to a host galaxy, and capturing its
microsecond-scale internal structure — requires the **raw voltage data** from
around the moment of the burst. Those voltages exist only fleetingly in a
memory ring buffer that is constantly being overwritten (a ~15-second retention
window; see §7). If you don't decide to *save* that slice within a few seconds,
it's gone forever. That is the entire reason this pipeline has to run in real
time: **detection within seconds is what makes a voltage dump — and therefore
localization and microstructure — possible at all.**

---

## 2. Image-plane FRB searching, explained from scratch

This is the conceptual heart of the project. Take your time here.

### 2.1 What an interferometer actually measures

When you cross-correlate the voltage streams from two antennas, you get a
complex number called a **visibility**. Each visibility corresponds to one
**baseline** (one pair of antennas) and one frequency channel. There is a deep
theorem (the van Cittert–Zernike theorem) that says: *the visibility measured on
a given baseline is one Fourier component of the sky brightness.* The baseline's
geometry — specifically its projection onto the plane perpendicular to the
pointing direction, measured in wavelengths, giving coordinates called
**(u, v)** — tells you *which* Fourier component. Short baselines sample
large-scale sky structure; long baselines sample fine detail.

With 96 antennas you have 96·97/2 = **4656 baselines**
(`docs/overview/dsa110-rt-overview.tex:360`; research packet 01 §2), i.e. 4656
samples of the sky's Fourier plane per channel per time step. To turn that back
into a picture of the sky, you *grid* those samples onto a regular 2-D grid in
(u, v) space and take an inverse 2-D Fourier transform. The result is a **dirty
image** — "dirty" because your incomplete sampling of the (u, v) plane convolves
the true sky with a messy point-spread function. For a *transient* search, the
dirty image is fine: a real point source still shows up as a bright compact blob.

### 2.2 The classic alternative: beamformed searching

Before you can appreciate what `dsart` does, you need to know the approach it
replaces. The legacy DSA-110 single-pulse search, called **Hella**
(`dsaX_hella`), used **beamforming** (research packet 03; `REALTIME_FRB_SEARCH.md:475`).

Beamforming means: take the antenna voltages, apply per-antenna phase weights,
and sum them so the array becomes maximally sensitive to *one specific direction
on the sky* — a "pencil beam." Choose 512 different sets of weights and you form
**512 pencil beams** tiling the field of view. For each beam you now have a
single time series; you dedisperse that time series at many trial DMs and run a
**boxcar match filter** (slide boxcars of various widths along the series,
looking for a signal-to-noise bump — a boxcar is the optimal matched filter for
a flat-topped pulse of unknown width). Hella ran 2 instances per search node and
shipped candidates to a coincidencer (`REALTIME_FRB_SEARCH.md:475-538`).

Beamforming is cheap and well-understood, but it has two real weaknesses:

- **The beam grid undersamples and tapers the field.** Pencil beams have finite
  width and gaps; a burst landing between beam centers is detected at reduced
  sensitivity, and its position is only known to "which beam(s) lit up."
- **Localization is coarse.** A beam is a fuzzy region, not a sky coordinate.

### 2.3 The image-plane approach `dsart` uses

`dsart` flips the problem around. Instead of forming a fixed grid of beams, it
**makes an actual image of the whole field for every time step and every trial
DM, and searches every pixel.** An image *pixel is a sky position*, so a
detection comes with its coordinates for free, at full field-of-view
sensitivity everywhere. The pipeline per corr node does, in order (fast path,
`docs/overview/dsa110-rt-overview.tex:602-766`):

1. **Correlate** the voltages into fast visibilities (a GEMM — a big
   matrix-multiply — on the GPU), summing the two polarizations into a single
   **Stokes-I** (total-intensity) product, `V_I = V_XX + V_YY`
   (`docs/overview/dsa110-rt-overview.tex:357`). Everything downstream is
   single-polarization.
2. **Dedisperse *before* imaging.** This is the clever bit. For each trial DM,
   you know exactly how much each frequency channel is delayed relative to the
   top of the band. So you apply per-channel integer *time shifts* to the
   visibilities to undo the dispersion sweep — *then* you image. If your DM guess
   is right, the burst's energy from all channels lands in the same time step.
3. **Grid** the dedispersed visibilities onto a 256×256 (u, v) grid using a
   static projection valid at meridian transit (HA = 0),
   `u_m = b_e`, `v_m = b_n·cos(φ_lat − δ₀)`
   (`docs/overview/dsa110-rt-overview.tex:695`).
4. **2-D FFT** to produce a dirty image for that time step and that DM trial.
5. **Search every pixel** of every image for a compact transient using boxcar
   matched filters in time.

So for each moment in time you produce a *stack* of images — one per trial DM —
and you hunt for point-like flashes across all of them. A candidate is a
(pixel = sky position, time, DM, width, S/N) tuple.

### 2.4 Why it's worth the pain

Full field-of-view sensitivity with **direct localization of every candidate**:
because a pixel *is* a sky coordinate, you don't have the beam-grid tapering
problem, and you know where every candidate came from without a follow-up step.
That is a big scientific win for an FRB survey whose whole point is host-galaxy
association.

The cost is **compute**. Making a full image for every DM trial for every time
step is enormously more arithmetic than dedispersing a few hundred beam time
series. The engineering of `dsart` is largely a story of making that affordable:

- **GPUs everywhere** — the correlation GEMM, the gridder, the FFT imager, and
  the detector all run on GPUs.
- **Aggressive numerics.** Visibilities are quantized to **int8** before
  transport and imaging; the FFTs run in **fp16** (half precision). This is a
  survey for bright transients, not precision spectroscopy, so the reduced
  dynamic range is acceptable — but it has a sharp edge you must respect (an
  fp16-cuFFT *overflow cliff* around fluence 1×10⁻³; see §6).
- **A strict FLOP budget.** The design pins roughly **~1 TFLOP per cube** and
  forbids expensive-but-tidy operations. For example the detector computes
  boxcars via a **cumulative-sum (`cumsum`) trick** — a prefix sum lets any
  boxcar width be read off as a subtraction of two array elements — *because a
  literal `conv1d` convolution would blow the FLOP budget*
  (`detector/forward.py`; research packet 00). When you see "conv1d forbidden"
  in the code, that's why.
- Real op-point: M7.7 hit **~166 ms/cube against a ~201 ms budget** (~17%
  headroom) (`docs/overview/dsa110-rt-overview.tex:114`; research packet 03).
  Note these are *two different budgets in different units*: the ~1 TFLOP/cube
  figure is a FLOP-count ceiling on the detector module specifically, while the
  166-vs-201.3 ms figure is a wall-clock latency budget on the whole
  search-side cube pipeline.

### 2.5 Dispersion, and the two-stage DM trick

The dispersion delay uses the standard pulsar-astronomy constant
(`docs/overview/dsa110-rt-overview.tex:344`):

```
τ(ν, DM) = K_DM · DM / ν²,     K_DM = 4.148808 ms·GHz²·pc⁻¹·cm³
```

Concretely, across the processed band a DM = 3000 pc·cm⁻³ source sweeps by about
1697.5 ms (`docs/overview/dsa110-rt-overview.tex:349`). You don't know the DM in
advance, so you must search many *trial* DMs — and, as §2.3 explained, each trial
means its own set of per-channel time shifts and its own image stack.

Here is the network-saving trick. Dedispersion is split into **two stages**:

- **Coarse-DM, on the corr side, *before* transport.** The 16 correlator nodes
  each handle a slice of the band. Each applies a small set of **8 coarse-DM
  trials** as per-channel integer shifts, sums channels, and — this is the point
  — ships only the *coarsely-dedispersed, quantized* product over the network
  (`docs/overview/dsa110-rt-overview.tex:673`; research packet 00). The
  "Option A" geometry has the corr side absorb the *entire* coarse-DM shift, so
  the residual the search side must handle spans only ±83 samples (down from
  ±1432) (`docs/overview/dsa110-rt-overview.tex:686`).
- **Fine-DM, on the search side.** The 4 search nodes combine the streams from
  all 16 corr nodes and apply the small *residual* fine shifts — 34 fine-DM
  trials per GPU half around each coarse-DM bucket
  (`fine_dm/combiner.py`; `docs/overview/dsa110-rt-overview.tex:841-935`).

Why split it this way? Because the expensive thing to move is *network
bandwidth between corr and search nodes*. If you shipped every fine-DM trial's
data across the fabric you'd need enormous bandwidth. By coarsely dedispersing
and channel-summing *first*, each corr node ships a compact int8 stream, and the
fine structure is reconstructed cheaply at the far end. The coarse stage
throws away almost no sensitivity because the fine stage fills in the residual.

---

## 3. The machines and where to find things

The system is a small fleet of Linux hosts at OVRO, reached from your laptop
through a chain of SSH jumps. Getting oriented on *which host does what* saves a
lot of confusion.

### 3.1 Host map

| Host (short) | DNS / role | What runs there |
|---|---|---|
| your laptop | — | `dsa110-operator` agent console (`python -m dsa_operator.web.app`), SSH tunnels |
| `ovro` | `ssh.ovro.caltech.edu` | outermost SSH gateway |
| `dsa110maas` | `dsa110maas.ovro.pvt` | second SSH hop |
| **`h23`** | `lxd110h23.pro.pvt` | **head node**: dashboard (`dsa_monitor`), C2 coincidencer, C3 voltage collector, etcd access, hiplot |
| 16 **corr nodes** | see below | capture, correlation, RFI/cal/GEMM/grid/coarse-DM, transport TX, slow-vis |
| 4 **search nodes** | see below | transport RX, fine-DM, imaging, detection, clustering, C1 emit |
| `h20` | `lxd110h20.pro.pvt` | Grafana + InfluxDB monitoring (read-only tier) |
| SNAPs | (physical location undocumented) | FPGA F-engines feeding corr nodes over 40 GbE UDP |

**Corr nodes (16), one per "chgroup":** `h03, h04, h05, h06, h07, h08, h10,
h11, h12, h14, h15, h16, h18, h19, h21, h22` — referred to in ops tooling as
`n03 … n22` (`REALTIME_FRB_SEARCH.md`; research packet 00). A *chgroup* (channel
group) is the slice of the frequency band that one corr node processes; there
are 16 of them (§4, §9).

**Search nodes (4):** `h01, h02, h09, h13` = `n01/n02/n09/n13`, two GPUs each
(research packet 00). In the legacy system these owned beam slices 0–127 /
128–255 / 256–383 / 384–511.

The SNAP boards stream UDP into the corr nodes over 40 GbE, but **where the
SNAPs physically sit (at the antennas vs. a central hut) is not documented** in
any of the source material (research packet 01 §1) — another team question.

### 3.2 Conda environments — and the golden rule about h23

Two Python environments coexist on the fleet (research packet 03; workspace-root
`dsa110-rt_revamp_7b1d2669.plan.md:167`):

- **`dsa110-rt`** — Python 3.11, the modern `dsart` pipeline.
- **`casa38`** — Python 3.8, the *legacy* environment, still used for
  `meridian_fringestop`/`dsamfs` (the calibration path — see §4).

**The golden rule from `CLAUDE.md`:** all GPU work and all `dsart` tests run on a
**corr or search node, never on h23.** The `dsa110-rt` environment on h23 exists
but is not usable for anything involving GPUs. If you try to run a GPU test on
the head node it will fail in confusing ways. Don't.

### 3.3 The web UIs and the SSH port-forward table

Nearly everything you'll want to look at is a web UI behind SSH. The forwards:

| UI | Where it lives | Forward |
|---|---|---|
| `dsa_monitor` dashboard | `h23:5778` (Flask, `dsa_monitor.service`) | `ssh -L 5778:localhost:5778 h23` |
| Grafana | `lxd110h20.pro.pvt:3000` | `ssh -L 3000:lxd110h20.pro.pvt:3000 h23` |
| InfluxDB 1.x (db `dsa110`) | `lxd110h20.pro.pvt:8086` | `ssh -L 8086:lxd110h20.pro.pvt:8086 h23` |
| hiplot | `h23:5027` | `ssh -L 5027:localhost:5027 h23` |
| operator console | your laptop `127.0.0.1:8787` | needs tunnels `12379→etcdv3service.pro.pvt:2379` and `15778→h23:5778` |
| etcd | `etcdv3service.pro.pvt:2379` | API, not a UI |

(Sources: research packet 00 "Web UIs / port forwarding".) InfluxDB has no web
page — you query it with e.g. `GET /query?db=dsa110&q=...`; a bare `/ping`
returns 204. The DNS aliases `grafanaservice.pro.pvt`, `influxdbservice.pro.pvt`,
and `lxd110h20` all point at the same box (`10.42.0.249`). The hiplot on port
5027 is specifically the **C2** instance (`hiplot_c2.service`); a second
instance for **C1** runs on port 5017.

**The ControlMaster gotcha (you *will* hit this).** OpenSSH multiplexes
connections via a ControlMaster socket, and it keeps your *old* `LocalForward`
settings alive even after you edit your config. If a forward seems stale or
wrong, don't just reconnect — explicitly cancel/add forwards on the live master:

```bash
ssh -O cancel -L 8086:old:8086 h23     # drop a stale forward
ssh -O forward -L 8086:influxdbservice.pro.pvt:8086 h23   # add the right one
ssh -O exit h23                        # or nuke the master entirely and restart
```

(One real bug we hit: a `LocalForward 8086 localhost:8086` on `dsa110maas`
pointed at nothing — the correct target is `influxdbservice.pro.pvt:8086`;
research packet 00.)

---

## 4. The journey of a voltage sample

Let's follow the data from an antenna to a saved candidate. Two facts frame
everything: **the data plane is PSRDADA shared-memory ring buffers, not function
calls**, and **the control plane is etcd, not RPC** (§5). Stages are *separate
processes* that attach to named ring buffers as readers or writers.

### 4.1 The front end: SNAPs, packets, and channels

A **SNAP** is an FPGA board acting as the array's **F-engine** ("F" for
Fourier/frequency): it digitizes each antenna's analog voltage and channelizes
it into frequency bins, then streams the result as UDP packets
(`docs/overview/dsa110-rt-overview.tex:248`; research packet 01 §1). The system
has `NSNAPS = 32` SNAPs, each carrying **3 antennas** (32×3 = 96 = the online
antenna count) (`REALTIME_FRB_SEARCH.md:236-291`).

Each SNAP UDP packet payload is `[3 ants, 384 chans, 2 times, 2 pols, 4-bit
complex]` = **4608 bytes** (`REALTIME_FRB_SEARCH.md:258`). The voltages are
4-bit complex, nibble-packed (low nibble = real, high = imag), two's-complement,
scaled by ×0.05 into fp16 range (research packet 01 §2). Native sample period is
32.768 µs; one **specnum** (the SNAP packet sequence number, §5.3) equals 2
native samples = 65.536 µs; a processing "block" is `0.134217728 s`
(`docs/overview/dsa110-rt-overview.tex:199-205`).

The full band is 1.53 GHz down, 250 MHz wide, 8192 native channels; the pipeline
processes only the contiguous middle **6144 channels (1024–7167)**, split into
**16 chgroups of 384 channels** — one chgroup per corr node — dropping 2048
noisy edge channels (`docs/overview/dsa110-rt-overview.tex:1249-1272`;
`configs/chgroup_assignments.yaml`). The processed band is ν_top = 1.49875 GHz
down to ν_bot ≈ 1.31128 GHz. The fast path further sums channels 8× down to 48
channels per chgroup (768 across the fleet), Δν_eff ≈ 244.14 kHz
(`docs/overview/dsa110-rt-overview.tex:667`).

Each corr node runs **two capture processes**, each ingesting 16 SNAPs (a "SNAP
pair" = 48 antennas). Pair A arrives on **UDP port 4011** and writes ring `dada`;
pair B on **UDP port 4012** and writes ring `eada`
(`REALTIME_FRB_SEARCH.md:275-286`; `configs/dsart_pipeline_rt.yaml:230,251`).
Capture is a C binary, `dsart_capture_manythread` (uses `recvmmsg` to drain
packets in batches), NUMA-pinned to the correct GPU socket (research packet 01
§1, §3).

### 4.2 The ring buffers and why reader counts are sacred

**PSRDADA** is a shared-memory ring-buffer library widely used in pulsar/FRB
instruments. A "DADA ring" (also called a *DADA buffer*) is a named chunk of
shared memory carved into `n` blocks of `b` bytes each, with one **writer** and
some number of **readers**. The writer fills blocks in a circle; readers consume
them. Crucially:

> **The number of readers is declared up front and is physics — not a
> suggestion.** If a ring is configured for `r` readers but only `r−1` actually
> attach, the *writer stalls* waiting for the missing reader to consume the
> block. This is **backpressure**, and it is by design: it guarantees no data is
> silently dropped. The configs call this out repeatedly — e.g. `fada` is `r=3`,
> `bada` is `r=2` with a null-drain standing in for an optional reader
> (`configs/dsart_pipeline_rt.yaml`; `CLAUDE.md`).

So if you ever add a stage that reads from a ring, you **must** bump `r` in the
config, or you'll wedge the whole pipeline. This is the single most common way to
break things by accident.

The named corr-side buffers (research packet 00):

| Ring | Size × blocks | Readers | Role |
|---|---|---|---|
| `dada` | 150,994,944 B × 20 (~2.7 s) | — | capture pair A (port 4011) |
| `eada` | 150,994,944 B × 20 (~2.7 s) | — | capture pair B (port 4012) |
| `fada` | 301,989,888 B × 70 (~9.4 s) | **r = 3** | merged voltages: slow, fast, voltage-retention |
| `bada` | 28,606,464 B × 300 | r = 2 → now sole `meridian_fringestop` | slow-vis for calibration |

The **three `fada` readers** are the fork in the road: (1) the **slow** path,
(2) the **fast** (search) path, and (3) **voltage retention** — an in-RAM
`VoltageRing` written by a single-writer seqlock, so that disk voltage dumps can
never back-pressure the capture front end (research packet 00). Note that the
VoltageRing is a *separate buffer* from `fada` with its own, deeper window: it
retains ~15 seconds of voltages (`--retention-s 15.0`,
`configs/dsart_pipeline_rt.yaml:539`; `src/dsart/dump/voltage_ring.py:23` —
~112 blocks ≈ 31.5 GiB resident), whereas the `fada` PSRDADA ring itself is
only ~9.4 s deep. Hold that third reader in mind; it's what makes §7's voltage
dumps possible.

### 4.3 The flow, end to end

```
 SNAP FPGAs (32, 4-bit complex, 2-pol UDP)
        │  40 GbE, port 4011 (pair A) / 4012 (pair B)
        ▼
   capture (C: dsart_capture_manythread, recvmmsg)
        │  writes
        ▼
   dada / eada rings ──► dsaX_merge ──► fada ring (r=3, ~9.4 s)
                                          │
        ┌─────────────────────────────────┼───────────────────────────────┐
        │ (reader 1: SLOW, GPU0)          │ (reader 2: FAST, GPU1)          │ (reader 3: voltage retention,
        ▼                                 ▼                                  in-RAM VoltageRing → §7)
  corr_slow_compute                 corr_fast_compute (plan §4.2)
        │  bada ring                       │  int4 unpack → autos
        ▼                                 │  → RFI flag (SK + bandpass + group
  meridian_fringestop (casa38)            │     + SumThreshold + flagants, OR-combined)
        │  UVH5                            │  → zero-fill flagged (ant,ch,pol)
        ▼                                 │  → cal + bandpass + DEC-phase weights
   calibration (SEFDs,                    │  → GEMM fast visibilities → Stokes-I
   beamformer weights)                    │  → GPU gridder (256×256 uv)
                                          │  → static-sky subtract (causal 8-block mean)
                                          │  → coarse-DM dedisperse (8 trials, int shifts)
                                          │  → int8 quantize
                                          ▼
                                    transport TX (UDP ProdFrame, 72-B header)
                                          │  port 6625 + chgroup
                                          ▼
                                    search_rx  (POSIX-shm ring /dsart-rxring-<cn>, ~9.8 GiB)
                                          │  reassembles from all 16 corr nodes
                                          ▼
                                    fine-DM combine (34 trials/GPU, residual shifts)
                                          ▼
                                    imager (dequant + irfft2 in fp16 → dirty images)
                                          ▼
                                    noise normalization (layer-1 global + layer-2 per-kernel EMA)
                                          ▼
                                    detector (boxcar via cumsum; NMS; 4D merge)
                                          ▼
                                    clusterer (HDBSCAN, cityblock)
                                          ▼
                                    C1 emit  ──TCP──►  h23:11500
                                          ▼
                                    C2 coincidencer (h23): cross-node coincidence + vetoes
                                          ▼
                                    C3 (h23): voltage-dump collection on trigger
```

A few notes on that diagram:

- **`dsaX_merge`** interleaves the two capture rings (`dada`, `eada`) into the
  combined `fada` ring, restoring antenna order (research packet 01 §5).
- The **slow path** deliberately produces *uncalibrated* visibilities in a
  byte-for-byte legacy-compatible format (`bada`), which
  **`meridian_fringestop`** — an unmodified legacy `dsamfs` tool running in the
  `casa38` env — turns into **UVH5** files. UVH5 is the HDF5-based radio-astronomy
  visibility format. This is the *calibration* branch: it derives system
  temperatures (SEFDs) and beamformer weights, and it is separate from the
  trigger path even though it shares the same capture front end (research packet
  00; `docs/overview/dsa110-rt-overview.tex:2670`). "Fringe-stopping" here means
  continuously re-phasing visibilities to track a fixed sky point as the Earth
  rotates — the slow path does it; the fast/search path deliberately does **not**
  (it accepts <0.01% S/N loss at DM=100, ~6% worst case), to save FLOPs
  (`docs/overview/dsa110-rt-overview.tex:706`).
- **Transport** ships int8 `ProdFrame` packets — a 72-byte production header,
  **no CRC by design** (integrity comes from `pattern_id` + sequence-number
  reorder + `n_filled`, not a checksum; `transport/prod_frame.py:49-60`; research
  packet 00). This is *not* a bug or placeholder. (For contrast, the older
  32-byte `FastVisFrame` in `transport/frame.py` *does* compute a real
  `zlib.crc32` — `frame.py:195-304`.)
- The search RX side is a **POSIX-shm SPMC ring** (`transport/_recv_ring`,
  `_recv_epoll` C extensions), named `/dsart-rxring-<cn_id>`, ~9.8 GiB in
  production, with a per-`(corr, dm)` reorder window (research packet 00).
- **C1 → C2 → C3** is the candidate cascade, detailed in §7.

---

## 5. Operating the system

### 5.1 The control plane is etcd, not RPC

There is **one orchestrator process per node**, `dsart_rt`
(`services/dsart_rt.py`), started as `-in pipeline_rt` on corr nodes and
`-in search_rt` on search nodes. It is deliberately *state-light*: it never
touches the data stream. All it does is (research packet 02 §1):

1. read its pipeline config from etcd,
2. watch command keys for operator verbs,
3. fork-exec the configured worker routines and create/destroy the PSRDADA
   buffers, and
4. publish heartbeats and buffer stats back to etcd.

**etcd** is a distributed key-value store (the same one Kubernetes uses). Here it
is the entire control bus. Three namespaces matter:

| Namespace | Purpose | Example keys |
|---|---|---|
| `/cnf/...` | **config** the orchestrator reads at startup | `/cnf/pipeline_rt`, `/cnf/search_rt`, `/cnf/spectral_line`, `/cnf/inject/active/<id>` |
| `/cmd/...` | **commands** (verbs) the orchestrator watches | `/cmd/corr_rt/<n>`, `/cmd/search_rt/<n>`, broadcast `/cmd/<ns>/0` |
| `/mon/...` | **monitoring** the orchestrator and services publish | `/mon/service/<ns>/<n>`, `/mon/corr_rt/<n>`, `/mon/array/dec` |

Config is pushed to etcd from the YAML files by
`tools/ops/push_dsart_to_etcd.py`, reading `configs/dsart_pipeline_rt.yaml` and
`configs/dsart_search_rt.yaml`. **Config is loaded once per `start`** — so a
config push only takes effect at the next `start` (research packet 02 §1). This
whole etcd surface deliberately *mirrors* the legacy `corr.py` control surface,
in a **disjoint key namespace**, so the legacy system and `dsart` can run
side-by-side without contention (`CLAUDE.md`; workspace-root
`dsa110-rt_revamp_7b1d2669.plan.md:2658`).

### 5.2 The verbs

A command is a JSON payload `{"cmd": "<verb>", "val": <any>}` written to a
`/cmd/...` key. The verbs (`services/dsart_rt.py:32-47`; research packet 02 §1):

| Verb | `val` | Effect |
|---|---|---|
| `start` | observing declination (deg), or `None` → resolve from `/mon/array/dec` | reload config, create buffers, spawn routines in two waves (compute first, then capture gated on sentinel files, 240 s timeout) |
| `stop` | — | SIGTERM the process group, 5 s grace, SIGKILL, destroy buffers in reverse |
| `utc_start` | first specnum (int) | arm capture (see §5.3) |
| `utc_stop` | int | disarm |

Verbs `record`, `trigger`, `ctrltrigger`, `inject`, `reload_cal`,
`reload_flagants` are **accepted but currently logged no-ops** on the
orchestrator side (research packet 02 §1; 03 — these are on the pending list).

There is also an **observation watchdog** inside the orchestrator: it reads
`/cmd/operator/control.max_obs_seconds`, and when elapsed time exceeds the cap it
auto-issues `utc_stop(0)`. This is enforced in the orchestrator itself,
independent of any operator agent (research packet 02 §1).

### 5.3 Arming and specnums (what "utc_start" really means)

A **specnum** is the SNAP packet **sequence number** — a counter incremented per
packet, *not* a wall-clock time (`REALTIME_FRB_SEARCH.md:1244`). The verb string
is literally `UTC_START-<seq>`, which is a legacy misnomer: `<seq>` is a specnum,
not a UTC timestamp. "Arming" means telling all capture processes fleet-wide:
*begin ingesting when you see this common specnum.* Because every SNAP shares the
sequence numbering, arming to a single specnum synchronizes the whole array.

Mechanically, the `utc_start` verb sends a UDP poke `UTC_START-<seq>` to
`127.0.0.1:11223` and `:11224` (the two capture control ports), and writes an
etcd "arm trio" (research packet 01 §4; 02 §1):

- `/mon/snap/1/utc_start_rt = {"val": seq}` — the armed specnum,
- `/mon/snap/1/armed_mjd` = now (MJD), refreshed,
- `/mon/snap/1/utc_start` = **0** — pinned to zero *on purpose*. (There is a unit
  mismatch: the legacy time-anchor formula expects native samples, so writing the
  raw specnum would corrupt the `meridian_fringestop` UVH5 time anchor. Setting
  0 makes the anchor evaluate to "now." This was a real bug found and fixed
  2026-06-02, commit `e6ee7cd`; research packet 01 §4.) Don't "fix" this to
  carry the specnum.

A caveat to avoid confusing yourself: you'll see both "16-bit" and "44-bit"
mentions around specnums — these refer to *different quantities* in different
parts of the legacy code and should not be conflated (research packet 01 §4).

### 5.4 The three ways to drive the system

**(1) The `dsa_monitor` dashboard Control tab (primary).** This is how fleet ops
actually happen. It's a Flask app on `h23:5778`. Every POST requires a typed
confirmation word plus a reason, and every action is audited to
`/mon/audit/control/...` (research packet 02 §2, §5). The Control-tab panels
(research packet 02 §5):

- Start fleet (`obs_dec_deg`) / Arm (utc_start with margin) / Disarm / Stop
  (confirm `stop`) / Restart-all (confirm `restart_all`, async job + poll) /
  Bounce-search
- Update fleet code (branch, force) / fstable build+deploy
- C2 restart + activity + candidates + decision log; restart h23 services;
  fleet services table
- Signal injection + SNR calibration (§6)
- Dump Now (confirm `dump_now`) / Dumps Enabled / Voltage Dumps Enabled / C3
  Reject Mode (§7)
- Operator Agent Authority panel (§5.5) / Spectral-line (SPL) per-subband table

**(2) The `tools/ops/dsart-rt` CLI (bash).** The canonical scriptable operator
tool (research packet 02 §2):

```bash
dsart-rt services {install|up|down|restart|status} [--corr LIST] [--search LIST]
dsart-rt pipeline {start|stop|status} [--dec D]
dsart-rt verb send VERB [--val V] [--corr LIST] [--search LIST]
dsart-rt mon show
dsart-rt push-config
```

For example, to arm just corr node n06:
`dsart-rt verb send utc_start --val 1234567 --corr n06`. The default fleet lists
are the 16 corr and 4 search short-names from §3.1.

**(3) The `dsa110-operator` agent console.** A separate repo — an LLM-driven
control/monitoring console that runs on your laptop and reaches the observatory
*only over SSH to h23*. Its `control/verbs.py` mirrors every dashboard POST as a
named Plan (`start_fleet`, `stop_fleet`, `utc_start`, `fire_injection`,
`set_dumps_enabled`, `dump_now`, …) and is the single source of truth for the
dashboard's actions (research packet 02 §2).

### 5.5 The authority model: humans always win

The agent console operates under a strict authority model
(`dsa110-operator/src/dsa_operator/control/authority.py`; research packet 02
§2, §5). The key `/cmd/operator/control` is written **only by the dashboard** —
the agent can *read* it but may **never write it**. It carries:

- `agents_enabled` — a master lockout (turn all agents off),
- `executor_email` — pins which human is authorized,
- `max_obs_seconds` — the watchdog cap from §5.2.

The agent may only write under `/operator/` and `/cmd/ant/`. If the control key
is absent, behavior is fail-open (enabled/unpinned/uncapped). The upshot: a
human at the dashboard can always override or disable the agent — humans win.

### 5.6 A typical observing session

Roughly (assembled from research packet 02 §2, §5):

1. **Push config** if it changed: `dsart-rt push-config` (or dashboard).
2. **Ensure services are up:** `dsart-rt services status` — expect all 16 corr +
   4 search orchestrators alive.
3. **Start the fleet** at the target declination: dashboard "Start fleet" with
   `obs_dec_deg`, or `dsart-rt pipeline start --dec D`. This creates buffers and
   spawns routines.
4. **Arm:** dashboard "Arm" (it computes `ARM_SEQ` from the capture
   `last_seq_no` plus a margin) or `dsart-rt verb send utc_start --val <seq>`.
5. **Watch.** On Grafana/Influx, follow the `/mon/...` keys — heartbeats
   (`corr_rt_heartbeat`, `search_rt_heartbeat`), RFI flag fraction (~8% is
   normal), buffer occupancy (`dada_dbmetric`), C2 activity. The workspace
   **`obs-status` skill** gives a one-shot health report from the dashboard +
   InfluxDB.
6. **Stop / disarm** when the session ends.

The Influx pusher (`dsart_rt_to_influx`) maps `/mon` keys to measurements you'll
see in Grafana: `corr_rt_heartbeat`, `corr_rt_routine`, `corr_rt_buffer`,
`corr_rt_capture`, `corr_rt_rfi`, and search equivalents, plus `c2_service`,
`c2_receiver`, `c2_inject_match`, `/mon/array/dec`, `/mon/array/gal_dm`
(research packet 02 §6).

---

## 6. Injections — how we know the search works

You cannot wait for a real FRB to test a real-time search. Instead the pipeline
can **inject** synthetic bursts and check that the detector finds them. There are
**three "altitudes"** at which you can inject, trading realism for convenience
(research packet 00; 02 §3):

1. **Cube-domain** (`inject/cube_injection.py`): a synthetic burst is placed
   directly into the detector's input cube, bypassing all upstream stages. This
   is the primary *detector-logic* gate and lets the search node be developed
   with no corr node, no transport, no telescope at all. It's how milestones were
   built in parallel.
2. **Voltage-domain, online** (`inject/online.py`): a properly *dispersed*,
   calibration-phased, per-polarization burst envelope is `.add_()`-ed into the
   real `voltages_real`/`voltages_imag` arrays **before** RFI flagging — so it
   traverses the *entire* real pipeline (RFI → cal → GEMM → grid → dedisperse →
   image → detect). This is the realistic end-to-end test.
3. **Replay** of recorded voltage dumps (`bench/replay_voltage_dump.py`, e.g. run
   `250924mptq`) or synthetic noise (`dada_junkdb`) injected into `fada` —
   re-running the pipeline over real recorded data.

### 6.1 The no-backchannel principle

This is a design point worth internalizing: **the detector is blind to the
injection.** No stage tells the detector "a burst is coming." The injected burst
is just data. Matching an injection to a detection is done **post-hoc, by
tolerance** — after the fact, the coincidencer compares detected candidates to
the registry of fired injections and pairs them if position/DM/time agree within
tolerances (`coinc/inject_match.py`). A specnum-proximity gate (default ±4096
samples, `DEFAULT_SPECNUM_TOL_SAMPLES` at `coinc/inject_match.py:190-201`) was
added to stop cross-attribution between near-simultaneous injections
(`coinc/inject_match.py:47-58`; research packet 00); the tolerance was widened
from the original ±2048 on 2026-06-10 after ±2048 was found to reject real
DM-2500 probes. This makes injection a genuine test rather than a rigged demo.

### 6.2 How to actually fire one

From the dashboard Control tab, panel `#panel-inject`
(`templates/control.html:896`; research packet 02 §3): fill in `inj_id`,
`dm_pc_cm3`, either `target_snr` **or** `fluence_jy_ms`, `width_samples`
(1–4096), `profile` (`gaussian`|`boxcar`), sky position `l_rad`/`m_rad`, and
optionally `apply_at_specnum`/`margin_blocks`/`chgroups`, then POST
`/control/inject`. The dashboard auto-arms `apply_at` to
`max(block_specnum_start) + margin_blocks × NPACKETS_PER_BLOCK` (default margin
32 blocks, `DEFAULT_INJECT_MARGIN_BLOCKS` at
`tools/dashboard/dsa_monitor/control_store.py:531`) so the burst lands slightly
in the future across all corr nodes.

Under the hood it fans out one etcd write per chgroup to
`/cmd/dsart/corr/<chgroup>/inject` with payload
`{"cmd":"inject","val":{inj_id, l_rad, m_rad, dm_pc_cm3, fluence_jy_ms,
width_samples, profile, apply_at_specnum}}`. Inside `corr_fast`,
`inject/etcd_watcher.py` watches that prefix and hands the config to the
`OnlineInjector` (`inject/online.py`). A live registry entry is written to
`/cnf/inject/active/<inj_id>` (TTL 60 s), and every fired injection is appended
to a durable JSONL log (default
`/dataz/dsa110/operations/inject/fired_injections.jsonl`) so C3 can still
recognize an injection after the TTL expires (research packet 02 §3).

Matching results are published to `/mon/dsart/inject/matches/<inj_id>` with the
best observed S/N, observed specnum, inferred `K`, and observed position/width.

### 6.3 SNR calibration (the `target_snr` path)

If you want to inject "a 12-σ burst" rather than "a burst of fluence X," the
system needs to know the mapping between fluence and observed S/N. That mapping
is calibrated per **DM bucket** with POST `/control/inject_calibrate`
(`app.py:1030`; research packet 02 §3). It fires a **laddered probe** and fits a
linear model:

```
K = observed_snr × sqrt(width_samples) / fluence_jy_ms
snr_to_fluence = target_snr × sqrt(width) / K
```

`K` is stored at `/cnf/inject/snr_calibration/<bucket>`, bucket =
`dm{round(dm/50)*50:04d}` (e.g. `dm0500`). If you request `target_snr` without a
stored calibration for that bucket you get **HTTP 412** ("run
`/control/inject_calibrate` first"). Important constants and their reasons:

- Default calibration fluence 7×10⁻⁴ Jy·ms, width 4 native samples.
- **`MAX_PROBE_FLUENCE = 1×10⁻³`** — this is the **fp16-cuFFT overflow cliff**.
  Push brighter and the half-precision FFT overflows and the numbers go
  nonsense. Respect this ceiling.
- **`SATURATION_OBSERVED_SNR = 240`** — the detector clips at about ±250σ, so
  observed S/N saturates around 240; the calibration is saturation-aware.
- The ladder steps ×(1, 2, 4) with 60 s between steps to let the per-kernel σ EMA
  recover between probes.
- A health pre-flight gates it: `corr_fast` heartbeats < 30 s old, search compute
  heartbeats present, and `c1_metering_active == 0`.

---

## 7. Voltage dumps and the trigger cascade

### 7.1 Why voltages, and the ~15-second window

The search path works on heavily reduced data (channel-summed, int8, Stokes-I).
That's enough to *detect* a burst, but not to *localize* it precisely or study
its microstructure — for that you need the **raw antenna voltages** around the
burst. Recall from §4.2 that the third reader of `fada` is an in-RAM
`VoltageRing` holding roughly the last **~15 seconds** of voltages
(`--retention-s 15.0`, `configs/dsart_pipeline_rt.yaml:539`;
`src/dsart/dump/voltage_ring.py:23` — ~112 blocks ≈ 31.5 GiB resident; note
this is a different, deeper buffer than the ~9.4 s `fada` ring itself). If a
candidate survives the trigger cascade fast enough, those voltages get copied
to disk before the ring overwrites them; otherwise they're lost. Everything
below is a race against that ~15-second clock.

### 7.2 C1 → C2 → C3

**C1 (per search node, `services/c1_emit.py`).** After clustering, each search
node emits **C1 candidates** over a persistent TCP connection to **h23 port
11500** (8 sockets fleet-wide) (research packet 00; wire schema
`docs/c1c2/C1C2_WIRE_SCHEMA.md`).

**C2 (the coincidencer, on h23, `services/coincidencer.py`).** C2 collects C1
candidates from all search nodes into a rolling MJD window and does
**coincidencing** — grouping candidates that are the *same event seen by
different nodes*. It builds a graph (union-find components; an edge exists
between two candidates iff `|t_i − t_j| ≤ (w_i + w_j)/2`) and applies vetoes and
YAML-defined trigger criteria (hot-reloaded on file change). On a trigger it
assigns an event name, writes a CSV audit trail, and fires **two distinct UDP
broadcast mechanisms**: a `TriggerBroadcaster` sends *cube-dump* packets to the
8 destinations (4 search nodes × 2 GPU halves), while a separate
`VoltageBroadcaster` sends `DUMP_VOLTAGE` packets **directly to each of the 16
corr nodes on port 11229**, telling them to stage their retained voltages
(`src/dsart/coinc/broadcast.py:41-51,146-156`;
`services/coincidencer.py:1263,1580`). The `DUMP_VOLTAGE` path does *not* route
through the C1 listeners.

**C3 (on h23, `services/c3.py`).** C3 polls the candidate archive for the C2
manifest (the "arrival sentinel"), runs a morphology **cube-veto** (a robust
z-score `(max − median)/(1.4826·MAD)`, but it *always keeps* injections and
ambiguous cases), and then either **KEEPs** — collecting all 16 voltage
fragments into `<event>/Level2/voltages/` — or **REJECTs** — conservatively
moving metadata to `candidates_rejected/` and cleaning up staged voltages. It
**never `rm -rf`s** (research packet 00; 02 §4).

### 7.3 Trigger criteria

`configs/c2_trigger_criteria.yaml` defines *ordered, first-match-wins*
`trigger_classes`, each a set of predicates (`snr_max_min`, `dm_median_min/max`,
`dm_iqr_max`, `width_median_max`, `lm_diag_max`, `dm_galactic_fraction_min/max`,
`n_events_min`, `n_search_nodes_min`) plus an action (`dump_all_gpus` or
`log_only`) and a `holdoff_s`. The stock classes (research packet 02 §4):

- **`bright_frb_extragalactic`** — DM ≥ 0.75× the NE2001 Galactic max along the
  line of sight, S/N ≥ 12, DM 115–2700 → `dump_all_gpus`. (NE2001 is a Galactic
  free-electron model; it predicts how much DM the Milky Way alone contributes,
  so an *excess* over it flags a plausibly extragalactic burst.)
- **`bright_galactic`** — S/N ≥ 15, DM ≥ 100 → dump.
- **`bright_pulsar`** — a pulse train → `log_only`.
- **`log_only`** fallback.

### 7.4 The three safety gates — and how to flip each

Voltage dumping is genuinely dangerous (it can fill NVMe disks and back up the
system), so there are three independent gates, each with a deliberately chosen
fail direction (research packet 02 §4):

| Gate | etcd key | Fail direction | Meaning | Dashboard control |
|---|---|---|---|---|
| 1. Dumps enabled | `/cmd/c2/dumps_enabled` | **fail-OPEN** (missing ⇒ enabled) | C2 actually broadcasts vs just logs "WOULD-DUMP" | POST `/control/dumps_enabled`, confirm `enable`\|`suppress`, reason required |
| 2. Voltages enabled | `/cmd/c2/voltages_enabled` | **fail-CLOSED** (missing ⇒ disabled) | gates the `DUMP_VOLTAGE` UDP broadcast to the 16 corr nodes | POST `/control/voltages_enabled`, confirm `enable` |
| 3. C3 flag-only | `/cmd/c3/flag_only` | default `True` = KEEP-only | `True` = collect+log, never delete; `False` enables conservative REJECT | panel `#panel-c3-mode`, confirm `delete` |

The fail directions are the interesting part. **Dumps-enabled fails open** so a
cold etcd doesn't silently make you miss a real FRB. **Voltages-enabled fails
closed** because voltage dumps write large files and a cold etcd should never
start filling NVMe on its own. Each key's value is
`{"enabled": bool, "ts", "actor", "reason"}`, with a ~200 ms cache in C2.

### 7.5 Disk paths

From `services/c3.py:76-145` (research packet 02 §4):

- archive: `/dataz/dsa110/candidates`
- rejected: `/dataz/dsa110/candidates_rejected`
- C3 state: `/dataz/dsa110/operations/c3/c3_state.json`
- per-corr-node staging (NVMe): `/home/ubuntu/data/voltage_staging`
- event layout: `<event>/Level2/voltages/` (fragments),
  `<event>/Level3/<event>.json` (the C2 manifest C3 polls),
  `<event>_voltages.json` after collection.

### 7.6 The M8 live test — and its honest caveats

The end-to-end voltage-dump path was live-tested on **2026-07-09 and PASSED**
(`scratch/M8_E2E_VOLTAGE_DUMP_TEST_20260709.md`; research packets 00, 02, 03).
Highlights: a single node dumped in ~24 s (fragment = 23 × 301,989,888 B =
6,945,767,424 B over window `[target−8, target+14]`); the full fleet did 16/16 in
~25 s with zero drops; C3's fail-open KEEP path fired within one 10 s scan; a
104 GiB rsync took ~11 min.

**But read these caveats before you trust it end to end:**

- **`voltages_enabled` stayed FALSE** during the test — it used a *synthetic*
  `DUMP_VOLTAGE` broadcast. So the **live C2-triggered path with
  `voltages_enabled` ON was never exercised in production.** The C2
  `_maybe_broadcast_voltage` sequence and the cube-veto REJECT/delete path are
  *untested live*. The guide (and the M8 report) flag this loudly.
- The C2 manifest's `mjd_target` is always **0.0** — a known *cosmetic* bug; a
  fix was identified but (as of the packet) **not confirmed landed** (open
  question #4 in packet 03).
- `dsart_c3.service` shows the unit **inactive** even while the C3 process runs —
  an unreconciled unit-vs-process mismatch (open question #5).

---

## 8. What works, what's partial, what's missing

Adapted from research packet 03. Being able to answer "is this actually done?"
honestly is more useful than assuming.

### 8.1 Works (implemented and validated)

- **Fast-vis search path, end to end** — the 10-step `corr_fast` hot path;
  M7.4 gate PASS (30-min soak, zero drops, 128/128 corr + 12/12 search routines
  alive, RFI ~8%); M7.7 op-point ~166 ms/cube vs ~201 ms budget.
- **Slow-vis UVH5 / calibration path** — `corr_slow → bada →
  meridian_fringestop` (unmodified `casa38` `dsamfs`) → UVH5, byte-identical
  legacy contract; sole `bada` reader since M7.4 Phase 9 (2026-05-30).
- **Injections** — all three altitudes, √-flux voltage convention, linear
  `K·F/√W` S/N model, per-DM buckets, saturation-aware ladder, post-cal hook.
- **C1/C2** — fleet gate PASSED 2026-05-28.
- **Voltage dumps (M8)** — live test PASSED 2026-07-09 (with the §7.6 caveats).
- **Dashboard** `dsa_monitor` Control tab + Influx/Grafana 89-panel generator.
- **RFI chain** (flagants, SK, bandpass, group, SumThreshold) active ~8% in gate.
- **Static-sky** `StaticSkyMean` causal 8-block sliding mean.
- **Real 40 GbE** — M4b pair-rate PASS at 1.752 Gb/s/pair, 16→4 corner-turn at
  28 Gb/s aggregate; loopback→real-fabric transition COMPLETE.

### 8.2 Partial / not yet / deferred

- **SPL (spectral-line) mode** — implemented and etcd-gated
  (`/cnf/spectral_line`), fail-safe disabled, applies only at next
  `restart_all`+`start`; postdates the 2026-06-10 doc freeze. **Unknown whether
  exercised on-sky** (open question #3).
- **M4a ProdFrame** (72-byte header, `prod_frame.py`) coexists with the older
  32-byte `FastVisFrame` (`frame.py`) while becoming canonical. Reminder:
  **ProdFrame has no CRC by design; `FastVisFrame` does have a real
  `zlib.crc32`** — neither is a stub.
- **`capture_supervisor.py`** — a 9-line `NotImplementedError` stub referenced by
  nothing (dead code).
- **Orchestrator-side verbs** `prepare`, `record`, `inject`, `reload_cal`,
  `reload_flagants`, `ctrltrigger`, `trigger` — unimplemented no-ops.
- **Detector v2 (learned)** — swap mechanism proven (M5 `IdentityDetector`),
  rollout deferred.
- Deferred: carry-over re-imaging (implemented+validated but OFF in production),
  Layer-3 per-fine-DM EMA, runtime detector-mode switching + atomic rollback.

### 8.3 Legacy coexistence and cutover — the honest picture

- **KEPT:** `meridian_fringestop` (unmodified legacy `dsamfs`) as the sole
  `bada` reader; the `bada` byte-for-byte contract preserved so legacy
  calibration/H5-archive/operator UIs at `/mon/corr/<n>` keep working "until the
  M7.6 cutover."
- **DELETED:** `dsaX_nsfrb` (and its `gen_nsfrb_fstable.py`, `caba` buffer, and
  `dada_dbnull -k caba` drain); the old `bada`/`dada` drain stand-ins.
- **Legacy Hella** (`dsaX_hella`, the beamformed single-pulse search) was the
  legacy production search. **Whether it still runs in production alongside
  `dsart` today, or has been retired, is UNKNOWN** — no doc states a retirement
  date (open question #1).
- **The formal M7.6 cutover** (rename `/mon/{corr,search}_rt → /mon/{corr,search}`
  + a parity sign-off gate — pulsar single-pulse detection in-beam, or
  trigger-rate parity vs legacy `corr.py`, or a real FRB) — **completion status
  is UNKNOWN** as of the 2026-06-10 doc; the docs are internally in tension
  (open question #2). **Check the dashboard / `obs-status` for the live truth.**

### 8.4 The audit verdict

An independent read of "do the claims match the code?" came back **GENUINE**
(research packet 00): the dispersion constant is consistent across
`common/constants.py`, `inject/online.py`, and `coinc/cube_veto.py`; the boxcar
√N S/N normalization is correct with real Welford/EMA statistics; there is no
injection backchannel; the DoD scripts run real pytests and numeric gates; and —
tellingly — a *failing* test artifact is preserved on purpose
(`m3-burst-correctness/summary.json` with `"stage": "FAIL"` from an 8-pixel
offset, later fixed), which is the mark of an honest test history rather than a
green-washed one. Disclosed weak spots: one soft `|| true` gate in `M3.sh:311`,
and the headline "5.84 Gb/s zero-loss 1-hour soak" is explicitly a **loopback**
result (labeled as such), not a real-fabric number.

---

## 9. Glossary

- **SNAP** — FPGA board that digitizes and channelizes antenna voltages and
  streams them as UDP; the array's F-engine. 32 in the system, 3 antennas each.
- **F-engine / X-engine** — F = the frequency-channelizing stage (the SNAPs);
  X = the cross-multiply/correlation stage (here, the GPU GEMM on corr nodes).
- **Visibility** — the complex cross-correlation of two antennas' signals in one
  channel; one Fourier component of the sky.
- **Baseline** — a pair of antennas; 96 antennas ⇒ 4656 baselines.
- **(u, v) plane** — the 2-D Fourier space of the sky; each baseline samples one
  (u, v) point (in wavelengths).
- **Gridding** — resampling the scattered visibility samples onto a regular
  (u, v) grid so you can FFT them.
- **Dirty image** — the inverse-FFT of the gridded visibilities; the true sky
  convolved with the array's point-spread function.
- **chgroup (channel group)** — the slice of the band processed by one corr
  node; 16 total, 384 channels each.
- **specnum** — a SNAP packet sequence number; used to synchronize/arm capture.
  *Not* a wall-clock time despite the `UTC_START-<seq>` naming.
- **DM (dispersion measure)** — integrated free-electron column along the line of
  sight; sets the frequency-dependent arrival delay `τ = K_DM·DM/ν²`.
- **Boxcar** — a flat rectangular matched filter slid along a time series to
  detect pulses of unknown width; here computed via `cumsum` for speed.
- **PSRDADA / DADA ring** — shared-memory ring-buffer library and its named
  buffers; the pipeline's data plane between stages.
- **Reader count** — the declared number of consumers on a DADA ring; a missing
  reader stalls the writer (backpressure). Sacred — bump `r` if you add a reader.
- **Fringe-stopping** — continuously re-phasing visibilities to track a fixed sky
  point as Earth rotates; done on the slow/cal path, deliberately *not* on the
  fast/search path.
- **Drift scan** — observing mode where the array is fixed and the sky drifts
  through; DSA-110 points in declination only, at meridian transit.
- **UVH5** — HDF5-based radio-interferometry visibility file format; the slow
  path's calibration output.
- **C1 / C2 / C3** — the candidate cascade: C1 = per-search-node candidate emit;
  C2 = cross-node coincidencer + trigger; C3 = voltage-dump collection/veto.
- **Coincidencing** — merging candidates that are the same event seen by
  different nodes (union-find on a time-overlap graph).
- **etcd** — distributed key-value store; the pipeline's entire control plane
  (`/cnf`, `/cmd`, `/mon`).
- **Stokes I** — total intensity, `V_XX + V_YY`; the single-polarization product
  the search path runs on.
- **NUMA** — Non-Uniform Memory Access; multi-socket memory locality. Capture and
  GPUs are pinned to sockets (`configs/numa_topology.yaml`) so data stays local.
- **Quantization** — reducing numeric precision (here to int8 for transport,
  fp16 for FFTs) to fit the compute/bandwidth budget.
- **Fluence** — time-integrated flux of a burst (Jy·ms); the injection brightness
  knob.
- **NE2001** — a model of the Milky Way's free-electron distribution; predicts
  the Galactic DM contribution, used by C2 to flag likely-extragalactic bursts.

---

## 10. Where to read next

In roughly the order that will help a newcomer:

- **`dsa110-rt/docs/overview/dsa110-rt-overview.pdf`** (source `.tex`) — the best
  single narrative of the `dsart` architecture, the channel plan, and the
  dedispersion/imaging math. Start here after this guide.
- **`REALTIME_FRB_SEARCH.md`** (workspace root) — the authoritative reference for
  the *legacy* `dsa110-xengine` system. The physics/topology ground truth: hosts,
  channel plan, UDP fabric, etcd verbs. Buffer sizes and reader counts are pinned
  to this.
- **`dsa110-rt/docs/c1c2/`** — `C1C2_DESIGN.md` and `C1C2_WIRE_SCHEMA.md` for the
  candidate cascade; `docs/voltage_dumps/VOLTAGE_DUMP_C3_DESIGN.md` for the C3
  state machine.
- **`PARALLEL_AGENTS.md`** — the binding coordination protocol for working in the
  repo alongside other agents (single-owner file classes, per-milestone branches,
  GPU pinning). Read before you touch shared code.
- **`CLAUDE.md`** (workspace root) — the workspace map, commands, and the h23/GPU
  rule.
- **The `research/` packets next to this guide** (`00`–`03`) — the condensed,
  cited findings this guide is built from; each ends with an explicit list of
  gaps and open questions.

And when in doubt about the *live* state of the system (is it observing? did the
M7.6 cutover happen? is Hella still running?), don't guess — run the
**`obs-status`** skill or open the dashboard. The docs lag reality; the dashboard
is reality.
