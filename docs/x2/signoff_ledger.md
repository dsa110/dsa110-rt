# x2 sign-off ledger (decisions reserved for the user at a checkpoint)

Integration keeps this list. Nothing here is decided or deployed until the user signs it off at a
checkpoint (CP1 unless marked). Each item: what, evidence needed, who supplies it, default until signed.

## From the plan (rev 4)
| id | item | evidence | owner | default until signed |
|---|---|---|---|---|
| S1 | Corner-turn format and its network-forced S/N cost (0.3-1.0% vs the model; Phase 0: today's wire itself costs 1.2%, so every 2x format is 0.2-1.0% better on the wire than today and the expected end-to-end gain is ~x1.21 at 1 ms) | G1 capacity sweep 10-29 (13 corr -> n01/n02, frame v2 8972 B): sender burst 512 KB -> 1-5.6% loss at 9-18 Gb/s, all rx_discards_phy (NIC port buffer, incast microbursts, PCIe Gen3 x4 drain); 64 KB -> clean to 13 Gb/s; **16 KB -> zero NIC discards up to 17.08 Gb/s per search node (sender-limited)**; remaining loss = pre-existing path-dependent PHY CRC (2.5-7e-4, moves between n01/n02 run to run). => every format needing <= 17 Gb/s clean fits, PROVIDED the production C sender paces with 16 KB bursts (WS-C). To confirm at G1 fleet load 16 -> 4. Recommendation: gridded ns2 cint4 (today's imaging exactly; 0.996 of model; needs 16.4) | WS-T, integration | none (2x does not run) |
| S2a | Hybrid row rule: sqrt(n) normalisation, robust per-stream scale + clip, 8-bit heavy values. Measured maps (WS-P, SPEC §3): heavy threshold n >= 1021; gridded n_sub 2: 70-78 heavy/stream = 2.9% of values, 42.4% of natural weight; n_sub 4: 21-25 = 1.13% / 20.7%; per-baseline: 1748 exact redundancy classes (not 1773), 80 heavy (46.3%) at n_sub 2, 26 (22.1%) at n_sub 4 | Phase 0 row-rule study incl. storm window and hot cells | Phase 0 | — |
| S2b | cfp16 corr storage of fast visibilities | Phase 0 U-precision study, CP1 corr exactness | Phase 0, WS-C | — |
| S2c | Fused-cal ULP differences in 2x | CP1 corr taps vs CPU reference | WS-C | 1x keeps separate kernels |
| S2d | fp32 detector accumulation, per-DM sigma_k, chunk-max decoder in floods | CPU twins + CP1 K3b compat | WS-S | — |
| S2e | B-spline time filters (adopt if modelled gain > 0 and CP1 cost within duty aim) | Phase 0 study 5 + CP1 K3b timing | Phase 0, WS-S | boxcar |
| S2f | Phase 0 recommendations (docs/x2/phase0/SUMMARY.md): mid-tread codes (s4 0.3534, s5 0.1932; code -2^(b-1) = no data); heavy threshold 3e-5 N (n >= 306; required by the plan's own 'clip >= 2x rail' rule, +0.2-1.4 Gb/s); sigma_hat tau 3 s; per-row guard (row robust scale > 1.5 sigma_hat -> coded at that scale); image in sigma units (L1 floor 0.078, L2 floor 0.5, clip 250); cfp16 accepted; b3x filter bank (x1.043 at 1 ms, 8 filters, 148 history rows) | Phase 0 studies 1-5 | WS-T, WS-S implement as defaults, user signs | as recommended (development default) |
| **S3** | **Bright-signal response.** Today's per-row max-abs wire compresses bright bursts (0.93 at 50 sigma, 0.80 at 100, 0.49 at 250, saturating ~150 sigma per sample; the 250 clip is never reached). 2x is linear to the clip, so bright events read up to 2x higher than today, the direction 3b43118 warns about. Choose: (a) linear 2x after a C1/C2 bright-class replay shows no storm, or (b) emulate today's compression | Phase 0 study 2 + C1/C2 bright-class replay (WS-D D13, DONE 10-22: dump-class triggers unchanged in all 5 variants — 108.6/day quiet 10-20..22, 89.4/day flood 10-03..05; rows >= 50 sigma +0.75%; docs/x2/results/) + a DETECTOR-level amplitude-regime replay (still owed: the 3b43118 storm came from the detector regime, likely the Layer-2 floor scaled 1/256, not the wire) | user | (a) in development; no deployment without the detector-level replay |
| E1 | Owner chain > 4.8 us/image after fallbacks -> tol 1.4 or FoV/2. Revision (WS-S, 10-22): the K2 column-pruning premise fails on today's grid layout (gridded ns2 occupies rows 5..210, cols 17..183): 112 of 129 half columns, not 57, so K2 ~0.7 us/image instead of 0.35 -> projected 2.75-3.85 us/image all-in = 0.42-0.59 GPU per half (gate 0.75, aim 0.6) | CP1 search track | WS-S | escalate, no change |
| E2 | Corr GEMM + stage-1 > 55 ms after fallbacks -> n_sub 4, then stage-1 at 0.52 ms, ... WS-C risk notes (10-22): the 2x GEMM re-reads each voltage tile ~6.5x and recomputes cal on load (may be load/ALU bound vs the 18 ms plan; fallbacks: more fast samples per program, smem-staged calibrated tiles); stage-1 does ~1.5e10 smem reads/block for gridded n_sub 2 and heavy values (n up to ~4000) serialise source chunks (14 ms plan; n_sub 4 halves it); per-tier Python loops + RFI/array-burst syncs on the host; the 1x-compat savings (24.5 ms serial H2D, ~10 ms zero-fill, ~6 ms copies) are unmeasured until CP1. GPUs are sm_75 | CP1 corr track | WS-C | escalate, no change |
| E4 | Clean capacity < 11.2 Gb/s/node -> hardware or delay | CP1 capacity sweep | WS-T | escalate |
| E5 | NUMA balancing vs the ring v4 consumers. Search nodes run kernel.numa_balancing=1 (~3,100 hint faults/s per node today, 17.5 G over 64 days); an extra mapper of the 16 GB 1x ring saw 2.4 ms minor faults (07:25 incident). Ring v4 (22-33 GB, per-DM, mmap readers) must be measured at CP1 under full load (system time, faults/s, scatter). If it costs real time: pread/pinned-copy readers, or kernel.numa_balancing=0 on search nodes (host change, sign-off) | CP1 search track + fleet load test | WS-T, WS-S | measure only |
| E6 | Ring v4 size: ~35 GB on a search node for gridded n_sub 2 (WS-T, rows padded to the largest stream), above the plan's 22-33 GB; must fit the node's 93 GiB budget with search_compute's GPU-host buffers and the freed 15.2 GiB v3 ring + ~20 GB pinned cube retention. A per-stream stride saves ~8%. Measure resident memory at CP1 fleet load | CP1 fleet load | WS-T | measure |
| CP2-K | Injection K with the b3x filter bank: the K bucket key stays (DM, injected width); the detecting filter is recorded in each K entry (WS-D D9/D10). Decide at CP2 from injection data whether a filter mismatch between calibration and probe should invalidate or re-key K | CP2 injection campaign | WS-D | record only |
| CP1-op | Install /etc/sysctl.d/99-dsart-ipfrag.conf on n01 (missing; n02/n09/n13 have it). **DECIDED by the user 2026-10-22: install and keep it permanently.** DONE 2026-10-28 23:29 UTC after n01 came back from the outage (n02's file, md5 a37bfd46..., applied with `sysctl -p` on that file only; previous live values 262144/196608/30); restore.n01_ipfrag_kept verifies | diagnostics_20261022.md | integration | installed, keep |

## Behaviour changes in 1x found by WS-D (all behind flags defaulting to today's behaviour)
Source: `docs/x2/downstream_inventory.md` §0 and §5.
| id | item | today | proposed |
|---|---|---|---|
| D-a | C2 partial events | `rmtree` at < 8 cube files in the window (~7% of sky candidates vanish) | `coinc.partial_event_policy: archive` in 1x too (2x never discards) |
| D-b | Duplicate dumps | class holdoff falls through to the next class: a >=35 sigma burst with DM/gal_dm in [0.75,1] dumps twice (`bright_galactic` has `dm_galactic_fraction_max: 1.0` vs its comment's < 0.75) | per-component "already fired" memory |
| D-c | C2 `cluster_rate` | counts evaluations (once per touched component per batch) | count distinct clusters |
| D-d | Noise-colour de-rate | `_DEDISP_CHAN_SUM_FACTOR` 8 x strength 4.0 divides S/N of every w>=2 candidate at DM ~1400-1496 by 1.13 (w2) to 1.26 (w64) — a live sensitivity cost at the top of today's DM range | user decision on strength/onset (no-op in 2x below DM ~2800) |
| D-e | Injection health gate | bot refuses when ANY half meters | owner-scoped gate |
| D-f | MJD of events | wall-clock derived | derived from the capture epoch |

## Pre-existing test failures on main (verified identical with and without x2, n02, 2026-10-28)
- tests/transport/test_async_tx.py: 4 failures (multi-worker DM split / coarse_dm_mask m72 / full mask).
- tests/transport/test_recv_ring_assemble_validity.py::TestParity::test_overrun_counter_bumped_on_lapped_slot.
- tests/transport/test_production_rx_ring_phaseb.py: hangs to the 900 s timeout.
- tests/test_c2_proactive_claim.py: was stale (old units bug), fixed by WS-D D4.

## Pre-existing physics/production bugs found by x2 work (1x today; user decision)
| id | item | evidence | proposed |
|---|---|---|---|
| P-a | `image/imager.py:167` `compute_edge_mask(drop_dc)` zeroes pixel (N/2, N/2) = the phase centre (l,m) = (0,0): an exactly on-axis burst is reported 1 px (45") off at 0.644 of its true S/N (-36%). Biases injections placed at (0,0) and the K calibration built on them; sky impact is 1 pixel of 65536 | I1 measured 0.641-0.646 in both 1x and 2x | mask the uv DC cell instead of the image centre, behind a flag; recalibrate K |
| P-b | **CONFIRMED in production 1x (10-29, read-only, checkout c953f98).** corr_fast_integration.py:4542/4552 wraps each fada page in a fresh ndarray -> slow_corr_kernel.py:291 pins it -> host_pin.py:132 binds the unregister finalizer to the per-block array, so the ring page is unpinned when that array is freed; plus host_pin.py:162 pops `addr` while entries are keyed by `id(base)` (:300/:354). Result: lazy pinning of all 70 pages on the critical path and ~173 registrations instead of 70, decaying after every restart (a post-restart transient by construction). corr_slow has the same pattern (corr_slow_compute.py:692/706). Per-registration cost on a corr node being measured (5-14 s per 288 MB page seen earlier) | WS-C helper analysis + CPU simulation | persistent (addr, size) pinning or one ndarray per page; fix the pop key; behind a flag |

## Pre-existing hardware/ops issues (independent of x2, reported)
- n22 loses 5.7% of SNAP packets to PHY CRC errors (optic/cable/port). **Worse after the outage (10-28):** jumbo (8972 B) pings n01 -> h22 lose 66%, small pings 0% — size-dependent CRC loss on h22's 40G port.
- n01/n02 corner-turn PHY loss 3-7e-4 (CRC + discard bursts).
- dsa110-operator injection probe never sends `inj_id` (400 from /control/inject) — WS-D D12 list.

## Incidents caused by x2 work (reported to the user)
- **2026-10-22 07:25-07:31 UTC, n13 search half 6 (coarse DM 6): 41.5 s of data unsearched.** Cause: integration
  ran the WS-T passive recorder (mmap PROT_READ of /dev/shm/dsart-rxring-13) and a copy-timing probe. Minor
  faults took ~2.4 ms each (numa_balancing=1), production scatter rose 55 -> 110-135 ms, and search_compute_0 made 12
  lag-recovery seeks. It recovered by itself. No recording was kept. Fix: tools/x2/prod_guard.sh (70eb50e) wraps every
  node run; no mmap of production segments; any further recording on production waits for the user's decision.
