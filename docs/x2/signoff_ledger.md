# x2 sign-off ledger (decisions reserved for the user at a checkpoint)

Integration keeps this list. Nothing here is decided or deployed until the user signs it off at a
checkpoint (CP1 unless marked). Each item: what, evidence needed, who supplies it, default until signed.

## From the plan (rev 4)
| id | item | evidence | owner | default until signed |
|---|---|---|---|---|
| S1 | Corner-turn format and its network-forced S/N cost (0.4-1.0% vs the model) | CP1 capacity sweep with v2 single-fragment frames | WS-T, WS-O runner | none (2x does not run) |
| S2a | Hybrid row rule: sqrt(n) normalisation, robust per-stream scale + clip, 8-bit heavy values | Phase 0 row-rule study incl. storm window and hot cells | Phase 0 | — |
| S2b | cfp16 corr storage of fast visibilities | Phase 0 U-precision study, CP1 corr exactness | Phase 0, WS-C | — |
| S2c | Fused-cal ULP differences in 2x | CP1 corr taps vs CPU reference | WS-C | 1x keeps separate kernels |
| S2d | fp32 detector accumulation, per-DM sigma_k, chunk-max decoder in floods | CPU twins + CP1 K3b compat | WS-S | — |
| S2e | B-spline time filters (adopt if modelled gain > 0 and CP1 cost within duty aim) | Phase 0 study 5 + CP1 K3b timing | Phase 0, WS-S | boxcar |
| E1 | Owner chain > 4.8 us/image after fallbacks -> tol 1.4 or FoV/2 | CP1 search track | WS-S | escalate, no change |
| E2 | Corr GEMM + stage-1 > 55 ms after fallbacks -> n_sub 4, then stage-1 at 0.52 ms, ... | CP1 corr track | WS-C | escalate, no change |
| E4 | Clean capacity < 11.2 Gb/s/node -> hardware or delay | CP1 capacity sweep | WS-T | escalate |
| CP1-op | Install /etc/sysctl.d/99-dsart-ipfrag.conf on n01 (missing; n02/n09/n13 have it) | diagnostics_20261022.md | runner | not installed |

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

## Pre-existing hardware/ops issues (independent of x2, reported)
- n22 loses 5.7% of SNAP packets to PHY CRC errors (optic/cable/port).
- n01/n02 corner-turn PHY loss 3-7e-4 (CRC + discard bursts).
- dsa110-operator injection probe never sends `inj_id` (400 from /control/inject) — WS-D D12 list.
