# G1 results (fleet, no SNAPs) — 2026-10-29

Plan rev 5, phase G1. Observing stopped (power outage); production stopped on every node. All runs from
`x2_scratch` copies; test shm `/dev/shm/dsart-x2-*`; ports 19100+ (sweep) / 6625+ (fleet-load).

## 1. Corner-turn capacity sweep (S1 evidence)
13 corr nodes (n03 n06 n07 n08 n10 n11 n12 n14 n15 n16 n18 n19 n21) -> n01, n02; frame v2, 8972 B
datagrams (never IP-fragmented); `bench/fake_corr_to_search.py sweep`; 60-120 s per rate.

| sender burst (token bucket per destination) | result |
|---|---|
| 512 KB (bench default before 10-29) | 1-5.6% loss at every rate 9-18 Gb/s; all `rx_discards_phy` (NIC port buffer), PAUSE frames; loss highest at 9 Gb/s |
| 64 KB | n02 clean to 13 Gb/s; discards return at 16 |
| **16 KB** | **zero NIC discards up to 17.08 Gb/s per search node** (sender-limited maximum) |

Cause: incast microbursts. 13 senders' bursts arrive at 40 Gb/s line rate; the ConnectX-4 drains to the host
over PCIe Gen3 x4 (~25 Gb/s), so its port buffer overflows whenever bursts coincide, even at 9 Gb/s
average. Fine-grained pacing removes it. `rx_out_of_buffer` (host ring) was 0 throughout.
Residual loss: pre-existing, path-dependent PHY CRC errors on n01/n02 (2.5-7e-4; which node shows them
changes run to run). Consequence: the production 2x C sender must pace with 16 KB bursts (WS-C), and the
bench defaults to 16 KB (d3e4924). **S1 decided by the user: 2x-gridded-ns4-cint4** (needs 14.0 Gb/s clean).

## 2. Fleet-load smoke runs (5 min)
16 TX chgroups (13 corr nodes; n06-n08 run two) -> `search_rx` v4 on n01 n02 n09 n13, format
2x-gridded-ns4-cint4 at load x1.1: **12.2 Gb/s and 211k datagrams/s per search node**; owner engines
(`b8_owner2x.py engine`, WS-S e16586d) on all 8 halves.

| node | datagram loss | PHY CRC fraction | NIC discards | engine duty | chain us/image | seeks |
|---|---|---|---|---|---|---|
| n01 | 7.3e-4 | 5.0e-4 | 0 | — | — | 15 |
| n02 | 3.6e-4 | 2.5e-4 | 0 | — | — | 8 |
| n09 | 0 | 0 | 0 | 0.550 / 0.554 | 3.59 / 3.68 | 1 / 1 (start-up) |
| n13 | 0 | 0 | 0 | — | — | 1 / 1 (start-up) |

- TX: every sender kept up (2459 blocks, 0 late, 0 drops).
- First run: engines refused the ring on PATTERN_MISMATCH: the ring spec and the bench TX built value
  maps with station heights, unlike corr_fast and the engine (SPEC §3: `antpos_u = None`). Fixed
  (c258db3). The fail-loudly check worked as designed.
- Second run: engines in real time (duty 0.55, wall p99 159 ms of the 201 ms cycle) but **lag-recovery
  seeks**: one at start-up on every half, and repeated seeks on n01/n02, where datagrams are lost to the PHY
  CRC errors. The ring reader waits on wall-clock grace per window when streams have holes. -> WS-S:
  data-time release with zero-fill flags, start at the data frontier, a loss/outage test.

## 3. Fleet-load, 60 min — PASSED (2026-10-29 14:26-15:32 UTC)
Same setup as section 2, engines at WS-S 90f9f01 (data-time fan-in release, sparse valid fraction, quorum
start), 16 KB sender pacing.

| node | halves | duty | wall p99 (gate <= 181 ms) | chain us/image (gate 4.8) | seeks | start-up skips | cycles | datagram loss (PHY CRC frac) |
|---|---|---|---|---|---|---|---|---|
| n01 | 0 / 1 | 0.574 / 0.563 | 163 / 162 | 3.63 / 3.60 | 0 / 0 | 0 / 0 | 19671 | 0 |
| n02 | 2 / 3 | 0.553 / 0.556 | 158 / 155 | 3.54 / 3.62 | 0 / 0 | 0 / 0 | 19671 | 1.04e-3 (7.2e-4) |
| n09 | 4 / 5 | 0.552 / 0.555 | 157 / 162 | 3.59 / 3.68 | 0 / 0 | 0 / 0 | 19671 | 0 |
| n13 | 6 / 7 | 0.528 / 0.522 | 147 / 146 | 3.55 / 3.55 | 0 / 0 | 0 / 0 | 19671 | 0 |

- 12.2 Gb/s, 211k datagrams/s per search node for 60 min; TX 29505 blocks per sender, 0 late, 0 drops;
  NIC discards 0 everywhere. Search real-time gates met on all 8 halves with duty <= 0.574 (aim 0.6).
- The only loss is pre-existing PHY CRC corruption on whichever of n01/n02 the run lands on (this hour all on
  n02, the previous run all on n01; n09/n13 always clean). The engines absorb it (0 seeks). Hardware
  recommendation: inspect/clean/replace the switch-side optics or cables of n01 and n02.
- E6: ring v4 per search node = 6 owned DMs (4 tier-A x 38400 rows + 2 tier-B x 19200 rows, 20 s retention),
  roughly 28-31 GB of the 97.6 GB node; no memory pressure in 60 min. E5: the engines read the ring with
  `pread` (default), so the mmap NUMA-fault risk does not apply; no lag at any point.

## 4. I2: real FRB 250924mptq through the full 2x chain on GPU — PASSED
Real voltages (16 sub-bands, 2025-09-21 cals) -> 2x corr (GPU) -> frame v2 -> ring v4 -> owner engine (GPU) -> C1, and
the same pages through today's 1x chain (n02, x2/main 6f1bd69):
- Position: 2x and 1x both at pixel (137, 171) (l 0.009381, m 0.001963 rad): **2x - 1x = (0, 0) px**.
- DM: 2x fine 126 (405.49, the predicted trial); 1x 402.19. Time: both within 0.2 sample of the expected arrival.
- S/N: 2x 266.4 (t2) vs 1x 144.8 (b2) = x1.84 = the model's x1.25 at 1 ms / today's wire compression ~0.68 at ~150
  sigma per sample (Phase 0 study 2, ledger S3), now seen on real sky data.
- The ~10 px offset of both chains from the TNS geometric prediction is pre-existing (ledger P-c).

## Open
- Production C sender pacing (16 KB) and corr real-time tail (WS-C), then corr 2x live from replayed
  voltages on all 16 nodes into the real chain.
