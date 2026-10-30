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

## 4. Full chain from replayed voltages, 60 min: real time end to end (2026-10-29 23:05 - 30 00:05 UTC)
The real 2x corr (`bench/timeres/b8_corr2x.py live`: corr_fast at 2x-gridded-ns4-cint4 on recorded voltages
250924mptq, each node its own sub-band, replayed at line rate through the TEST DADA key 0xd2f0, the C
sendmmsg sender with 16 KB pacing, common `--start-at` / `--block0`) on all 16 corr nodes, into `search_rx`
v4 and the owner engines on all 4 search nodes (8 halves). Driver: `bench/fake_corr_to_search.py
fleet-replay` (x2/g1-chain); every node process under `tools/x2/prod_guard.sh` (stopped mode). Full
tables: `results/g1_fleet_replay_60min.md`, `results/g1_fleet_replay_10min.md`.

| | 60 min (g1c-full60) | 10 min (g1c-full10b) |
|---|---|---|
| corr block time mean / p99 / max (16 nodes) | 69.8-75.9 / 83.6-97.0 / <= 112 ms (n04 142 ms in its first 32 blocks) | 69.7-76.2 / 85.0-96.2 / <= 112 ms |
| corr GPU block (stage timer) mean / p99 | 46.6-50.6 / <= 52.8 ms | 46.4-49.7 / <= 56.6 ms |
| blocks > 134.2 ms (steady state), overruns, TX drops | 0, 0, 0 on every node (26822 blocks each) | 0, 0, 0 (4470 blocks each) |
| per search node | 11.09 Gb/s, 192k datagrams/s | 11.08 Gb/s |
| NIC discards / rejected datagrams | 0 / 0 everywhere | 0 / 0 |
| loss n09, n13 | 0 | 0 |
| loss n01 / n02 (PHY CRC fraction) | 6.4e-4 (4.5e-4) / 3.2e-4 (2.2e-4) | 5.7e-4 (4.0e-4) / 5.6e-4 (4.0e-4) |
| fan-in windows complete / quorum / deadline | n09, n13 all complete; n01 106330/974/0, n02 106623/669/0 | n01 17711/175/0 |
| halves: duty / wall p99 (gate 181 ms) / chain us/image / seeks | 0.520-0.572 / 145-169 / 3.53-3.66 / 0 | 0.518-0.570 / 147-165 / 3.51-3.66 / 0 |

- Real time is met everywhere: every corr real-time gate, and every search gate on all 8 halves.
- The only gate miss is the corner-turn loss on n01/n02, and it is not the pipeline. The loss sits on 2-3 of
  the 64 corr -> search PATHS per run at ~5e-3 each, and ~1e-6 on every other path into the same node:
  - 60 min: n05 and n21 -> n01, n03 -> n02;
  - 10 min: n08 and n11 -> n01, n04 and n08 -> n02.

  The same senders are lossless to n09/n13 and to the other search node. The lossy paths change from run
  to run, consistent with flow hashing over a faulty inter-switch member link: the senders' source ports
  are ephemeral. About 1.42-1.45 datagrams are lost per CRC error counted at the receiving NIC; the excess
  is the corrupted frames a store-and-forward hop drops before the end NIC can count them. The fleet-load
  runs (section 2) showed the same ratio. Hardware recommendation unchanged (n01/n02 switch side and the
  inter-switch links).
- n22 (known jumbo PHY fault) was lossless in this window: <= 2.4e-6 to every search node.
- The engines absorb the loss: 0 seeks, 0 deadline windows. Rows missing on the clean nodes (3e-4 in 60 min,
  2e-3 in 10 min) are the start-up history before the first data; they scale with 1/duration.
- The first 10-min attempt (g1c-full10) was stopped at the start. corr_fast on n11 and n16 exited 0.2 s
  into its first blocking `getNextPage` with `ipcbuf_get_next_read: error decrement FULL`. Both nodes were
  running the merged WS-C code for the first time (cold kernel caches, 13 s warm-up versus 6 s). With
  56 of 64 streams the engines' 60-stream quorum cannot be met, so `fleet-replay` now aborts (with
  teardown) when a corr worker is dead at the start. The repeat runs (warm caches) had no failure.
- Teardown left nothing: no marker process, no dsart-x2 shm, no 0xd2f0, no sentinel, no GPU process on the
  20 nodes, checked by the driver and independently afterwards.
- Set-up notes:
  - the search side (ring spec, engines) must build its value maps at the corr's pointing DEC: 54.5737 for
    250924mptq (the pattern ids depend on it); the x2 antpos fixture equals the 250924mptq cal blob antpos;
  - n18's fixture lacked the 250924mptq cal, so corr_fast would have used the production cal. It was staged
    in `/home/ubuntu/x2_scratch/voltages/250924mptq` (cals from n03, antennas.out = sb12).

## Open
- corr_fast treats one failed `getNextPage` (semop) as fatal (corr_fast_integration.py:5079); the cause of
  the cold-start failure is not visible because psrdada's `ipc_semop` never prints errno
  (dsa110-psrdada/src/ipcutil.c:67, `!(flag | IPC_NOWAIT)` is always false) -> WS-C.
