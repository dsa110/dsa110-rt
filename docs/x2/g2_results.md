# G2 results (SNAPs on, observing off) — 2026-11-06/07

Everything below ran from the scratch copy `/home/ubuntu/x2_scratch/x2_g2` (x2/main, deployed with
`tools/x2/fleet_scratch.sh -t x2_g2 -b`), under `tools/x2/prod_guard.sh` STOPPED mode, on test DADA keys and
test control ports. No production checkout, etcd key, service or Slack channel was touched.

## 1. SNAP sync and capture drops (after the SNAP restart; replaces the operator's Grafana check)

`tools/x2/snap_capture_check.py drive` on all 16 corr nodes: two test captures per node (scratch build of
`dsart_capture_manythread`, test buffers, test control ports), one common `UTC_START` arm, 5 min armed,
plus a tcpdump header sample per port for per-SNAP sequence offsets.

- Received / expected packets 0.9998-1.0000 on all 32 captures, 7.45 blocks/s, 16 SNAPs on every port.
- Payload drop fraction 0 to 1.5e-4 on every capture, at or below the pre-outage Influx baseline
  (2026-10-21: 1e-8 to 1e-2, median ~1.6e-3); kernel drops 0. n22: 4-8e-5 on its f1 link (0.13 before the
  outage on f0).
- Sync: every SNAP carries the same seq_no at the same time to within 1.12 ms on every node (a SNAP armed on
  another PPS edge would be ~30.5k counts = 1 s off). The 0.2-1.1 ms per-SNAP spread differs per corr node
  = the SNAPs' transmit interleaving across their 16 destinations; harmless (a capture block is 134 ms).
- SNAP arm record: all 32 `/mon/snap/N/armed_mjd` = 61350.93100694 (2026-11-06 22:20:39 UTC).

## 2. Corr 2x on live SNAP data, one node (n03, 10 min, TX framed + drained)

`b8_corr2x.py live --live-capture` (capture x2 -> dsaX_merge -> test fada 0xd2f0 -> corr_fast with the
production cal, DEC 16.2734 = `/mon/array/dec`): mean 50.9 ms, p99 55.6, max 72.6 (budget 134.2), GPU 49.2 ms,
4470 blocks, 0 dropped, fada never above 0/24; capture 0 dropped of 298 M packets (4e-6 late); SNAP NIC
discards 0, CRC 0; 26.8 M v2 datagrams framed, 0 rejected.

## 3. Full chain on live SNAP data, 16 corr -> 4 search, 60 min (g2fl60b)

`fake_corr_to_search.py fleet-replay --live-capture` -- report: `results/g2_fleet_live_60min.md`.

| track | result |
|---|---|
| corr (16 nodes) | **PASS**: mean 72.5-78.4 ms, p99 84.5-92.8, 26822 blocks each, 0 overruns, 0 TX drops, 2.7-2.8 Gb/s each; one block > 134.2 ms on n11 (149 ms, absorbed) |
| search (8 halves) | **PASS**: duty 0.51-0.57, wall p99 148.6-159.5 ms (gate 181.2), **0 seeks, 0 start-up skips**, 17881 cycles each |
| fan-in | 107267 complete / 19 quorum / 0 deadline windows per search node |
| corner turn | n09/n13 loss 0; n01/n02 2.1e-4 = one path n21 -> n01/n02 at 3.3-3.4e-3 (every other path ~1e-6), 1.45 datagrams lost per CRC error: the known fabric fault (ledger), formally fails `loss <= max(1e-4, CRC)` as in G1 |

Value maps: the search side used its own production `antennas.out` (E/N/core identical to the corr nodes'
10-21 blobs; pattern id 0x61e371f9ebde9332 on all four rings and engines).

## 4. Found and fixed during G2

- **Arm deadlock** (n03): corr_fast blocks in the PSRDADA reader holding the GIL until data flows, so an arm
  *thread* never ran -> the arm (and fada sampling) moved to a helper process `b8_corr2x.py live-arm`.
- **Search start-up seek** (g2fl1, n01 half 1): the first live cycle took 0.7-0.9 s of the 0.2 s cycle on every
  half (first-use costs: allocator growth, FFT plans, first launches, plus the full-window first cycle); one
  half's 2 s backlog counted as a lag-recovery seek. Fix: `dsart.x2.search.service.prime_engine` runs 2 cycles
  of a twin engine on in-memory data before the live engine (production service and bench); 60 min since: 0 seeks.
- **False "capture already runs"** (n05): `pgrep -f dsart_capture_manythread` matched prod_guard's own pgrep ->
  `pgrep -x dsart_capture_m` (exact comm).
- **Teardown gaps**: the driver KILLs 3 s after TERM; live buffers / capture shm were left and not reported ->
  corr teardown runs b8's own teardown, leftovers() checks those keys; its "other b8 running" test matched its
  own shell -> anchored on the python executable.
- `snap_capture_check` keys c2a0/c2a2 (a DADA key K also takes K+1).

## Open (G2)

- Downstream live (C1 -> C2 -> C3 / cutouts / plots) with the production 2x search service: needs a C2 test
  instance isolated from production etcd/Slack (user decision on how).
- 1x golden RX recordings (incl. the storm window) need the 1x pipeline producing a 1x RX ring.
