# Read-only fabric diagnostics during observing, 2026-10-22

All measurements read-only (ethtool -S / sysctl / /proc/net), 10 s deltas, production observing.
Script: copy of `nic_diag.sh` (job tmp), run on each node.

## Search nodes (corner-turn ingress at 1x)
| node | rx Gb/s | packets/10 s | CRC errors | NIC discards | loss | UDP/reassembly errors | tx PAUSE |
|---|---|---|---|---|---|---|---|
| n01 | 4.29 | 1,221,324 | 264 | 88 | 2.9e-4 | 0 / 0 | 25 |
| n02 | 4.27 | 1,215,858 | 538 | 275 | 6.7e-4 | 0 / 0 | 33 |
- Production packet rate ~122k/s per search node (64 streams x 2 DMs x 954 Hz), as estimated in the plan.
- **Pre-existing loss already exceeds 1e-4** (physical-layer CRC errors on n01/n02 + bursts of NIC
  discards with PAUSE frames). The corner-turn loss gate must therefore be relative to the 1x rate
  measured in the same window (plan already states this for SNAP discards; extend it to the corner
  turn).

## n01 "88% loss" in the b5 / b5c fabric tests is an artifact
- The corner-turn harness (`bench/net_pair.py`) defaults to n_filled 5000 x 2 B = 10 kB payloads with
  `--max-frag-payload-bytes 8964`, i.e. a 9036 B + 1162 B datagram pair (loopback average 5099 B). The
  9036 B datagram exceeds the 9000 B MTU and is IP-fragmented on the fabric.
- n01 lacks `/etc/sysctl.d/99-dsart-ipfrag.conf` (ipfrag_high_thresh 262144 vs 268435456 on n02,
  ipfrag_time 30 vs 10) so it cannot reassemble at rate: 88% loss. n02/n09/n13 (tuned) reassembled until
  ~13 Gb/s (3.7% loss). n01 carries the data IP directly on enp129s0f0 (n02: on br2); NIC, firmware,
  queues (40 combined), RSS and coalescing are otherwise identical.
- In production (single-fragment ~4.2 kB datagrams) n01 is clean: 0 reassembly failures, 0 UDP errors.
- Consequences: (1) the "11 Gb/s clean per search node" figure is fragmentation-limited, not a NIC
  limit — the CP1 capacity sweep must use non-fragmenting v2 frames (September: clean to 16 Gb/s with
  8972 B datagrams); (2) E3 ("n01 not fixable") is moot; install the ipfrag sysctl on n01 at CP1 anyway.

## Corr nodes (SNAP ingress, production)
| node | rx Gb/s | CRC errors / packets | NIC discards / packets | tx PAUSE |
|---|---|---|---|---|
| n04 | 18.25 | 0 | 0 | 11 |
| n11 | 18.25 | 0 | 5311 / 4.89 M = 0.11% | 1245 |
| n05 | 18.24 | 0 | 5468 / 4.89 M = 0.11% | 1530 |
| n22 | 17.25 | **263,698 / 4.63 M = 5.7%** (+19,842 symbol errors) | 0 | 0 |
- b5b's 4-6% SNAP discards on n04/n11 were with the pipeline STOPPED; with capture running, discards are
  0-0.11%. The 2x gate (SNAP discards with 2x TX <= 1x in the same window) stands.
- **n22 loses 5.7% of its SNAP packets to physical-layer CRC errors** (optic/cable/port) — pre-existing,
  costs real data on chgroup of n22 today; independent of x2. Needs hardware attention.
