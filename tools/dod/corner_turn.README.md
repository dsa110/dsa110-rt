# `corner_turn.sh` — full-fleet 16→4 corner-turn bench

Multi-corr fan-in test that extends `bench/net_pair.py` (a single TX→RX
pair) to the production topology: **16 corr (TX) nodes → 4 search (RX)
nodes** at the production per-pair rate, sustained for 5 minutes.

This is the natural follow-up to M4b core DoD (single n01↔n02 pair at 4×
per-pair rate) once Phase-2 fan-out reached the full 18-of-20 fleet. The
M4b core DoD verified one pair could sustain 4× per-pair rate; this bench
verifies the entire 64-stream fabric sustains the actual production
shape at the actual production per-pair rate.

## Production topology (per `REALTIME_FRB_SEARCH.md` §1)

```
   TX nodes (corr farm, 16):    RX nodes (search farm, 4):
     n03 (chgroup  0)             n01 (10.41.0.205)
     n04 (chgroup  1)             n02 (10.41.0.222)
     n05 (chgroup  2)             n09 (10.41.0.253)
     n06 (chgroup  3)             n13 (10.41.0.238)
     n07 (chgroup  4)             ────────────────
     n08 (chgroup  5)
     n10 (chgroup  6)           Each TX → all 4 RX
     n11 (chgroup  7)           Each RX ← all 16 TX
     n12 (chgroup  8)           Total: 64 unidirectional UDP streams
     n14 (chgroup  9)
     n15 (chgroup 10)
     n16 (chgroup 11)           Ports: 6625 + chgroup
     n18 (chgroup 12)             n03 sends on port 6625 to all 4 RX
     n19 (chgroup 13)             n04 sends on port 6626 to all 4 RX
     n21 (chgroup 14)             ...
     n22 (chgroup 15)             n22 sends on port 6640 to all 4 RX
```

The chgroup → node assignment mirrors `dsaX_dbnic.c` / `dsaX_nicdb.c` in
the legacy stack: each corr node carries one chgroup of the 16-chgroup
total band, and announces itself by sending on a chgroup-specific port.
Each search node listens on all 16 ports (one per corr).

## Rate math

* **Per-flow rate**: 0.073 Gb/s (matches `bench/net_pair.py` default).
* **Per-pair**: 6 flows × 0.073 = **0.438 Gb/s**.
  This is the per-pair production rate (plan §11 line 2654: "0.44 / 6"
  is the per-flow rate). M4b's core DoD ran a single pair at 24 flows
  (1.752 Gb/s); the corner-turn runs each pair at the canonical 6 flows.
* **Per-RX aggregate ingress**: 16 × 0.438 ≈ **7.01 Gb/s**.
  This matches the M4a chunk-6 C-epoll RX ceiling validated on h01
  loopback (7.885 Gb/s — the corner-turn lands at ~89% of that ceiling).
* **Total fabric load**: 64 × 0.438 ≈ **28.2 Gb/s** across the 10.41/24
  data plane (40 GbE NIC per node, MTU 9000).

## DoD invariants

Per pair (each evaluated on the 64 pairs):

| id | criterion | rationale |
|----|-----------|-----------|
| **I1** | RX rate = 0.438 ± 5% Gb/s | sustained at production rate |
| **I2** | `fragment_loss_estimate_fraction < 1e-4` | budget from M4b §M4b DoD I1 |
| **I3** | `pattern_mismatch_count == 0` | no codec / chgroup-id corruption |
| **I4** | `tx_dropped_payloads_total + sendto_errors_total == 0` | no upstream backpressure into the TX queue (plan §4.3 line 1447 "drop oldest, don't block") |

Per RX (each evaluated on the 4 RX nodes):

| id | criterion | rationale |
|----|-----------|-----------|
| **I5** | aggregate ingress = 7.01 ± 5% Gb/s | RX node can sustain full 16-source fan-in |

**All 5 invariants pass on every pair / RX node → CORNER-TURN PASS at
production rate.**

Short smoke runs (`--smoke`, 10 s) relax I2 to `< 5e-4` since the startup
transient (RX socket warm-up, reassembly hash priming) can push a single
pair just over 1e-4 in the first ~5 s; 5-min sustained runs cleanly meet
1e-4.

## Synchronization

All 128 net_pair processes (64 TX + 64 RX) are launched in a single
parallel wave from h23 via `ssh nohup`, then sleep until a common
`--start-at` (a unix UTC float). The orchestrator picks T0 = `now + 60 s`
so the launches have a 60 s quiet window. All nodes are NTP-synced;
50 ms skew tolerance is fine since net_pair's rate-limiter is per-flow.

RX side is launched first (with a 3 s settle), then TX. This guarantees
every listening socket is bound before the wire starts.

## Per-node resource footprint

* **RX node** (n01, n02, n09, n13): 16 net_pair processes × ~1 core CPU +
  ~512 MiB RAM (incl. 256 MiB `SO_RCVBUF` per socket). Aggregate: ~16
  cores + ~8 GiB RAM. Each node has 24 cores + 192 GiB RAM — easy fit.
* **TX node** (16 corr nodes): 4 net_pair processes × ~2 cores + ~256
  MiB RAM. Aggregate: ~8 cores + ~1 GiB RAM. Each node has 24 cores +
  192 GiB RAM — trivial.

Kernel buffer budget per RX node: 16 × 256 MiB = 4 GiB. Within
`net.core.rmem_max = 256 MiB` (Phase-2 sysctl target; n18 already at
512 MiB, even more headroom) and `net.core.netdev_max_backlog = 100000`.

## Usage

```bash
# Full 5-min run at production rate (default):
bash tools/dod/corner_turn.sh

# Smoke test (1×1, 10 s):
bash tools/dod/corner_turn.sh --smoke

# Custom duration:
bash tools/dod/corner_turn.sh --duration 60

# Subset of nodes:
bash tools/dod/corner_turn.sh --tx-list "n03 n04" --rx-list "n01 n02"

# Tweak rate (e.g. 4× per-pair production = M4b core DoD rate):
bash tools/dod/corner_turn.sh --n-flows 24 --rate 0.073
```

## Outputs

On h23 under `~/dsart-corner-turn-logs/<UTC>/`:

```
banner.txt                          run config + targets
launch_(rx|tx)_<from>_<to>.log      per-launch ssh transcripts
rx_<RX>_from_<TX>.json              per-pair RX counters (64 files)
tx_<TX>_to_<RX>.json                per-pair TX counters (64 files)
summary.json                        aggregated verdict + per-pair rows
verdict.txt                         human-readable PASS / FAIL summary
```

On each node (cumulative across runs — not auto-cleaned):

```
~/dsart-corner-turn/<UTC>/          per-run counter staging
~/dsart-ct-(rx|tx)-<peer>.log       net_pair stdout/err (overwritten per run)
```

## Known gaps

These are **out of scope** for this bench; they live in M7 or follow-up
work:

* **30-min lying-pipeline DoD** (plan §11.6, `bench/derisk/lying_pipeline.py`
  does not exist) — M7-owned.
* **RX-hold backpressure on every pair** — M4b's I2 (single-pair) verified
  the no-backpressure-into-TX property. Here we observe `tx_dropped_payloads`
  and `sendto_errors` stay zero at production rate, which is the same
  invariant from the other direction, but we don't actively SIGSTOP any RX.
* **Sustained > 5 min** — operationally a 10-min or 1-hour soak would
  catch slow leaks. We pick 5 min to keep the bench fast and to match the
  user's specified duration.
* **Mixed traffic** — only the value-channel uv-grid payload is sent.
  Control-channel (trigger / status / heartbeat) traffic isn't in this
  bench's scope; that's covered separately by the M5 / M6 control-plane
  benches.

## Why this is the right test to run before M7

M4b core DoD certified the *single-pair* steady-state property: one
real 40 GbE pair can sustain 4× per-pair production rate (= per-search
aggregate from 4 corrs) with the new transport.

The corner-turn certifies the *fleet-wide* steady-state property: the
full 64-stream fabric at production shape and rate, with every RX node
seeing every TX node. If this passes, then any subset (e.g. a 4→1
fan-in soak, or a 16→1 per-search test) is implicitly covered. The only
remaining transport question is the 30-min lying-pipeline DoD, which is
explicitly M7 scope.
