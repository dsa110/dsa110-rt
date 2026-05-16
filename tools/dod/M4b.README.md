# M4b — pair-rate transport DoD

`tools/dod/M4b.sh` orchestrates the M4b real-fabric transport bench across
two LXD-fleet hosts (default n01 → n02). Bench code lives in
`bench/net_pair.py`; this directory holds the runner.

For the milestone narrative + DoD invariant text see plan §M4b
(line 2526), §6.3 (M4b fleet bring-up runbook), §11.4 (pair-rate UDP
probe), and the n02 Phase-2 report at
`/home/ubuntu/vikram/dev/m4b-deploy/PHASE2-REPORT.md`.

## What this DoD asserts

The single-pair production-receive-rate-equivalent test drives one
(corr, search) UDP pair on the `br2` 100 GbE data plane at:

```
24 dm_idx flows × 0.073 Gb/s/flow ≈ 1.76 Gb/s aggregate
```

— that is, **4× the 6-flow per-pair production rate**, which equals
"what one production search node would receive in aggregate from 4
corrs". Three invariants:

| # | Step | Invariant |
|---|------|-----------|
| I1 | STEP 2 | 60 s sustained, fragment-loss < 1e-4, `pattern_mismatch_count == 0`, `tx_dropped_payloads == 0`, achieved Gb/s within ±5% of target |
| I2 | STEP 3 | Mid-run RX `SIGSTOP` (1 s) → `SIGCONT`: TX-side `tx_dropped_payloads` increments during the hold; aggregate TX rate stays ≥ 50% target (no collapse, no upstream backpressure into the gridder per plan §4.3 line 1447) |
| I3 | STEP 4 | 10-min soak: aggregate ±5% target, `pattern_mismatch_count == 0`, fragment-loss < 1e-4 (skipped with `--quick`) |

## Defaults (baked in for n01 → n02)

| Flag | Default | Note |
|---|---|---|
| `--tx-host` | `n01` | corr-host short name |
| `--rx-host` | `n02` | search-host short name |
| `--tx-ip` | `10.41.0.205` | n01's `br2` IP |
| `--rx-ip` | `10.41.0.222` | n02's `br2` IP (per Phase-2 report) |
| `--port` | `19000` | RX bind port |
| `--duration` | `60` | STEP 2 + STEP 3 duration |
| `--soak` | `600` | STEP 4 soak duration |
| `--n-flows` | `24` | full coarse-DM range = 4× per-pair production |
| `--rate-gbps-per-flow` | `0.073` | matches `0.44 / 6` per plan §11 line 2654 |
| `--n01-repo` | `~/proj/dsa110-rt-integration` | tx-host dsart checkout |
| `--n02-repo` | `~/proj/dsa110-rt` | rx-host dsart checkout (Phase-2 path) |
| `--allow-branch` | `m4b/host-bringup-fixes` | both nodes' working branch |

The n01 / n02 repo-path asymmetry is intentional: n01 has historically
hosted the integration checkout under
`~/proj/dsa110-rt-integration`, while the Phase-2 fan-out (per
`PHASE2-REPORT.md`) installs the canonical checkout at
`~/proj/dsa110-rt` on every other node. Override per-host with
`--n01-repo` / `--n02-repo` if your layout differs.

## How RX-hold backpressure (STEP 3) works

The bench needs to verify "TX drops at TX rather than blocking" when
the receiver stalls. The orchestrator side-channels this via:

1. The remote RX bash-wrapper writes its own `$$` (about-to-become-
   python PID) to a `/tmp/m4b_rx_*.pid` file *before* `exec python ...`.
   `exec` replaces the bash without changing the PID, so the file
   contains the python process's PID.
2. At `start_at + duration / 3`, the orchestrator runs
   `ssh rx-host 'kill -STOP $(cat <pidfile>)'`, sleeps 1 s, then
   `kill -CONT`.
3. While the RX is paused, the kernel `SO_RCVBUF` fills up; the TX
   side's wire-loop sees `sendto` errors (kernel `ENOBUFS` / dropped
   packets), counted in `sendto_errors_total` ⇒ surfaced as
   `tx_dropped_payloads_total` in the TX counters JSON.

If a future net_pair TX impl uses `TransportTx` (with an
application-level token-bucket pacer that drops at TX), the same
field will be populated by the pacer's `tx_dropped_payloads` counter
instead. Either signal satisfies I2.

## Known gaps / follow-ups

- **§11.6 lying-pipeline 30-min DoD** (plan §M4b line 2538 last
  sentence) is **not** wired. `bench/derisk/lying_pipeline.py` does not
  exist yet; there is no `bench/derisk/` directory. **M7 owns this**;
  add a `STEP 6` to `M4b.sh` once that bench lands.
- **Multi-corr fan-in** (16 TX hosts → 1 search node) is not yet
  exercised on real fabric. Only n02 is Phase-2 complete per the n02
  Phase-2 report; once n03..n22 finish Phase-2 fan-out, a sibling
  bench (`bench/net_fan_in.py`?) can drive the full 16-corr ingress
  per plan §11.4 line 2802.
- **Intra-soak time-series** for STEP 4: `bench/net_pair.py` currently
  summarises at end-of-run. The "no monotonic drop_rate climb" verdict
  in I3 is approximated via end-of-run aggregate ±5% + fragment-loss
  < 1e-4. Adding 10 s windowed stats to `net_pair.py` (and updating
  `assert_step4` to inspect them) is a clean follow-up.
- **`epoll_rx_latency_hist`** is `null` in the RX counters JSON — the
  M4a chunk-6 C epoll loop does not yet expose a per-datagram latency
  histogram. Add when the chunk-6 C path grows it.

## Local pre-flight smoke

`bench/net_pair.py --smoke` runs a 5 s, 1-flow, loopback round-trip on
either side independently — useful for verifying the env is healthy
before invoking `M4b.sh`:

```bash
# On n02:
python -m bench.net_pair --mode rx --smoke

# On n01 (in another shell):
python -m bench.net_pair --mode tx --smoke
```
