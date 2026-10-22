# CP1 runbook: first observing interruption for the 2x search

Plan: `/home/ubuntu/.claude-vikram/plans/velvety-jingling-crescent.md` (rev 4), section CP1.
Runner: `tools/x2/cp_runner.py` with the step list in `tools/x2/cp_runner_plans.py`.
Run `cp_runner.py run cp1 --track all --dry-run` to print every command.
Owner: WS-O.

## What CP1 is

CP1 lasts about 6.5 h: about 4.9 h of steps (runner critical path) plus about 1.5 h of contingency.
It runs short GPU and network measurements while observing is stopped.

- **Production code and configs are not changed.** Every test runs from a scratch copy of each
  node's checkout at `/home/ubuntu/x2_scratch/<tag>`. No etcd key is written, and no op-point is
  switched.
- **There is one planned host change.** n01 gets n02's `/etc/sysctl.d/99-dsart-ipfrag.conf`
  (diagnostics 10-22).
- **The window ends with the fleet stopped in 1x.** The runner verifies this state. You then press
  Start (op-point 1x) in the web UI.
- **What CP1 produces:**
  - the gate results;
  - the format choice (S1);
  - the S2 evidence;
  - any escalations (E1, E2, E4);
  - the O2 schedule.

  A failed gate leads to a short CP1b of about 3 h.

## Safety rules (after the 2026-10-22 n13 incident)

On 2026-10-22, an mmap reader of n13's production RX ring left 41.5 s of data unsearched. The runner
now follows these rules:

1. **Before the window, everything is read-only.** The `pre` track only reads:
   - git on h23;
   - etcd;
   - `git --no-optional-locks` on the nodes;
   - `ethtool -S`;
   - the search_compute logs.

   The CPU tests go through `tools/x2/node_cpu_tests.sh`, which wraps every test in
   `tools/x2/prod_guard.sh`. Nothing mmaps a production segment.
2. **Production must be stopped before anything is touched.** Some steps touch GPUs, shm, NICs,
   services or a host setting; the dry run tags these `touches ...`. Right before each such step,
   the runner checks:
   - etcd reads READY (`check fleet-stopped`: corr state READY, and no search orchestrator running
     or starting);
   - on every node of the step, no production pipeline process is running. This means no
     corr_fast, corr_slow, search_compute, search_rx, voltage_retention, capture, merge or
     fringestop process without this run's marker;
   - no unmarked process is using a GPU.

   If any check fails, the runner aborts (see Abort). The runner never stops or starts production
   itself.
3. **Every runner process is marked.** Anything the runner starts on a node carries
   `DSART_X2_CP=<tag>`. Teardown kills exactly those processes and removes `/dev/shm/dsart-x2-*`.
4. **NIC interfaces are read from each node.** The runner uses the interface that holds the
   10.41.0.x address, through the bridge (n02 uses br2). This is normally `enp129s0f0`. n14 runs on
   f1 after an in-memory failover, which reverts on reboot.

## Before the window (entry criteria)

Integration requests the window only when all of these hold:
- all CPU tests are green on `x2/main`;
- I1 has passed;
- the runner dry run is clean;
- this runbook has been sent;
- the kernels have passed their CPU twins and the O4 compile checks.

Then, while observing, run:

```
cd <x2/main checkout on h23>
tools/x2/cp_runner.py run cp1 --track pre          # ~45 min, read-only + guarded CPU tests
```

| step | what it records (in `~/x2_checkpoints/<tag>/pre/`) |
|---|---|
| pre.repo_state | the sha under test; the checkout must be clean |
| pre.static_checks | pyflakes over the x2 code; every .py parses (no bytecode written) |
| pre.compile_only | O4 compile-only kernel checks on h23 (WS-C / WS-S scripts) |
| pre.cpu_tests | every `tests/x2/test_*.py` on n13 under prod_guard |
| pre.etcd_snapshot | sha256 of /cnf/pipeline_rt, /cnf/search_rt, /cnf/opoint and the injection K prefix |
| pre.node_checkouts | the production checkout sha and dirty count on all 20 nodes |
| pre.search_health_1x | the 1x baseline of lag seeks, cube rate and scatter, 2 min on 8 halves |
| pre.nic_baseline_1x | 5 min of 1x NIC counters on all 20 nodes, used for the relative loss/discard gates |

The tag is `cp1_<sha>`. Use the same `--tag` for every later command, so all results land in
`~/x2_checkpoints/<tag>/`.

## The window

Run all tracks in sequence with one command:

```
tools/x2/cp_runner.py run cp1 --track all --tag <tag> [--corr-node n04] [--search-node n09]
```

Or run one track per terminal: `--track window`, then `corr` and `search` in parallel, then
`fleet`, then `restore`. Steps marked CONFIRM or USER ACTION wait for you at the terminal.
To answer them in advance, pass `--yes <step>`.

| t (h) | track | step(s) | min | what happens | you | abort point |
|---|---|---|---|---|---|---|
| 0:00 | window | stop_observing | 5 | You stop observing; the runner verifies READY. | **Stop** in the Control tab (confirm=stop), wait for READY, type `done`. | yes, if not READY |
| 0:05 | window | gpus_idle, stage_scratch | 16 | Builds the scratch copy and C extensions on 20 nodes in parallel. | none | yes |
| 0:21 | corr (n04) | env_check | 5 | Checks torch, the GPUs, and that dsart imports from scratch. | none | yes |
|  | corr | prod_profile_golden | 30 | Records the production 1x stage profile and ≥64 golden blocks (RFI, warm-up, array-burst, injection). | none | yes |
|  | corr | kernels_2x | 25 | Runs stage-1/GEMM for both maps at n_sub 2 and 4, and measures peak memory and exactness. | none | yes; E2 if GEMM+stage-1 > 55 ms |
|  | corr | compat_1x | 20 | Runs the 1x-compat taps against the golden blocks. | none | yes |
|  | corr | live_2x | 30 | Runs corr 2x LIVE on n04: only capture, corr_slow and voltage_retention start; the test corr_fast takes the production reader identity; TX goes to a drain. | **CONFIRM** | yes |
|  | corr | collect, nic_gate | 3 | Compares SNAP NIC discards with the 1x baseline. | none | yes |
| 0:21 | search (n09) | env_check | 5 | Same checks as corr env_check. | none | yes |
|  | search | kernels | 40 | Runs the FFT variants, K3b box + b3x (Phase 0 S2e), and K1 per format (1x CSR bit-exact). | none | yes; E1 if the chain > 4.8 µs/image |
|  | search | compat_1x | 20 | Runs the 1x-compat search taps. | none | yes |
|  | search | engine_synthetic | 15 | Runs the owner engine on both GPUs from a synthetic ring. | none | yes |
| 2:14 | fleet | n01_ipfrag | 10 | Installs n02's ipfrag sysctl file on n01 and checks that n01's values equal n02's. | **CONFIRM** | yes |
|  | fleet | nic_detect | 1 | Records each node's data interface. | none | none |
|  | fleet | capacity_sweep | 35 | Sends v2 single-fragment frames, 16 corr to 4 search, at 9, 11, 13, 15, 16, 17 and 18 Gb/s per search node, recording NIC counters, Gb/s and datagrams/s. | none | yes |
|  | fleet | format_choice | 5 | Prints the S1 table (Phase 0 needs vs measured clean ingress) and suggests a format. | **S1 sign-off:** type the format id | yes; E4 if nothing fits |
|  | fleet | plan_build, plan_distribute | 4 | Builds and self-verifies the v3 plan of the chosen format, then copies it into every scratch copy. | none | yes |
|  | fleet | search_full_load | 45 | Sends v2 synthetic UDP at the chosen load +10% into search_rx v4, with owner engines on all 8 halves. | none | yes |
|  | fleet | c_sender_snap_live | 15 | Runs the C sendmmsg sender at the per-corr load with the SNAPs live; gate is SNAP NIC discards ≤ 1x per corr node. | **CONFIRM** | yes |
|  | fleet | collect | 3 | Pulls the node results. | none | none |
| 4:12 | restore | kill_marked … h20_pusher | 9 | See "Restore" below. | none | yes |
|  | restore | user_start | 15 | Waits until the fleet is OBSERVING. | **Start** (op-point 1x, the default), wait for PREPARED, **Arm**, then type `done`. | none |
|  | restore | search_health, nic_after | 19 | Checks the incident signature for 10 min after a 2 min settle; compares NIC counters with the 1x baseline. | none | none (reported) |
| ~4:55 | | end | | Contingency to ~6.5 h. Review the results. | | |

## Gates (no compromise; a miss is escalated, never waived)

- **corr live:**
  - block time mean ≤ 85% of 134.218 ms, and p99 ≤ 100%;
  - zero fada/eada overruns;
  - SNAP NIC discards ≤ the 1x rate.
- **Kernels:**
  - corr exact against the CPU reference (fp32 bit-exact; GEMM ≤ 1e-5 relative, then 1 fp16 ULP);
  - K1 1x CSR bit-exact against `combine_sparse_v4`;
  - FFT |d| ≤ 2e-3 σ against float64;
  - K3b compat |ΔS/N| ≤ 0.02.
- **Corner turn:**
  - a node is "clean" when loss ≤ max(1e-4, that node's 1x loss in the pre baseline) and NIC
    discards do not increase over the dwell;
  - the operating load is ≤ 80% of the clean rate;
  - fan-in is complete for ≥ 99.9% of cubes per DM;
  - Phase 0 puts the needs at 11.4–17.2 Gb/s per node, and gridded ns2 cint5 at about 305k
    datagrams/s.
- **Search:**
  - GPU duty ≤ 0.75 per half (aim ≤ 0.6);
  - bounded lag and zero ring overruns;
  - host Python ≤ 30 ms per cycle.

## S1: the format choice

`check format-table` prints each row of the Phase 0 table (`docs/x2/phase0/SUMMARY.md`): total
vs model, and the clean ingress needed per node at ≤ 80% load. It also prints the measured clean
ingress (the minimum over the search nodes) and a suggestion: the highest total that fits, with
gridded preferred on ties.

| format | total vs model | need (Gb/s/node) |
|---|---|---|
| 2x-gridded-ns4-cint5 | 0.997 | 17.2 |
| 2x-gridded-ns2-cint4 | 0.996 | 16.4 |
| 2x-baselines-ns4-cint5 | 0.995 | 14.0 |
| 2x-gridded-ns4-cint4 | 0.994 | 14.0 |
| 2x-baselines-ns4-cint4 | 0.993 | 11.4 |
| 2x-baselines-ns2-cint4 | 0.990 | 11.5 |

You type the op-point id. It is validated against `dsart.x2.opoint.opoint_from_id`. Everything
afterwards (the plan, the full-load test, and later the 2x config set and `/cnf/opoint`) is keyed
by that id. Also check the search GPU duty from the search track before signing off.

## Abort

The runner aborts in any of these cases:
- a step fails;
- a gate script exits non-zero;
- a stopped-production assertion fails;
- a production guard trips;
- you press Ctrl-C;
- you decline a CONFIRM.

You can also abort explicitly with `tools/x2/cp_runner.py abort cp1 --tag <tag>`.

An abort does the following:
1. **Teardown** of every track it started:
   - each bench's own `teardown` subcommand;
   - kill the `DSART_X2_CP=<tag>` processes;
   - remove `/dev/shm/dsart-x2-*`;
   - on the corr test node only, destroy leftover PSRDADA buffers (dada, eada, fada, bada), and
     only when no production process runs;
   - verify that no test process, shm or GPU use is left.
2. **Restore checks**, without your start:
   - every node is clean;
   - the production checkouts are unchanged (sha and dirty count on all 20 nodes);
   - the live etcd configs, /cnf/opoint and injection K are unchanged;
   - the fleet is READY with 16/16 corr and 4/4 search orchestrators up;
   - the h20 pusher is active.
3. **End state.** The fleet is stopped in 1x. Once the checks pass, Start from the web UI.

If you run one track per terminal, a failure tears down only that track. Run
`cp_runner.py abort cp1` to restore the whole fleet.

## Restore (normal end)

The restore track runs:
1. `kill_marked`, `no_test_leftovers`, `checkouts_unchanged`, `etcd_unchanged`, `fleet_ready`,
   `h20_pusher` (all as in an abort).
2. `user_start`: you press Start with op-point 1x, wait for PREPARED, then Arm.
3. `search_health`: the 2026-10-22 incident signature, checked on all 8 halves for 10 min after a
   2 min settle. The step reads the search_compute logs with `tail -F` only. It requires:
   - 0 `lag-recovery seek`;
   - never two consecutive `cubes/s last` values below 4.9 (the cadence is 4.97);
   - the cube_progress `scatter` mean ≤ 90 ms (normally about 54 ms; 110–135 ms in the incident).

   The pre baseline is in `pre/search_health_1x.json`.
4. `nic_after`: 5 min of NIC counters, compared node by node with the 1x baseline. n01 should
   improve after the ipfrag sysctl.

## The n01 ipfrag sysctl (the only host change)

The `fleet.n01_ipfrag` step does the following:
1. Records n01's current `ipfrag_high_thresh`, `ipfrag_low_thresh`, `ipfrag_time` and
   `ipfrag_max_dist`, and lists `/etc/sysctl.d/`. All of this goes into the step log.
2. Copies n02's `99-dsart-ipfrag.conf` through h23, installs it as root:root 0644, and runs
   `sudo -n sysctl --system`. If a different file already exists, it refuses. If sudo needs a
   password, the step fails; install by hand, then continue with
   `--track fleet --from fleet.nic_detect`.
3. Checks that every key in the file reads the same on n01 as on n02.

To revert: `sudo rm /etc/sysctl.d/99-dsart-ipfrag.conf` and
`sudo sysctl -w <key>=<old value from the step log>` for each key.

## Results

Everything is under `~/x2_checkpoints/<tag>/`:

| path | contents |
|---|---|
| `pre/` | the pre-window baselines (above) |
| `<action>_<track>_<UTC>/results.json` | per step: status, gate, guard results, commands (rc, timing, output tails) |
| `<action>_<track>_<UTC>/logs/<step>.log` | full output of each step |
| `corr/`, `search/`, `fleet/` | bench JSON, NIC samples, `capacity.json`, `plan_<format>.npz`, the gate JSONs |
| `nodes/<node>/` | node-side results |
| `restore/` | `search_health.json`, the NIC gate |

Step statuses:
- `passed`;
- `failed` (a gate missed, or an error);
- `refused-guard` (production was not stopped);
- `guard-tripped` (prod_guard killed the step);
- `missing-prereq` (a placeholder script does not exist yet);
- `declined`.

## Scripts other workstreams still owe (placeholders in the dry run)

The command lines in `cp_runner_plans.py` are the contract. Each script must:
- exit 0 when all gates pass, 1 when a gate fails, and 2 on an error;
- write JSON under `--out`;
- name test shm `dsart-x2-*`;
- implement `teardown` (except the h23 drivers).

| script | owner | subcommands used |
|---|---|---|
| `bench/timeres/b8_corr2x.py` | WS-C | `compile-check`, `profile-1x`, `golden-capture --blocks`, `kernels --maps --n-sub --check-ref`, `compat-1x --golden`, `live --opoint --minutes --tx drain --reader-identity corr_fast`, `teardown` |
| `bench/timeres/b8_owner2x.py` | WS-S | `compile-check`, `kernels --fft --k3b --k1-formats --check-1x-csr`, `compat-1x`, `engine --gpus --input synthetic-ring/rx-v4 --opoint --minutes`, `teardown` |
| `bench/fake_corr_to_search.py` v2 | WS-T | `sweep --frame-version 2 --rates-gbps --dwell-s --record-pps --nic-counters` (writes `clean_gbps_per_node`), `fleet-load --format --load-factor --plan --engine`, `snap-live --sender c --format` (all take `--tx-nodes --rx-nodes --scratch --marker --out`) |

## Outside CP1, for O2 and CP2

These items are recorded here so they are not forgotten:
- **The h20 Influx pusher** (`dsart_rt_to_influx.service`, a system unit on lxd110h20 running from
  `/home/ubuntu/proj/dsa110-rt` on h20) is outside `update_dsart` and outside the dashboard service
  inventory. Any deploy must update it by hand: `git pull` there, then
  `sudo systemctl restart dsart_rt_to_influx`.
- **C2 class ids.** Install `systemd/dsart_c2.service.d/x2-pythonhashseed.conf` into
  `~/.config/systemd/user/dsart_c2.service.d/`, then run `systemctl --user daemon-reload`. This
  keeps `trigger_class_id` stable across C2 restarts, including op-point switches. The Start
  control warns until this is done.
- **Op-point sets** are pushed to `/cnf/opoints/<sel>/*` with
  `tools/ops/push_dsart_to_etcd.py --opoint <sel> [--plan ...] --write`. This is safe while
  observing, because it never writes a live key. The web-UI Start control switches only when the
  fleet is stopped (docs/x2/SPEC.md §10).
