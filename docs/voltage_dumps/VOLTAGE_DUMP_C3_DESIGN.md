# Voltage dumps + C3 collector/veto — design

Status: implementation in progress (branch `feature/voltage-dumps-c3`).
Author: voltage-dump bring-up.
Supersedes the legacy `dsaX_trigger` / `dsaX_store` / `look_after_dumps.py`
(corr) + `dsa110-T3` (h23) chain, re-expressed in the `dsa110-rt` idiom.

---

## 0. Goal

Capture raw voltages on the 16 corr nodes for C2-triggered events, exactly
the way cubes are already captured — same C2 UDP trigger, same per-event
candidate directory — and add an h23 service ("C3") that collects the dumps,
runs the cube-morphology veto developed in the B1933/false-positive analysis
(`reports/b1933_fp_analysis_20260618/REPORT.md`), and either **keeps** the
voltages or **conservatively rejects** the event (delete bulky cubes +
voltages, move metadata/plots aside — never a full `rm -rf`).

### Operator-locked decisions (2026-06-18)

| Decision | Choice |
|---|---|
| Retention target | ~15 s / ~32 GiB per node |
| Retention mechanism | Dedicated isolated RAM ring fed by a **3rd** `fada` reader |
| Dump window length | ~3 s (margin for high-DM sweep) |
| C3 conservativeness | Delete **only** on high-confidence cube vetoes; KEEP anything ambiguous |
| Injections | **No** voltage dumps for injections (synthetic; saves ~69 GiB/event) |
| Branch | `feature/voltage-dumps-c3` off current HEAD |

---

## 1. Numbers (physics-pinned)

From `dsart.common.constants`:

* `FADA_BYTES_PER_BLOCK = 2048 × 96 × 384 × 2 × 2 × 1 = 301,989,888` B (288 MiB).
* `BLOCK_SAMPLES_SPECNUM = 2048` specnums / block.
* `1 specnum = 2 native samples = 65.536 µs` → `1 block = 134.218 ms`.
* `block_n = event_specnum // 2048`  (same anchor convention corr_fast uses:
  its first `fada` page is `block_n = 1`, so `block_specnum_start = block_n*2048`).

Derived sizing (per corr node):

| Quantity | Blocks | Bytes |
|---|---|---|
| Retention ring (15 s) | `ceil(15 / 0.134218) = 112` | 112 × 288 MiB = **31.5 GiB** |
| Dump window (3 s) | `N_PRE + N_POST + 1 = 8 + 14 + 1 = 23` | 23 × 288 MiB = **6.47 GiB** |
| Full event (16 nodes) | — | **~103 GiB** |

Latency budget: C2 trigger latency is ~0.5–2 s nominal, ~25–30 s worst case
(RFI backlog). The 15 s ring covers nominal + moderate backlog; in a severe
RFI flood the oldest blocks roll off and that event's dump is lost (acceptable
— those are exactly the events C3 would veto anyway). The dump window is placed
`[block_n − N_PRE, block_n + N_POST]` so it brackets the dispersion sweep
(DM≈3000 across the 250 MHz subband band is ~1.1 s, well inside 2 s of post).

RAM headroom: the new pipeline frees the legacy `eaea` (200×288 MiB ≈ 56 GiB)
+ `fafa` (~9 GiB) rings, so a 31.5 GiB retention ring fits comfortably.

---

## 2. Architecture

```
                C2 (h23 coincidencer)
                   │  dump_all_gpus, NOT an injection, voltages_enabled
                   │
   ┌───────────────┼────────────────────────────┐
   │ cube path (existing)        │ voltage path (NEW)
   ▼                             ▼
 8× C1 cube listeners      16× corr VoltageTriggerListener   (UDP :11229)
 (search nodes)                  │  flag DUMP_VOLTAGE
                                 ▼
                         voltage_retention service (corr)
                          ├─ reader thread: 3rd fada reader → RAM ring
                          └─ dump worker: ring → NVMe staging .out + .json
                                 │
                                 ▼  (per node) /home/ubuntu/data/voltage_staging/
                                 │     <event>_sb<NN>_data.out (+ .json)
                                 ▼
                         C3 (h23): cube_veto → KEEP / REJECT
                          ├─ KEEP   : rsync 16 .out → <cand>/Level2/voltages/
                          └─ REJECT : signal corr nodes to delete staging,
                                      delete <cand>/cubes/*, move metadata/
                                      plots → candidates_rejected/<event>/
```

### 2.1 Trigger packet (no schema bump)

`coinc/wire.py` already carries a 64-byte LE trigger packet with a `flags`
field. We add **one bit**:

```
C2_TRIGGER_FLAG_DUMP_CUBE    = 1 << 0   (existing — search nodes)
C2_TRIGGER_FLAG_DUMP_VOLTAGE = 1 << 1   (NEW    — corr nodes)
```

`schema_version` stays 1: the wire struct is unchanged, only a previously-zero
bit is now meaningful. The cube broadcast keeps `flags = DUMP_CUBE`; the
voltage broadcast sends `flags = DUMP_VOLTAGE` to the corr nodes.

### 2.2 C2 changes (`services/coincidencer.py`)

On `dump_all_gpus`, after all existing vetoes pass and `dumps_enabled`:

1. Compute `is_injection = bool(member_inj_ids)` (moved **before** broadcast —
   the inject registry is time-sensitive and must be read at fire time).
2. Voltage gate: a second etcd kill-switch `/cmd/c2/voltages_enabled`,
   **default-CLOSED** (missing key → disabled). Voltages are an expensive,
   brand-new capability; the operator opts in via the dashboard.
3. If `voltages_enabled and not is_injection`: `voltage_broadcaster.broadcast(
   event_name, peak_event_specnum, mjd_target, flags=DUMP_VOLTAGE)` to all 16
   corr nodes (best-effort UDP, mirrors the cube broadcaster).
4. Durable injection marker: the L3 metadata now carries
   `"injection": {"is_injection": bool, "inj_ids": [...]}` so **C3 can tell an
   injection from a real event long after the registry expired** (the matcher's
   `/cnf/inject/active` entries live only ~60 s; C3 runs minutes later).

### 2.3 Corr-node `voltage_retention` service (NEW)

One long-running Python service per corr node, spawned by `dsart_rt` like the
other corr routines. Three cooperating parts:

* **Reader thread** — attaches `fada` as the **3rd** reader (`r=3`). Loop:
  `getNextPage()` → `memcpy` into the next RAM-ring slot → record its
  `block_n` → `markCleared()`. The memcpy of 288 MiB is ~30–50 ms ≪ the
  134 ms block cadence, so this reader never back-pressures capture. It does
  **nothing else** — this is the isolation guarantee.
* **RAM ring** (`dump/voltage_ring.py`, pure + tested) — a preallocated
  `(N_BLOCKS, FADA_BYTES_PER_BLOCK)` uint8 array + a parallel `block_n[N]`
  index. O(1) slot = `block_n % N_BLOCKS`; a slot is "valid for block b" iff
  `index[slot] == b`. Lock-free single-writer / single-reader-per-request via
  a generation counter (the dump worker copies out under a short lock that the
  reader does not hold during its memcpy).
* **Dump worker** (thread) — consumes requests from the listener. For a
  request `(event_name, event_specnum)`:
  1. `target = event_specnum // 2048`; window `[target−N_PRE, target+N_POST]`.
  2. Wait (≤ `dump_wait_s`) until the ring's newest `block_n ≥ target+N_POST`
     (covers trigger-before-post-blocks-captured).
  3. Copy the in-range blocks out of the ring (skipping any already rolled off)
     and stream them to `staging/<event>_sb<NN>_data.out` (raw concatenated
     fada bytes, **no header** — matches the legacy `.out` convention) plus a
     sidecar `staging/<event>_sb<NN>.json` manifest
     (`event_name, cn_id, chgroup, block_n_start, n_blocks, first_specnum,
     mjd_start, bytes_per_block, dropped_blocks`).

A delete request (`event_name`) removes the staged `.out`+`.json` (C3 REJECT
path). Disk staging is on local NVMe (`/home/ubuntu/data/voltage_staging`),
so a slow C3 pull never touches the RAM ring or capture.

### 2.4 Corr-node `VoltageTriggerListener` (NEW)

Mirror of `dump/c2_trigger_listener.py` but on the corr side:

* binds `(corr-net-ip, 11229)`, decodes the 64-byte `C2TriggerPacket`,
* ignores packets without `DUMP_VOLTAGE`, drops bad magic/size,
* enqueues `(event_name, event_specnum)` to the dump worker,
* exposes mon counters (`received, dump_flagged, enqueued, queue_full,
  bad_magic, bad_size`).

It also serves the **delete** control: C3 deletes staged voltages by sending a
`DUMP_VOLTAGE`-flagged packet with `event_specnum = 0` (sentinel) — the
listener routes `event_specnum==0` to the worker's delete path keyed by
`event_name`. (Keeps the corr side single-port, no extra socket.)

### 2.5 C3 service on h23 (NEW, `services/c3.py`)

Watches `event_archive_root` (`/dataz/dsa110/candidates`) for events that have
(a) a complete cube set (`cubes/cube_s*_g*_*.npz`, ≥ expected count) and (b)
a Level3 JSON (i.e. C2 already materialised the dir). For each new event:

1. **Veto** (`coinc/cube_veto.py`, pure + tested) on the dumped cubes +
   L3/C1 metadata. Injection-exempt: if L3 `injection.is_injection` is true
   (or any C1 row has an `inj_id`), force KEEP.
2. **KEEP** → collect voltages: poll each corr node's staging over the corr-net
   (`rsync`), wait for up to 16 fragments (or `collect_min_fragments` after
   `collect_timeout_s`, mirroring legacy `MIN_CT=8`), land them in
   `<cand>/Level2/voltages/<event>_sb<NN>_data.out`. Record a
   `Level3/<event>_voltages.json` collection report.
3. **REJECT** (high-confidence veto only) → conservative cleanup:
   * tell every corr node to delete its staged voltage for this event,
   * delete `<cand>/cubes/*.npz` (bulky, regenerable from voltages we won't keep),
   * **move** the event dir's metadata + plots to
     `candidates_rejected/<event>/` (Level2 CSVs, Level2/plots, Level3 JSON),
   * write `candidates_rejected/<event>/REJECT.json` (the veto rule(s) that
     fired + every metric, for audit).

C3 never deletes a directory outright and never deletes on an ambiguous veto.

---

## 3. The veto (`coinc/cube_veto.py`)

Productionises the injection-safe scheme from the analysis report. Tier-1
rules (high-confidence) — an event is **REJECT** only if it trips a tier-1 rule
**and is not an injection**:

* **R1 image offset** — global cube apex (l,m) far from the C2/C1 trigger (l,m).
* **R2 time shift** — global cube apex time far from trigger time.
* **R3 DM shift** — global cube apex fine-DM far from trigger DM.
* **R4 no time peak** — peak_grid time-series prominence below threshold
  (robust z-score of the apex row vs the cube's own time baseline).
* **R5 DM-edge railing** — apex pinned to the first/last fine-DM bin.
* **R10 cube-doesn't-confirm** — global apex lands in a *different* cube than
  the trigger's search node AND the trigger feature is weak (`tz_trig < 10`).

Everything else → **KEEP**. Thresholds are module constants with the values
validated in the report (0 injection / 0 B1933 loss; 74/89 true FPs removed).
The veto is a pure function `decide(metrics, is_injection) -> VetoDecision`
with `keep: bool`, `rules_fired: list[str]`, `metrics: dict` so C3 just records
the result.

---

## 4. Config + wiring

* `configs/config_corr.yaml`: `fada.num_readers: 2 → 3`.
* `configs/dsart_pipeline_rt.yaml`: `fada r: 2 → 3` + new `voltage_retention`
  routine (NUMA-1 pinned, near `fada`).
* `configs/numa_topology.yaml`: pin `voltage_retention` reader/worker to NUMA 1.
* `configs/dsart_search_rt.yaml`: `coinc.voltage_broadcast` (corr host map +
  port 11229) and a top-level `c3:` block (archive root, corr host map,
  collect timeouts, veto thresholds, rejected root).
* systemd unit for C3 on h23 (sibling of the coincidencer unit).

---

## 5. Safety properties

1. **Capture can never stall on the dump path.** The 3rd `fada` reader only
   memcpys + `markCleared`s; all slow work (disk, network) is downstream of the
   isolated RAM ring.
2. **No voltage dumps for injections** — gated at C2 fire time on the live
   inject registry; double-checked at C3 via the durable L3 marker.
3. **Conservative deletion** — REJECT requires a tier-1 cube veto; KEEP is the
   default for anything ambiguous; deletes are cubes+voltages only; metadata
   and plots are *moved*, never destroyed, with a full audit JSON.
4. **Operator kill-switch** — `/cmd/c2/voltages_enabled` defaults CLOSED; the
   feature is dark until explicitly enabled.

### 5.1 Runtime operator controls (dashboard Control tab)

Two etcd-backed toggles drive the pipeline at runtime; both write an audited
flip row under `/mon/audit/control/…` (who/why/when) and need a typed
confirm word as a speed-bump. Implemented in
`tools/dashboard/dsa_monitor/voltage_controls.py` + routes in `app.py`.

* **Voltage dumps enabled** → `/cmd/c2/voltages_enabled`. The C2
  voltage-broadcast kill-switch (separate from the cube `dumps_enabled`
  switch). **Fail-CLOSED**: a missing key means DISABLED, so a cold etcd or a
  fresh deploy never fills corr NVMe. C2 picks the new value up within its
  ~200 ms gate-cache TTL.
* **C3 reject mode (keep/delete)** → `/cmd/c3/flag_only`. C3 re-reads this on
  every event: `flag_only=true` (the safe default) = collect + log only, NO
  destructive cleanup; `flag_only=false` = enable the conservative REJECT
  delete path. A missing/malformed key (or etcd error) falls back to C3's
  configured `c3.flag_only` (also `true`). This makes the flag-first → soak →
  auto-delete transition a deliberate, audited, fail-safe flip rather than a
  config redeploy + restart.

C3 ships installed-and-enabled (flag-only) via `tools/c2/install.sh`
(`dsart_c3.service`); the corr-side `voltage_retention` reader is fada's 3rd
reader (`fada num_readers: 3` — keep `configs/config_corr.yaml` and the
authoritative `configs/dsart_pipeline_rt.yaml` `buffers:` block in lockstep).

---

## 6. Test plan

Pure/unit (no hardware):
* `wire`: DUMP_VOLTAGE bit round-trips; cube path unaffected.
* `voltage_ring`: store/lookup, wraparound, rolled-off detection, window extract.
* `voltage_broadcast`: fan-out to N corr hosts, per-host success map.
* `cube_veto`: each rule fires on a crafted metric set; injections always KEEP;
   a clean burst always KEEP.
* `c3`: KEEP vs REJECT decision drives the right filesystem actions (tmp dirs).
* `voltage_collect`: wait-for-N + timeout-min-fragments logic.

Integration (bench, opt-in): retention reader against a synthetic `fada`
(reuse `bench/replay_voltage_dump.py`), trigger → staged `.out` byte-for-byte
equals the replayed blocks.
