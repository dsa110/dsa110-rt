# C1 → C2 TCP wire schema and C2 → C1 UDP trigger packet

Status: locked 2026-05-21.  Implementations on both sides MUST agree
on this file; changes require a `schema_version` bump and a sub-agent
review.

## 0. Versioning

`schema_version = 1` (first lock).  The C1 emitter prints
`schema_version` in every header line so a future C2 can multi-version
parse.  The C2 receiver currently accepts only `schema_version == 1`
and logs+drops anything else (mon-point `c2_bad_schema`).


## 1. C1 → C2 TCP transport

- Direction: search node → h23.  Persistent, one TCP socket per
  `(search_node_id, gpu_half)` (so 8 sockets total).
- Server bind: `0.0.0.0:11500` on h23 (configurable, see
  `coinc.bind`).
- Client connect: from each search node's search-net interface
  (10.41.0.x) to `h23:11500`.
- Reconnect policy: client retries with exponential backoff
  (250 ms → 30 s cap).
- Heartbeat: at most every 10 s, the client sends an empty cube
  batch (`# HBeat ...`) — purely so the server can detect dead
  clients via select-timeout.  Per-cube batches are themselves
  sufficient liveness if cubes are flowing.

### 1.1 Payload framing

ASCII, line-delimited, one "cube batch" per atomic write:

```
# C1 <schema_version> <cube_id> <event_specnum_start> <mjd_start> <sample_period_specnum> <sample_period_us> <n_grid> <n_fdm_in_cube> <search_node_id> <gpu_half> <n_candidates>
<candidate row 1>
<candidate row 2>
...
<candidate row N>
# END
```

Lines:

- `#`-prefixed header line:
  - `schema_version` — integer, currently `1`.
  - `cube_id`, `event_specnum_start`, `sample_period_specnum`,
    `n_grid`, `n_fdm_in_cube`, `search_node_id`, `gpu_half` —
    integers.  `sample_period_specnum` is the number of raw SNAP
    spec-num units per detector sample (= 16 at default ops);
    same value on every batch in a fleet but carried explicitly so
    C2 needs no out-of-band knobs.
  - `mjd_start` — fixed-point `%.11f` (so the round-trip preserves
    µs-precision; mirrors the T1 ASCII logger's `_FMT_MJD`).
  - `sample_period_us` — fixed-point `%.6f` µs per detector
    sample (= 1048.576 µs at default ops; carried explicitly so
    C2 can convert ``event_specnum`` → MJD without an out-of-band
    config).
  - `n_candidates` — integer; informational (C2 reads exactly
    `n_candidates` rows then the END line).

  Candidate MJD recovery on the C2 side:

  ```
  samples_since_cube_start = (row.event_specnum - header.event_specnum_start) // header.sample_period_specnum
  candidate_mjd            = header.mjd_start + samples_since_cube_start * header.sample_period_us / 1e6 / 86400.0
  ```
- Candidate row schema (one space between fields, no trailing
  space, terminated `\n`):

  ```
  <snr> <l_rad> <m_rad> <l_pix> <m_pix> <dm_pc_cc> <dm_idx_global> <fine_dm_idx> <event_specnum> <width_samples> <kernel_id> <flags>
  ```

  Field types and formats:

  | field            | type   | format             | notes                                           |
  | ---------------- | ------ | ------------------ | ----------------------------------------------- |
  | snr              | float  | `%.6e`             | SNR in σ                                        |
  | l_rad            | float  | `%.9e`             | radians, computed via CubeGeometry              |
  | m_rad            | float  | `%.9e`             | radians                                         |
  | l_pix            | int    | `%d`               | pixel index ∈ [0, n_grid)                       |
  | m_pix            | int    | `%d`               | pixel index ∈ [0, n_grid)                       |
  | dm_pc_cc         | float  | `%.6f`             | fine-DM in pc cm⁻³                              |
  | dm_idx_global    | int    | `%d`               | absolute index into the full plan fine_dm grid  |
  | fine_dm_idx      | int    | `%d`               | per-cube local fine-DM index ∈ [0, n_fdm)       |
  | event_specnum    | int    | `%d`               | absolute spec num of the candidate's t          |
  | width_samples    | int    | `%d`               | matched-filter width in detector samples        |
  | kernel_id        | string | `k_img:k_dm:k_time`| no spaces, see Candidate._check_kernel_id       |
  | flags            | int    | `%d`               | CandidateFlags bit mask                         |

- `# END` line terminates the batch.

Empty batches (n_candidates = 0) carry no candidate rows; header +
END only.  These also serve as heartbeats.

### 1.2 Atomicity

The full batch (header + rows + END) is written via a single
`socket.sendall()` to avoid intra-batch interleave.  C2's parser
uses a buffered line iterator and only commits the batch when END is
seen.  A torn batch (TCP close mid-batch) is dropped with
`c2_torn_batch` incremented.

### 1.3 dm_idx_global vs fine_dm_idx

`fine_dm_idx` is the per-cube local index 0..n_fdm-1 (matches
`CubeGeometry.fine_dm_pc_cc[fine_dm_idx]`).  `dm_idx_global` is the
absolute index into the full DM plan
(`DmPlan.fine_dm[dm_idx_global]` corresponds globally).  This lets
C2 compare DM identity across (s, g) halves trivially (they each own
disjoint slices of the global grid, so two candidates with the same
`dm_idx_global` denote the exact same DM trial).


## 2. C2 → C1 UDP trigger packet

- Direction: h23 → search nodes (each `(search_node, gpu_half)`).
- Server bind on C1 side: `<search-net IPv4>:(11227 + gpu_half)`.
- Send from C2: connectionless, source from h23's preferred IPv4.
- Packet is fixed-size 64 bytes, little-endian.

```c
struct C2TriggerPacket {
    uint32_t magic;            //  0   4    'DSRT' = 0x44535254 little-endian = 0x54525344
    uint16_t version;          //  4   2    1
    uint16_t trigger_class_id; //  6   2    operator-defined; opaque to C1
    char     event_name[16];   //  8  16    NUL-padded ASCII (e.g. "260521abcd\0...")
    int64_t  event_specnum;    // 24   8    cube anchor specnum
    double   mjd_target;       // 32   8    event MJD (informational; sanity check)
    uint32_t flags;            // 40   4    bit 0 = dump_cube
    uint32_t reserved;         // 44   4    zero
    uint8_t  pad[16];          // 48  16    zero (to 64 bytes total)
};
```

`magic` is the literal ASCII `'D','S','R','T'` (= `0x44535254`
big-endian, `0x54525344` little-endian).  The C1 listener checks the
magic on receive and silently drops mismatches.

### 2.1 Cube lookup semantics on the C1 side

On receive, the listener computes:

```
sample_period_samples = cube_cadence_samples * sample_period_specnum
candidate_cube = find(ring,
    lambda r: r.event_specnum_start <= event_specnum < r.event_specnum_start + sample_period_samples)
```

- Hit: submit a `CubeDumpWriter` job; path
  `${dump_root}/<event_name>/cube_s<sid>_g<g>_<event_specnum>.npz`.
  Increment `c1_trigger_dumped`.
- Miss (older than the ring): increment `c1_trigger_too_late`; log
  WARNING with the ring's oldest specnum.
- Miss (newer than the ring — should not happen): increment
  `c1_trigger_too_early`; log WARNING.

### 2.2 Cube file naming

The trigger packet's `event_name` is part of the path; the writer
preserves the existing `cube_s<sid>_g<g>_<event_specnum>.npz`
template (so a per-event subdir contains 8 NPZs, one per (s, g),
all with the same `event_specnum`).

A side `cube_uploader` per search node rsyncs new files under
`${dump_root}/<event_name>/` to
`h23:/dataz/dsa110/candidates/<event_name>/cubes/`.  The uploader is
a separate systemd user service (`dsart_cube_uploader.service`) to
keep the search-compute hot path free.


## 3. Configuration knobs (single source of truth)

```yaml
c1:
  snr_min: 9.0                       # detector NMS threshold AND emit floor
  cube_ring_depth: 8                 # CPU-side cube tensors retained per half
  merger:
    lm_max_cells: 3
    dm_max_trials: 2
    t_frac: 1.0
  c2_endpoint:
    host: h23
    port: 11500
  emit_queue_depth: 16
  dump_listener:
    bind_host: ""                     # filled per-host via hostargs in dsart_search_rt.yaml
    base_port: 11227                  # actual port = base_port + gpu_half
  dump_root: /home/ubuntu/data/c2/cube_dump
  uploader:
    remote_root: ubuntu@lxd110h23:/dataz/dsa110/candidates
    bandwidth_limit_kbps: 0           # 0 = unlimited

coinc:
  bind:
    host: "0.0.0.0"
    port: 11500
  window_s: 5.0                       # rolling buffer time depth
  csv_retention_hours: 48
  csv_dir_c1: /dataz/dsa110/operations/C1/cluster_output
  csv_dir_c2: /dataz/dsa110/operations/C2/cluster_output
  event_archive_root: /dataz/dsa110/candidates
  trigger_criteria_path: /home/ubuntu/vikram/dev/dsa110-rt/configs/c2_trigger_criteria.yaml
  dump_broadcast:
    port_base: 11227                  # actual port = port_base + gpu_half
    hosts:
      # (search_node_id, ipv4) — derived from dsart_search_rt.yaml::hostargs
      "1": "10.41.0.205"
      "2": "10.41.0.222"
      "9": "10.41.0.253"
      "13": "10.41.0.238"
  plotter:
    n_workers: 2
    per_event_timeout_s: 30.0
```
