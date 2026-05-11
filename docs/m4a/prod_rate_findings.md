# M4a production-rate findings on h01 (UDP loopback, 2026-05-11)

This doc records the production-rate bench results that motivate **M4a
chunk 6** (C epoll receive loop). The findings answer the question
"does the M4a transport work at production speed on h01 over loopback?"
The short answer is: the **protocol is bit-exact correct at every rate
we measured, but the Python RX path tops out at ~1 Gb/s — 7× short of
the 7 Gb/s per-search-node ingress requirement**.

## Production topology (plan §11 line 2654)

* 16 corr nodes, 4 search nodes; each corr × search pair is a unicast
  UDP flow on a single 40 GbE link.
* Each corr emits its chgroup's 24 coarse DMs split 4 ways across the 4
  search destinations (~6 DMs per (corr, search) pair).
* Per-flow rate: **0.438 Gb/s** = 6 × 0.073 Gb/s/DM.
* Per-corr egress: **1.75 Gb/s** = 4 × 0.438.
* Per-search ingress: **7.00 Gb/s** = 16 × 0.438.

## Op-point used by the bench (matches §9 default)

| Knob | Value |
|---|---|
| `N_grid` | 256 |
| `bits_per_cell` | 16 (cint8 complex) |
| `t_int_factor` | 16 |
| `n_filled` | 5000 |
| `n_fast_vis` | 1 (per cube tile) |
| `n_dm_per_tx` | 6 |
| `target_gbps_per_dm` | 0.073 |
| `max_frag_payload_bytes` | 8964 (jumbo MTU 9000 − 28 IP/UDP − 8 header) |
| `rcvbuf_mib` | 256 (plan §4.3) |
| Bytes per cube per TX | 60 000 (6 DMs × 1 fv × 5000 cells × 2 B) |
| Fragments per (dm, t) payload | 2 (10 000 B payload at 8964-B MTU) |

## Bench runs (60 s each, on h01)

| Test | Topology | observed Gb/s | target Gb/s | % of target | loss % | committed / expected | mismatches | TX dropped |
|---|---|---|---|---|---|---|---|---|
| **P1 per-flow** | 1 TX (6 DMs) → 1 RX, single thread | 0.192 | 0.438 | 44% | 0.00 | 143 892 / 143 892 | 0 | 0 |
| **P2-mp fan-in** | 16 TX subprocs → 1 RX | 5.841 | 7.008 | 83% | 82.05 | 786 218 / 4 380 570 | 0 | 0 |
| **P3-mp fan-out** | 4 TX subprocs → 4 RX (1 proc) | 1.638 | 1.752 | 93% | 73.71 | 322 907 / 1 228 452 | 0 | 0 |

Sanity counters (all three runs): `bad_magic_count = 0`,
`bad_field_range_count = 0`, `bad_length_count = 0`,
`out_of_order_drop_count = 0`, `pattern_mismatches = 0`. The protocol
is *correct* — the failures are pure rate-limit failures.

`window_slide_zerofill_count` shows the RX correctly accounted for the
kernel-dropped seq numbers as zero-fills (3.58 M for P2-mp;
560 K for P3-mp), which is the designed behaviour under loss.

## What each test tells us

### P1 (1 TX, 1 RX, single thread): Python TX encode ceiling

* TX achieved 0.192 Gb/s = 44% of the 0.438 Gb/s per-(corr, search)
  target.
* Loss: **zero**. The TX is the bottleneck — it cannot emit faster than
  it does, and the RX consumes everything it gets.
* Per-cube transmit cost: ~2.67 ms (12 fragment encodes + 12 sendto
  syscalls per cube; `n_dm=6`, `n_fv=1`, 2 frags per (dm, t)).
* **Implication**: a single Python TX process cannot meet the per-flow
  production rate.

### P2-mp (16 TX subprocs, 1 RX): kernel + Python RX ceiling

* TX aggregate: 5.841 Gb/s = 83% of the 7.008 Gb/s target. Per-process
  rate dropped to 0.337 – 0.378 Gb/s (vs 0.41 in P3-mp) because 16
  subprocs share the same CPU pool.
* Kernel UDP loopback drops: **82% of datagrams were dropped by the
  kernel** before Python could `recvfrom` them.
* `nstat UdpRcvbufErrors` rose by ~7.2 M during the run.
* The Python recv loop *did* consume everything the kernel delivered
  (1.57 M datagrams over 60 s = **~26 k pkt/s, or ~1 Gb/s of payload
  through the Python recv path**).
* **Implication**: the Python recv path tops out at ~1 Gb/s for our
  fragment sizes. The 7 Gb/s production target requires roughly **7×**
  more throughput than a single Python recv loop can deliver.

### P3-mp (4 TX subprocs, 4 RX in 1 proc): GIL-bound RX threads

* TX aggregate: 1.638 Gb/s = 93% of the 1.752 Gb/s target. With only
  4 TX subprocs each got more CPU, so per-proc rate rose to
  0.384 – 0.419 Gb/s (mean 0.409 — close to the 0.438 target).
* Kernel-side loss: 73.7%. With 4 RX socket threads in one Python
  process, the GIL serialises them across CPUs, so per-RX drain is
  *lower* than the 1 Gb/s ceiling P2-mp showed.
* **Implication**: a single Python process can host one viable RX
  loop. Putting multiple RX loops in one process is GIL-bound and
  underperforms.

## Where the bottleneck really lives

The h01 loopback drops *look* like a kernel `SO_RCVBUF` overflow, and
they are — but the *root cause* is that the Python recv loop drains
slower than the wire. Per-datagram cost in the current Python recv
path:

```
recvfrom syscall           ~5  µs
header unpack (struct)     ~5  µs
pattern_id check           ~1  µs
fragment book-keeping      ~5  µs
dequant on commit          ~25 µs / fragment (amortised across the 2-frag payload)
                           ─────
total                      ~40 µs / datagram → ~25 k pkt/s → ~1 Gb/s
```

At 97 k pkt/s (the 7 Gb/s target), the budget per datagram is **~10 µs**.

A real 40 GbE NIC with hardware multi-queue + RPS + GRO would smooth
the bursty arrival pattern and let `SO_RCVBUF=256 MiB` absorb the
short-term jitter, but the **per-datagram steady-state Python cost is
the same** — the kernel-side drop on loopback is a symptom, not the
root cause. Production with a real NIC will hit the same wall.

## What it would take to hit the production rate

### TX side (per-process: 0.19 Gb/s single thread, 0.41 Gb/s with whole CPU)

* The plan already accepts that production corr nodes run one process
  with 4 TX threads, one per search destination. The GIL serialises
  Python threads, so a single corr process would aggregate
  4 × (1 / n_threads_sharing_GIL) ≈ 1 × per-thread rate — far below
  the 1.75 Gb/s per-corr target.
* **Required**: either (a) one corr process spawns 4 TX subprocesses
  (one per search dest), each on its own CPU, getting ~0.41 Gb/s and
  aggregating to ~1.6 Gb/s (still 7% short, and structurally ugly), or
  (b) move the TX encode + send hot path into C / Cython, so a single
  process with 4 threads can hit 0.44 Gb/s per thread.

### RX side (per-process: ~1 Gb/s with current Python recv path)

* **Required**: the production search-node process must run a **C
  epoll receive loop** (M4a "chunk 6" in plan vocabulary) that
  ingests datagrams, parses the 72-byte header, advances the per-(
  corr, dm) reorder window, and writes the *quantised* payload into
  the existing chunk-4 SPMC shm ring **without crossing into Python
  for every datagram**.
* Python should only see *committed slots*, not individual datagrams.
* Dequantisation moves from `ingest_datagram` to the **compute reader**
  (`ProductionRxRingSource.peek_next_block`) — i.e. it happens once
  per used slot, not 2× per fragment. This was already approved as
  D4 in `M4a_PLAN_FIXES.md`.

## Proposed chunk 6 design

Drop a new C library `src/dsart/transport/recv_epoll.{c,py}` that:

1. **Owns the listening UDP socket** (creates + binds + sets
   `SO_RCVBUF`). The Python side passes only the bind address + port.
2. **Runs an `epoll` loop in its own pthread** (started from Python).
   On every `EPOLLIN`:
   1. Batched `recvmmsg(64)` to drain the socket without one syscall
      per datagram.
   2. For each datagram: validate magic / version / length /
      field-range (same checks as `unpack_frame` today).
   3. Check `pattern_id` against the per-chgroup expected map (kept
      in a small atomic-replaceable struct so Python can `cmd: prepare`
      mid-run without locks).
   4. Advance the per-(chgroup, dm_idx) reorder window (depth W,
      committed-bit logic per chunk-3 `_ReorderWindow`).
   5. On commit / pattern-mismatch / zerofill: call
      `rx_ring_publish_slot()` (chunk-4 C API) with the *quantised*
      payload bytes — no dequant on the recv path.
3. **Exposes counters** (`bad_magic_count`, `pattern_mismatch_count`,
   `window_slide_zerofill_count`, …) as `_Atomic` u64 fields readable
   from Python via ctypes — same mon-key set as `TransportRxProd`.
4. **Replaces `_RxLoop` + `TransportRxProd.ingest_datagram`** for
   production. The chunk-3 `TransportRxProd` stays as the Python
   fallback / reference impl used by unit tests.

### Estimated performance (per published numbers + back-of-envelope)

* `recvmmsg` reduces syscall overhead by ~10×: from ~5 µs/datagram to
  ~0.5 µs/datagram batched.
* C header parse: ~50 ns vs ~5 µs in Python.
* C reorder-window update: ~100 ns vs ~5 µs in Python.
* No dequant on the recv path (moved to compute reader).
* Total: **~1 µs / datagram → ~1 M pkt/s → ~70 Gb/s** for our
  fragment sizes. With margin, this comfortably absorbs the 7 Gb/s
  target on real hardware.

### Acceptance criteria for chunk 6

Reuse `bench/net_loopback.py --prod fan-in-mp --json …` after chunk-6
lands. Pass when:

| Metric | Pass threshold | Failure means |
|---|---|---|
| `observed_gbps_aggregate` | ≥ 6.66 Gb/s (95% of 7.008) | TX side still short or kernel overflow |
| `loss_pct` | < 0.5% | RX path can't drain at wire rate |
| `pattern_mismatches` | 0 | Protocol regression |
| `window_slide_zerofill_count` | ≤ 0.5% of `expected_committed` | Confirms low loss, not low rate |
| `bad_magic / length / field` | 0 | Header validation regression |

Add a `P2c` test variant (post-chunk-6) that drives the C recv loop
specifically.

## What did *not* change after this measurement

* The six protocol invariants `I1..I6` in `bench/net_loopback.py` still
  pass at the toy rate they were designed for. They were never
  rate-stress tests; they verify the wire format, fragmentation,
  pattern_id, reorder window, and restart semantics, and they did
  catch the SIGSEGV + reorder-window bugs during the M4a chunks 3+4+5
  integration.
* `prod_frame.py`, `tx.py`, `rx.py`, `recv_ring.{c,py}`, and
  `production_rx_ring.py` need **no protocol changes** to meet the
  performance target — chunk 6 is a pure performance refactor of the
  recv hot path, behind the same Python `TransportRxProd` API.

## What to do next

1. Land this bench + this doc on `chore/net-loopback-prod-rates`,
   then merge to `main` (no behavioural change to production code).
2. Open chunk 6 as the next M4a work item; promote it from
   "deferred" to "required" in `M4a_PLAN_FIXES.md` with the bench
   numbers above as motivation.
3. After chunk 6 lands, rerun `--prod fan-in-mp` on h01 and (if it
   passes) on multi-host hardware to confirm the kernel-side drop
   was the only loopback artefact.
