# M4a plan fixes + decisions tracker

This file mirrors the M2/M3/M5/M6 pattern (see git log for those equivalents):
F-items are *fixes* (corrections to plan.md or implementation choices that
deviate from the plan as written and need to be folded back), D-items are
*decisions* locked during implementation. Both accumulate here during M4a
development and are folded into `dsa110-rt_revamp_7b1d2669.plan.md` during
M4a hardening (the final M4a chunk), at which point this file is deleted.

For binding M3 / M5 / M6 context that M4a inherits verbatim — including
the `SparseCOOPayload` in-process contract (M1 §3, post-F1/F2/F3), the
input-hashed `pattern_id` semantics (plan §3 line 307, plan §4.3 Option C),
the `SparsityPattern.build_pattern` deterministic-rebuild contract
(M3 chunk-3a, F28/F33 folded), the F26 sparse-COO vs image-cube transport
auto-detect (M3 chunk-8), and the captured-NPZ schema (M3 chunk-7 / F6)
— see plan.md `§8.M3-carryover` (~lines 2320-2376) and the M3 chunk-8
TX/RX modules in `src/dsart/transport/`.

For parallel-agent file-ownership / branch-model / h01-test-isolation
conventions, see `PARALLEL_AGENTS.md`. M4a is **transport-only**, GPU
contention with M3/M5 is negligible (per plan §8 line 408 — M4a is
"transport-only, low GPU usage; could share GPU 0 with M3 or GPU 1 with
M5 with negligible contention"). The natural M4a split is TX-side ∥
RX-side; PARALLEL_AGENTS.md §8 documents the pattern.

---

## Scope reminder (plan §M4a, lines 2377-2384)

M4a replaces the chunk-8 simplified 32-byte FastVisFrame with the production
§4.3 wire format + receive-ring, on-host loopback only. Phase B (M4b) flips
`transport.target_addr` from `127.0.0.1` to the real NIC IP of the second
machine, no other code changes. M4a does not extend etcd (M7 owns
`services/control/etcd_watcher.py`); the chunk-7 bench uses an env-var
prepare-reload shim (`DSART_M4A_RELOAD_PATTERN=1`) to simulate `cmd: prepare`.

Six DoD invariants (plan §M4a line 2383):
1. 60 s sustained 0% loss at the largest §9 op-point over loopback.
2. Mid-run RX socket hold → TX drops at TX (`tx_dropped_payloads` ↑), no
   upstream backpressure into the gridder.
3. `pattern_mismatch_count == 0` for 60 s when both ends started with the
   same `/cnf/corr_setup_96` + `dec_deg`.
4. Mutate corr-side `dec_deg` to a stale value → `pattern_mismatch_count`
   increments steadily (every datagram).
5. Synchronised re-`cmd: prepare` on both ends → `pattern_mismatch_count`
   returns to 0 within < 1 cube.
6. Mid-run, restart `dsart-search-rx@01` only → rebuilds patterns
   locally on next `cmd: prepare`, resumes < 5 s with
   `pattern_mismatch_count` returning to 0.

---

## F-items (fixes — corrections / additions to plan.md)

### F1 (chunk 2): `specnum` plumbing into `TransportTx.transmit`

Plan §4.3 header line 1422 specifies `specnum` (uint64, SNAP block-start
counter, cross-corr time alignment). The chunk-8 `TransportTx.transmit`
Protocol signature is

```python
def transmit(self, cubes_for_tx, *, block_n: int, rfi_warming_up: bool) -> int
```

— there is no `specnum` argument; `block_n` is the chunk-4 corr-side
block counter (NOT the F-engine packet seq). The chunk-2 prod-frame
path needs `specnum` from upstream block metadata.

Chunk-2 mitigation: `TransportTx.transmit` is extended with an
optional `specnum: int | None = None` argument. When
`use_prod_frame=True` and `specnum is None`, `transmit` raises
`NotImplementedError` with a clear message about wiring upstream.

`services/corr_fast_integration.py` should add `specnum` to its
contract, and the chunk-4 orchestrator should pass the SNAP block-start
seq derived from the upstream voltage-block metadata (chunk-4 already
has access to specnum-equivalent fields via the fada reader; see
`F-engine packet block` plumbing in `slow_corr_kernel.py`).

### F2 (chunk 2): image-cube ndim=4 deferred in prod path

The chunk-8 path supports both sparse-COO (ndim=3) and image-cube
(ndim=4) cubes per F26 (M3 chunk-8). The chunk-2 prod-frame path
deliberately implements only the sparse-COO case (the §4.3 wire format
is COO + `pattern_id` round-trip; image-cube delivery would require a
peer wire format).

`_transmit_one_cube_prod` raises `NotImplementedError` on `ndim==4`
with a clear pointer to the chunk-7 net-loopback bench. If the
chunk-7 bench needs image-cube delivery, the M4a driver should add a
dedicated wire format (likely outside the prod-frame envelope, since
N_grid² fp16 = 65 536 × 4 B / 8964 ≈ 30 fragments per (dm, t) pair
puts heavy pressure on the seq counter).

### F3 (chunk 4): C-extension build system not pinned in plan

Plan §4.4 line 1465 states "Owner module: `src/dsart/transport/recv_ring.py`
(Python ctypes wrapper) + `src/dsart/transport/recv_ring.c` (struct
definitions; ...)" but does not pin the C extension build mode. D6 locks
it to setuptools + `setup.py` (with `Extension('dsart.transport._recv_ring',
...)`). pyproject.toml `[tool.setuptools]` cannot declare `ext_modules` in
setuptools<64; the setup.py exists alongside pyproject.toml only for that
purpose. Fold into plan §4.4 line 1465 in M4a hardening.

### F4 (chunk 5): cudaHostRegister wiring deferred to chunk 7

Plan §4.4 line 1467 says "cudaHostRegister(addr, size,
cudaHostRegisterMapped | cudaHostRegisterPortable) so both compute
processes' GPU contexts can cudaHostGetDevicePointer for zero-copy DMA."
`ProductionRxRingSource._try_register_ring` is currently a no-op because
the chunk-4 C API does not expose the mmap base address
(`rx_ring_get_base_ptr` is missing). D-item D7 records the deferral to
the chunk-7 GPU-integration bench. Fold into plan §4.4 line 1467 with a
pointer to D7 when chunk 7 lands.

---

## D-items (decisions — implementation-level choices locked during M4a)

### D1 (chunk 2): `FLAG_QUANTIZED` is set only for cint8, not cfp16

Plan §4.3 line 1414 reads `bit0=quantized` (vs cfp16 — implicit in
§4.3 wire-format bullet line 1408 "operational cint8 cell"). The
chunk-1 spec (`prod_frame.py` `FLAG_QUANTIZED` constant docstring)
calls bit0 "payload is quantized cint8 (vs cfp16)" — distinct from
`bits_per_cell`, which is the encoding selector.

Chunk-2 locks: `FLAG_QUANTIZED` is set iff `bits_per_cell ==
BITS_CINT8_COMPLEX` (16). For cfp16 (32), the wire payload is the
"no quantisation" fp16 path and the bit is clear. `scale` and
`offset` are fixed at `1.0` and `0.0` respectively for cfp16
(identity dequant), so the receiver's `x = scale*q + offset` rule
applies uniformly across both paths.

Tests: `test_c2_cfp16_scale_is_identity_flag_quantized_clear` and
`test_x8_cint8_flag_quantized_set_cfp16_not_set` pin this behaviour.

### D2 (chunk 2): per-flow pacer is keyed by `dm_idx` only

Plan §4.3 line 1447 specifies "per-flow token-bucket rate-limiter".
The §4.3 line 1402 "per (corr, search) pair" UDP flow vocabulary
suggests one flow per corr-side TX (since each `TransportTx` binds
to one destination IP:port already). Within a single `TransportTx`,
the "per-flow" cardinality is therefore per-`dm_idx` (matching the
`seq_by_flow: dict[(corr_idx, dm_idx), int]` plan §4.3 line 1421
contract for sequence numbers).

Chunk-2 locks: `TransportTx._bucket_by_flow: dict[int, _TokenBucket]`
keyed by `dm_idx`, lazily created when a `dm_idx` is first seen.
The `(corr_idx, ...)` half of the (corr, dm_idx) key is implicit
because each TransportTx is bound to one corr's chgroup, and the
mon-key path `/mon/corr/<corr_idx>/transport/tx_dropped_payloads`
already discriminates by corr.

### D3 (chunk 2): seq advances even when all fragments dropped

Plan §4.3 line 1421: "seq is monotonically increasing per (corr,
dm_idx) flow". When the pacer drops every fragment of a payload
(extreme back-pressure), the seq counter still advances by one
(the payload is "emitted" from the corr's POV — the drop is
accounted via `tx_dropped_payloads`, not as a "didn't emit a seq").

This matches the chunk-1 receiver-side contract: missing fragments
are detected via the seq reorder window slide-out (§4.3 line 1473).
If the TX skipped seq values, the receiver could not distinguish a
"silent TX" from a "lost-on-wire" event.

Chunk-2 locks: in `_transmit_one_cube_prod`, the `_next_seq(dm_idx)`
call is made before the per-fragment send loop, and `seq` is reused
across all fragments of the same payload. `tx_dropped_payloads`
increments at most once per (dm_idx, t_idx) payload regardless of
how many fragments were dropped.

### D4 (chunk 2): token-bucket "drop-oldest" semantics

Plan §4.3 line 1447: "if the bucket throttles, the sender drops the
oldest queued payload". The chunk-2 `_TokenBucket` implements this
literally: when the FIFO is full and a new item arrives that cannot
be sent immediately, the FIFO's oldest entry is popped and
`drop_count` is incremented. The new item replaces it. The fixed
bucket capacity is `max_frag_payload_bytes * 4` (4-fragment burst
absorption), and the FIFO depth is configurable
(`TransportTxProdConfig.bucket_fifo_depth`, default 4).

### D5 (chunk 2): mon-key emission is in-process counters (M0 pattern)

The plan §3.7 mon-key registry expects `/mon/corr/<n>/transport/*`
keys writable by the M4a driver. Per `M4a_PLAN_FIXES.md` "Mon-keys
M4a is responsible for emitting", M4a uses `tools.mon_key_emitter`
for benches and real `/mon/` writes are wired in M7.

`tools/mon_key_emitter.py` does not yet exist in the repo (the M3
chunks emit their mon-keys via direct attribute access on the
service object, drained by chunk-7-and-later wiring). Chunk-2
follows the M3 pattern: expose `tx_dropped_payloads` and
`cube_seq_emitted` as `int64` attributes on the `TransportTx`
instance; an external drainer (chunk-7 bench, M7 etcd writer) reads
them at the 2 s mon-key cadence. The attribute names map 1:1 onto
the `/mon/corr/<n>/transport/*` key suffixes.

### D6 (chunk 4): huge-page fallback — plain 4 KiB pages for v1

Plan §4.4 line 1467 prescribes `mmap(MAP_SHARED | MAP_LOCKED |
MAP_HUGETLB | (1 GiB shift if available))`. On h01 the kernel may not
have huge pages reserved (`/proc/sys/vm/nr_hugepages == 0`); requesting
`MAP_HUGETLB` from a non-privileged process is also rejected on most
distributions. The chunk-4 C implementation uses plain `mmap(MAP_SHARED,
...)` for v1 and documents the 4 KiB-page fallback here. If chunk-7
perf gate exposes cross-NUMA bandwidth shortfall, revisit with hugetlbfs
mount + setuid helper. The 2-NUMA p99-latency budget (≤ 134 ms per cube)
is met by 4 KiB pages at default ops per the chunk-4 cross-NUMA latency
probe test.

### D7 (chunk 5): cudaHostRegister deferred to chunk 7

See F4 above. Chunk 5's `ProductionRxRingSource._try_register_ring` is a
documented no-op; the actual cudaHostRegister call lives in chunk-7 once
the C API exposes a base-ptr accessor. Tests for chunk 5 pass
`enable_cuda_register=False`.

### D8 (chunk 4): build system = setuptools + setup.py, not cmake

setuptools `Extension('dsart.transport._recv_ring', ...)` is the build
mode for the C extension. `pyproject.toml [tool.setuptools]` cannot
declare `ext_modules` in setuptools<64; an explicit `setup.py` lives
alongside `pyproject.toml`. Rejected alternatives: cmake (adds a
build-system dependency for a ~400-line C file; pip would need
scikit-build-core), meson (same; unfamiliar build orchestration for this
repo), bare gcc Makefile (defers the build to deploy time; defeats
`pip install -e .` ergonomics). Setuptools is already the build backend.

### D9 (chunk 3): `corr_idx = chgroup` convention

Plan §4.3 / §4.4 references per-`(corr, dm_idx)` flows but does not pin
whether `corr_idx == chgroup` or `corr_idx == sender_node_id`. M4a fixes
`corr_idx = chgroup` because in the §2.2 / §6.1 split each corr node owns
exactly one chgroup. The per-(corr, dm_idx) reorder window keys on
`(header.chgroup, header.dm_idx)`. Fold into plan §4.3 in M4a hardening.

### D10 (chunk 3): reorder-window slide policy

Plan §4.3 line 1473 says "Out-of-order seq arrivals within the window
land in their correct slot; arrivals beyond the window fall through
`pattern_mismatch=false, data_present=false`". The chunk-3 implementation
interprets this as:

- `seq < head` → silently dropped (`out_of_order_drop_count++`), no
  zero-fill slot emitted to the ring.
- `seq > tail` → slide window, zero-fill any displaced incomplete slots
  (`window_slide_zerofill_count++`, `seq_gap_count_per_flow[(corr,dm)]++`).
- `seq ∈ [head, tail]` and bitmap not yet full → buffer; commit only on
  full reassembly.

The reasoning: late arrivals are silent noise (a retransmission past the
window's lifetime), while window-slide drops are budgeted data loss that
downstream cube-validity must see.

### D11 (chunk 3): dequant at COO-store time

Plan §4.4 line 1462 pins dequantisation at COO-store time. Chunk 3 emits
`np.complex64` arrays on the `RxProdSlot` (not raw int8 / fp16). The
chunk-4 ring stores raw bytes; the dequantisation happens in
`dequantise_payload` inside the reorder-window commit callback. Per the
plan this means the **ring carries float values, not raw quantized ints**.

An alternative considered: leave bytes in the ring and dequantise on the
GPU compute side. Rejected because (a) plan §4.4 line 1462 explicitly
pins dequant to COO-store, (b) the GPU sparse-scatter kernel runs against
complex64 cells (G5 +uv-only single-pol), and (c) doing dequant on the
RX side keeps the per-payload `scale`/`offset` close to the per-payload
header rather than threading them through to the GPU.

### D12 (chunk 4): shm name format

`/dsart_rx_ring_<s>_<utc_ns>` is the plan-suggested format (plan §4.4
line 1467 cites `/dsart_rx_ring_<s>`). M4a chunk 4 leaves the naming to
the caller — the C API takes `name` verbatim. The chunk-7 bench will fix
the format with the utc_ns suffix.

### D13 (chunk 4): build artifact path (PEP 3149)

The C extension `.so` is named `_recv_ring.cpython-<ver>-<arch>.so`
(PEP 3149). `recv_ring.py` globs for `_recv_ring*.so` so both
`pip install -e .` (installed) and `python setup.py build_ext --inplace`
(inplace) paths work.

---

## Chunk ledger

M4a chunks (deps in parens):

| # | Chunk | Status | Owner |
|---|---|---|---|
| 1 | `prod_frame_72b` — `src/dsart/transport/prod_frame.py` + tests | merged (42/42 on h01) | M4a driver (synchronous) |
| 2 | `tx_prod_header` — `transport/tx.py` extend to 72-byte header + fragmentation + token-bucket pacer | merged (35/35 chunk-2; 42/42 chunk-1 regression; 16/16 chunk-8 regression on h01) | TX agent |
| 3 | `rx_defrag` — `transport/rx.py` per-(corr, dm_idx) reorder window + bitmap + pattern_id verify | merged (27 tests; h01 verify pending post-merge integration run) | RX agent |
| 4 | `recv_ring_shm` — POSIX-shm SPMC sparse ring (`transport/recv_ring.{c,py}`); CONC-1 contract | merged (14 tests; C-ext build verify pending post-merge integration run on h01) | RX agent |
| 5 | `production_source` — `transport/production_rx_ring.py` satisfying `services/rx_ring.RxRingSource` Protocol | merged (12 tests; h01 verify pending post-merge integration run) | RX agent (deps 3 + 4) |
| 6 | `c_epoll_loop` — **conditional**, only if chunk-7 perf gate fails Python `recvmmsg` at target rate | not started | RX agent (on-call) |
| 7 | `bench_net_loopback` — `bench/net_loopback.py` with the 6 DoD invariants + env-var reload shim | not started | M4a driver (post-merge) |
| 8 | `dod_orchestrator` — `tools/dod/M4a.sh` + `M4a_preflight.sh` + status JSON; retire `M4a_PLAN_FIXES.md` | not started | M4a driver |

---

## Chunk-1 wire-format freeze (the synchronization point)

Chunk 1 is the joint file the TX and RX agents both depend on. Its delivery
unblocks the parallel TX/RX work. Key contracts (all from plan §4.3 line
1409-1444):

- **Header**: 72 bytes total, little-endian, packed. Field layout pinned in
  the plan; chunk-1 must produce byte-identical bytes for the same input
  tuple regardless of caller.
- **Magic**: `0xD5A1107E` ("DSA110 7E"). Distinct from chunk-8's `0xD5A0FA57`.
- **Version**: `1`.
- **Reserved fields**: sender writes `0`; receiver verifies for forward-compat
  (a future v2 may carve out reserved bytes).
- **`pattern_id`**: 8-byte BLAKE2b input-hash from
  `grid/sparsity_pattern._pattern_id_payload` — chunk 1 does NOT re-implement
  the hash, it consumes the existing M3 helper.
- **Validity-mask bits in `flags`**: bit0=quantized, bit1=last_in_block,
  bit2=reserved (DEDISP no-emit; v1 senders MUST NOT set), bit3=noise_warmup
  (search-side; RX ignores), bit4=rfi_warming_up (RFI Stat-B burn-in; not
  no-emit).
- **Sequencing semantics**:
  - `seq` — uint64, per-(corr, dm_idx) flow ordering for fragment reassembly
    + drop accounting. The plan's "per (corr, dm_idx) flow" terminology is
    canonical; do not collapse to a single per-corr counter. (See M4a kickoff
    discussion 2026-05-11 for the rationale: per-flow `seq` makes
    "which fragment is missing" answerable from the gap alone.)
  - `specnum` — uint64, F-engine block-start counter, identical across all
    16 corrs for the same time block. The search-side combiner uses
    `specnum` for cross-corr time alignment.

Chunk 1 also publishes `predict_pattern_id` re-export sugar so the TX side
can fill the header without importing the gridder module directly.

---

## Open hooks for the chunk-7 bench

The chunk-7 bench needs an env-var-driven simulator for `cmd: prepare` since
real etcd is deferred to M7. Plan:

- `DSART_M4A_RELOAD_PATTERN=1` env flag triggers an in-process reload of the
  cached `SparsityPattern` on both TX and RX sides on next loop iteration.
- The bench fires this in invariants (3) → (5) by mutating
  `os.environ['DSART_M4A_DEC_DEG_OVERRIDE']` and then setting
  `DSART_M4A_RELOAD_PATTERN=1`.
- The chunk-7 bench owns the env-var protocol; chunks 2 and 3 implement the
  reload-on-env-flag check (cheap: one `os.environ.get(...)` per block).
- M7's real `etcd_watcher.py` replaces the env-var path with an etcd watch;
  the production reload path inside `transport/{tx,rx}.py` is unchanged.

---

## Mon-keys M4a is responsible for emitting

Per plan §3.7 mon-key registry:

| Key | Owner | Cadence | Type | Notes |
|---|---|---|---|---|
| `/mon/corr/<n>/transport/tx_dropped_payloads` | TX (chunk 2) | 2 s | int64 counter | token-bucket drops |
| `/mon/corr/<n>/transport/cube_seq_emitted` | TX (chunk 2) | 2 s | int64 | last emitted UDP value-channel `seq` |
| `/mon/corr/<n>/transport/clip_count` | M3 (already) | 2 s | int64 counter | M4a does NOT re-emit |
| `/mon/search/<s>/rx/pattern_mismatch_count` | RX (chunk 3) | 2 s | int64 counter | per-(chgroup, dm_idx) keyed; rolled up |
| `/mon/search/<s>/rx/seq_gap_count_per_flow` | RX (chunk 3) | 2 s | int64 histogram | per-(corr, dm_idx) reorder-window slide-out |
| `/mon/search/<s>/rx/window_slide_zerofill_count` | RX (chunk 3) | 2 s | int64 counter | reorder-window slide → zero-fill events |

M4a uses the existing M0 `tools.mon_key_emitter` (in-process for benches);
real `/mon/` writes are wired in M7.
