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

- **F1 (chunk 4 build system)**: plan §4.4 line 1465 states "Owner module:
  `src/dsart/transport/recv_ring.py` (Python ctypes wrapper) +
  `src/dsart/transport/recv_ring.c` (struct definitions; ...)" but does
  not pin the C extension build mode. D-item D3 locks it to setuptools
  + `setup.py` (with `Extension('dsart.transport._recv_ring', ...)`).
  pyproject.toml [tool.setuptools] cannot declare ext_modules in
  setuptools<64; the setup.py exists alongside pyproject.toml only for
  that purpose. Fold into plan §4.4 line 1465 in M4a hardening.

- **F2 (chunk 5 CUDA host-register)**: plan §4.4 line 1467 says
  "cudaHostRegister(addr, size, cudaHostRegisterMapped | cudaHostRegisterPortable)
  so both compute processes' GPU contexts can cudaHostGetDevicePointer for
  zero-copy DMA." `ProductionRxRingSource._try_register_ring` is currently
  a no-op because the chunk-4 C API does not expose the mmap base address
  (`rx_ring_get_base_ptr` is missing). D-item D2 records the deferral to
  the chunk-7 GPU-integration bench. Fold into plan §4.4 line 1467 with a
  pointer to D2 when chunk 7 lands.

---

## D-items (decisions — implementation-level choices locked during M4a)

- **D1 (chunk 4 huge-page fallback)**: plan §4.4 line 1467 prescribes
  `mmap(MAP_SHARED | MAP_LOCKED | MAP_HUGETLB | (1 GiB shift if available))`.
  On h01 the kernel may not have huge pages reserved
  (`/proc/sys/vm/nr_hugepages == 0`); requesting `MAP_HUGETLB` from a
  non-privileged process is also rejected on most distributions. The
  chunk-4 C implementation uses plain `mmap(MAP_SHARED, ...)` for v1 and
  documents the 4 KiB-page fallback here. If chunk-7 perf gate exposes
  cross-NUMA bandwidth shortfall, revisit with hugetlbfs mount + setuid
  helper. The 2 NUMA p99-latency budget (≤ 134 ms per cube) is met by 4 KiB
  pages at default ops per the chunk-4 cross-NUMA latency probe test.

- **D2 (chunk 5 cudaHostRegister deferral)**: see F2 above. Chunk 5's
  `ProductionRxRingSource._try_register_ring` is a documented no-op; the
  actual cudaHostRegister call lives in chunk-7 once the C API exposes a
  base-ptr accessor. Tests for chunk 5 pass `enable_cuda_register=False`.

- **D3 (chunk 4 build system: setuptools, not cmake)**: setuptools
  `Extension('dsart.transport._recv_ring', ...)` is the build mode for the
  C extension. pyproject.toml [tool.setuptools] cannot declare ext_modules
  in setuptools<64; an explicit `setup.py` lives alongside pyproject.toml.
  Rejected alternatives: cmake (adds a build-system dependency for a
  ~400-line C file; pip would need scikit-build-core), meson (same;
  unfamiliar build orchestration for this repo), bare `gcc` Makefile
  (defers the build to deploy time; defeats `pip install -e .`
  ergonomics). Setuptools is already the build backend.

- **D4 (chunk 3 corr_idx convention)**: plan §4.3 / §4.4 references
  per-`(corr, dm_idx)` flows but does not pin whether `corr_idx == chgroup`
  or `corr_idx == sender_node_id`. M4a fixes `corr_idx = chgroup` because
  in the §2.2 / §6.1 split each corr node owns exactly one chgroup. The
  per-(corr, dm_idx) reorder window keys on `(header.chgroup, header.dm_idx)`.
  Fold into plan §4.3 in M4a hardening.

- **D5 (chunk 3 reorder-window slide policy)**: plan §4.3 line 1473 says
  "Out-of-order seq arrivals within the window land in their correct slot;
  arrivals beyond the window fall through `pattern_mismatch=false,
  data_present=false`". The chunk-3 implementation interprets this as:
  - `seq < head` → silently dropped (`out_of_order_drop_count++`),
    no zero-fill slot emitted to the ring.
  - `seq > tail` → slide window, zero-fill any displaced incomplete slots
    (`window_slide_zerofill_count++`, `seq_gap_count_per_flow[(corr,dm)]++`).
  - `seq ∈ [head, tail]` and bitmap not yet full → buffer; commit only on
    full reassembly.
  The reasoning: late arrivals are silent noise (a retransmission past the
  window's lifetime), while window-slide drops are budgeted data loss
  that downstream cube-validity must see.

- **D6 (chunk 3 dequant at COO-store)**: plan §4.4 line 1462 pins
  dequantisation at COO-store time. Chunk 3 emits `np.complex64` arrays
  on the `RxProdSlot` (not raw int8 / fp16). The chunk-4 ring stores raw
  bytes; the dequantisation happens in `dequantise_payload` inside the
  reorder-window commit callback. Per the plan this means the **ring carries
  float values, not raw quantized ints**.

  An alternative considered: leave bytes in the ring and dequantise on the
  GPU compute side. Rejected because (a) plan §4.4 line 1462 explicitly
  pins dequant to COO-store, (b) the GPU sparse-scatter kernel runs against
  complex64 cells (G5 +uv-only single-pol), and (c) doing dequant on the
  RX side keeps the per-payload `scale`/`offset` close to the per-payload
  header rather than threading them through to the GPU.

- **D7 (chunk 4 shm name format)**: `/dsart_rx_ring_<s>_<utc_ns>` is the
  plan-suggested format (plan §4.4 line 1467 cites `/dsart_rx_ring_<s>`).
  M4a chunk 4 leaves the naming to the caller — the C API takes `name`
  verbatim. The chunk-7 bench will fix the format with the utc_ns suffix.

- **D8 (chunk 4 build artifact path)**: the C extension `.so` is named
  `_recv_ring.cpython-<ver>-<arch>.so` (PEP 3149). `recv_ring.py` globs
  for `_recv_ring*.so` so both `pip install -e .` (installed) and
  `python setup.py build_ext --inplace` (inplace) paths work.

---

## Chunk ledger

M4a chunks (deps in parens):

| # | Chunk | Status | Owner |
|---|---|---|---|
| 1 | `prod_frame_72b` — `src/dsart/transport/prod_frame.py` + tests | scaffolding | M4a driver (synchronous) |
| 2 | `tx_prod_header` — `transport/tx.py` extend to 72-byte header + fragmentation + token-bucket pacer | not started | TX agent |
| 3 | `rx_defrag` — `transport/rx.py` per-(corr, dm_idx) reorder window + bitmap + pattern_id verify | landed (m4a/rx-defrag-and-ring) | RX agent |
| 4 | `recv_ring_shm` — POSIX-shm SPMC sparse ring (`transport/recv_ring.{c,py}`); CONC-1 contract | landed (m4a/rx-defrag-and-ring) | RX agent |
| 5 | `production_source` — `transport/production_rx_ring.py` satisfying `services/rx_ring.RxRingSource` Protocol | landed (m4a/rx-defrag-and-ring) | RX agent |
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
