# M4a chunk 2 — DoD STEP draft

This is the chunk-2 STEP block the M4a driver should fold into
`tools/dod/M4a.sh`. The driver owns `M4a.sh`; this file is a draft
suggestion, NOT a direct edit.

```bash
# ------------------------------------------------------------------
# chunk_2_tx_prod_header — M4a chunk-2 TX path
# ------------------------------------------------------------------
STEP "chunk_2_tx_prod_header" \
  "M4a chunk 2: TransportTx prod-frame path (72-byte header, fragmentation, token-bucket pacer)" \
  bash -c '
    set -euo pipefail
    cd "$DSART_REPO"
    # Tests must all pass on h01 (CPU-only path; no GPU required).
    python -m pytest tests/transport/test_tx_prod.py -q --tb=short
    # Chunk-1 + chunk-8 regression gate.
    python -m pytest tests/transport/test_prod_frame.py \
                     tests/test_transport_loopback.py -q --tb=short
  '
```

## Expected pass counts on h01

* `tests/transport/test_tx_prod.py`: 35/35
* `tests/transport/test_prod_frame.py`: 42/42
* `tests/test_transport_loopback.py`: 16/16

## Chunk-2 deliverables

1. `src/dsart/transport/tx.py`:
   * `TransportTxProdConfig` (frozen+slots dataclass).
   * `_TokenBucket` per-flow drop-oldest pacer.
   * `TransportTx.__init__(use_prod_frame, prod_config)`.
   * `TransportTx.prepare_prod(pattern_id_by_chgroup, n_grid)`.
   * `TransportTx._transmit_one_cube_prod(...)`.
   * `TransportTx.tx_dropped_payloads` / `cube_seq_emitted` mon-key
     counters (in-process int64).
2. `tests/transport/test_tx_prod.py`: 35 tests (groups a/b/c/d/e/f
   plus 10 extra edge cases).
3. `M4a_PLAN_FIXES.md`: F1, F2, D1-D5 logged; chunk-2 status in ledger.

## Mon-keys M4a chunk-2 emits

| Path | Source attribute | Cadence |
|------|------------------|---------|
| `/mon/corr/<n>/transport/tx_dropped_payloads` | `TransportTx.tx_dropped_payloads` | 2 s |
| `/mon/corr/<n>/transport/cube_seq_emitted` | `TransportTx.cube_seq_emitted` | 2 s |

Both are `int64` counters; `tx_dropped_payloads` is monotone
non-decreasing; `cube_seq_emitted` is the highest seq number emitted
so far across all dm flows.

## Hook points for chunk-7 bench

The chunk-7 net-loopback bench wires the TX up like this:

```python
from dsart.transport.tx import TransportTx, TransportTxProdConfig
from dsart.transport.prod_frame import BITS_CINT8_COMPLEX

cfg = TransportTxProdConfig(
    target_gbps_per_flow=plan_op_point_gbps,  # from §9 op-table
    bits_per_cell=BITS_CINT8_COMPLEX,
    t_int_factor=op_point_t_int,
)
tx = TransportTx(
    host=transport_target_addr,   # 127.0.0.1 for loopback
    port=9000 + chgroup,
    chgroup=chgroup,
    use_prod_frame=True,
    prod_config=cfg,
)
tx.prepare_prod(
    pattern_id_by_chgroup={cg: predict_pattern_id(...) for cg in range(16)},
    n_grid=n_grid,
)
# Per-block:
tx.transmit(
    cubes_for_tx,
    block_n=block_n,
    rfi_warming_up=rfi_warming_up_now,
    specnum=snap_block_start_specnum,  # NEW: chunk-2 requires this
)
# At each 2 s tick:
mon_emit("/mon/corr/<n>/transport/tx_dropped_payloads", tx.tx_dropped_payloads)
mon_emit("/mon/corr/<n>/transport/cube_seq_emitted", tx.cube_seq_emitted)
```

## Open hand-off to RX agent (chunks 3+4+5)

The chunk-2 TX emits frames the chunk-3 RX must parse. Wire-format
compliance is verified by chunk-1's `test_prod_frame.py` round-trip
tests; chunk-2 additionally pins:

* `pattern_id` in every header == `prepare_prod()` cache value.
* `scale` / `offset` computed over **filled cells only** (cells with
  value 0 in the sparse-COO vector are *part of the data*, while
  cells absent from the pattern are *not*; only the latter are
  ignored).
* `seq` is per `dm_idx`, monotone increasing within a single
  `TransportTx` lifetime; resets to 0 on `prepare_prod()`.
* `FLAG_QUANTIZED` is set iff `bits_per_cell == 16` (D1).
* `FLAG_LAST_IN_BLOCK` is set on the final fragment of each
  (dm_idx, t_idx) payload (D3 / plan §4.3 line 1414 bit1).
