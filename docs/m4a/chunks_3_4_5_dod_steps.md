# M4a chunks 3 + 4 + 5 — suggested DoD `STEP` blocks

Source for `tools/dod/M4a.sh` integration. The M4a driver should append
the following STEPs (the RX agent does NOT edit `tools/dod/M4a.sh` per the
coordination contracts; this doc is the hand-off).

---

## STEP chunk_3_rx_defrag

```bash
STEP chunk_3_rx_defrag "RX prod-frame path: reorder window + pattern_id verify + dequant" '
  cd "$REPO_ROOT"
  python -m pytest tests/transport/test_rx_prod.py -q --tb=short -x
  python -m pytest tests/transport/test_prod_frame.py -q --tb=short
'
```

Pass criteria: chunk-3 27 tests pass; chunk-1 42 tests still pass; no
chunk-8 regressions (`test_transport_loopback.py` still 16/16 pass).

## STEP chunk_4_recv_ring_shm

```bash
STEP chunk_4_recv_ring_shm "POSIX-shm SPMC sparse ring (CONC-1 contract)" '
  cd "$REPO_ROOT"
  # Build C extension. On h01 with dsa110-rt env activated:
  pip install -e . 2>&1 | tail -20
  test -f src/dsart/transport/_recv_ring*.so || {
    echo "ERROR: _recv_ring.so not built. Falling back to inplace build."
    python setup.py build_ext --inplace
  }
  python -m pytest tests/transport/test_recv_ring_spmc.py -q --tb=short -x
'
```

Pass criteria: `_recv_ring.so` builds with no warnings; all 14 chunk-4 tests
pass (cross-NUMA test may be skipped on single-NUMA dev hosts).

## STEP chunk_5_production_source

```bash
STEP chunk_5_production_source "ProductionRxRingSource impl of RxRingSource Protocol" '
  cd "$REPO_ROOT"
  python -m pytest tests/transport/test_production_rx_ring.py -q --tb=short -x
'
```

Pass criteria: 10+ chunk-5 tests pass (CUDA host-register smoke skips if
cupy unavailable).

## Full chunks-3+4+5 invariant

```bash
STEP chunks_3_4_5_full "M4a RX-side full suite" '
  cd "$REPO_ROOT"
  python -m pytest tests/transport/ -q --tb=short
  # Expect chunk-1 (42), chunk-3 (27), chunk-4 (14), chunk-5 (10+) all pass.
  # chunk-8 path (test_transport_loopback.py) must still pass: 16/16.
'
```

---

## Pre-flight checks (M4a_preflight.sh additions)

- gcc available (`which gcc` returns 0; needed for the C extension build).
- POSIX shm mounted at `/dev/shm` (`mountpoint -q /dev/shm`).
- POSIX shm free space ≥ 4 GiB (`df /dev/shm | awk 'NR==2 {print $4}'`) —
  default-ops ring is ~2.3 GiB at N_grid=256.
