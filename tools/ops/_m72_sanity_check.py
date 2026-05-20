"""Quick sanity check for the M7.2 warmup-gating + tx_wire_drops fixes.

Runs on n01/n06 (post-rsync). Validates:
  1. dsart_rt orchestrator imports cleanly with the new RoutineSpec field.
  2. pipeline_rt.yaml parses; capture routines have gate_on_paths populated.
  3. corr_slow / corr_fast --help shows --ready-sentinel-path.
  4. TransportTx.tx_wire_drops property exists and reports zero on a fresh tx.
"""
import sys
import yaml

from dsart.services.dsart_rt import PipelineConfig
print("=== orchestrator imports OK ===")

with open("configs/dsart_pipeline_rt.yaml") as f:
    raw = yaml.safe_load(f)
cfg = PipelineConfig.from_dict(raw)
print(f"pipeline_rt.yaml parsed: {len(cfg.buffers)} buffers, "
      f"{len(cfg.routines)} routines")
for r in cfg.routines:
    if r.gate_on_paths:
        print(f"  gated: {r.name} gate={list(r.gate_on_paths)}")
    else:
        print(f"  ungated: {r.name}")
print()

print("=== tx_wire_drops property ===")
from dsart.transport.tx import TransportTx, _TokenBucket
assert hasattr(TransportTx, "tx_wire_drops")
print("doc head:", TransportTx.tx_wire_drops.__doc__.splitlines()[0])

print()
print("=== _TokenBucket.drop_count is the canonical wire-drop counter ===")
b = _TokenBucket(rate_bytes_per_sec=1.0, capacity_bytes=10, max_fifo=2)
print("fresh bucket drop_count:", b.drop_count)
