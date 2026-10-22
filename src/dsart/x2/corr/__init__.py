"""x2 corr side (WS-C): the 2x time-resolution corr_fast chain and the
1x-compat restructure of today's block loop.

Modules (import them directly; nothing is re-exported so importing the
package never pulls in torch / triton / cupy):

* :mod:`.pipeline1x`  -- 1x-compat orchestrator (async H2D, 2-block ring,
  deferred TX); bit-identical to ``process_block`` at today's parameters.
* :mod:`.plan2x`      -- corr view of WS-P's tiered 2x DM plan (stage-1 tables).
  Value maps are WS-P's ``dsart.grid.baseline_grid``.
* :mod:`.gemm2x`      -- Triton fast GEMM (core cross-baselines, cal + mask
  on load, cfp16 time-minor ring output, unique-baseline reduce).
* :mod:`.stage1_cuda` -- CSR-driven stage-1 (all DMs of a tier, x2/x4
  decimation in registers) as a cupy RawModule, with a numpy twin.
* :mod:`.static_sky2x` -- batched static-sky subtraction.
* :mod:`.packer2x`    -- row-rule packer (SPEC §4) and the v1 cint8 mode.
* :mod:`.pipeline2x`  -- 2x block orchestrator behind the op-point switch.
"""
