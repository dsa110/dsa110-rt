"""Polarisation-sum re-export shim (M3 chunk 3a; plan §4.2 line 1295).

The Stokes-I pol-sum (``V_I = V_xx + V_yy``) collapses the 2-pol axis of
:meth:`dsart.services.corr_fast_kernel.FastCorrKernel.compute_split`'s
output into a single-pol Stokes-I tensor. It is the "last point in the
pipeline that sees a pol axis" (plan §3 line 301 + §4.2 line 1295).

The actual implementation lives one module up in
:mod:`dsart.services.corr_fast_kernel` because chunk 2a needed it
co-located with the kernel for unit-testability. Plan §4.2 places
``pol_sum.py`` logically inside the ``grid/`` subpackage, however —
both because the pol-sum runs in the same fused cupy kernel as the
gridder in production (§4.2 line 1294) and because downstream
``grid/`` importers reach for it next to the gridder kernel.

This module re-exports :func:`stokes_i_pol_sum` so downstream code can
write::

    from dsart.grid.pol_sum import stokes_i_pol_sum

without having to know the chunk-2a → chunk-3a layering, and without
having to add a ``services`` import for what is logically a ``grid/``
operation.

References
==========

* Plan §3 line 301 — Stokes-I pol-sum convention.
* Plan §4.2 line 1295 — pol-sum placement in the per-micro-batch
  pipeline.
* :mod:`dsart.services.corr_fast_kernel` — canonical implementation.
"""

from __future__ import annotations

from dsart.services.corr_fast_kernel import stokes_i_pol_sum

__all__ = ["stokes_i_pol_sum"]
