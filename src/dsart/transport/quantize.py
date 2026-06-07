"""src/dsart/transport/quantize.py — host-side cf64 -> cint8 quantiser
(plan section 4.4 lines 1471-1493 + section 4.2 cint8 wire payload).

The M3 -> M5 receive ring delivers cint8 streams in the on-the-wire
``[N_chgroup, T_stream, 2, N_grid, N_grid] int8`` (split-plane re/im)
layout (see ``image/imager_gpu.py::GpuImager.process_cube`` +
``image/fused_combine_cuda.py::fused_dequant_combine_per_fdm`` for the
canonical kernel-side reader). The bench paths that consume the M3
captured-NPZ fixtures (which carry cf64 visibilities, NOT cint8) need
to round-trip through cint8 before feeding the production hot path so
the imager exercises exactly the same dequant + accumulate kernel that
production runs.

This module is the single source of truth for that quantisation step.
Both ``bench/captured_burst_recovery.py`` and the production
``services/cube_pipeline.py`` GPU backend (``image_backend="gpu"``) call
``quantise_streams_global_cint8``. The function uses a SINGLE GLOBAL
max-abs scale across all chgroups + re/im planes; per-chgroup scaling
would distort the cross-chgroup magnitude balance the imager's coherent
sum step depends on.

Production note: the live M4a receive-ring delivers cint8 streams that
are ALREADY quantised (with per-chgroup scale + offset baked in by the
M3 stage-2 quantiser). This host-side helper is bench-only / pre-cint8
fallback.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np

__all__ = [
    "quantise_streams_global_cint8",
    "quantise_per_chgroup_into_cint8",
]


def quantise_streams_global_cint8(
    streams_cf: np.ndarray,
    *,
    target_max: int = 120,
) -> Tuple[np.ndarray, float]:
    """Quantise ``[N_chg, T_stream, N_grid, N_grid] complex`` ->
    ``[N_chg, T_stream, 2, N_grid, N_grid] int8`` using a SINGLE
    global max-abs scale (re/im are split into the inner-2 axis).

    A single scale preserves the cross-chgroup relative magnitude that
    the imager's coherent-sum step depends on. Per-chgroup scaling
    would distort the relative weighting and corrupt the dirty image.

    Args:
        streams_cf: dense complex visibility stream stack, ndim=4
            ``[N_chg, T_stream, N_grid, N_grid]``. Accepts any
            complex dtype; values are cast to fp64 internally for the
            scale-and-round step.
        target_max: post-scale absolute clip target (<= 127 for int8;
            120 leaves a small headroom against rounding overflow).

    Returns:
        ``(cint8, scale)``:
        - cint8: ``[N_chg, T_stream, 2, N_grid, N_grid] int8`` array
          in M3 wire layout (re plane at axis_2=0, im plane at
          axis_2=1).
        - scale: the global multiplicative scale applied; the
          inverse-quantisation rule is
          ``cf_re = cint8[..., 0, :, :] / scale``,
          ``cf_im = cint8[..., 1, :, :] / scale``.

    Raises:
        ValueError if ``streams_cf`` is not 4-D or not complex.
    """
    if streams_cf.ndim != 4:
        raise ValueError(
            f"streams_cf must be 4-D [N_chg, T_stream, N_grid, N_grid]; "
            f"got shape {streams_cf.shape}"
        )
    if not np.iscomplexobj(streams_cf):
        raise ValueError(
            f"streams_cf must be complex; got dtype {streams_cf.dtype}"
        )
    if not (1 <= int(target_max) <= 127):
        raise ValueError(
            f"target_max must be in [1, 127]; got {target_max!r}"
        )

    re = np.ascontiguousarray(streams_cf.real, dtype=np.float64)
    im = np.ascontiguousarray(streams_cf.imag, dtype=np.float64)
    global_max = float(max(np.abs(re).max(initial=0.0),
                           np.abs(im).max(initial=0.0)))
    if global_max <= 0.0:
        scale = 1.0
    else:
        scale = float(target_max) / global_max

    n_chg, t_stream, n_grid, _ = streams_cf.shape
    cint8 = np.empty(
        (n_chg, t_stream, 2, n_grid, n_grid), dtype=np.int8,
    )
    np.clip(np.rint(re * scale), -127, 127, out=re)
    np.clip(np.rint(im * scale), -127, 127, out=im)
    cint8[:, :, 0] = re.astype(np.int8)
    cint8[:, :, 1] = im.astype(np.int8)
    return cint8, scale


def quantise_per_chgroup_into_cint8(
    per_chgroup_streams: Mapping[int, np.ndarray],
    *,
    out_cint8: np.ndarray,
    target_max: int = 120,
    zero_fill_missing: bool = True,
    fixed_scale: Optional[float] = None,
) -> float:
    """Streaming per-chgroup cf32 -> cint8 quantiser writing into a
    caller-supplied output buffer.

    Equivalent (up to fp32 vs fp64 rounding) to
    ``quantise_streams_global_cint8`` but avoids three large transient
    allocations the dense path makes per cube: (a) the
    ``[N_chg, T_stream, N_grid, N_grid] cf32`` stack the caller would
    otherwise have to materialise, (b) the ``cf64`` real/imag copies
    the dense path uses for the rint/clip step, and (c) the implicit
    ``re * scale`` and ``np.rint`` temp buffers (each ~5 GiB at
    production T_det=256/N_fdm=32/N_grid=256/N_chg=16). Working set
    here is one chgroup at a time (~80 MiB) regardless of N_chg.

    The function makes two passes over the streams:

      1. Global max-abs scan (fp32 reductions per chgroup; no large
         temp allocs).
      2. Scale + rint + clip + cast to int8 directly into the
         caller-supplied ``out_cint8`` buffer, one chgroup at a time.

    A single global scale across all chgroups + re/im planes is used,
    matching ``quantise_streams_global_cint8`` (per-chgroup scales
    would distort the cross-chgroup magnitude balance the imager's
    coherent-sum step depends on).

    Args:
        per_chgroup_streams: ``{chgroup_idx -> [T_stream, N_grid,
            N_grid] complex}``. Any complex dtype accepted; the inner
            ``re * scale`` math runs in the array's native float
            precision (typically fp32 for cf32 inputs).
        out_cint8: pre-allocated ``[N_chg, T_stream, 2, N_grid,
            N_grid] int8`` output buffer. Caller owns the buffer; can
            be re-used across cubes.
        target_max: post-scale absolute clip target (<= 127 for int8).
        zero_fill_missing: if True, chgroups absent from
            ``per_chgroup_streams`` have their slice in ``out_cint8``
            zeroed (the imager kernel reads these as zero
            contributions). If False, missing chgroups are left
            untouched (caller responsibility to pre-zero).

    Returns:
        The single global scale applied (``target_max / global_max``).
    """
    if out_cint8.ndim != 5 or out_cint8.dtype != np.int8:
        raise ValueError(
            f"out_cint8 must be 5-D int8; got shape={out_cint8.shape} "
            f"dtype={out_cint8.dtype}"
        )
    n_chg, t_stream, two, n_grid_h, n_grid_w = out_cint8.shape
    if two != 2:
        raise ValueError(
            f"out_cint8 axis-2 must have size 2 (re/im); got {two}"
        )
    if not (1 <= int(target_max) <= 127):
        raise ValueError(
            f"target_max must be in [1, 127]; got {target_max!r}"
        )

    # Per-chgroup shape/dtype validation always runs (so callers get
    # the same error semantics regardless of fixed_scale). The g_max
    # scan is skipped when fixed_scale is provided.
    g_max = 0.0
    for g, stream in per_chgroup_streams.items():
        if not (0 <= int(g) < n_chg):
            raise ValueError(
                f"per_chgroup_streams contains chgroup={g}; expected "
                f"0..{n_chg - 1}"
            )
        s = np.asarray(stream)
        if not np.iscomplexobj(s):
            raise ValueError(
                f"per_chgroup_streams[{g}].dtype={s.dtype}; expected complex"
            )
        if s.shape != (t_stream, n_grid_h, n_grid_w):
            raise ValueError(
                f"per_chgroup_streams[{g}].shape={s.shape}; expected "
                f"({t_stream}, {n_grid_h}, {n_grid_w})"
            )
        if fixed_scale is None:
            re_max = float(np.abs(s.real).max(initial=0.0))
            im_max = float(np.abs(s.imag).max(initial=0.0))
            if re_max > g_max:
                g_max = re_max
            if im_max > g_max:
                g_max = im_max

    if fixed_scale is not None:
        # M7.7.2: caller supplies a constant scale so cint8 across
        # consecutive cubes is mutually consistent (required for the
        # carry-over numerical-equivalence gate, where overlapping
        # absolute time must produce identical cint8 in both cubes).
        # Out-of-range cells are clipped to ±127.
        scale = float(fixed_scale)
    else:
        scale = (float(target_max) / g_max) if g_max > 0.0 else 1.0
    scale_dtype: np.dtype
    sample_stream = next(iter(per_chgroup_streams.values()), None)
    if sample_stream is not None and np.asarray(sample_stream).dtype == np.complex64:
        scale_dtype = np.float32
    else:
        scale_dtype = np.float64
    scale_typed = scale_dtype.type(scale) if hasattr(scale_dtype, "type") else np.array(
        scale, dtype=scale_dtype
    )

    if zero_fill_missing:
        present = set(int(g) for g in per_chgroup_streams.keys())
        for g in range(n_chg):
            if g not in present:
                out_cint8[g].fill(0)

    for g, stream in per_chgroup_streams.items():
        s = np.asarray(stream)
        re = s.real
        im = s.imag
        re_q = np.rint(re * scale_typed)
        np.clip(re_q, -127, 127, out=re_q)
        out_cint8[int(g), :, 0] = re_q.astype(np.int8)
        im_q = np.rint(im * scale_typed)
        np.clip(im_q, -127, 127, out=im_q)
        out_cint8[int(g), :, 1] = im_q.astype(np.int8)

    return scale
