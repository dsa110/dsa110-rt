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

from typing import Tuple

import numpy as np

__all__ = ["quantise_streams_global_cint8"]


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
