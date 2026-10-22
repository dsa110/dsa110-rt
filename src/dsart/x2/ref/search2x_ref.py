"""Float64 numpy reference of the owner search chain (plan "Correctness 2").

Independent of every production / engine code path it checks:

* decode: SPEC §4 owner decode ``(code + offset) * c_v`` in float64
  (1x: offset 0, ``c_v = 1`` -- codes summed raw as today);
* gridding: ``np.add.at`` of every (stream, value) into its FULL-plane cell
  at the stream row ``t - shift[f, s]`` (no CSR, no Hermitian fold);
* imaging: the textbook ``fftshift(Re(ifft2(G)))`` (not the half-plane
  irfft2 + checkerboard form the GPU uses), times today's edge mask
  (``image.imager.compute_edge_mask``, which carries the (-1)^(i+j) DC
  correction) and a scalar image scale (1x: the uv prescale 1/256);
* Layer-1: 3 x 3-sigma median-centred clipped std over the active cells
  (optionally over a given subsample, to compare with production);
* filters: cascades of integer box stages -- boxcar (``b``), triangle
  (``t``), FWHM-matched B3 (``q``) and the Phase 0 ``b3x`` bank -- with taps
  by explicit convolution, centred ``a = L // 2`` (today's boxcar
  convention ``[c - w//2, c - w//2 + w - 1]``), evaluated as float64
  prefix-sum box stages on ``clip(x / sigma, +-clip)``;
* chunk maxima over 64 consecutive centres and every filter, packed
  ``(t, k)``; and today's decoder (top-k over cells, merge-radius
  suppression ``|dt|<=128, |df|<=5, |dl|,|dm|<=3``, AND semantics).

Plus synthetic-data helpers (point-source dispersed bursts in the stream
domain; a reference row-rule quantiser) used by tests and the I1 harness.
Geometries are small (N = 16..64): this is for exactness, not speed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "RefFilter",
    "ref_filter",
    "ref_filter_bank",
    "ref_decode",
    "ref_grid_dedispersed",
    "ref_image",
    "ref_edge_mask",
    "ref_sigma_clipped_std",
    "ref_layer1",
    "ref_filter_scores",
    "ref_chunk_max",
    "ref_decode_topk",
    "RefCycleResult",
    "ref_search_cycle",
    "point_source_values",
    "inject_dispersed_burst",
    "quantize_rowrule",
]


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefFilter:
    name: str
    width: int           # nominal width (tier samples)
    stages: tuple        # box stage lengths of the cascade (integer sums)
    taps: np.ndarray     # float64 [L]
    a: int               # output at centre c = sum_j taps[j] x[c - a + j]

    @property
    def length(self) -> int:
        return int(self.taps.shape[0])

    @property
    def norm(self) -> float:
        return float(np.sqrt(np.sum(self.taps ** 2)))


def ref_filter(name: str, width: int, stages) -> RefFilter:
    """A cascade of integer box stages; taps by explicit convolution."""
    taps = np.ones(1)
    for L in stages:
        taps = np.convolve(taps, np.ones(int(L)))
    return RefFilter(name, int(width), tuple(int(L) for L in stages), taps,
                     int(taps.shape[0]) // 2)


#: Phase 0 study 5 quadratic B-spline stage lengths (FWHM-matched).
REF_B3 = {2: (1, 2, 2), 4: (3, 3, 3), 8: (6, 6, 7), 16: (12, 13, 13), 32: (25, 25, 26),
          64: (50, 50, 51)}


def ref_filter_bank(widths: Sequence[int], family: str = "b") -> List[RefFilter]:
    """``b`` boxcar, ``t`` equal triangle (w, w), ``q`` FWHM-matched B3
    (Phase 0 table), ``b3x`` the 8-filter Phase 0 bank (widths ignored)."""
    if family == "b3x":
        return ([ref_filter("b1", 1, (1,)), ref_filter("b2", 2, (2,)),
                 ref_filter("t2", 2, (2, 2))]
                + [ref_filter(f"q{w}", w, REF_B3[w]) for w in (4, 8, 16, 32, 64)])
    out = []
    for w in widths:
        w = int(w)
        st = {"b": (w,), "t": (w, w),
              "q": REF_B3.get(w, (max(1, round(w / 1.268)),) * 3)}[family]
        out.append(ref_filter(f"{family}{w}", w, st))
    return out


# ---------------------------------------------------------------------------
# decode + gridding + imaging
# ---------------------------------------------------------------------------


def ref_decode(re: np.ndarray, im: np.ndarray, c_v: np.ndarray, offset: float) -> np.ndarray:
    """codes ``[..., V]`` -> complex128 ``(re + off) c_v + i (im + off) c_v``."""
    c = np.asarray(c_v, dtype=np.float64)
    return ((np.asarray(re, np.float64) + offset) * c
            + 1j * (np.asarray(im, np.float64) + offset) * c)


def ref_grid_dedispersed(
    values: Sequence[np.ndarray],
    cells: Sequence[np.ndarray],
    shifts: np.ndarray,
    *,
    row0: int,
    out_rows: Sequence[int],
    n_grid: int,
) -> np.ndarray:
    """Full-plane dedispersed uv grids ``G[f, t, N, N]`` (complex128).

    Args:
        values: per stream ``[T_s, V_s]`` complex decoded values; row ``i``
            is absolute stream row ``row0 + i``.
        cells: per stream ``[V_s]`` full-plane linear cell (``u*N + w``),
            ``-1`` = unused.
        shifts: ``[F, S]``; out row ``t`` of fine DM ``f`` reads stream ``s``
            at absolute row ``t - shifts[f, s]`` (zero outside the data).
        out_rows: absolute output rows.
    """
    F, S = shifts.shape
    out_rows = np.asarray(out_rows, dtype=np.int64)
    G = np.zeros((F, out_rows.size, n_grid * n_grid), dtype=np.complex128)
    for s in range(S):
        val = np.asarray(values[s])
        cel = np.asarray(cells[s], dtype=np.int64)
        use = np.nonzero(cel >= 0)[0]
        for f in range(F):
            src = out_rows - int(shifts[f, s]) - int(row0)
            ok = (src >= 0) & (src < val.shape[0])
            ti = np.nonzero(ok)[0]
            if ti.size == 0 or use.size == 0:
                continue
            blk = val[src[ti]][:, use]                              # [nt, nv]
            tt = np.repeat(ti, use.size)
            cc = np.tile(cel[use], ti.size)
            np.add.at(G[f], (tt, cc), blk.reshape(-1))
    return G.reshape(F, out_rows.size, n_grid, n_grid)


def ref_edge_mask(n_grid: int, kernel_support: int = 1) -> np.ndarray:
    """Today's multiplicative edge mask (float64, carries the sign flip)."""
    from ...image.imager import compute_edge_mask
    return compute_edge_mask(n_grid=n_grid, kernel_support=kernel_support,
                             dtype=np.float64)


def ref_image(G: np.ndarray, mask: np.ndarray, scale: float) -> np.ndarray:
    """``fftshift(Re(ifft2(G))) * mask * scale`` on the last two axes."""
    img = np.fft.fftshift(np.fft.ifft2(G, axes=(-2, -1)).real, axes=(-2, -1))
    return img * mask * float(scale)


# ---------------------------------------------------------------------------
# noise normalisation
# ---------------------------------------------------------------------------


def _lower_median(s: np.ndarray) -> float:
    """``torch.nanmedian`` semantics: the LOWER middle value for even counts."""
    return float(np.partition(s, (s.size - 1) // 2)[(s.size - 1) // 2])


def ref_sigma_clipped_std(x: np.ndarray, n_sigma: float = 3.0, n_iter: int = 3) -> float:
    """Median-centred iterative clip, exactly the production estimator's
    semantics (``noise_norm.layer1.sigma_clipped_std_batched``: lower
    median, rms about it over the kept set), float64."""
    s = np.asarray(x, dtype=np.float64).reshape(-1)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0
    med = _lower_median(s)
    sig = np.sqrt(np.mean((s - med) ** 2))
    act = np.ones(s.size, dtype=bool)
    for _ in range(int(n_iter)):
        act &= (s >= med - n_sigma * sig) & (s <= med + n_sigma * sig)
        if not act.any():
            return 0.0
        med = _lower_median(s[act])
        sig = np.sqrt(np.mean((s[act] - med) ** 2))
    return float(sig)


def ref_layer1(img: np.ndarray, active: np.ndarray, *, sample_idx: Optional[np.ndarray] = None,
               floor: float = 0.0) -> np.ndarray:
    """Per-fine-DM sigma of ``img [T, F, N, N]`` over active cells.

    ``sample_idx [F, M]`` (flat ``(t, pixel)`` indices into one fine DM's
    ``[T, N*N]`` slab) reproduces a subsampled production estimate.
    """
    T, F = img.shape[:2]
    P = img.shape[2] * img.shape[3]
    act = np.asarray(active, dtype=bool).reshape(-1)
    out = np.zeros(F)
    for f in range(F):
        slab = img[:, f].reshape(T, P)
        if sample_idx is not None:
            idx = np.asarray(sample_idx[f], dtype=np.int64)
            v = slab.reshape(-1)[idx]
            v = np.where(act[idx % P], v, np.nan)
        else:
            v = slab[:, act]
        out[f] = ref_sigma_clipped_std(v)
    return np.maximum(out, floor)


# ---------------------------------------------------------------------------
# detector
# ---------------------------------------------------------------------------


def _box_axis0(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing box sum along axis 0 (zeros before row 0): out[e] = sum x[e-w+1 .. e]."""
    P = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)], axis=0)
    lo = np.maximum(np.arange(x.shape[0]) + 1 - w, 0)
    return P[1:] - P[lo]


def ref_filter_scores(x_norm: np.ndarray, row0: int, filters: Sequence[RefFilter],
                      centres: Sequence[int]) -> np.ndarray:
    """Filter outputs ``[K, n_c, F, N, N]`` at absolute centres.

    ``x_norm [R, F, N, N]`` is the clipped, sigma-normalised input; row ``i``
    is absolute row ``row0 + i``; rows outside are zero. Each B-spline is
    evaluated as ``m`` cascaded float64 prefix-sum box filters over the
    whole (zero-extended) row axis, then read at the window end
    ``c - a + L - 1`` (independent of the engine's fp32 running sums).
    """
    centres = np.asarray(centres, dtype=np.int64)
    R = x_norm.shape[0]
    out = np.zeros((len(filters), centres.size) + x_norm.shape[1:], dtype=np.float64)
    pad = max(f.length for f in filters)
    xp = np.concatenate([np.asarray(x_norm, dtype=np.float64),
                         np.zeros((pad,) + x_norm.shape[1:])], axis=0)
    for k, flt in enumerate(filters):
        y = xp
        for L in flt.stages:
            y = _box_axis0(y, int(L))
        e = centres - flt.a + flt.length - 1 - row0
        ok = (e >= 0) & (e < R + pad)
        out[k, ok] = y[e[ok]]
    return out


def ref_chunk_max(snr: np.ndarray, chunk: int = 64) -> tuple:
    """``snr [K, n_c, ...]`` -> per (chunk, cell) max over (centre, filter).

    Returns ``(max [n_ch, ...], t_in_chunk [n_ch, ...], k [n_ch, ...])``;
    ties resolve to the earliest centre, then the lowest filter index.
    """
    K, n_c = snr.shape[:2]
    n_ch = n_c // chunk
    mx = np.empty((n_ch,) + snr.shape[2:])
    tin = np.empty((n_ch,) + snr.shape[2:], dtype=np.int64)
    kk = np.empty((n_ch,) + snr.shape[2:], dtype=np.int64)
    for ci in range(n_ch):
        v = np.moveaxis(snr[:, ci * chunk:(ci + 1) * chunk], 0, 1)     # [chunk, K, ...]
        flat = v.reshape((chunk * K,) + snr.shape[2:])
        arg = np.argmax(flat, axis=0)                                  # earliest (t, k)
        mx[ci] = np.take_along_axis(flat, arg[None], axis=0)[0]
        tin[ci] = arg // K
        kk[ci] = arg % K
    return mx, tin, kk


def ref_decode_topk(score: np.ndarray, *, threshold: float, n_top: int,
                    radius_t: int = 128, radius_f: int = 5, radius_lm: int = 3,
                    t_of=None) -> list:
    """Today's ``decode_topk_argmax_lowmem`` selection on a max cube.

    ``score [T, F, N, N]``. Returns ``[(t, f, l, m, value), ...]`` in
    descending value. ``t_of`` optionally maps the flat index's ``t`` to an
    absolute time (chunk-max decoding passes the packed centre).
    """
    T, F, H, W = score.shape
    flat = score.reshape(-1)
    k = min(max(1, int(n_top)), flat.size)
    idx = np.argsort(-flat, kind="stable")[:k]
    out, acc = [], []
    for ix in idx:
        v = float(flat[ix])
        if v <= threshold:
            break
        t, rem = divmod(int(ix), F * H * W)
        f, rem = divmod(rem, H * W)
        l_, m_ = divmod(rem, W)
        tt = t_of(t, f, l_, m_) if t_of is not None else t
        if any(abs(tt - a) <= radius_t and abs(f - b) <= radius_f
               and abs(l_ - c) <= radius_lm and abs(m_ - d) <= radius_lm
               for a, b, c, d in acc):
            continue
        acc.append((tt, f, l_, m_))
        out.append((tt, f, l_, m_, v))
    return out


# ---------------------------------------------------------------------------
# one cycle of one DM, end to end
# ---------------------------------------------------------------------------


@dataclass
class RefCycleResult:
    G: np.ndarray            # [F, T, N, N] full-plane dedispersed uv
    image: np.ndarray        # [T, F, N, N] scaled masked image (float64)
    sigma: np.ndarray        # [F] Layer-1 sigma used
    x_norm: np.ndarray       # [T, F, N, N] clip(image / sigma)
    scores: Optional[np.ndarray]   # [K, n_c, F, N, N] raw filter outputs (keep_scores)
    full_max: np.ndarray     # [n_c, F, N, N] max over filters of score / s_k
    full_k: np.ndarray       # winning filter (first on ties)
    chunk_max: tuple         # (max, t_in_chunk, k), ties -> earliest centre, lowest k
    centres: np.ndarray
    image_rows: np.ndarray
    candidates: list = field(default_factory=list)


def ref_search_cycle(
    *,
    values: Sequence[np.ndarray],
    cells: Sequence[np.ndarray],
    shifts: np.ndarray,
    row0: int,
    t0: int,
    hop: int,
    delay: int,
    history: int,
    n_grid: int,
    mask: np.ndarray,
    image_scale: float,
    filters: Sequence[RefFilter],
    s_k: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    clip: float = 250.0,
    chunk: int = 64,
    threshold: float = 9.55,
    n_top: int = 224,
    decode: str = "chunk",
    keep_scores: bool = False,
) -> RefCycleResult:
    """Reference for one owner cycle of one DM.

    Images rows ``[t0 - history, t0 + hop)``; detects centres
    ``[t0 - delay, t0 + hop - delay)``. ``sigma`` (Layer-1) defaults to the
    cycle's own new rows ``[t0, t0 + hop)`` (2x same-cycle semantics).
    ``decode="chunk"`` decodes the per-(pixel, chunk) maxima (2x);
    ``"full"`` decodes the full max-over-filters cube exactly as today's
    ``decode_topk_argmax_lowmem`` does (its top-k can saturate on one bright
    burst -- the S2 "chunk-max decoder" item).
    """
    rows = np.arange(t0 - history, t0 + hop)
    G = ref_grid_dedispersed(values, cells, shifts, row0=row0, out_rows=rows, n_grid=n_grid)
    img = np.moveaxis(ref_image(G, mask, image_scale), 0, 1)       # [T, F, N, N]
    if sigma is None:
        sigma = ref_layer1(img[history:], mask != 0)
    sigma = np.asarray(sigma, dtype=np.float64)
    x = np.clip(img / sigma[None, :, None, None], -clip, clip)
    centres = np.arange(t0 - delay, t0 + hop - delay)
    s_k = np.asarray(s_k, dtype=np.float64)
    full = None
    kfull = None
    scores = [] if keep_scores else None
    for k, flt in enumerate(filters):                      # max over filters, first k on ties
        sc = ref_filter_scores(x, int(rows[0]), [flt], centres)[0]
        if keep_scores:
            scores.append(sc)
        v = sc / s_k[k]
        if full is None:
            full, kfull = v, np.zeros(v.shape, dtype=np.int64)
        else:
            b = v > full
            full = np.where(b, v, full)
            kfull = np.where(b, k, kfull)
    n_ch = hop // chunk
    fv = full[:n_ch * chunk].reshape((n_ch, chunk) + full.shape[1:])
    arg = np.argmax(fv, axis=1)                             # earliest centre on ties
    mx = np.take_along_axis(fv, arg[:, None], axis=1)[:, 0]
    kk = np.take_along_axis(kfull[:n_ch * chunk].reshape(fv.shape), arg[:, None], axis=1)[:, 0]
    cm = (mx, arg, kk)
    tin = arg
    cands = []
    if decode == "chunk":
        def t_of(ci, f, l_, m_):
            return int(centres[0] + ci * chunk + tin[ci, f, l_, m_])

        raw = ref_decode_topk(mx, threshold=threshold, n_top=n_top, t_of=t_of)
        for (tt, f, l_, m_, v) in raw:
            ci = (tt - int(centres[0])) // chunk
            cands.append(dict(t=tt, f=f, l=l_, m=m_, snr=v, k=int(kk[ci, f, l_, m_])))
    elif decode == "full":
        raw = ref_decode_topk(full, threshold=threshold, n_top=n_top,
                              t_of=lambda t, f, l_, m_: int(centres[t]))
        for (tt, f, l_, m_, v) in raw:
            cands.append(dict(t=tt, f=f, l=l_, m=m_, snr=v,
                              k=int(kfull[tt - int(centres[0]), f, l_, m_])))
    else:
        raise ValueError(f"decode={decode!r}")
    if keep_scores:
        scores = np.stack(scores)
    return RefCycleResult(G=G, image=img, sigma=sigma, x_norm=x, scores=scores, full_max=full,
                          full_k=kfull, chunk_max=cm, centres=centres, image_rows=rows,
                          candidates=cands)


# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------


def point_source_values(cells: np.ndarray, n_grid: int, l_pix: int, m_pix: int,
                        amp: float = 1.0) -> np.ndarray:
    """Per-value visibility of a point source that images at pixel
    ``(l_pix, m_pix)`` (row, column), POSITIVE, through :func:`ref_image`
    with today's mask. The gridder puts uv = 0 at array index (N/2, N/2)
    (``compute_edge_mask`` docstring), so a value at array cell ``(u, w)``
    has physical uv ``(U, W) = (u - N/2, w - N/2)`` and carries
    ``amp * exp(-2 pi i (U y' + W x') / N)``, ``(y', x') = (l - N/2,
    m - N/2)``; the resulting ``(-1)^(l+m)`` is undone by the mask sign."""
    cells = np.asarray(cells, dtype=np.int64)
    h = n_grid // 2
    u = cells // n_grid - h
    w = cells % n_grid - h
    yp = l_pix - h
    xp = m_pix - h
    ph = -2.0 * np.pi * (u * yp + w * xp) / n_grid
    v = amp * np.exp(1j * ph)
    return np.where(cells >= 0, v, 0.0)


def inject_dispersed_burst(values: Sequence[np.ndarray], cells: Sequence[np.ndarray], *,
                           shifts_true: np.ndarray, row0: int, t_burst: int, n_grid: int,
                           l_pix: int, m_pix: int, amp: float, width: int = 1,
                           per_value_gain: Optional[Sequence[np.ndarray]] = None) -> None:
    """Add a dispersed point-source burst in place: stream ``s`` carries it
    at rows ``t_burst - shifts_true[s] + [0, width)``, so fine DM trials
    with ``shifts == shifts_true`` dedisperse it to rows
    ``t_burst .. t_burst + width - 1``."""
    for s, (val, cel) in enumerate(zip(values, cells)):
        vis = point_source_values(cel, n_grid, l_pix, m_pix, amp)
        if per_value_gain is not None:
            vis = vis * per_value_gain[s]
        for j in range(int(width)):
            r = int(t_burst) - int(shifts_true[s]) + j - int(row0)
            if 0 <= r < val.shape[0]:
                val[r] += vis


def quantize_rowrule(y: np.ndarray, step: float, bits: int, midrise: bool = False) -> tuple:
    """Reference quantiser (SPEC §4.4, test-data generator only). Mid-tread
    (default, ``x2/quant.py``): ``clip(rint(y / step), -(2^(b-1) - 1),
    2^(b-1) - 1)``; mid-rise: ``clip(floor(y / step), -2^(b-1), 2^(b-1) - 1)``."""
    hi = (1 << (bits - 1)) - 1
    lo = -(1 << (bits - 1)) if midrise else -hi
    op = np.floor if midrise else np.rint
    re = np.clip(op(np.real(y) / step), lo, hi).astype(np.int64)
    im = np.clip(op(np.imag(y) / step), lo, hi).astype(np.int64)
    return re, im


# ---------------------------------------------------------------------------
# offline re-imaging of archived raw windows (WS-D tools/x2/reimage_raw_window)
# ---------------------------------------------------------------------------

_VALUE_MAPS: dict = {}


def register_value_map(vmap) -> None:
    """Make a value map (``grid.baseline_grid.ValueMap`` or a duck-typed
    equivalent) available to :func:`reimage_from_raw` by its ``pattern_id``
    (value maps are archived once per pattern, not per event)."""
    _VALUE_MAPS[int(vmap.pattern_id)] = vmap


#: Phase 0 decode steps (sigma_hat units) by complex bits / n_sub.
_S_MAIN = {8: 0.3534, 10: 0.1932, 12: 0.1055}
_S_HEAVY = {2: 0.0773, 4: 0.0547}


def reimage_from_raw(slab, fine_local: int, t0_ext: int, n_t: int, *, ext_factor: int,
                     value_map=None, s_main: Optional[float] = None,
                     s_heavy: Optional[float] = None, image_scale: Optional[float] = None,
                     sigma=None, kernel_support: int = 1) -> np.ndarray:
    """Images ``[n_t, N, N]`` (float64) of fine ``fine_local`` of one archived
    coarse-DM slab (``x2.cutout.raw_archive.RawSlab``) over external rows
    ``[t0_ext, t0_ext + n_t)``; a tier-B tier sample fills ext rows 2k and
    2k+1 with the SAME image.

    Independent float64 path: frame-v2 rows unpacked (``x2.search.codec``,
    the K0 reference), mid-tread decode ``code * s`` (``s_heavy`` for heavy
    values) times ``sqrt(n)`` when the map is normalised, the ``"no data"``
    sentinel and invalid rows as 0, ``np.add.at`` gridding at
    ``stream[t - shift]``, ``fftshift(Re(ifft2))``, today's mask and the 2x
    image scale ``N^2 / sqrt(sum n)`` (raw image in sigma; Phase 0 option B)
    unless ``image_scale`` is given; divided by ``sigma`` (the Layer-1 sigma
    applied at detection: scalar or ``[n_t]``) when given.
    """
    from ..search.codec import RowFormat, unpack_cint8_rows, unpack_frame2_rows

    ex = getattr(slab, "extra", {}) or {}
    vm = value_map if value_map is not None else _VALUE_MAPS.get(int(ex.get("pattern_id", -1)))
    if vm is None:
        raise ValueError(f"no value map registered for pattern_id {ex.get('pattern_id')!r}")
    N = int(vm.n_grid)
    r = int(ext_factor)
    e_rows = np.arange(int(t0_ext), int(t0_ext) + int(n_t))
    t_rows = e_rows // r
    out_rows = np.arange(int(t_rows[0]), int(t_rows[-1]) + 1)
    sh = np.asarray(slab.shifts)[int(fine_local)]                     # [S]
    is_2x = bool(getattr(vm, "normalise", False)) or getattr(vm, "heavy_threshold", None) is not None
    bits_c = int(ex.get("bits_main", 8))
    n_sub = len(vm.streams) // 16 if len(vm.streams) >= 16 else 1
    sm = float(np.float32(s_main if s_main is not None else _S_MAIN.get(bits_c, 1.0)))
    shv = float(np.float32(s_heavy if s_heavy is not None else _S_HEAVY.get(n_sub, 0.1)))
    vals, cells, sig2 = [], [], 0.0
    lo = int(out_rows[0] - sh.max())
    hi = int(out_rows[-1] - sh.min()) + 1
    for s, sv in enumerate(vm.streams):
        nv = int(sv.n_values)
        nh = int(getattr(sv, "n_heavy", 0))
        codes, valid = slab.rows(s, lo, hi - lo)
        if is_2x:
            fmt = RowFormat("frame2", n_values=nv, n_heavy=nh, bits=bits_c // 2)
            re, im = unpack_frame2_rows(codes, fmt)
            sent = -(1 << (fmt.bits - 1))
            nd = np.zeros(re.shape, bool)
            nd[:, :nh] = (re[:, :nh] == -128) | (im[:, :nh] == -128)
            nd[:, nh:] = (re[:, nh:] == sent) | (im[:, nh:] == sent)
            step = np.full(nv, sm)
            step[:nh] = shv
        else:
            re, im = unpack_cint8_rows(codes, RowFormat("cint8", n_values=nv))
            nd = np.zeros(re.shape, bool)
            step = np.ones(nv)
        n = np.asarray(sv.n_contrib, dtype=np.float64)
        w = np.sqrt(n) if bool(getattr(vm, "normalise", False)) else np.ones(nv)
        v = (re.astype(np.float64) + 1j * im.astype(np.float64)) * (step * w)[None, :]
        v[nd] = 0.0
        v[np.asarray(valid) == 0] = 0.0
        vals.append(v)
        cel = np.asarray(sv.cell_row, np.int64) * N + np.asarray(sv.cell_col, np.int64)
        cel = np.where(np.asarray(sv.cell_row, np.int64) < 0, -1, cel)   # unused values
        cells.append(cel)
        sig2 += float(np.sum(w[cel >= 0] ** 2))
    G = ref_grid_dedispersed(vals, cells, sh[None, :], row0=lo, out_rows=out_rows, n_grid=N)
    scale = float(image_scale) if image_scale is not None else N * N / np.sqrt(sig2)
    img = ref_image(G, ref_edge_mask(N, kernel_support), scale)[0]        # [T_tier, N, N]
    out = img[t_rows - out_rows[0]]
    if sigma is not None:
        sg = np.asarray(sigma, dtype=np.float64)
        out = out / (sg[:, None, None] if sg.ndim else sg)
    return out
