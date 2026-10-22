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
* filters: boxcar (``b``), B2 triangle (``t``), B3 (``g``) as explicit
  integer FIR taps, centred ``a = L // 2`` (today's boxcar convention
  ``[c - w//2, c - w//2 + w - 1]``), on ``clip(x / sigma, +-clip)``;
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
    family: str          # "b" boxcar, "t" B2 triangle, "g" B3
    width: int           # base boxcar width (tier samples)
    taps: np.ndarray     # float64 [L]
    a: int               # output at centre c = sum_j taps[j] x[c - a + j]

    @property
    def length(self) -> int:
        return int(self.taps.shape[0])

    @property
    def norm(self) -> float:
        return float(np.sqrt(np.sum(self.taps ** 2)))


def ref_filter_bank(widths: Sequence[int], family: str = "b") -> List[RefFilter]:
    """B-spline order 1/2/3 = 1/2/3-fold convolution of a width-w boxcar."""
    order = {"b": 1, "t": 2, "g": 3}[family]
    out = []
    for w in widths:
        box = np.ones(int(w))
        taps = box
        for _ in range(order - 1):
            taps = np.convolve(taps, box)
        out.append(RefFilter(f"{family}{int(w)}", family, int(w), taps.astype(np.float64),
                             int(taps.shape[0]) // 2))
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


def ref_sigma_clipped_std(x: np.ndarray, n_sigma: float = 3.0, n_iter: int = 3) -> float:
    """Median-centred iterative clip, exactly the production estimator's
    semantics (``noise_norm.layer1.sigma_clipped_std_batched``), float64."""
    s = np.asarray(x, dtype=np.float64).reshape(-1)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0
    med = np.median(s)
    sig = np.sqrt(np.mean((s - med) ** 2))
    act = np.ones(s.size, dtype=bool)
    for _ in range(int(n_iter)):
        act &= (s >= med - n_sigma * sig) & (s <= med + n_sigma * sig)
        if not act.any():
            return 0.0
        med = np.median(s[act])
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


def ref_filter_scores(x_norm: np.ndarray, row0: int, filters: Sequence[RefFilter],
                      centres: Sequence[int]) -> np.ndarray:
    """Filter outputs ``[K, n_c, F, N, N]`` at absolute centres.

    ``x_norm [R, F, N, N]`` is the clipped, sigma-normalised input; row ``i``
    is absolute row ``row0 + i``; rows outside are zero.
    """
    centres = np.asarray(centres, dtype=np.int64)
    R = x_norm.shape[0]
    out = np.zeros((len(filters), centres.size) + x_norm.shape[1:], dtype=np.float64)
    for k, flt in enumerate(filters):
        for j, tap in enumerate(flt.taps):
            rows = centres - flt.a + j - row0
            ok = (rows >= 0) & (rows < R)
            out[k, ok] += tap * x_norm[rows[ok]]
    return out


def ref_chunk_max(snr: np.ndarray, chunk: int = 64) -> tuple:
    """``snr [K, n_c, ...]`` -> per (chunk, cell) max over (centre, filter).

    Returns ``(max [n_ch, ...], t_in_chunk [n_ch, ...], k [n_ch, ...])``;
    ties resolve to the earliest centre, then the lowest filter index.
    """
    K, n_c = snr.shape[:2]
    n_ch = n_c // chunk
    v = snr[:, :n_ch * chunk].reshape((K, n_ch, chunk) + snr.shape[2:])
    v = np.moveaxis(v, 0, 2)                                     # [n_ch, chunk, K, ...]
    flat = v.reshape((n_ch, chunk * K) + snr.shape[2:])
    arg = np.argmax(flat, axis=1)                                # first max = earliest (t, k)
    mx = np.take_along_axis(flat, arg[:, None], axis=1)[:, 0]
    return mx, arg // K, arg % K


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
    scores: np.ndarray       # [K, n_c, F, N, N] raw filter outputs
    snr: np.ndarray          # scores / s_k
    chunk_max: tuple         # (max, t_in_chunk, k)
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
) -> RefCycleResult:
    """Reference for one owner cycle of one DM.

    Images rows ``[t0 - history, t0 + hop)``; detects centres
    ``[t0 - delay, t0 + hop - delay)``. ``sigma`` (Layer-1) defaults to the
    cycle's own new rows ``[t0, t0 + hop)`` (2x same-cycle semantics).
    """
    rows = np.arange(t0 - history, t0 + hop)
    G = ref_grid_dedispersed(values, cells, shifts, row0=row0, out_rows=rows, n_grid=n_grid)
    img = np.moveaxis(ref_image(G, mask, image_scale), 0, 1)       # [T, F, N, N]
    if sigma is None:
        sigma = ref_layer1(img[history:], mask != 0)
    sigma = np.asarray(sigma, dtype=np.float64)
    x = np.clip(img / sigma[None, :, None, None], -clip, clip)
    centres = np.arange(t0 - delay, t0 + hop - delay)
    scores = ref_filter_scores(x, int(rows[0]), filters, centres)
    snr = scores / np.asarray(s_k, dtype=np.float64)[:, None, None, None, None]
    cm = ref_chunk_max(snr, chunk)
    mx, tin, kk = cm
    n_ch = mx.shape[0]

    def t_of(ci, f, l_, m_):
        return int(centres[0] + ci * chunk + tin[ci, f, l_, m_])

    raw = ref_decode_topk(mx, threshold=threshold, n_top=n_top, t_of=t_of)
    cands = []
    for (tt, f, l_, m_, v) in raw:
        ci = (tt - int(centres[0])) // chunk
        cands.append(dict(t=tt, f=f, l=l_, m=m_, snr=v, k=int(kk[ci, f, l_, m_])))
    del n_ch
    return RefCycleResult(G=G, image=img, sigma=sigma, x_norm=x, scores=scores, snr=snr,
                          chunk_max=cm, centres=centres, image_rows=rows, candidates=cands)


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


def quantize_rowrule(y: np.ndarray, step: float, bits: int) -> tuple:
    """Reference mid-rise quantiser (SPEC §4.3, test-data generator only):
    ``code = clip(floor(y / step), -2^(b-1), 2^(b-1) - 1)`` per component."""
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    re = np.clip(np.floor(np.real(y) / step), lo, hi).astype(np.int64)
    im = np.clip(np.floor(np.imag(y) / step), lo, hi).astype(np.int64)
    return re, im
