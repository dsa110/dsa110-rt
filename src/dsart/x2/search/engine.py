"""Owner engine: one search half, all its owned coarse DMs (plan "Search").

Per owned DM, per cycle (201.326592 ms, aligned to absolute native sample
``k * 6144``; ``hop`` = 384 tier-A / 192 tier-B / 192 1x samples):

1. **K0** fetch the stream rows the cycle needs that are not yet resident
   (``ring.wait_window``) into the DM's time-minor code ring; flagged rows
   are zero-filled;
2. **K1** CSR combine every fine DM of the DM at its stage-2 shifts
   (variable ``n_fdm`` per DM, straight from the plan);
3. **K2/K3a** column/row inverse FFT -> masked, scaled fp16 rows of the
   DM's image ring (``hop`` + the widest filter's support; the first cycle
   also images the history -- no re-imaged overlap afterwards);
4. **Layer-1** per fine DM on this cycle's new rows; the sigma the detector
   divides by travels in the cycle's work record (``same_cycle``: this
   cycle's own estimate; ``lagged``: the previous cycle's, today's fused
   1-cube lag) -- never a shared "previous" variable another in-flight
   cycle could overwrite (the 2026-09-23 sigma-feedback flood class);
5. **K3b** streaming detector: per (pixel, 64-centre chunk) max over
   (centre, filter) + packed ``(t, k)``, plus Layer-2 taps;
6. **Layer-2** per DM (``noise_norm.layer2.Layer2State``): decode with
   the state before this cycle, update after (today's single-pass order);
7. **decode** today's top-k + merge-radius selection on the chunk maxima
   -> :class:`Candidate2x` in external-clock units with global fine-DM
   indices; optional C1 cross-kernel merge across the half's DMs.

Amplitude regime (Phase 0 study 2, option B): the image scale defaults to
``1 / expected_image_sigma`` = ``N^2 / sqrt(sum n V)`` on top of irfft2's
1/N^2, so the raw image is in sigma (1.00); Layer-1 floor 0.078 in those
image units (today's 5e-3/256 relative to the live sigma); Layer-2 floor
0.5 in post-Layer-1 units for boxcars, ``0.5 * seed / sqrt(w)`` per filter
otherwise, NEVER rescaled with image units; input clip 250; fp32
accumulation; the 2x wire is linear to the clip (no emulation of today's
compression, ledger S3). In 1x-compat the image scale is today's uv
prescale (1/256) and the floors today's ``5e-3 / 256`` and 0.5.

Backends: ``"cpu"`` runs the numpy twins (bit-exact with the kernels
where the kernels are exact: K0, K1, K3b); ``"gpu"`` dispatches to
``kernels_triton`` / ``kernels_cuda`` / the cuFFT fallback on one compute
stream (+ a copy stream for H2D and a low-priority stream for cutouts).
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .candidates import (NOISE_WARMUP_FLAG, PARTIAL_FANIN_FLAG, Candidate2x,
                         decode_chunkmax, decode_topk, merge_c1)
from .combine import StreamRing, k0_rows_to_ring, k1_twin
from .csr import OwnerCsr, expected_image_sigma
from .fft import FftPlan, image_from_uv_torch, image_from_uv_twin
from .filters import FilterBank, k3b_twin
from .geometry import CYCLE_NATIVE, DmSpec, OwnerPlan

__all__ = ["EngineConfig", "DmInputs", "DmState", "DmCycleResult", "CycleResult",
           "OwnerEngine"]

_LOG = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    n_grid: int = 256
    widths: tuple = (1, 2, 4, 8, 16, 32, 64)
    families: tuple = ("b3x",)            # Phase 0 bank; ("b",) = today's boxcars
    clip_sigma: float = 250.0
    chunk: int = 64
    delay: Optional[int] = None
    image_scale: Optional[float] = None
    edge_kernel_support: int = 1
    layer1_mode: str = "same_cycle"
    layer1_floor: float = 0.078           # image units (Phase 0 study 2, option B)
    layer1_max_samples: int = 10000
    layer1_n_burnin: int = 1
    layer1_seed: int = 0
    layer2_floor: Optional[float] = None  # None: per filter 0.5 * seed / sqrt(w)
    layer2_max_ratio: float = 4.0
    layer2_escape: int = 64
    layer2_valid_min_fraction: float = 0.9
    layer2_tau_s: float = 30.0
    layer2_n_burnin: int = 30
    layer2_cadence_s: Optional[float] = None
    layer2_max_samples: int = 100000
    n_tap_pixels: int = 256
    threshold: float = 9.55
    n_top_per_192: int = 224
    radius_t: int = 128
    radius_f: int = 5
    radius_lm: int = 3
    c1_merge: bool = True
    c1_lm_max_cells: int = 5
    c1_dm_max_trials: int = 2
    c1_t_frac: float = 1.0
    c1_sample_period_specnum: int = 16
    dm_index_mode: str = "global"
    fft_mode: str = "twin"
    backend: str = "cpu"
    rows_per_batch: int = 64
    wait_timeout_s: Optional[float] = 0.0
    sigma_history_cycles: int = 160          # >= 30 s of 201 ms cycles

    @classmethod
    def compat_1x(cls, **kw) -> "EngineConfig":
        """Today's search at today's parameters (``configs/dsart_search_rt.yaml``)."""
        base = dict(widths=(1, 2, 4, 8, 16, 32, 64), families=("b",), delay=32,
                    image_scale=1.0 / 256.0, layer1_floor=5e-3 / 256.0, layer2_floor=0.5,
                    layer1_mode="lagged", layer2_cadence_s=0.134217728,
                    dm_index_mode="local", threshold=9.55)
        base.update(kw)
        return cls(**base)


@dataclass
class DmInputs:
    """Static per-DM inputs: row format, owner CSR and decode constants."""

    fmt: object                  # RowFormat or codec.StreamFormats (per stream)
    csr: OwnerCsr
    c_v: np.ndarray              # [S, V] float32
    offset: float = 0.0
    value_sigma: Optional[list] = None   # per stream [V] decoded per-component std
    pattern_id: Optional[int] = None     # value-map pattern_id (SPEC §3)


@dataclass
class DmCycleResult:
    d: int
    cycle: int
    t0: int
    sigma_applied: np.ndarray
    s_k_applied: np.ndarray
    sigma_new: np.ndarray
    valid_fraction: float
    warming_up: bool
    candidates: List[Candidate2x]
    best: Optional[np.ndarray] = None
    tk: Optional[np.ndarray] = None
    timings_ms: dict = field(default_factory=dict)


@dataclass
class CycleResult:
    cycle: int
    per_dm: Dict[int, DmCycleResult]
    candidates: List[Candidate2x]
    wall_ms: float


class DmState:
    """Everything one owned DM carries across cycles."""

    def __init__(self, spec: DmSpec, inputs: DmInputs, cfg: EngineConfig, bank: FilterBank,
                 mask: np.ndarray, image_scale: float) -> None:
        from ...noise_norm.layer1 import Layer1State
        from ...noise_norm.layer2 import Layer2State
        import torch

        self.spec = spec
        self.inp = inputs
        self.cfg = cfg
        self.bank = bank
        N = cfg.n_grid
        if inputs.csr.n_grid != N:
            raise ValueError(f"DM {spec.d}: CSR n_grid {inputs.csr.n_grid} != {N}")
        if inputs.csr.n_stream != spec.n_stream:
            raise ValueError(f"DM {spec.d}: CSR has {inputs.csr.n_stream} streams, "
                             f"plan {spec.n_stream}")
        if float(inputs.offset) != 0.0:
            raise ValueError(f"DM {spec.d}: mid-rise decode is not supported -- flagged / "
                             "zero-filled rows must decode to 0 (SPEC §4, Phase 0 study 1)")
        self.fft = FftPlan.build(inputs.csr)
        self.delay = bank.max_d if cfg.delay is None else int(cfg.delay)
        self.history = bank.history(self.delay)
        if spec.hop % cfg.chunk:
            raise ValueError(f"DM {spec.d}: hop {spec.hop} not a multiple of chunk {cfg.chunk}")
        self.R_img = spec.hop + self.history
        self.R_s = spec.hop + self.history + spec.max_pos + spec.max_neg + 1
        V = int(inputs.c_v.shape[1])
        on_host = cfg.backend == "cpu"            # the GPU backend owns device rings
        self.streams = StreamRing(spec.n_stream, V, self.R_s, with_codes=on_host)
        self.image = (np.zeros((self.R_img, spec.n_fdm, N * N), dtype=np.float16) if on_host
                      else None)
        self.image_hi: Optional[int] = None          # exclusive absolute row imaged
        self.image_lo: Optional[int] = None
        self.mask_scaled = (mask * image_scale).astype(np.float32)
        self.active = (mask != 0).reshape(-1)
        self.image_scale = float(image_scale)
        act_idx = np.nonzero(self.active)[0]
        # Layer-1: fixed per-fine-DM subsample of (t_rel, pixel) over the hop
        M = int(min(cfg.layer1_max_samples, spec.hop * act_idx.size))
        rng = np.random.default_rng(cfg.layer1_seed + 7919 * spec.d)
        self.l1_t = rng.integers(0, spec.hop, size=(spec.n_fdm, M))
        self.l1_p = act_idx[rng.integers(0, act_idx.size, size=(spec.n_fdm, M))]
        self.layer1 = Layer1State(spec.n_fdm, n_burnin_cubes=cfg.layer1_n_burnin,
                                  sigma_floor=float(cfg.layer1_floor))
        # Layer-2 taps: a fixed, spread set of active pixels
        n_tap = int(min(cfg.n_tap_pixels, act_idx.size))
        self.tap_pixels = act_idx[np.linspace(0, act_idx.size - 1, n_tap).astype(np.int64)]
        cad = cfg.layer2_cadence_s or (CYCLE_NATIVE * 32.768e-6)
        self.layer2 = Layer2State(
            n_kernels=len(bank), cube_cadence_s=float(cad), tau_s=cfg.layer2_tau_s,
            n_burnin=cfg.layer2_n_burnin, n_kernel_max_t=1,
            sigma_floor=(bank.layer2_floors if cfg.layer2_floor is None
                         else float(cfg.layer2_floor)),
            sigma_max_ratio=cfg.layer2_max_ratio,
            clamp_escape_cubes=cfg.layer2_escape,
        )
        self.layer2._s_k.copy_(torch.as_tensor(bank.seeds, dtype=self.layer2._s_k.dtype))
        self.sigma_prev: Optional[np.ndarray] = None
        self.cycles = 0
        # applied Layer-1 sigma per detected centre range, >= 30 s (cutouts
        # are normalised with the sigma used at detection, SPEC §9)
        self.sigma_history: "deque" = deque(maxlen=int(cfg.sigma_history_cycles))

    # ------------------------------------------------------------------
    def stream_rows_needed(self, img_lo: int, img_hi: int) -> tuple:
        return img_lo - self.spec.max_pos, img_hi + self.spec.max_neg

    def summary(self) -> dict:
        return dict(d=self.spec.d, tier=self.spec.tier, n_fdm=self.spec.n_fdm,
                    hop=self.spec.hop, delay=self.delay, history=self.history,
                    R_img=self.R_img, R_s=self.R_s, csr=self.inp.csr.summary(),
                    image_scale=self.image_scale)


class OwnerEngine:
    """All owned DMs of one half. ``process_cycle(k)`` runs one cycle."""

    def __init__(self, plan: OwnerPlan, inputs: Dict[int, DmInputs], ring,
                 cfg: Optional[EngineConfig] = None) -> None:
        from ...image.imager import compute_edge_mask

        self.plan = plan
        self.cfg = cfg or EngineConfig()
        self.ring = ring
        self.bank = FilterBank.build(self.cfg.widths, self.cfg.families)
        mask = compute_edge_mask(n_grid=self.cfg.n_grid,
                                 kernel_support=self.cfg.edge_kernel_support,
                                 dtype=np.float64)
        self.mask = mask
        self.states: Dict[int, DmState] = {}
        self.build_status: Dict[int, dict] = {}
        for spec in plan.dms:
            if spec.d not in inputs:
                self.build_status[spec.d] = dict(ok=False, error="no CSR/format for this DM")
                continue
            inp = inputs[spec.d]
            scale = self._image_scale(inp)
            st = DmState(spec, inp, self.cfg, self.bank, mask, scale)
            self.states[spec.d] = st
            pid = inp.pattern_id if inp.pattern_id is not None else inp.csr.pattern_id
            self.build_status[spec.d] = dict(ok=True, pattern_id=pid, **st.summary())
        missing = [d for d, s in self.build_status.items() if not s["ok"]]
        if missing:
            # Fail loudly: a half must never run deaf to part of its DMs
            # (the n01 "deaf below DM 600" trap).
            raise RuntimeError(f"half {plan.half}: CSR/pattern build failed for DMs {missing}")
        self._gpu = None
        if self.cfg.backend == "gpu":
            from .gpu_backend import GpuBackend
            self._gpu = GpuBackend(self)

    def _image_scale(self, inp: DmInputs) -> float:
        if self.cfg.image_scale is not None:
            return float(self.cfg.image_scale)
        vs = inp.value_sigma
        if vs is None:
            vs = [np.asarray(inp.c_v[s], dtype=np.float64) for s in range(inp.csr.n_stream)]
        sig = expected_image_sigma(inp.csr, vs)
        if not sig > 0:
            raise ValueError("expected image sigma is zero (empty CSR?)")
        return 1.0 / sig

    # ------------------------------------------------------------------
    def status(self) -> dict:
        """Published at start (plan "Fail loudly"): per-DM pattern/CSR build."""
        return dict(half=self.plan.half, opoint=self.plan.opoint, plan_id=self.plan.plan_id,
                    filters=self.bank.kernel_ids(), dms=self.build_status)

    def process_cycle(self, k: int, *, keep_maps: bool = False) -> CycleResult:
        t_start = time.perf_counter()
        per_dm: Dict[int, DmCycleResult] = {}
        cands: List[Candidate2x] = []
        for spec in self.order_dms():
            st = self.states[spec.d]
            res = self._process_dm(st, int(k), keep_maps=keep_maps)
            per_dm[spec.d] = res
            cands.extend(res.candidates)
        if self.cfg.c1_merge and cands:
            cands = merge_c1(cands, lm_max_cells=self.cfg.c1_lm_max_cells,
                             dm_max_trials=self.cfg.c1_dm_max_trials, t_frac=self.cfg.c1_t_frac,
                             sample_period_specnum=self.cfg.c1_sample_period_specnum,
                             dm_index_mode=self.cfg.dm_index_mode)
            cands = [c for c in cands if c.snr >= self.cfg.threshold]
        return CycleResult(cycle=int(k), per_dm=per_dm, candidates=cands,
                           wall_ms=1e3 * (time.perf_counter() - t_start))

    def order_dms(self) -> List[DmSpec]:
        """Tier A (finest sample, most work) first, then by global index."""
        return sorted((s for s in self.plan.dms), key=lambda s: (s.t_native, s.d))

    # ------------------------------------------------------------------
    def _fetch(self, st: DmState, lo: int, hi: int) -> None:
        """K0: make stream rows ``[lo, hi)`` resident (append when
        contiguous with what is loaded, reload on a gap)."""
        sr = st.streams
        if sr.loaded_hi is not None and sr.loaded_lo <= lo and hi <= sr.loaded_hi:
            return
        if sr.loaded_hi is None or lo < sr.loaded_lo or lo > sr.loaded_hi:
            start = lo
            sr.loaded_lo = sr.loaded_hi = lo
        else:
            start = sr.loaded_hi
        n = hi - start
        if n > sr.n_slots:
            raise ValueError(f"DM {st.spec.d}: {n} rows > stream ring {sr.n_slots}")
        raw, valid = self.ring.wait_window(st.spec.d, int(start), int(n),
                                           timeout_s=self.cfg.wait_timeout_s)
        if self._gpu is not None:
            self._gpu.k0(st, raw, valid, start)              # codes live on the device
            sr.valid[:, sr.slots(np.arange(start, start + n))] = np.asarray(valid, bool)
        else:
            k0_rows_to_ring(sr, raw, valid, row0=int(start), fmt=st.inp.fmt)
        sr.loaded_hi = hi
        sr.loaded_lo = max(sr.loaded_lo, hi - sr.n_slots)

    def _image_rows(self, st: DmState, lo: int, hi: int) -> None:
        """K1 + K2/K3a for absolute rows ``[lo, hi)`` into the image ring."""
        spec = st.spec
        N = self.cfg.n_grid
        if self._gpu is not None:
            self._gpu.image_rows(st, lo, hi)
            return
        for r0 in range(lo, hi, self.cfg.rows_per_batch):
            n = min(self.cfg.rows_per_batch, hi - r0)
            uv = k1_twin(st.streams.codes, c_v=st.inp.c_v, offset=st.inp.offset, csr=st.inp.csr,
                         shifts=spec.shifts, t_lo=r0, n_t=n)               # [F, n, n_act]
            uvb = np.ascontiguousarray(np.swapaxes(uv, 0, 1)).reshape(n * spec.n_fdm, -1)
            if self.cfg.fft_mode == "twin":
                img = image_from_uv_twin(uvb, st.fft, st.mask_scaled)
            else:
                import torch
                img = image_from_uv_torch(torch.from_numpy(uvb), st.fft,
                                          torch.from_numpy(st.mask_scaled),
                                          mode=self.cfg.fft_mode).numpy().astype(np.float16)
            slots = np.mod(np.arange(r0, r0 + n), st.R_img)
            st.image[slots] = img.reshape(n, spec.n_fdm, N * N)
        st.image_hi = hi if st.image_hi is None else max(st.image_hi, hi)
        st.image_lo = lo if st.image_lo is None else max(st.image_lo, hi - st.R_img)

    def _layer1(self, st: DmState, t0: int) -> np.ndarray:
        """Per-fine-DM sigma of this cycle's new rows (image units)."""
        import torch
        from ...noise_norm.layer1 import sigma_clipped_std_batched

        if self._gpu is not None:
            sig = self._gpu.layer1_sigma(st, t0)
        else:
            slots = np.mod(t0 + st.l1_t, st.R_img)                         # [F, M]
            f_idx = np.arange(st.spec.n_fdm)[:, None]
            samples = st.image[slots, f_idx, st.l1_p].astype(np.float32)
            sig = sigma_clipped_std_batched(torch.from_numpy(samples)).numpy()
        out = st.layer1.update_and_query(per_fdm_sigma=torch.as_tensor(sig, dtype=torch.float32))
        return out.detach().cpu().numpy().astype(np.float64)

    def _valid_fraction(self, st: DmState, t0: int) -> float:
        rows = t0 + np.arange(st.spec.hop)
        sl = np.mod(rows[None, :, None] - st.spec.shifts[:, None, :], st.R_s)  # [F, hop, S]
        ok = st.streams.valid[np.arange(st.spec.n_stream)[None, None, :], sl].all(axis=2)
        return float(ok.mean())

    def _process_dm(self, st: DmState, k: int, *, keep_maps: bool) -> DmCycleResult:
        import torch
        from ...noise_norm.layer1 import sigma_clipped_std_batched

        spec, cfg = st.spec, self.cfg
        tm = {}
        t0, t1 = spec.cycle_rows(k)
        img_lo = t0 - st.history
        if st.image_hi is not None and st.image_hi >= img_lo and st.image_hi <= t1:
            img_lo = max(img_lo, st.image_hi)
        if img_lo < t1:
            s_lo, s_hi = st.stream_rows_needed(img_lo, t1)
            a = time.perf_counter()
            self._fetch(st, s_lo, s_hi)
            tm["k0_ms"] = 1e3 * (time.perf_counter() - a)
            a = time.perf_counter()
            self._image_rows(st, img_lo, t1)
            tm["image_ms"] = 1e3 * (time.perf_counter() - a)
        a = time.perf_counter()
        sigma_new = self._layer1(st, t0)
        if cfg.layer1_mode == "same_cycle" or st.sigma_prev is None:
            sigma_applied = sigma_new
        elif cfg.layer1_mode == "lagged":
            sigma_applied = st.sigma_prev
        else:
            raise ValueError(f"layer1_mode={cfg.layer1_mode!r}")
        st.sigma_prev = sigma_new
        st.sigma_history.append((t0 - st.delay, t1 - st.delay, np.asarray(sigma_applied).copy()))
        tm["layer1_ms"] = 1e3 * (time.perf_counter() - a)
        s_k_applied = st.layer2._s_k.detach().cpu().numpy().astype(np.float64)
        a = time.perf_counter()
        sigma_inv = (1.0 / np.maximum(sigma_applied, np.finfo(np.float32).tiny)).astype(np.float32)
        s_inv = (1.0 / np.maximum(s_k_applied, 1e-30)).astype(np.float32)
        c_first = t0 - st.delay
        n_top = int(round(cfg.n_top_per_192 * spec.hop / 192.0))
        topk = None
        if self._gpu is not None:
            vals, idx, tkv, shape, taps = self._gpu.k3b_topk(st, c_first, sigma_inv, s_inv,
                                                              n_top)
            topk = (vals, idx, tkv, shape)
            best = tk = None
        else:
            best, tk, taps = k3b_twin(st.image, ring_row0=0, c_first=c_first, n_c=spec.hop,
                                      sigma_inv=sigma_inv, s_inv=s_inv, bank=self.bank,
                                      clip=cfg.clip_sigma, chunk=cfg.chunk,
                                      tap_pixels=st.tap_pixels)
        tm["k3b_ms"] = 1e3 * (time.perf_counter() - a)
        # Layer-2 (single pass: decode used the state BEFORE this update)
        K = len(self.bank)
        flat = taps.reshape(K, -1)
        if flat.shape[1] > cfg.layer2_max_samples:
            idx = np.linspace(0, flat.shape[1] - 1, cfg.layer2_max_samples).astype(np.int64)
            flat = flat[:, idx]
        sig_k = sigma_clipped_std_batched(torch.from_numpy(np.ascontiguousarray(flat)))
        vfrac = self._valid_fraction(st, t0)
        cube_valid = vfrac >= cfg.layer2_valid_min_fraction
        _sk, warming = st.layer2.update_and_query(per_kernel_sigma=sig_k, valid=cube_valid)
        flags = (NOISE_WARMUP_FLAG if warming else 0) | (PARTIAL_FANIN_FLAG if vfrac < 1.0 else 0)
        a = time.perf_counter()
        dkw = dict(c_first=c_first, chunk=cfg.chunk, n_grid=cfg.n_grid, bank=self.bank,
                   threshold=cfg.threshold, spec=spec, half=self.plan.half,
                   ext_factor=self.plan.ext_factor(spec), radius_t=cfg.radius_t,
                   radius_f=cfg.radius_f, radius_lm=cfg.radius_lm, flags=flags)
        if topk is not None:
            cands = decode_topk(topk[0], topk[1], topk[2], shape=topk[3], **dkw)
        else:
            cands = decode_chunkmax(best, tk, n_top=n_top, **dkw)
        tm["decode_ms"] = 1e3 * (time.perf_counter() - a)
        st.cycles += 1
        return DmCycleResult(d=spec.d, cycle=k, t0=t0, sigma_applied=sigma_applied,
                             s_k_applied=s_k_applied, sigma_new=sigma_new,
                             valid_fraction=vfrac, warming_up=bool(warming), candidates=cands,
                             best=best if keep_maps else None, tk=tk if keep_maps else None,
                             timings_ms=tm)

    # ------------------------------------------------------------------
    # on-demand re-imaging (cutouts, SPEC §9) -- never touches live state
    # ------------------------------------------------------------------
    def sigma_at(self, d: int, rows: np.ndarray) -> tuple:
        """Layer-1 sigma applied at detection for tier rows ``rows``:
        ``(sigma [len(rows), F], found bool [len(rows)])``; rows outside the
        kept history take the nearest cycle's sigma (``found`` False)."""
        st = self.states[d]
        rows = np.asarray(rows, dtype=np.int64)
        hist = list(st.sigma_history)
        F = st.spec.n_fdm
        out = np.ones((rows.size, F))
        found = np.zeros(rows.size, dtype=bool)
        if not hist:
            return out, found
        lo = np.array([h[0] for h in hist])
        hi = np.array([h[1] for h in hist])
        for i, r in enumerate(rows):
            j = np.nonzero((lo <= r) & (r < hi))[0]
            if j.size:
                out[i] = hist[int(j[-1])][2]
                found[i] = True
            else:
                out[i] = hist[int(np.argmin(np.minimum(np.abs(lo - r), np.abs(hi - 1 - r))))][2]
        return out, found

    def reimage_iter(self, d: int, t_lo: int, t_hi: int, *, raw: Optional[np.ndarray] = None,
                     valid: Optional[np.ndarray] = None, raw_row0: Optional[int] = None,
                     batch_rows: int = 32, normalise: bool = True):
        """Re-image tier rows ``[t_lo, t_hi)`` of DM ``d`` for every fine DM,
        from a raw snapshot (``raw [S, n, row_bytes]`` covering stream rows
        ``[t_lo - max_pos, t_hi + max_neg)``) or the ring, into a private
        scratch stream ring (the live DM state is untouched; on the GPU this
        runs on the low-priority stream). Yields ``(rows, images, valid_frac)``
        batches, ``images [n, F, N, N]`` float32, divided by the Layer-1
        sigma applied at detection when ``normalise``."""
        st = self.states[d]
        spec = st.spec
        s_lo, s_hi = t_lo - spec.max_pos, t_hi + spec.max_neg
        if raw is None:
            raw, valid = self.ring.wait_window(d, s_lo, s_hi - s_lo,
                                               timeout_s=self.cfg.wait_timeout_s)
        else:
            if valid is None:
                valid = np.ones(raw.shape[:2], dtype=bool)
            if raw_row0 is not None and (raw_row0 != s_lo or raw.shape[1] != s_hi - s_lo):
                # cut the needed rows out of a larger snapshot (zero / invalid outside)
                sub = np.zeros((raw.shape[0], s_hi - s_lo, raw.shape[2]), dtype=raw.dtype)
                vsub = np.zeros((raw.shape[0], s_hi - s_lo), dtype=bool)
                a = max(s_lo, raw_row0)
                b = min(s_hi, raw_row0 + raw.shape[1])
                if b > a:
                    sub[:, a - s_lo:b - s_lo] = raw[:, a - raw_row0:b - raw_row0]
                    vsub[:, a - s_lo:b - s_lo] = valid[:, a - raw_row0:b - raw_row0]
                raw, valid = sub, vsub
        if valid is None:
            valid = np.ones(raw.shape[:2], dtype=bool)
        if self._gpu is not None and hasattr(self._gpu, "reimage_iter"):
            yield from self._gpu.reimage_iter(st, t_lo, t_hi, raw, valid, s_lo, batch_rows,
                                              normalise)
            return
        scratch = StreamRing(spec.n_stream, int(st.inp.c_v.shape[1]), s_hi - s_lo + 1)
        k0_rows_to_ring(scratch, raw, valid, row0=int(s_lo), fmt=st.inp.fmt)
        N = self.cfg.n_grid
        for r0 in range(t_lo, t_hi, batch_rows):
            n = min(batch_rows, t_hi - r0)
            uv = k1_twin(scratch.codes, c_v=st.inp.c_v, offset=st.inp.offset, csr=st.inp.csr,
                         shifts=spec.shifts, t_lo=r0, n_t=n)
            uvb = np.ascontiguousarray(np.swapaxes(uv, 0, 1)).reshape(n * spec.n_fdm, -1)
            if self.cfg.fft_mode == "twin":
                img = image_from_uv_twin(uvb, st.fft, st.mask_scaled, out_dtype=np.float16)
            else:
                import torch
                img = image_from_uv_torch(torch.from_numpy(uvb), st.fft,
                                          torch.from_numpy(st.mask_scaled),
                                          mode=self.cfg.fft_mode).numpy().astype(np.float16)
            img = img.astype(np.float32).reshape(n, spec.n_fdm, N, N)
            rows = np.arange(r0, r0 + n)
            if normalise:
                sig, _found = self.sigma_at(d, rows)
                img = img / sig[:, :, None, None]
            sl = scratch.slots(rows[None, :, None] - spec.shifts[:, None, :])  # [F, n, S]
            vf = float(scratch.valid[np.arange(spec.n_stream)[None, None, :], sl].mean())
            yield rows, img, vf

    # ------------------------------------------------------------------
    def detector_input(self, d: int, t_lo: int, t_hi: int, sigma: np.ndarray) -> np.ndarray:
        """Diagnostic: ``clip(image / sigma)`` rows ``[t_lo, t_hi)`` as float32
        ``[T, F, N, N]`` (what K3b consumed for those rows)."""
        st = self.states[d]
        slots = np.mod(np.arange(t_lo, t_hi), st.R_img)
        x = st.image[slots].astype(np.float32) * (1.0 / np.asarray(sigma, np.float32))[None, :, None]
        x = np.clip(x, -self.cfg.clip_sigma, self.cfg.clip_sigma)
        N = self.cfg.n_grid
        return x.reshape(len(slots), st.spec.n_fdm, N, N)
