"""GPU backend of the owner engine (CP1: first execution on a GPU node).

Device state per owned DM: the time-minor code ring, its validity, the fp16
image ring, the CSR / FFT tables, the per-pixel ``fl32(mask x scale / N^2)``
and the twiddles. Streams: ONE compute stream (K0 .. K3b, top-k), a copy
stream (H2D of new ring rows from pinned host memory; D2H of the top-k) and
a low-priority stream for on-demand re-imaging (cutouts).

Kernels: K0 / K2 / K3a CUDA (``kernels_cuda``), K1 / K3b Triton
(``kernels_triton``); ``fft_mode="pruned"`` / ``"irfft2"`` use the cuFFT
fallback (``fft.image_from_uv_torch``; per Phase 0 an fp16 cuFFT path would
need the 1/256 prescale -- these run complex64, so they do not).

Every method mirrors a CPU-path step of :class:`~dsart.x2.search.engine.
OwnerEngine` whose numpy twin is tested on CPU; the CP1 ``x2_gpu`` tests
(``tests/x2/test_search_gpu.py``) compare the two. Timing: CUDA events
around each DM-cycle's compute-stream work (``gpu_ms``) for the duty gate.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .codec import as_stream_formats
from .fft import twiddles

__all__ = ["GpuBackend"]


class _DmDev:
    pass


class GpuBackend:
    def __init__(self, engine) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("backend='gpu' needs CUDA (CP1 only; CPU CI uses backend='cpu')")
        self.eng = engine
        self.dev = torch.device("cuda")
        lo, hi = torch.cuda.Stream.priority_range()          # (low, high) = (0, -k)
        self.compute = torch.cuda.Stream(device=self.dev, priority=hi)
        self.copy = torch.cuda.Stream(device=self.dev)
        self.low = torch.cuda.Stream(device=self.dev, priority=lo)
        self.dm = {}
        self.gpu_ms = {}
        N = engine.cfg.n_grid
        self.tw = torch.as_tensor(twiddles(N).view(np.float32).reshape(-1, 2), device=self.dev)
        for d, st in engine.states.items():
            self.dm[d] = self._build(st)

    # ------------------------------------------------------------------
    def _build(self, st) -> _DmDev:
        from .kernels_triton import csr_tensors

        cfg = self.eng.cfg
        N = cfg.n_grid
        x = _DmDev()
        spec, inp = st.spec, st.inp
        S, V = spec.n_stream, int(inp.c_v.shape[1])
        x.codes = torch.zeros((S, V, st.R_s, 2), dtype=torch.int8, device=self.dev)
        x.valid = torch.zeros((S, st.R_s), dtype=torch.uint8, device=self.dev)
        x.nodata = torch.zeros((S,), dtype=torch.int64, device=self.dev)
        x.image = torch.zeros((st.R_img, spec.n_fdm, N * N), dtype=torch.float16, device=self.dev)
        x.c_v = torch.as_tensor(inp.c_v, dtype=torch.float32, device=self.dev)
        x.csr = csr_tensors(inp.csr, self.dev)
        x.shifts = torch.as_tensor(spec.shifts, dtype=torch.int32, device=self.dev)
        x.pix = torch.as_tensor((st.mask_scaled.astype(np.float64) / (N * N)).astype(np.float32),
                                device=self.dev)
        x.mask_scaled = torch.as_tensor(st.mask_scaled, device=self.dev)
        fp = st.fft
        sel = np.nonzero(fp.cell_line_slot >= 0)[0]
        order = sel[np.lexsort((fp.cell_pos[sel], fp.cell_line_slot[sel]))]
        cnt = np.bincount(fp.cell_line_slot[order], minlength=fp.fft_lines.size)
        x.line_ptr = torch.as_tensor(np.concatenate([[0], np.cumsum(cnt)]).astype(np.int32),
                                     device=self.dev)
        x.line_idx = torch.as_tensor(order.astype(np.int32), device=self.dev)
        x.line_pos = torch.as_tensor(fp.cell_pos[order].astype(np.int32), device=self.dev)
        x.fl = torch.as_tensor(fp.fft_lines.astype(np.int32), device=self.dev)
        csel = np.nonzero(fp.cell_const_slot >= 0)[0]
        x.cc_idx = torch.as_tensor(csel.astype(np.int32), device=self.dev)
        x.cc_line = torch.as_tensor(fp.cell_line[csel].astype(np.int32), device=self.dev)
        x.transpose = int(fp.fold == "row")
        fm = as_stream_formats(inp.fmt, S)
        x.kind = fm.kind
        x.nv = torch.as_tensor([f.n_values for f in fm.per_stream], dtype=torch.int32,
                               device=self.dev)
        x.nh = torch.as_tensor([getattr(f, "n_heavy", 0) for f in fm.per_stream],
                               dtype=torch.int32, device=self.dev)
        x.bits = torch.as_tensor([getattr(f, "bits", 8) for f in fm.per_stream],
                                 dtype=torch.int32, device=self.dev)
        x.row_bytes = fm.row_bytes
        x.pinned: Optional[torch.Tensor] = None
        x.tap = torch.as_tensor(st.tap_pixels, dtype=torch.long, device=self.dev)
        return x

    def _cp(self):
        import cupy as cp
        return cp

    def _ext(self, stream):
        cp = self._cp()
        return cp.cuda.ExternalStream(stream.cuda_stream)

    @staticmethod
    def _ptr(t: torch.Tensor):
        from ...image.tiled_combine_cuda import _ptr
        return _ptr(t)

    # ------------------------------------------------------------------
    def k0(self, st, raw: np.ndarray, valid: np.ndarray, start: int) -> None:
        """H2D (copy stream, pinned) + K0 unpack into the device code ring."""
        from .kernels_cuda import get_module

        x = self.dm[st.spec.d]
        S, n, rb = raw.shape
        if x.pinned is None or x.pinned.numel() < raw.size:
            x.pinned = torch.empty(raw.size, dtype=torch.uint8, pin_memory=True)
        hraw = x.pinned[:raw.size].view(S, n, rb)
        hraw.copy_(torch.from_numpy(np.ascontiguousarray(raw)))
        hval = torch.from_numpy(np.ascontiguousarray(valid, dtype=np.uint8))
        with torch.cuda.stream(self.copy):
            draw = hraw.to(self.dev, non_blocking=True)
            dval = hval.to(self.dev, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.copy)
        self.compute.wait_event(ev)
        mod = get_module()
        with self._ext(self.compute):
            if x.kind == "frame2":
                k = mod.get_function("k0_unpack_frame2")
                k((n, S), (128,), (self._ptr(draw), self._ptr(dval), self._ptr(x.nv),
                                   self._ptr(x.nh), self._ptr(x.bits), self._ptr(x.codes),
                                   self._ptr(x.valid), self._ptr(x.nodata), np.int32(S),
                                   np.int32(n), np.int32(rb), np.int32(x.codes.shape[1]),
                                   np.int32(st.R_s), np.int64(start)))
            else:
                k = mod.get_function("k0_unpack_cint8")
                k((n, S), (128,), (self._ptr(draw), self._ptr(dval), self._ptr(x.codes),
                                   self._ptr(x.valid), np.int32(S), np.int32(n), np.int32(rb),
                                   np.int32(x.codes.shape[1]), np.int32(st.R_s),
                                   np.int64(start)))
        draw.record_stream(self.compute)
        dval.record_stream(self.compute)

    # ------------------------------------------------------------------
    def _image_batch(self, st, x, r0: int, n: int, stream, ring: torch.Tensor,
                     ring_rows: int) -> None:
        from .kernels_triton import k1_combine
        from .kernels_cuda import get_module
        from .fft import image_from_uv_torch

        cfg = self.eng.cfg
        N = cfg.n_grid
        F = st.spec.n_fdm
        with torch.cuda.stream(stream):
            uv = k1_combine(x.codes, x.c_v, x.csr, x.shifts, t_lo=r0, n_t=n,
                            offset=float(st.inp.offset))                  # [F, n, A, 2]
            B = F * n
            A = int(x.csr["n_act"])
            if cfg.fft_mode in ("pruned", "irfft2"):
                img = image_from_uv_torch(torch.view_as_complex(uv.view(B, A, 2)), st.fft,
                                          x.mask_scaled, mode=cfg.fft_mode)
                img = img.view(F, n, N * N).transpose(0, 1).to(torch.float16)
                slots = torch.remainder(torch.arange(r0, r0 + n, device=self.dev), ring_rows)
                ring[slots] = img
                return
            mod = get_module()
            n_fl = int(x.fl.numel())
            Y = torch.empty((B, max(n_fl, 1), N, 2), dtype=torch.float32, device=self.dev)
            warps = 4
            with self._ext(stream):
                k2 = mod.get_function("k2_colfft_256")
                k2((int((B * n_fl + warps - 1) // warps),), (32 * warps,),
                   (self._ptr(uv), self._ptr(x.line_ptr), self._ptr(x.line_idx),
                    self._ptr(x.line_pos), self._ptr(self.tw), self._ptr(Y), np.int32(B),
                    np.int32(A), np.int32(n_fl)), shared_mem=warps * N * 8)
                k3 = mod.get_function("k3a_rowc2r_256")
                k3((int((B * N + warps - 1) // warps),), (32 * warps,),
                   (self._ptr(Y), self._ptr(uv), self._ptr(x.fl), self._ptr(x.cc_idx),
                    self._ptr(x.cc_line), self._ptr(self.tw), self._ptr(x.pix), self._ptr(ring),
                    np.int32(B), np.int32(A), np.int32(n_fl), np.int32(int(x.cc_idx.numel())),
                    np.int32(n), np.int32(F), np.int64(r0), np.int32(ring_rows),
                    np.int32(x.transpose)), shared_mem=warps * (N // 2 + 1) * 8)

    def image_rows(self, st, lo: int, hi: int) -> None:
        x = self.dm[st.spec.d]
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record(self.compute)
        for r0 in range(lo, hi, self.eng.cfg.rows_per_batch):
            n = min(self.eng.cfg.rows_per_batch, hi - r0)
            self._image_batch(st, x, r0, n, self.compute, x.image, st.R_img)
        ev1.record(self.compute)
        self.gpu_ms.setdefault(st.spec.d, []).append(("image", ev0, ev1))
        st.image_hi = hi if st.image_hi is None else max(st.image_hi, hi)

    def layer1_sigma(self, st, t0: int) -> np.ndarray:
        from ...noise_norm.layer1 import sigma_clipped_std_batched

        x = self.dm[st.spec.d]
        with torch.cuda.stream(self.compute):
            slots = torch.as_tensor(np.mod(t0 + st.l1_t, st.R_img), device=self.dev)
            f_idx = torch.arange(st.spec.n_fdm, device=self.dev)[:, None]
            p = torch.as_tensor(st.l1_p, device=self.dev)
            smp = x.image[slots, f_idx, p].float()
            sig = sigma_clipped_std_batched(smp)
        return sig.cpu().numpy()

    def k3b_topk(self, st, c_first: int, sigma_inv: np.ndarray, s_inv: np.ndarray,
                 n_top: int) -> tuple:
        """K3b + top-k on the device; ONE D2H of the top-k and the taps."""
        from .kernels_triton import k3b_detect

        x = self.dm[st.spec.d]
        cfg = self.eng.cfg
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.compute):
            ev0.record(self.compute)
            best, tk, taps = k3b_detect(
                x.image, ring_row0=0, c_first=c_first, n_c=st.spec.hop,
                sigma_inv=torch.as_tensor(sigma_inv, device=self.dev), s_inv=s_inv,
                bank=self.eng.bank, clip=cfg.clip_sigma, chunk=cfg.chunk,
                tap_pixels=st.tap_pixels)
            flat = best.view(-1)
            k = int(min(max(1, n_top), flat.numel()))
            vals, idx = torch.topk(flat, k)
            tkv = tk.view(-1)[idx]
            ev1.record(self.compute)
        self.gpu_ms.setdefault(st.spec.d, []).append(("k3b", ev0, ev1))
        out = (vals.cpu().numpy(), idx.cpu().numpy(), tkv.cpu().numpy(), tuple(best.shape),
               None if taps is None else taps.cpu().numpy())
        return out

    def pop_gpu_ms(self) -> dict:
        """Per DM: summed compute-stream ms since the last call (syncs)."""
        out = {}
        for d, evs in self.gpu_ms.items():
            if evs:
                evs[-1][2].synchronize()
            out[d] = float(sum(a.elapsed_time(b) for _n, a, b in evs))
            evs.clear()
        return out

    # ------------------------------------------------------------------
    def reimage_iter(self, st, t_lo, t_hi, raw, valid, s_lo, batch_rows, normalise):
        """Re-imaging on the LOW-priority stream into private buffers."""
        from .kernels_cuda import get_module

        spec = st.spec
        x = self.dm[spec.d]
        S, n_rows, rb = raw.shape
        R = n_rows + 1
        codes = torch.zeros((S, int(x.codes.shape[1]), R, 2), dtype=torch.int8, device=self.dev)
        vld = torch.zeros((S, R), dtype=torch.uint8, device=self.dev)
        nod = torch.zeros((S,), dtype=torch.int64, device=self.dev)
        priv = _DmDev()
        priv.__dict__.update(x.__dict__)
        priv.codes = codes
        N = self.eng.cfg.n_grid
        with torch.cuda.stream(self.low):
            draw = torch.from_numpy(np.ascontiguousarray(raw)).to(self.dev)
            dval = torch.from_numpy(np.ascontiguousarray(valid, dtype=np.uint8)).to(self.dev)
            mod = get_module()
            with self._ext(self.low):
                if x.kind == "frame2":
                    mod.get_function("k0_unpack_frame2")(
                        (n_rows, S), (128,),
                        (self._ptr(draw), self._ptr(dval), self._ptr(x.nv), self._ptr(x.nh),
                         self._ptr(x.bits), self._ptr(codes), self._ptr(vld), self._ptr(nod),
                         np.int32(S), np.int32(n_rows), np.int32(rb), np.int32(codes.shape[1]),
                         np.int32(R), np.int64(s_lo)))
                else:
                    mod.get_function("k0_unpack_cint8")(
                        (n_rows, S), (128,),
                        (self._ptr(draw), self._ptr(dval), self._ptr(codes), self._ptr(vld),
                         np.int32(S), np.int32(n_rows), np.int32(rb), np.int32(codes.shape[1]),
                         np.int32(R), np.int64(s_lo)))
        for r0 in range(t_lo, t_hi, batch_rows):
            n = min(batch_rows, t_hi - r0)
            ring = torch.zeros((n, spec.n_fdm, N * N), dtype=torch.float16, device=self.dev)
            # a private n-row ring: row r lands in slot r % n; roll back to row order
            self._image_batch(st, priv, r0, n, self.low, ring, n)
            with torch.cuda.stream(self.low):
                if r0 % n:
                    ring = torch.roll(ring, -(r0 % n), dims=0)
            self.low.synchronize()
            img = ring.float().cpu().numpy().reshape(n, spec.n_fdm, N, N)
            rows = np.arange(r0, r0 + n)
            if normalise:
                sig, _f = self.eng.sigma_at(spec.d, rows)
                img = img / sig[:, :, None, None]
            yield rows, img, float(valid.mean())
