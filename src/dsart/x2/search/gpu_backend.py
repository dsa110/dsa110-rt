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
        x.w_v = torch.as_tensor(inp.weights(), dtype=torch.float32, device=self.dev)
        x.csr = csr_tensors(inp.csr, self.dev)
        x.shifts = torch.as_tensor(spec.shifts, dtype=torch.int32, device=self.dev)
        if cfg.k1_impl not in ("triton", "staged"):
            raise ValueError(f"k1_impl={cfg.k1_impl!r} (triton|staged)")
        from .k1_staged import staged_geometry
        x.k1geom = staged_geometry(spec.shifts, n_t=cfg.rows_per_batch)
        x.csr["smax"] = torch.as_tensor(x.k1geom.smax, dtype=torch.int32, device=self.dev)
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
    def _ev(stream) -> torch.cuda.Event:
        e = torch.cuda.Event(enable_timing=True)
        e.record(stream)
        return e

    def _mark(self, d: int, name: str, a, b) -> None:
        self.gpu_ms.setdefault(d, []).append((name, a, b))

    @staticmethod
    def _ptr(t: torch.Tensor):
        from ...image.tiled_combine_cuda import _ptr
        return _ptr(t)

    # ------------------------------------------------------------------
    def staging(self, st, n: int, row_bytes: int) -> np.ndarray:
        """The DM's pinned host staging as ``uint8 [S, n, row_bytes]`` (after
        the previous H2D from it finished): the ring copies straight into it
        (``RingV4Source.wait_window_into``) and :meth:`k0` skips its copy."""
        x = self.dm[st.spec.d]
        S = st.spec.n_stream
        size = S * int(n) * int(row_bytes)
        if getattr(x, "pinned_ev", None) is not None:
            x.pinned_ev.synchronize()
            x.pinned_ev = None
        if x.pinned is None or x.pinned.numel() < size:
            x.pinned = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        return x.pinned[:size].numpy().reshape(S, int(n), int(row_bytes))

    def k0(self, st, raw: np.ndarray, valid: np.ndarray, start: int) -> None:
        """H2D (copy stream, pinned) + K0 unpack into the device code ring."""
        from .kernels_cuda import get_module

        x = self.dm[st.spec.d]
        S, n, rb = raw.shape
        staged = (x.pinned is not None and raw.flags["C_CONTIGUOUS"]
                  and raw.ctypes.data == x.pinned.data_ptr())
        if getattr(x, "pinned_ev", None) is not None:
            x.pinned_ev.synchronize()                 # previous H2D done with the buffer
        if staged:
            hraw = x.pinned[:raw.size].view(S, n, rb)     # the ring copied into it already
        else:
            if x.pinned is None or x.pinned.numel() < raw.size:
                x.pinned = torch.empty(raw.size, dtype=torch.uint8, pin_memory=True)
            hraw = x.pinned[:raw.size].view(S, n, rb)
            hraw.copy_(torch.from_numpy(np.ascontiguousarray(raw)))
        hval = torch.from_numpy(np.ascontiguousarray(valid, dtype=np.uint8))
        with torch.cuda.stream(self.copy):
            h0 = self._ev(self.copy)
            draw = hraw.to(self.dev, non_blocking=True)
            dval = hval.to(self.dev, non_blocking=True)
            ev = torch.cuda.Event(enable_timing=True)
            ev.record(self.copy)
        self._mark(st.spec.d, "h2d", h0, ev)
        x.pinned_ev = ev
        self.compute.wait_event(ev)
        mod = get_module()
        k0a = self._ev(self.compute)
        with self._ext(self.compute):
            if x.kind == "frame2":
                from .kernels_cuda import launch_k0_tile
                launch_k0_tile(draw, dval, x.nv, x.nh, x.bits, x.codes, x.valid, x.nodata, S=S,
                               n=n, row_bytes=rb, V=int(x.codes.shape[1]), R=st.R_s, row0=start,
                               stream=self.compute)
            else:
                k = mod.get_function("k0_unpack_cint8")
                k((n, S), (128,), (self._ptr(draw), self._ptr(dval), self._ptr(x.codes),
                                   self._ptr(x.valid), np.int32(S), np.int32(n), np.int32(rb),
                                   np.int32(x.codes.shape[1]), np.int32(st.R_s),
                                   np.int64(start)))
        self._mark(st.spec.d, "k0", k0a, self._ev(self.compute))
        draw.record_stream(self.compute)
        dval.record_stream(self.compute)

    # ------------------------------------------------------------------
    def _image_batch(self, st, x, r0: int, n: int, stream, ring: torch.Tensor,
                     ring_rows: int) -> None:
        from .kernels_triton import k1_combine
        from .kernels_cuda import get_module
        from .fft import image_from_uv_torch
        from .k1_staged import k1_staged

        cfg = self.eng.cfg
        N = cfg.n_grid
        F = st.spec.n_fdm
        timed = stream is self.compute
        with torch.cuda.stream(stream):
            ea = self._ev(stream) if timed else None
            if cfg.k1_impl == "staged":
                uv = k1_staged(x.codes, x.c_v, x.csr, x.shifts, t_lo=r0, n_t=n,
                               offset=float(st.inp.offset), w_v=x.w_v, geom=x.k1geom,
                               stream=stream)                                      # [F, n, A, 2]
            else:
                uv = k1_combine(x.codes, x.c_v, x.csr, x.shifts, t_lo=r0, n_t=n,
                                offset=float(st.inp.offset), w_v=x.w_v)            # [F, n, A, 2]
            if timed:
                eb = self._ev(stream)
                self._mark(st.spec.d, "k1", ea, eb)
            B = F * n
            A = int(x.csr["n_act"])
            if cfg.fft_mode in ("pruned", "irfft2"):
                img = image_from_uv_torch(torch.view_as_complex(uv.view(B, A, 2)), st.fft,
                                          x.mask_scaled, mode=cfg.fft_mode)
                img = img.view(F, n, N * N).transpose(0, 1).to(torch.float16)
                slots = torch.remainder(torch.arange(r0, r0 + n, device=self.dev), ring_rows)
                ring[slots] = img
                if timed:
                    self._mark(st.spec.d, "fft", eb, self._ev(stream))
                return
            if cfg.fft_mode == "fused":                    # K2 + K3a in one kernel
                from .kernels_cuda import k23_tables, launch_k23
                if getattr(x, "k23", None) is None:
                    x.k23 = k23_tables(dict(line_ptr=x.line_ptr, line_idx=x.line_idx,
                                            line_pos=x.line_pos, fl=x.fl, cc_idx=x.cc_idx,
                                            cc_line=x.cc_line), int(x.fl.numel()), self.dev,
                                       pix=x.pix)
                launch_k23(uv, x.k23, self.tw, x.pix, ring, B=B, n_act=A, n_t=n, F=F, t0=r0,
                           R_img=ring_rows, transpose=x.transpose, stream=stream)
                if timed:
                    self._mark(st.spec.d, "fft", eb, self._ev(stream))
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
            if timed:
                self._mark(st.spec.d, "fft", eb, self._ev(stream))

    def image_rows(self, st, lo: int, hi: int) -> None:
        x = self.dm[st.spec.d]
        for r0 in range(lo, hi, self.eng.cfg.rows_per_batch):
            n = min(self.eng.cfg.rows_per_batch, hi - r0)
            self._image_batch(st, x, r0, n, self.compute, x.image, st.R_img)
        st.image_hi = hi if st.image_hi is None else max(st.image_hi, hi)

    def layer1_launch(self, st, t0: int) -> torch.Tensor:
        """Layer-1 per-fine-DM sigma of the cycle's new rows, left on the
        device (compute stream; :meth:`to_host` brings it back)."""
        from ...noise_norm.layer1 import sigma_clipped_std_batched

        x = self.dm[st.spec.d]
        with torch.cuda.stream(self.compute):
            a = self._ev(self.compute)
            if st.l1_t.shape[1] <= 10240:               # one launch (kernels_cuda.l1_clipstd)
                from .kernels_cuda import launch_l1
                if getattr(x, "l1_tables", None) is None:
                    x.l1_tables = (
                        torch.as_tensor(np.ascontiguousarray(st.l1_t), dtype=torch.int32,
                                        device=self.dev),
                        torch.as_tensor(np.ascontiguousarray(st.l1_p), dtype=torch.int32,
                                        device=self.dev))
                lt, lp = x.l1_tables
                sig = torch.empty(st.spec.n_fdm, dtype=torch.float32, device=self.dev)
                launch_l1(x.image, lt, lp, sig, F=st.spec.n_fdm, P=int(x.image.shape[2]),
                          R_img=st.R_img, t0=int(t0), M=int(st.l1_t.shape[1]),
                          stream=self.compute)
            else:
                slots = torch.as_tensor(np.mod(t0 + st.l1_t, st.R_img), device=self.dev)
                f_idx = torch.arange(st.spec.n_fdm, device=self.dev)[:, None]
                p = torch.as_tensor(st.l1_p, device=self.dev)
                smp = x.image[slots, f_idx, p].float()
                sig = sigma_clipped_std_batched(smp)
            self._mark(st.spec.d, "l1", a, self._ev(self.compute))
        return sig

    def to_host(self, tensors: dict) -> dict:
        """ONE wait for every device tensor in ``{key: tensor | tuple of
        tensors}``: non-blocking D2H copies into pinned memory ON the compute
        stream (ordered after the work that produced them; a copy on another
        stream could read the buffers before the compute stream wrote them),
        one event sync; returns the same structure as numpy arrays."""
        out = {}
        with torch.cuda.stream(self.compute):
            for key, v in tensors.items():
                if isinstance(v, (tuple, list)):
                    out[key] = tuple(t.to("cpu", non_blocking=True) for t in v)
                else:
                    out[key] = v.to("cpu", non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.compute)
        ev.synchronize()
        return {k: (tuple(t.numpy() for t in v) if isinstance(v, tuple) else v.numpy())
                for k, v in out.items()}

    def layer1_sigma(self, st, t0: int) -> np.ndarray:
        return self.to_host({0: self.layer1_launch(st, t0)})[0]

    def k3b_topk_launch(self, st, c_first: int, sigma_inv: np.ndarray, s_inv: np.ndarray,
                        n_top: int) -> tuple:
        """K3b + top-k on the device (compute stream, nothing comes back):
        ``(vals, idx, tk_at_idx, best.shape, taps)``."""
        from .kernels_triton import k3b_detect
        from .k3b_fused import k3b_fused

        x = self.dm[st.spec.d]
        cfg = self.eng.cfg
        from .k3b_cuda import k3b_cuda
        from .k3b_reg import k3b_reg
        impls = dict(v1=k3b_detect, fused=k3b_fused, cuda=k3b_cuda, reg=k3b_reg)
        if cfg.k3b_impl not in impls:
            raise ValueError(f"k3b_impl={cfg.k3b_impl!r} (v1|fused|cuda|reg)")
        detect = impls[cfg.k3b_impl]
        with torch.cuda.stream(self.compute):
            ev0 = self._ev(self.compute)
            # small per-cycle inputs go up pinned + non-blocking (a pageable
            # torch H2D syncs the host with the stream); "reg" takes device
            # tensors for all of them
            si = torch.from_numpy(np.ascontiguousarray(sigma_inv, dtype=np.float32))
            si = si.pin_memory().to(self.dev, non_blocking=True)
            tap_pixels, sk = st.tap_pixels, s_inv
            if cfg.k3b_impl == "reg":
                if getattr(x, "tap32", None) is None:
                    x.tap32 = x.tap.to(torch.int32)
                tap_pixels = x.tap32 if st.tap_pixels is not None else None
                sk = torch.from_numpy(np.ascontiguousarray(s_inv, dtype=np.float32))
                sk = sk.pin_memory().to(self.dev, non_blocking=True)
            best, tk, taps = detect(
                x.image, ring_row0=0, c_first=c_first, n_c=st.spec.hop,
                sigma_inv=si, s_inv=sk, bank=self.eng.bank, clip=cfg.clip_sigma,
                chunk=cfg.chunk, tap_pixels=tap_pixels)
            evm = self._ev(self.compute)
            flat = best.view(-1)
            k = int(min(max(1, n_top), flat.numel()))
            vals, idx = torch.topk(flat, k)
            tkv = tk.view(-1)[idx]
            ev1 = self._ev(self.compute)
        self._mark(st.spec.d, "k3b", ev0, evm)
        self._mark(st.spec.d, "topk", evm, ev1)
        return vals, idx, tkv, tuple(best.shape), taps

    def k3b_topk(self, st, c_first: int, sigma_inv: np.ndarray, s_inv: np.ndarray,
                 n_top: int) -> tuple:
        """K3b + top-k on the device; ONE D2H of the top-k (taps stay on the
        device for :meth:`layer2_sigma`)."""
        vals, idx, tkv, shape, taps = self.k3b_topk_launch(st, c_first, sigma_inv, s_inv, n_top)
        h = self.to_host({0: (vals, idx, tkv)})[0]
        return h[0], h[1], h[2], shape, taps

    def layer2_launch(self, st, taps: torch.Tensor, max_samples: int) -> torch.Tensor:
        """Per-filter sigma of the raw tap scores on the device (the CPU path's
        ``sigma_clipped_std_batched`` over the same ``linspace`` subsample),
        left on the device."""
        x = self.dm[st.spec.d]
        K = int(taps.shape[0])
        with torch.cuda.stream(self.compute):
            a = self._ev(self.compute)
            flat = taps.reshape(K, -1)
            n = int(flat.shape[1])
            if n > max_samples:
                key = (n, int(max_samples))
                if getattr(x, "l2_idx", (None, None))[0] != key:
                    x.l2_idx = (key, torch.as_tensor(
                        np.linspace(0, n - 1, max_samples).astype(np.int64), device=self.dev))
                flat = flat[:, x.l2_idx[1]]
            from .kernels_cuda import launch_clipstd_rows
            work = flat.float().contiguous()
            if work.data_ptr() == taps.data_ptr():
                work = work.clone()                      # the kernel marks samples in place
            sig = torch.empty(K, dtype=torch.float32, device=self.dev)
            launch_clipstd_rows(work, sig, stream=self.compute)
            self._mark(st.spec.d, "l2", a, self._ev(self.compute))
        return sig

    def layer2_sigma(self, st, taps: torch.Tensor, max_samples: int) -> torch.Tensor:
        return torch.from_numpy(self.to_host({0: self.layer2_launch(st, taps, max_samples)})[0])

    def pop_gpu_ms(self, by_stage: bool = False) -> dict:
        """Per DM since the last call (syncs): the compute-stream ms summed
        over stages (the duty numerator; ``h2d`` runs on the copy stream and
        overlaps, so it is excluded), or with ``by_stage`` ``{stage: ms}``
        for ``h2d k0 k1 fft l1 k3b topk``."""
        out = {}
        for d, evs in self.gpu_ms.items():
            st = {}
            for name, a, b in evs:
                b.synchronize()
                st[name] = st.get(name, 0.0) + float(a.elapsed_time(b))
            evs.clear()
            out[d] = st if by_stage else float(sum(v for k, v in st.items() if k != "h2d"))
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
