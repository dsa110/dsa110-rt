"""Frame-v2 framing of one corr block of one stream, vectorised (SPEC §5).

Byte-equal to ``prod_frame2.iter_block_datagrams`` flow by flow (the
reference framing), for every flow of a stream at once:

* a datagram carries ``k <= max_k(bytes_per_sample)`` consecutive rows of ONE
  (stream, DM) flow (<= 72 + 8900 = 8972 B);
* rows are framed in runs of identical (step_main, step_heavy, ROW_GUARD)
  -- bytes compared, as the reference does -- so a datagram never straddles
  a corr block or a guarded row; ROW_GUARD is set on guarded runs;
* the block's last datagram of each flow carries LAST_IN_BLOCK.

Output is what ``MmsgSender.submit`` takes: one contiguous ``uint8`` buffer
plus per-datagram ``offsets`` / ``lengths``, listed flow by flow in row
order (``dms`` order, then ascending rows) so the receiver sees every flow in
sequence. Datagrams of equal k are laid out together in the buffer (one
row gather per k, headers from a structured array), so the regular case
(no guarded rows) costs two copies per stream: the full-k datagrams and the
per-flow tails. The payload may be a numpy array (host assembly) or a torch
tensor on a device: then the layout and headers are computed on the host
from the small per-row step / guard arrays, the buffer is assembled on the
device and comes back in one copy (pinned ``hbuf`` on CUDA).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dsart.transport import prod_frame2 as f2

__all__ = ["FramedBlock", "HEADER_DTYPE", "StreamFormat", "frame_stream_block"]

HEADER_DTYPE = np.dtype([
    ("magic", "<u4"), ("version", "<u2"), ("flags", "<u2"), ("seq_first", "<u8"),
    ("t0_native", "<u8"), ("stream_id", "<u2"), ("dm_idx", "<u2"), ("k", "u1"), ("n_sub", "u1"),
    ("bits_main", "u1"), ("bits_heavy", "u1"), ("n_grid", "<u2"), ("t_int_factor", "u1"),
    ("value_kind", "u1"), ("n_values", "<u4"), ("pattern_id", "<u8"), ("n_heavy", "<u4"),
    ("step_main", "<f4"), ("step_heavy", "<f4"), ("payload_bytes", "<u4"), ("s_main", "<f4"),
    ("s_heavy", "<f4"),
])
assert HEADER_DTYPE.itemsize == f2.HEADER_BYTES
assert all(HEADER_DTYPE.fields[k][1] == v for k, v in f2.OFFSETS.items())
_EXTRA_OK = f2.FLAG_NOISE_WARMUP | f2.FLAG_RFI_WARMING_UP


@dataclass(frozen=True)
class StreamFormat:
    """Static frame fields of one stream (one per local stream of a chgroup)."""

    stream_id: int
    n_values: int
    n_heavy: int
    bits_main: int
    n_sub: int
    n_grid: int
    value_kind: int
    pattern_id: int
    s_main: float
    s_heavy: float

    @property
    def bits_heavy(self) -> int:
        return 16 if self.n_heavy else 0

    @property
    def bytes_per_sample(self) -> int:
        return f2.bytes_per_sample(self.n_values, self.n_heavy, self.bits_main, self.bits_heavy)

    @property
    def max_k(self) -> int:
        return f2.max_k(self.bytes_per_sample)

    @classmethod
    def from_value_map(cls, vm, stream: int, rule, *, n_values: int | None = None,
                       n_heavy: int | None = None) -> "StreamFormat":
        """From WS-P's ``ValueMap`` stream ``stream`` and a ``quant.RowRule``
        (``n_values`` / ``n_heavy`` override for value subsets, benches only)."""
        st = vm.streams[stream]
        return cls(stream_id=int(stream),
                   n_values=int(st.n_values if n_values is None else n_values),
                   n_heavy=int(st.n_heavy if n_heavy is None else n_heavy),
                   bits_main=int(rule.bits_main), n_sub=int(vm.n_sub), n_grid=int(vm.n_grid),
                   value_kind=f2.VALUE_KIND_CELLS if vm.kind == "gridded" else f2.VALUE_KIND_BASELINES,
                   pattern_id=int(vm.stream_pattern_ids[stream]),
                   s_main=float(rule.s_main_f32), s_heavy=float(rule.s_heavy_f32))


@dataclass
class FramedBlock:
    """Datagrams of one block of one stream, ready for ``MmsgSender.submit``."""

    buf: np.ndarray            # uint8, every datagram (header + payload) contiguous
    offsets: np.ndarray        # uint64 [n], in send order
    lengths: np.ndarray        # uint32 [n]
    flow: np.ndarray           # int64 [n]: index into ``dms``
    j0: np.ndarray             # int64 [n]: first row of the datagram in the block
    k: np.ndarray              # int64 [n]

    def __len__(self) -> int:
        return int(self.lengths.size)

    @property
    def n_bytes(self) -> int:
        return int(self.lengths.sum(dtype=np.int64))

    def as_submit(self) -> tuple:
        return self.buf, self.offsets, self.lengths

    def datagrams(self) -> list[bytes]:
        b = self.buf
        return [b[int(o): int(o) + int(n)].tobytes() for o, n in zip(self.offsets, self.lengths)]


def frame_stream_block(payload, step_main: np.ndarray, step_heavy: np.ndarray,
                       guarded: np.ndarray | None, *, fmt: StreamFormat, dms, seq_first: int,
                       t_int_factor: int, extra_flags: int = 0, k: int | None = None,
                       dbuf=None, hbuf=None) -> FramedBlock:
    """Frame ``payload`` uint8 ``(n_d, S, bytes_per_sample)`` (flow ``i`` = DM
    ``dms[i]`` of stream ``fmt.stream_id``) with per-row steps ``(n_d, S)``
    float32 (or ``(n_d,)`` / scalars) and the per-row guard ``(n_d, S)``.
    ``seq_first`` is the tier-clock row of row 0 (``t0_native = seq *
    t_int_factor``)."""
    on_device = not isinstance(payload, np.ndarray) and hasattr(payload, "device")
    pay = payload if on_device else np.ascontiguousarray(payload, dtype=np.uint8)
    if pay.ndim != 3:
        raise ValueError(f"payload must be (n_d, S, bps), got {tuple(pay.shape)}")
    n_d, S, bps = (int(x) for x in pay.shape)
    if bps != fmt.bytes_per_sample:
        raise ValueError(f"payload rows are {bps} B, the stream format needs {fmt.bytes_per_sample}")
    dms = np.asarray(dms, dtype=np.int64).reshape(-1)
    if dms.size != n_d:
        raise ValueError(f"{dms.size} dms for {n_d} flows")
    kk = fmt.max_k if k is None else int(k)
    if not 1 <= kk <= fmt.max_k:
        raise ValueError(f"k={kk} not in [1, {fmt.max_k}]")
    seq_first = int(seq_first)
    if seq_first < 0 or seq_first + S >= f2.SEQ_LIMIT:
        raise ValueError(f"seq_first={seq_first} out of range")
    if n_d == 0 or S == 0:
        z = np.zeros(0, np.int64)
        return FramedBlock(np.zeros(0, np.uint8), z.astype(np.uint64), z.astype(np.uint32), z, z, z)

    sm = np.ascontiguousarray(np.broadcast_to(np.asarray(step_main, np.float32).reshape(
        (n_d, -1) if np.ndim(step_main) else (1, 1)), (n_d, S)))
    sh = np.ascontiguousarray(np.broadcast_to(np.asarray(step_heavy, np.float32).reshape(
        (n_d, -1) if np.ndim(step_heavy) else (1, 1)), (n_d, S)))
    gd = (np.zeros((n_d, S), bool) if guarded is None
          else np.ascontiguousarray(np.broadcast_to(np.asarray(guarded, bool), (n_d, S))))

    # ---- datagram starts: run starts + every kk rows inside a run
    smb, shb = sm.view(np.uint32), sh.view(np.uint32)
    brk = np.ones((n_d, S), bool)
    brk[:, 1:] = (gd[:, 1:] != gd[:, :-1]) | (smb[:, 1:] != smb[:, :-1]) | (shb[:, 1:] != shb[:, :-1])
    col = np.arange(S, dtype=np.int64)
    run0 = np.maximum.accumulate(np.where(brk, col[None, :], 0), axis=1)
    start = brk | ((col[None, :] - run0) % kk == 0)
    flow, j0 = np.nonzero(start)                                    # flow-major, rows ascending
    flat0 = flow * S + j0
    flat1 = np.empty_like(flat0)
    flat1[:-1] = flat0[1:]
    flat1[-1] = n_d * S
    j1 = flat1 - flow * S
    j1 = np.where(flat1 // S != flow, S, j1)                        # flow end (next start is j=0)
    kdg = j1 - j0
    n = flow.size

    # ---- headers
    if int(extra_flags) & ~_EXTRA_OK:
        raise ValueError(f"extra_flags=0x{int(extra_flags):04x}: only NOISE_WARMUP / RFI_WARMING_UP")
    flags = np.full(n, f2.FLAG_QUANTIZED | int(extra_flags), np.uint16)
    if fmt.n_heavy:
        flags |= np.uint16(f2.FLAG_HEAVY_PRESENT)
    flags[j1 == S] |= np.uint16(f2.FLAG_LAST_IN_BLOCK)
    flags[gd[flow, j0]] |= np.uint16(f2.FLAG_ROW_GUARD)
    hdr = np.zeros(n, HEADER_DTYPE)
    hdr["magic"] = f2.MAGIC
    hdr["version"] = f2.VERSION
    hdr["flags"] = flags
    seq = seq_first + j0
    hdr["seq_first"] = seq
    hdr["t0_native"] = seq * int(t_int_factor)
    hdr["stream_id"] = fmt.stream_id
    hdr["dm_idx"] = dms[flow]
    hdr["k"] = kdg
    hdr["n_sub"] = fmt.n_sub
    hdr["bits_main"] = fmt.bits_main
    hdr["bits_heavy"] = fmt.bits_heavy
    hdr["n_grid"] = fmt.n_grid
    hdr["t_int_factor"] = int(t_int_factor)
    hdr["value_kind"] = fmt.value_kind
    hdr["n_values"] = fmt.n_values
    hdr["pattern_id"] = np.uint64(int(fmt.pattern_id) & 0xFFFF_FFFF_FFFF_FFFF)
    hdr["n_heavy"] = fmt.n_heavy
    hdr["step_main"] = sm[flow, j0]
    hdr["step_heavy"] = sh[flow, j0]
    hdr["payload_bytes"] = kdg * bps
    hdr["s_main"] = np.float32(fmt.s_main)
    hdr["s_heavy"] = np.float32(fmt.s_heavy)
    _validate_static(fmt, t_int_factor, dms)

    # ---- buffer: groups of equal k, one row gather each
    lengths = (f2.HEADER_BYTES + kdg * bps).astype(np.uint32)
    offsets = np.zeros(n, np.uint64)
    if on_device:
        buf = _assemble_device(pay, hdr, flat0, kdg, n_d, S, bps, offsets, dbuf, hbuf,
                               int(lengths.sum(dtype=np.int64)))
        return FramedBlock(buf=buf, offsets=offsets, lengths=lengths, flow=flow.astype(np.int64),
                           j0=j0.astype(np.int64), k=kdg.astype(np.int64))
    buf = np.empty(int(lengths.sum(dtype=np.int64)), np.uint8)
    rows2 = pay.reshape(n_d * S, bps)
    hdr_u8 = hdr.view(np.uint8).reshape(n, f2.HEADER_BYTES)
    pos = 0
    for kv in np.unique(kdg):
        sel = np.flatnonzero(kdg == kv)
        dlen = f2.HEADER_BYTES + int(kv) * bps
        g = buf[pos: pos + sel.size * dlen].reshape(sel.size, dlen)
        g[:, : f2.HEADER_BYTES] = hdr_u8[sel]
        f0 = flat0[sel]
        if (n_d * S) % kv == 0 and not np.any(f0 % kv):            # regular: one row gather
            g[:, f2.HEADER_BYTES:] = rows2.reshape(-1, int(kv) * bps)[f0 // kv]
        else:
            idx = f0[:, None] + np.arange(int(kv), dtype=np.int64)[None, :]
            g[:, f2.HEADER_BYTES:] = np.take(rows2, idx, axis=0).reshape(sel.size, int(kv) * bps)
        offsets[sel] = pos + np.arange(sel.size, dtype=np.uint64) * np.uint64(dlen)
        pos += sel.size * dlen
    return FramedBlock(buf=buf, offsets=offsets, lengths=lengths, flow=flow.astype(np.int64),
                       j0=j0.astype(np.int64), k=kdg.astype(np.int64))


def max_buffer_bytes(n_d: int, S: int, bps: int) -> int:
    """Upper bound of a block's datagram bytes (every row its own datagram)."""
    return int(n_d) * int(S) * (f2.HEADER_BYTES + int(bps))


def _assemble_device(pay, hdr, flat0, kdg, n_d, S, bps, offsets, dbuf, hbuf, total):
    """The same group layout as the host path, assembled on the payload's device; returns
    the host numpy buffer (via pinned ``hbuf`` on CUDA)."""
    import torch  # noqa: PLC0415
    dev = pay.device
    if dbuf is None or dbuf.numel() < total:
        dbuf = torch.empty(max(total, 1), dtype=torch.uint8, device=dev)
    db = dbuf[:total]
    rows2 = pay.reshape(n_d * S, bps)
    hdr_t = torch.from_numpy(hdr.view(np.uint8).reshape(-1, f2.HEADER_BYTES)).to(dev, non_blocking=True)
    pos = 0
    for kv in np.unique(kdg):
        kv = int(kv)
        sel = np.flatnonzero(kdg == kv)
        dlen = f2.HEADER_BYTES + kv * bps
        g = db[pos: pos + sel.size * dlen].view(sel.size, dlen)
        sel_t = torch.from_numpy(sel).to(dev, non_blocking=True)
        g[:, : f2.HEADER_BYTES] = hdr_t.index_select(0, sel_t)
        f0 = flat0[sel]
        if (n_d * S) % kv == 0 and not np.any(f0 % kv):
            src = rows2.reshape(-1, kv * bps)
            if sel.size == src.shape[0]:
                g[:, f2.HEADER_BYTES:] = src                       # every datagram full: a copy
            else:
                g[:, f2.HEADER_BYTES:] = src.index_select(0, torch.from_numpy(f0 // kv).to(dev))
        else:
            idx = torch.from_numpy((f0[:, None] + np.arange(kv)[None, :]).reshape(-1)).to(dev)
            g[:, f2.HEADER_BYTES:] = rows2.index_select(0, idx).view(sel.size, kv * bps)
        offsets[sel] = pos + np.arange(sel.size, dtype=np.uint64) * np.uint64(dlen)
        pos += sel.size * dlen
    if db.is_cuda:
        if hbuf is None or hbuf.numel() < total:
            hbuf = torch.empty(max(total, 1), dtype=torch.uint8, pin_memory=True)
        hb = hbuf[:total]
        hb.copy_(db, non_blocking=True)
        torch.cuda.current_stream(dev).synchronize()
        return hb.numpy()
    return db.numpy()


def _validate_static(fmt: StreamFormat, t_int_factor: int, dms: np.ndarray) -> None:
    """The per-stream header fields once, through the reference validator."""
    h = f2.Frame2Header(
        seq_first=0, t0_native=0, stream_id=fmt.stream_id, dm_idx=int(dms.max(initial=0)), k=1,
        n_sub=fmt.n_sub, bits_main=fmt.bits_main, bits_heavy=fmt.bits_heavy, n_grid=fmt.n_grid,
        t_int_factor=int(t_int_factor), value_kind=fmt.value_kind, n_values=fmt.n_values,
        pattern_id=int(fmt.pattern_id), n_heavy=fmt.n_heavy, step_main=0.0, step_heavy=0.0,
        payload_bytes=fmt.bytes_per_sample, s_main=fmt.s_main, s_heavy=fmt.s_heavy,
        flags=f2.FLAG_QUANTIZED | (f2.FLAG_HEAVY_PRESENT if fmt.n_heavy else 0))
    f2.validate_header(h)
    if dms.size and (dms.min() < 0 or dms.max() > 0xFFFF):
        raise ValueError("dm_idx out of u16 range")
