"""2x corr TX: frame v2 + WS-T's C sendmmsg sender (SPEC §5, §6).

:class:`FramedTx2x` is the :class:`~dsart.x2.corr.pipeline2x.Tx2xStage` the
2x corr_fast runs (the pipeline calls it on its D2H stream, one block
behind). Per tier and stream it builds the datagrams byte-equal to
``prod_frame2.iter_block_datagrams`` and hands all of them to a sink in ONE
call. CUDA blocks are framed ON THE DEVICE (guarded rows included): the
per-row steps / guard flags come to the host (small), the layout and headers
are built there (numpy, vectorised), the payload rows are gathered into the
datagram slots of a persistent device buffer and the whole buffer comes back
in one D2H into pinned memory. Sinks:

* :func:`build_mmsg_sender` -- ``transport.send_mmsg.MmsgSender``: a C
  pthread drains per-destination queues with ``sendmmsg``; ``submit`` is a
  ctypes call (GIL released, no Python per datagram). Destinations are the
  search nodes (``SEARCH_NODES`` order) at ``port_base + chgroup`` (search_rx
  v4 binds one port per chgroup); routes are the plan's explicit owner map,
  ``dm -> owner_half[dm] // 2`` (``owner_half[d] = d % 8``); each destination
  is token-bucket paced at ``pace`` (1.25) x its regular bytes per block over
  one block (134.2 ms).
* :class:`DrainSink` -- counts what would be sent and sends nothing (the CP1
  ``live --tx drain`` step: the full corr chain incl. D2H and framing, no
  packet on the fabric).

Both expose ``counters()`` in the ``MmsgSender`` schema, so
``TxRateMeter().update(tx.counters())`` feeds the corr_fast heartbeat.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch

from dsart.transport import prod_frame2 as f2
from dsart.x2.corr.framer2x import (FramedBlock, StreamFormat, frame_stream_block, max_buffer_bytes,
                                    staging_bytes)

LOG = logging.getLogger("dsart.x2.corr.tx2x")

__all__ = [
    "BLOCK_NATIVE",
    "BLOCK_S",
    "DrainSink",
    "FramedTx2x",
    "block_tx_budget",
    "build_mmsg_sender",
    "corr_routes",
    "frame_regular_device",
]

BLOCK_NATIVE = 4096
BLOCK_S = BLOCK_NATIVE * 32.768e-6


def _np(v) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    return np.asarray(v)


class DrainSink:
    """A sink that counts and discards (no socket). ``counters()`` mirrors
    ``MmsgSender.counters()`` with one destination named ``drain``."""

    def __init__(self) -> None:
        self.submitted = self.n_bytes = self.calls = 0

    def submit(self, datagrams, *, block_timeout_s: float = 0.0) -> int:
        _buf, _offs, lens = datagrams
        n = int(np.asarray(lens).size)
        self.submitted += n
        self.n_bytes += int(np.asarray(lens).sum(dtype=np.int64))
        self.calls += 1
        return n

    def counters(self) -> dict:
        d = {"enqueued": self.submitted, "sent": self.submitted, "bytes": self.n_bytes,
             "full_drops": 0, "errors": 0, "eagain": 0, "refused": 0, "max_depth": 0,
             "depth": 0, "last_errno": 0, "dest": "drain"}
        return {"submitted": self.submitted, "accepted": self.submitted, "unrouted": 0,
                "bad_length": 0, "sendmmsg_calls": 0, "idle_sleeps": 0, "per_dest": [d]}

    def flush(self, timeout_s: float = 1.0) -> bool:
        return True

    def close(self) -> None:
        pass


def frame_regular_device(payload, step_main, step_heavy, *, fmt: StreamFormat, dms, seq_first: int,
                         t_int_factor: int, extra_flags: int = 0, dbuf=None, hbuf=None) -> FramedBlock:
    """A block whose every flow has one (step_main, step_heavy) and no guard, framed with the
    payload where it is (torch, any device): :func:`frame_stream_block` with per-flow steps."""
    n_d, S = int(payload.shape[0]), int(payload.shape[1])
    sm = np.repeat(np.asarray(step_main, np.float32).reshape(n_d, 1), S, axis=1)
    sh = np.repeat(np.asarray(step_heavy, np.float32).reshape(n_d, 1), S, axis=1)
    return frame_stream_block(payload, sm, sh, None, fmt=fmt, dms=dms, seq_first=seq_first,
                              t_int_factor=t_int_factor, extra_flags=extra_flags, dbuf=dbuf, hbuf=hbuf)


class FramedTx2x:
    """Tx2xStage: frame each stream of a tier and submit it to ``sink``.

    ``formats[s]`` is the :class:`StreamFormat` of local stream ``s`` (in the
    packer's stream order). ``transmit_tier`` returns the datagrams the sink
    accepted (an ``MmsgSender`` drops and counts on full queues). Regular blocks
    on a CUDA device are framed on the device (``device_framing``)."""

    def __init__(self, sink: Any, formats: list[StreamFormat], *, device_framing: bool = True) -> None:
        self.sink = sink
        self.formats = list(formats)
        self.device_framing = bool(device_framing)
        self.n_datagrams = 0
        self.n_bytes = 0
        self.n_rejected = 0
        self.n_device_framed = 0
        self.frame_s = 0.0
        self._dbuf: dict = {}
        self._hbuf: dict = {}
        self._sbuf: dict = {}

    @staticmethod
    def _row_params(packets) -> list | None:
        """Per stream ``(step_main, step_heavy, guarded)`` as numpy from ONE device-to-host
        copy for the whole tier (instead of three per stream); None unless every stream's
        rows are CUDA tensors of one shape."""
        if not packets:
            return None
        shp = None
        rows = []
        for pk in packets:
            ts = [pk["step_main"], pk["step_heavy"]] + ([pk["guarded"]] if "guarded" in pk else [])
            if not all(isinstance(t, torch.Tensor) and t.is_cuda for t in ts):
                return None
            shp = tuple(ts[0].shape) if shp is None else shp
            if any(tuple(t.shape) != shp for t in ts):
                return None
            rows += [ts[0].to(torch.float32), ts[1].to(torch.float32),
                     ts[2].to(torch.float32) if len(ts) == 3
                     else torch.zeros(shp, dtype=torch.float32, device=ts[0].device)]
        host = torch.stack(rows).cpu().numpy()
        return [(host[3 * s], host[3 * s + 1], (host[3 * s + 2] > 0.5) if "guarded" in pk else None)
                for s, pk in enumerate(packets)]

    def transmit_tier(self, *, block_n, tier, dms, t0_native, T_native, stream_ids, packets,
                      rfi_warming_up) -> int:
        T = int(T_native)
        if int(t0_native) % T:
            raise ValueError(f"t0_native={t0_native} is not a multiple of T_native={T}")
        seq_first = int(t0_native) // T
        extra = f2.FLAG_RFI_WARMING_UP if rfi_warming_up else 0
        if len(packets) != len(self.formats):
            raise ValueError(f"{len(packets)} streams handed to TX, {len(self.formats)} formats")
        for s in range(len(packets)):
            if int(stream_ids[s]) != self.formats[s].stream_id:
                raise ValueError(f"stream {s}: id {stream_ids[s]} != format {self.formats[s].stream_id}")
        t0 = time.perf_counter()
        dev_pay = [self.device_framing and isinstance(pk["payload"], torch.Tensor)
                   and pk["payload"].is_cuda for pk in packets]
        rp = self._row_params(packets) if any(dev_pay) else None
        # phase 1: every stream framed; device streams leave their D2H copy in flight
        blocks = []
        pending = None
        for s, pk in enumerate(packets):
            fmt = self.formats[s]
            pay = pk["payload"]
            if rp is not None:
                sm, sh, gd = rp[s]
            else:
                sm, sh = _np(pk["step_main"]), _np(pk["step_heavy"])
                gd = _np(pk["guarded"]) if "guarded" in pk else None
            if dev_pay[s]:
                key = (s, tuple(pay.shape))
                if key not in self._dbuf:
                    n_b = max_buffer_bytes(*(int(x) for x in pay.shape))
                    self._dbuf[key] = torch.empty(n_b, dtype=torch.uint8, device=pay.device)
                    self._hbuf[key] = torch.empty(n_b, dtype=torch.uint8, pin_memory=True)
                    self._sbuf[key] = torch.empty(staging_bytes(int(pay.shape[0]), int(pay.shape[1])),
                                                  dtype=torch.uint8, pin_memory=True)
                fb = frame_stream_block(pay, sm, sh, gd, fmt=fmt, dms=np.asarray(dms),
                                        seq_first=seq_first, t_int_factor=T, extra_flags=extra,
                                        dbuf=self._dbuf[key], hbuf=self._hbuf[key],
                                        sbuf=self._sbuf[key], sync=False)
                pending = pay.device
                self.n_device_framed += 1
            else:
                fb = frame_stream_block(_np(pay), sm, sh, gd, fmt=fmt, dms=np.asarray(dms),
                                        seq_first=seq_first, t_int_factor=T, extra_flags=extra)
            blocks.append(fb)
        if pending is not None:                  # phase 2: one sync for the tier's D2H copies
            torch.cuda.current_stream(pending).synchronize()
        self.frame_s += time.perf_counter() - t0
        n_ok = 0
        for fb in blocks:
            n = int(self.sink.submit(fb.as_submit())) if len(fb) else 0
            n_ok += n
            self.n_datagrams += len(fb)
            self.n_bytes += fb.n_bytes
            self.n_rejected += len(fb) - n
        return n_ok

    def counters(self) -> dict:
        return self.sink.counters()

    def close(self, drain_timeout_s: float = 1.0) -> None:
        try:
            if hasattr(self.sink, "flush"):
                self.sink.flush(drain_timeout_s)
            if hasattr(self.sink, "stop"):
                self.sink.stop(drain_timeout_s)
        finally:
            self.sink.close()


def corr_routes(owner_half, dms, *, n_dest: int) -> dict[int, int]:
    """``{dm: destination}`` = search node ``owner_half[dm] // 2`` (the plan's
    explicit owner map; ``owner_half[d] = d % 8``)."""
    from dsart.transport.send_mmsg import routes_for_owner_halves  # noqa: PLC0415
    routes = routes_for_owner_halves([int(d) for d in dms], np.asarray(owner_half))
    bad = {d: n for d, n in routes.items() if not 0 <= n < n_dest}
    if bad:
        raise ValueError(f"routes outside the {n_dest} destinations: {bad}")
    return routes


def block_tx_budget(formats: list[StreamFormat], dm_t_native: dict[int, int],
                    routes: dict[int, int], n_dest: int) -> tuple[list[int], list[int]]:
    """Regular framing (no guarded rows) per destination: (datagram bytes,
    datagrams) per corr block."""
    nbytes = [0] * n_dest
    ndg = [0] * n_dest
    for d, tn in dm_t_native.items():
        S = BLOCK_NATIVE // int(tn)
        for f in formats:
            n = -(-S // f.max_k)
            nbytes[routes[d]] += n * f2.HEADER_BYTES + S * f.bytes_per_sample
            ndg[routes[d]] += n
    return nbytes, ndg


#: Token-bucket depth per destination of the 2x corr sender. G1 corner-turn sweep (13 corr ->
#: n01/n02, frame v2 8972 B): 512 KB lost 1-5.6% at every rate 9-18 Gb/s (rx_discards_phy, NIC
#: port-buffer overflow from incast microbursts), 64 KB was clean to 13 Gb/s, 16 KB had ZERO
#: discards up to the sender-limited 17.08 Gb/s per search node. (MmsgSender's class default,
#: 256 KB, discards.)
X2_TX_BURST_BYTES = 16 << 10
#: largest datagram the sender queues (slot size); a bucket shallower than one datagram never
#: accumulates enough tokens to send it
X2_TX_MAX_DATAGRAM = 9000


def block_tx_worst_datagrams(formats: list[StreamFormat], dm_t_native: dict[int, int], routes,
                              n_dest: int) -> list[int]:
    """Per destination, the most datagrams one corr block can need: every row its own
    datagram (each guarded row carries its own steps, so a level step that guards whole
    blocks for a few seconds multiplies the datagram count by up to ``max_k`` while the
    bytes barely change)."""
    ndg = [0] * n_dest
    for d, tn in dm_t_native.items():
        S = BLOCK_NATIVE // int(tn)
        ndg[routes[d]] += S * len(formats)
    return ndg


#: the sender's queue memory (slots x 9000 B x destinations) is bounded by this many slots
X2_TX_MAX_SLOTS = 1 << 15


def build_mmsg_sender(*, formats: list[StreamFormat], dm_t_native: dict[int, int], owner_half,
                      hosts: list[str], port: int, bind_host: str = "", cpu: int = -1,
                      pace: float = 1.25, burst_bytes: int = X2_TX_BURST_BYTES,
                      start: bool = True):
    """The 2x corr sender: one destination per search node (``hosts`` in
    ``SEARCH_NODES`` order) at ``port`` (= port_base + chgroup), explicit
    routes from the plan's owner map, paced at ``pace`` x the regular bytes
    per block with a ``burst_bytes`` token bucket per destination, queues
    sized for >= 2 blocks."""
    from dsart.transport.send_mmsg import MmsgSender  # noqa: PLC0415
    if int(burst_bytes) < X2_TX_MAX_DATAGRAM:
        raise ValueError(f"burst_bytes {burst_bytes} < one datagram ({X2_TX_MAX_DATAGRAM} B): "
                         "the paced sender would never send")
    n_dest = len(hosts)
    routes = corr_routes(owner_half, dm_t_native.keys(), n_dest=n_dest)
    nbytes, ndg = block_tx_budget(formats, dm_t_native, routes, n_dest)
    rates = [float(pace) * 8.0 * b / BLOCK_S for b in nbytes]
    # queues hold 2 blocks of the WORST datagram count (a guard storm), not of the regular one:
    # on 2026-10-29 a 16 s storm overflowed 4096 regular-sized slots (41k datagrams dropped)
    worst = block_tx_worst_datagrams(formats, dm_t_native, routes, n_dest)
    slots = 1024
    while slots < 2 * max(max(ndg, default=0), max(worst, default=0)) and slots < X2_TX_MAX_SLOTS:
        slots *= 2
    tx = MmsgSender([(h, int(port), r) for h, r in zip(hosts, rates)], routes=routes,
                    bind_host=bind_host, queue_slots=slots, burst_bytes=int(burst_bytes), cpu=int(cpu),
                    slot_bytes=X2_TX_MAX_DATAGRAM)
    LOG.info("x2 TX: %d destinations port %d, %.2f Gb/s paced total (x%.2f of %.2f Gb/s), "
             "burst %d B/destination, %d datagrams/block (worst %d/destination), queue %d slots "
             "(%.0f MB), bind %r",
             n_dest, port, sum(rates) / 1e9, pace, 8.0 * sum(nbytes) / BLOCK_S / 1e9,
             int(burst_bytes), sum(ndg), max(worst, default=0), slots,
             slots * X2_TX_MAX_DATAGRAM * n_dest / 1e6, bind_host or "any")
    if start:
        tx.start()
    return tx


def frame_counts(fb: FramedBlock) -> dict:
    """Small summary of one framed block (tests / logs)."""
    return {"datagrams": len(fb), "bytes": fb.n_bytes,
            "k": sorted(set(int(k) for k in fb.k))}
