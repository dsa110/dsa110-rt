"""Python wrapper for ``send_mmsg.c``: paced multi-destination UDP sender.

The 2x corr side hands every block's frame-v2 datagrams to
:meth:`MmsgSender.submit` in ONE call (contiguous buffer + offsets +
lengths, or a list of ``bytes``); a C pthread drains per-destination queues
with ``sendmmsg`` at a per-destination rate. ctypes releases the GIL for the
call, and no Python runs per datagram.

Routing is explicit: ``routes[dm] = destination index``, read from each
datagram's u16 ``dm_idx`` (byte 26 in frame v1 and v2). Datagrams of an
unrouted DM are dropped and counted.

    tx = MmsgSender([("10.41.0.205", 6625, 3.2e9), ("10.41.0.222", 6625, 3.2e9)],
                    routes={0: 0, 1: 0, 2: 1, 3: 1})
    tx.start()
    tx.submit(datagrams)          # list[bytes] or (buf, offsets, lengths)
    tx.flush(1.0)
    print(tx.counters())
    tx.close()
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

DM_OFFSET = 26
DEST_COUNTERS = ("enqueued", "sent", "bytes", "full_drops", "errors", "eagain",
                 "refused", "max_depth", "depth", "last_errno")
GLOBAL_COUNTERS = ("submitted", "accepted", "unrouted", "bad_length",
                   "sendmmsg_calls", "idle_sleeps")
MAX_DEST = 64


class TxmConfig(ctypes.Structure):
    _fields_ = [
        ("n_dest", ctypes.c_uint32), ("batch", ctypes.c_uint32),
        ("queue_slots", ctypes.c_uint32), ("slot_bytes", ctypes.c_uint32),
        ("burst_bytes", ctypes.c_uint64), ("so_sndbuf", ctypes.c_int32),
        ("cpu", ctypes.c_int32), ("dm_offset", ctypes.c_uint32),
        ("_pad", ctypes.c_uint32), ("bind_host", ctypes.c_char * 64),
    ]


class TxmDest(ctypes.Structure):
    _fields_ = [
        ("host", ctypes.c_char * 64), ("port", ctypes.c_uint16),
        ("_pad", ctypes.c_uint16 * 3), ("rate_bps", ctypes.c_double),
    ]


_LIB: Optional[ctypes.CDLL] = None


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    here = Path(__file__).resolve().parent
    abi = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    cands = sorted(here.glob(f"_send_mmsg.{abi}-*.so")) + sorted(here.glob("_send_mmsg*.so"))
    if not cands:
        raise RuntimeError(f"_send_mmsg*.so not found in {here}; run "
                           "`python setup.py build_ext --inplace`")
    lib = ctypes.CDLL(str(cands[0]))
    vp = ctypes.c_void_p
    lib.txm_create.restype = vp
    lib.txm_create.argtypes = [ctypes.POINTER(TxmConfig), ctypes.POINTER(TxmDest),
                               ctypes.c_char_p, ctypes.c_size_t]
    lib.txm_set_route.restype = ctypes.c_int
    lib.txm_set_route.argtypes = [vp, ctypes.c_uint32, ctypes.c_int32]
    lib.txm_set_rate.restype = ctypes.c_int
    lib.txm_set_rate.argtypes = [vp, ctypes.c_uint32, ctypes.c_double]
    lib.txm_get_route.restype = ctypes.c_int32
    lib.txm_get_route.argtypes = [vp, ctypes.c_uint32]
    lib.txm_submit.restype = ctypes.c_int64
    lib.txm_submit.argtypes = [vp, vp, vp, vp, ctypes.c_uint32, ctypes.c_double]
    lib.txm_start.restype = ctypes.c_int
    lib.txm_start.argtypes = [vp]
    lib.txm_stop.restype = ctypes.c_int
    lib.txm_stop.argtypes = [vp, ctypes.c_double]
    lib.txm_flush.restype = ctypes.c_int
    lib.txm_flush.argtypes = [vp, ctypes.c_double]
    lib.txm_counters.restype = None
    lib.txm_counters.argtypes = [vp, vp, vp]
    lib.txm_destroy.restype = None
    lib.txm_destroy.argtypes = [vp]
    lib.txm_abi.restype = ctypes.c_int
    lib.txm_abi.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32]
    abi_v = (ctypes.c_uint64 * 8)()
    n = lib.txm_abi(abi_v, 8)
    want = [ctypes.sizeof(TxmConfig), ctypes.sizeof(TxmDest), len(GLOBAL_COUNTERS) + 2,
            len(DEST_COUNTERS), MAX_DEST]
    if list(abi_v[:n]) != want:
        raise RuntimeError(f"send_mmsg ABI mismatch: {list(abi_v[:n])} != {want}")
    _LIB = lib
    return lib


def pack_datagrams(datagrams: Sequence[bytes]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``list[bytes]`` -> (contiguous uint8 buffer, u64 offsets, u32 lengths)."""
    lengths = np.fromiter((len(d) for d in datagrams), dtype=np.uint32, count=len(datagrams))
    offsets = np.zeros(len(datagrams), dtype=np.uint64)
    if len(datagrams) > 1:
        offsets[1:] = np.cumsum(lengths[:-1], dtype=np.uint64)
    buf = np.frombuffer(b"".join(datagrams), dtype=np.uint8) if datagrams else np.zeros(0, np.uint8)
    return buf, offsets, lengths


class MmsgSender:
    """Paced multi-destination sender (see module docstring).

    Args:
        destinations: ``[(ipv4, port, rate_bps), ...]``; ``rate_bps`` is bits/s
            of datagram bytes (UDP payload), 0 = unpaced.
        routes: ``{dm_idx: destination index}`` (more via :meth:`set_route`).
        bind_host: source IPv4 (the corr's data NIC); "" = any.
        batch: datagrams per ``sendmmsg``; queue_slots: per destination.
        burst_bytes: token-bucket depth per destination.
        cpu: pin the sender thread (-1 = no pinning).
    """

    def __init__(self, destinations: Sequence[tuple[str, int, float]], *,
                 routes: Optional[dict[int, int]] = None, bind_host: str = "",
                 so_sndbuf: int = 16 << 20, batch: int = 64, queue_slots: int = 4096,
                 slot_bytes: int = 9000, burst_bytes: int = 256 << 10,
                 dm_offset: int = DM_OFFSET, cpu: int = -1) -> None:
        lib = _lib()
        if not 1 <= len(destinations) <= MAX_DEST:
            raise ValueError(f"1..{MAX_DEST} destinations")
        cfg = TxmConfig(n_dest=len(destinations), batch=batch, queue_slots=queue_slots,
                        slot_bytes=slot_bytes, burst_bytes=burst_bytes, so_sndbuf=so_sndbuf,
                        cpu=cpu, dm_offset=dm_offset, bind_host=bind_host.encode())
        dests = (TxmDest * len(destinations))()
        for i, (host, port, rate) in enumerate(destinations):
            dests[i].host = host.encode()
            dests[i].port = int(port)
            dests[i].rate_bps = float(rate)
        err = ctypes.create_string_buffer(256)
        h = lib.txm_create(ctypes.byref(cfg), dests, err, len(err))
        if not h:
            raise OSError(f"txm_create failed: {err.value.decode(errors='replace')}")
        self._h = h
        self._lib = lib
        self.destinations = [tuple(d) for d in destinations]
        for dm, dest in (routes or {}).items():
            self.set_route(dm, dest)

    def set_route(self, dm: int, dest: int) -> None:
        if self._lib.txm_set_route(ctypes.c_void_p(self._h), int(dm), int(dest)) != 0:
            raise ValueError(f"bad route dm={dm} -> dest={dest}")

    def clear_route(self, dm: int) -> None:
        self.set_route(dm, -1)

    def set_rate(self, dest: int, rate_bps: float) -> None:
        """Change destination ``dest``'s pacing rate (bits/s; 0 = unpaced)."""
        if self._lib.txm_set_rate(ctypes.c_void_p(self._h), int(dest), float(rate_bps)) != 0:
            raise ValueError(f"bad destination {dest}")

    def route(self, dm: int) -> int:
        return int(self._lib.txm_get_route(ctypes.c_void_p(self._h), int(dm)))

    def start(self) -> None:
        if self._lib.txm_start(ctypes.c_void_p(self._h)) != 0:
            raise RuntimeError("txm_start failed")

    def submit(self, datagrams, *, block_timeout_s: float = 0.0) -> int:
        """Queue datagrams (``list[bytes]`` or ``(buf, offsets, lengths)``);
        returns how many were accepted (the rest are counted as drops)."""
        if isinstance(datagrams, tuple) and len(datagrams) == 3 and isinstance(datagrams[0], np.ndarray):
            buf, offs, lens = datagrams
        else:
            buf, offs, lens = pack_datagrams(list(datagrams))
        buf = np.ascontiguousarray(buf, dtype=np.uint8)
        offs = np.ascontiguousarray(offs, dtype=np.uint64)
        lens = np.ascontiguousarray(lens, dtype=np.uint32)
        if offs.shape != lens.shape:
            raise ValueError("offsets and lengths must have the same length")
        if lens.size and int((offs + lens).max()) > buf.size:
            raise ValueError("a datagram extends past the buffer")
        return int(self._lib.txm_submit(ctypes.c_void_p(self._h), buf.ctypes.data,
                                        offs.ctypes.data, lens.ctypes.data, int(lens.size),
                                        float(block_timeout_s)))

    def flush(self, timeout_s: float = 1.0) -> bool:
        """Wait until every queue is empty (True) or ``timeout_s`` (False)."""
        return self._lib.txm_flush(ctypes.c_void_p(self._h), float(timeout_s)) == 0

    def counters(self) -> dict:
        g = (ctypes.c_uint64 * 8)()
        n = len(self.destinations)
        d = (ctypes.c_uint64 * (n * len(DEST_COUNTERS)))()
        self._lib.txm_counters(ctypes.c_void_p(self._h), g, d)
        per = []
        for i in range(n):
            row = d[i * len(DEST_COUNTERS):(i + 1) * len(DEST_COUNTERS)]
            c = dict(zip(DEST_COUNTERS, (int(x) for x in row)))
            c["last_errno"] = int(np.int64(np.uint64(c["last_errno"])))
            c["dest"] = f"{self.destinations[i][0]}:{self.destinations[i][1]}"
            per.append(c)
        out = dict(zip(GLOBAL_COUNTERS, (int(x) for x in g[:len(GLOBAL_COUNTERS)])))
        out["per_dest"] = per
        return out

    def stop(self, drain_timeout_s: float = 1.0) -> None:
        if self._h:
            self._lib.txm_stop(ctypes.c_void_p(self._h), float(drain_timeout_s))

    def close(self) -> None:
        if self._h:
            self._lib.txm_destroy(ctypes.c_void_p(self._h))
            self._h = 0

    def __enter__(self) -> "MmsgSender":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:          # best effort; prefer close()
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


class TxRateMeter:
    """Turns successive :meth:`MmsgSender.counters` snapshots into the small
    monitoring dict WS-D reads from ``/mon/corr_rt/<chgroup>/corr_fast``:
    ``tx_gbps`` (datagram bytes), ``tx_datagrams_per_s``, ``tx_drops_per_s``
    (queue-full + send errors + unrouted) and ``tx_queue_depth_max``."""

    def __init__(self) -> None:
        self._prev = None

    def update(self, counters: dict, now: float | None = None) -> dict | None:
        import time as _time
        t = _time.monotonic() if now is None else float(now)
        snap = (t, sum(d["bytes"] for d in counters["per_dest"]),
                sum(d["sent"] for d in counters["per_dest"]),
                sum(d["full_drops"] + d["errors"] for d in counters["per_dest"]) + counters["unrouted"])
        prev, self._prev = self._prev, snap
        if prev is None:
            return None
        dt = max(1e-9, snap[0] - prev[0])
        return {"tx_gbps": round((snap[1] - prev[1]) * 8 / dt / 1e9, 4),
                "tx_datagrams_per_s": round((snap[2] - prev[2]) / dt, 1),
                "tx_drops_per_s": round((snap[3] - prev[3]) / dt, 2),
                "tx_queue_depth_max": max((d["depth"] for d in counters["per_dest"]), default=0)}


def routes_for_owner_halves(dms: Iterable[int], owner_half: Sequence[int],
                            node_of_half=lambda h: h // 2) -> dict[int, int]:
    """``{d: search-node index}`` for the plan's owner map (half ``d % 8`` ->
    node ``half // 2``); pair with one destination per search node."""
    return {int(d): int(node_of_half(int(owner_half[int(d)]))) for d in dms}


__all__ = ["MmsgSender", "TxRateMeter", "pack_datagrams", "routes_for_owner_halves", "DEST_COUNTERS",
           "GLOBAL_COUNTERS", "DM_OFFSET"]
