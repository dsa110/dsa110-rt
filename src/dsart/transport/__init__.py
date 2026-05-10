"""DSA-110 real-time pipeline transport plane (M3 chunk 8 + M4a).

Chunk 8 (this milestone) ships the **fast-vis cube TX/RX layer** with a
loopback capture test:

* :class:`FastVisFrame` — 32-byte header + payload wire format
  (:mod:`dsart.transport.frame`).
* :class:`TransportTx` — UDP unicast transmitter, satisfies the
  chunk-4 ``TransportTxStage`` Protocol
  (:mod:`dsart.transport.tx`).
* :class:`TransportRx` — UDP receiver with magic + CRC validation +
  per-(chgroup) sequence-gap accounting
  (:mod:`dsart.transport.rx`).
* :class:`LoopbackCaptureService` — headless capture-and-persist
  service for the loopback bench + the M3 voltage-fixture sub-DoDs
  (:mod:`dsart.transport.loopback_capture`).

M4a will extend this with the production 72-byte header
(``pattern_id``, ``scale``, ``offset``, fragment reassembly) per plan
§4.3 — the chunk-8 ``FastVisFrame`` is the simpler peer kept for
intra-host loopback testing.
"""

from dsart.transport.frame import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    DTYPE_CFP16,
    DTYPE_CINT8,
    FLAG_RFI_WARMING_UP,
    HEADER_BYTES,
    MAGIC,
    FastVisFrame,
    FrameCRCError,
    FrameMagicError,
    FramePayloadOversizeError,
)
from dsart.transport.loopback_capture import LoopbackCaptureService
from dsart.transport.rx import RxStats, TransportRx
from dsart.transport.tx import TransportTx


__all__ = [
    "DEFAULT_MAX_PAYLOAD_BYTES",
    "DTYPE_CFP16",
    "DTYPE_CINT8",
    "FLAG_RFI_WARMING_UP",
    "HEADER_BYTES",
    "MAGIC",
    "FastVisFrame",
    "FrameCRCError",
    "FrameMagicError",
    "FramePayloadOversizeError",
    "LoopbackCaptureService",
    "RxStats",
    "TransportRx",
    "TransportTx",
]
