"""DSA-110 real-time pipeline subpackage: trigger.

Per plan §3 line 97 + §4.4 lines 1669-1718 + M5 PARALLEL_AGENTS.md
§3 Class A ownership. The package decomposes as:

  - ``ndjson_codec.py`` — TriggerPacket / TriggerAck JSON codec.
  - ``predicate.py``    — TriggerCondition Protocol, TriggerContext,
                          TriggerDecision, evaluate_chain.
  - ``conditions/``     — one file per TriggerCondition (SnrThreshold,
                          PerCubePerKernelCap, PerCubeTotalCap,
                          RateLimitTokenBucket; future:
                          LearnedClassifier, PointingExclusion, ...).
  - ``holdoff.py``      — single-process per-(l, m, kernel) holdoff
                          state machine (the production posix-shm
                          mmap-shared form is wired by Chunk-6
                          search-rx).
  - ``emitter.py``      — async TCP fan-out + ACK demux + in-flight
                          tracker.
  - ``mock_listener.py``— asyncio TCP server for unit tests + the
                          Chunk-5 cube_injection_detector bench.

Deferred to M6 (out of M5 scope):
  - ``listener.py``     — corr-side TCP listener (hosted in
                          ``corr_fast_compute``).
  - ``dedup_cache.py``  — corr-side dedup cache.
  - ``ctrltrigger_bridge.py`` — operator etcd → trigger bridge.
"""

from .emitter import (
    DEFAULT_BACKOFF_CAP_S,
    DEFAULT_BACKOFF_INITIAL_S,
    DEFAULT_COMPLETION_TIMEOUT_S,
    DEFAULT_TX_QUEUE_DEPTH,
    ConnectionEndpoint,
    ConnState,
    EmitRecord,
    InFlightTracker,
    TriggerEmitter,
    TriggerEmitterConfig,
)
from .holdoff import (
    DEFAULT_HOLDOFF_MS,
    DEFAULT_LM_ROUND,
    HoldoffStateMachine,
    HoldoffCellKey,
    make_cell_key,
)
from .mock_listener import (
    MockListenerConfig,
    MockTriggerListener,
    MockTriggerListenerFan,
    ReceivedRecord,
)
from .ndjson_codec import (
    decode_ack,
    decode_packet,
    encode_ack,
    encode_packet,
    iter_lines,
    split_ndjson_buffer,
)
from .predicate import (
    TriggerCondition,
    TriggerContext,
    TriggerDecision,
    evaluate_chain,
)

__all__ = [
    "ConnectionEndpoint",
    "ConnState",
    "DEFAULT_BACKOFF_CAP_S",
    "DEFAULT_BACKOFF_INITIAL_S",
    "DEFAULT_COMPLETION_TIMEOUT_S",
    "DEFAULT_HOLDOFF_MS",
    "DEFAULT_LM_ROUND",
    "DEFAULT_TX_QUEUE_DEPTH",
    "EmitRecord",
    "HoldoffCellKey",
    "HoldoffStateMachine",
    "InFlightTracker",
    "MockListenerConfig",
    "MockTriggerListener",
    "MockTriggerListenerFan",
    "ReceivedRecord",
    "TriggerCondition",
    "TriggerContext",
    "TriggerDecision",
    "TriggerEmitter",
    "TriggerEmitterConfig",
    "decode_ack",
    "decode_packet",
    "encode_ack",
    "encode_packet",
    "evaluate_chain",
    "iter_lines",
    "make_cell_key",
    "split_ndjson_buffer",
]
