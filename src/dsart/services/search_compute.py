"""Search-node compute service entry (RX ring → detector → cluster → dump).

M5 chunk 6b-α + M6 chunk 5: long-running asyncio orchestrator that
drives the ``CubePipeline`` from a pluggable ``RxRingSource``, then
hands per-cube candidates to:

  * the M6 :class:`ClustererService` (HDBSCAN/DBSCAN; ThreadPoolExecutor);
  * the M6 :class:`BrightPulsePredicate` + :class:`CubeDumpWriter`
    (bright-pulse cluster predicate → NPZ dump on a writer thread);
  * the M6 :class:`UdpTriggerListener` (one-shot "dump next cube" flag);
  * the M6 :class:`CandsLogger` (T1 per-candidate + T2 per-cluster
    hourly-rotated ASCII rows).

All four sub-systems are *optional*. If their config is ``None`` the
service still processes cubes (for unit-test wiring + the M5-only
detector path). Production wires all four.

Per-cube driver order (M6 D7 / D9 / chunk 5):

  1. ``slot = await source.acquire()``
  2. ``geom = self._geom_from_slot(slot)``
  3. ``udp_armed = listener.consume_dump_next_cube_flag()``  ← BEFORE
     pipeline.process so we can decide whether to dump cube N before
     committing CPU/GPU work that may release the cube tensor.
  4. ``result = pipeline.process(slot)``
  5. If ``udp_armed``: ``cube_dump.submit(result.cube, manifest_udp)``
  6. ``cluster_result = clusterer.submit(...).result()``  ← blocks
     up to a configurable timeout (default = cube cadence). The chunk-6
     bench (`bench/clusterer_throughput.py`) gates on the p99 budget so
     this should never time out at production load.
  7. For each cluster record: if predicate fires, submit auto dump.
  8. ``cands_logger.write_cube(...)`` writes T1 + T2 rows.
  9. ``await source.release(slot.cube_id)``.

Production path runs ``main()`` from a CLI; benches subclass / wire
the service directly to inject a synthetic source. Per M-defer (plan
§M-defer) the M5 trigger emitter is retired; voltage triggering is
delegated to the corr-side ``dsa110-xengine`` framework.
"""

from __future__ import annotations

import argparse
import asyncio
import heapq
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch

from ..cluster import (
    CandsLogger,
    CandsLoggerConfig,
    ClustererConfig,
    ClustererService,
)
from ..coinc.wire import build_header
from ..common.constants import (
    CUBE_CADENCE_SAMPLES_DEFAULT,
    DELTA_NU_CH_GHZ,
    DETECTOR_IMAGE_KERNELS,
    DETECTOR_DM_KERNELS,
    DETECTOR_TIME_KERNELS,
    K_DM_MS_GHZ2_PC,
    NU_BOT_PROC_GHZ,
    NU_TOP_PROC_GHZ,
    T_INT_SEARCH_US_DEFAULT,
)
from ..common.contracts import Candidate, CubeGeometry
from ..detector.forward import DeterministicDetector
from ..detector.kernels import build_kernel_bank
from ..detector.merger import MergerConfig
from ..dump import (
    BrightPulsePredicate,
    BrightPulsePredicateConfig,
    C2TriggerListener,
    C2TriggerListenerConfig,
    CubeDumpWriter,
    CubeDumpWriterConfig,
    UdpTriggerListener,
    UdpTriggerListenerConfig,
)
from ..common.contracts import CubeDumpManifest
from ..noise_norm.layer1 import Layer1State
from .c1_emit import C1EmitConfig, C1TcpEmitter, candidate_to_c1_row
from .search_ring_mon import SearchRingMonPublisher
from .search_compute_mon import SearchComputeMonPublisher
from .cube_pipeline import (
    CubePipeline,
    CubePipelineConfig,
    CubeRetentionRing,
    PrefetchedCube,
)
from .rx_ring import CubeRingSlot, RxRingSource

__all__ = [
    "SearchComputeConfig",
    "SearchComputeService",
    "meter_candidates",
    "main",
]

#: M7.6 C1→C2 metering: number of cubes (blocks) to average before
#: publishing the metering rollup to etcd. 16 ≈ 2.1 s at 7.45 cubes/s,
#: keeping the etcd/influx PUT rate low (per the operator's request).
_METER_WINDOW_BLOCKS = 16


def meter_candidates(
    candidates: List[Candidate],
    cap: Optional[int],
) -> "tuple[List[Candidate], int]":
    """C1→C2 metering selection: keep at most ``cap`` candidates, ordered
    narrow-first (``width_samples`` ascending) then bright-first (``snr``
    descending). Returns ``(kept, n_dropped)``.

    ``cap`` of ``None`` / ``<= 0`` disables metering (returns the input
    unchanged with ``n_dropped == 0``). RT-safe: when at or under the cap
    we return immediately without sorting; above the cap we use
    ``heapq.nsmallest`` (O(k log cap), cap ≪ k during RFI floods) so the
    hot loop never pays an O(k log k) full sort. Selection is a no-op on
    the candidates' identity/order when not over the cap, so cube dump +
    retention (which run off the full candidate list upstream) are
    unaffected — this only bounds what is shipped to C2."""
    if cap is None or cap <= 0:
        return candidates, 0
    n = len(candidates)
    if n <= cap:
        return candidates, 0
    kept = heapq.nsmallest(
        int(cap),
        candidates,
        key=lambda c: (int(c.width_samples), -float(c.snr)),
    )
    return kept, n - len(kept)


# corr_fast F33 op-point: fine channels collapsed into one effective
# channel BEFORE dedispersion (``CorrFastConfig.chan_sum_factor``). The
# intra-summed-channel dispersion smearing — the irreducible pulse
# broadening a genuine dispersed burst suffers — scales with this.
_DEDISP_CHAN_SUM_FACTOR = 8


def dm_smear_samples(
    dm_pc_cc: float,
    *,
    chan_sum_factor: int = _DEDISP_CHAN_SUM_FACTOR,
    t_search_us: float = T_INT_SEARCH_US_DEFAULT,
) -> float:
    """Band-averaged intra-(summed-)channel dispersion smearing at
    ``dm_pc_cc``, in search-sample units.

    A real cold-plasma-dispersed burst at this DM cannot be narrower
    than ~this many samples: each summed channel (``chan_sum_factor`` ×
    the native 30.5 kHz channel) smears the pulse by the differential
    dispersion delay across its own width, an irreducible floor that
    incoherent dedispersion does not remove. Verified against the
    codebase's own characterisation (corr_fast note: ~2.7 ms at DM 3000,
    ν = 1.31 GHz). Returns 0.0 for non-positive DM. Pure + RT-cheap."""
    if dm_pc_cc <= 0.0:
        return 0.0
    dnu = DELTA_NU_CH_GHZ * float(chan_sum_factor)  # summed-channel BW (GHz)

    def _smear_ms(nu: float) -> float:
        lo = nu - dnu / 2.0
        hi = nu + dnu / 2.0
        return K_DM_MS_GHZ2_PC * dm_pc_cc * (lo ** -2 - hi ** -2)

    avg_ms = 0.5 * (_smear_ms(NU_BOT_PROC_GHZ) + _smear_ms(NU_TOP_PROC_GHZ))
    return avg_ms * 1000.0 / float(t_search_us)


def filter_unphysical_narrow(
    candidates: List[Candidate],
    floor_frac: Optional[float],
    *,
    t_search_us: float = T_INT_SEARCH_US_DEFAULT,
) -> "tuple[List[Candidate], int]":
    """Drop candidates far narrower than the DM-smearing floor permits.

    A candidate is rejected when its boxcar ``width_samples`` is below
    ``floor_frac × dm_smear_samples(dm_fine, t_search_us=...)`` — i.e.
    much narrower than intra-channel dispersion smearing allows for its
    DM. Returns ``(kept, n_dropped)``.

    ``t_search_us`` MUST be the actual detection-sample period (the
    production op-point is 1048.576 µs = 32 native samples; the service
    passes the per-cube ``geom.sample_period_us``). At that cadence the
    smearing floor at DM≈2500 is ~1.8 samples, so width-2 high-DM
    detections are *consistent* with smearing and only clearly-narrow
    (width-1) high-DM detections are rejected — the safely-discardable
    set. (The decisive artifact discriminator is spatial/temporal
    incoherence, handled by the trigger criteria + dump-rate cap, not by
    width alone.)

    ``floor_frac`` of ``None`` / ``<= 0`` disables the filter. RT-safe:
    a single linear pass with no sort/heap. The floor is sub-sample at
    low DM, so low-DM candidates are never affected (genuine narrow
    low-DM events pass untouched)."""
    if floor_frac is None or floor_frac <= 0.0 or not candidates:
        return candidates, 0
    kept = [
        c
        for c in candidates
        if float(c.width_samples) >= floor_frac * dm_smear_samples(
            float(c.dm_fine), t_search_us=t_search_us,
        )
    ]
    return kept, len(candidates) - len(kept)


_LOG = logging.getLogger("dsart.services.search_compute")


# ---------------------------------------------------------------------------
# Service config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SearchComputeConfig:
    """Static (per-process-lifetime) service config (plan §9 O-4 +
    ``configs/config_compute_search.yaml`` + M6 chunks 1-5 sub-system
    configs).

    The M5 chunk-6b-α scope ships fields needed by the pipeline +
    Layer-1 state. The M6 chunk-5 wiring adds optional sub-system
    configs (clusterer, cube_dump, udp_listener, cands_logger) and the
    cube-geometry hyperparameters needed to construct a
    :class:`CubeGeometry` per cube.

    Sub-system gating: each ``*_config`` field defaults to ``None``.
    If ``None`` the corresponding sub-system is not instantiated and
    its per-cube hook is a no-op. Tests that only exercise the
    detector path leave them all None; production wires all four.

    Args:
        pipeline: ``CubePipelineConfig`` (M5 chunk 6a).
        n_fdm: full-cube fine-DM count (M5 chunk 6b).
        detector_*: detector knobs (M5 chunk 4-6).
        layer1_*: Layer-1 σ-clip knobs (M5 chunk 3).
        search_node_id, gpu_half: process identity.
        cube_cell_l_rad, cube_cell_m_rad: gridder cell pitch in radians
            (production: derived from λ / (n_grid · cell_λ); for v1
            we let the operator pin a constant per-deployment).
        cube_l0_rad, cube_m0_rad: (l, m) at pixel index 0. Default 0.0.
        cube_sample_period_us: time between adjacent samples in µs
            (defaults to ``T_INT_SEARCH_US_DEFAULT`` from constants).
        cube_sample_period_specnum: spec-nums per detector sample
            (default 16 = ``t_int_search_us / t_int_fast_us`` at
            production ops point).
        mjd_at_specnum_0: MJD at specnum 0 of this run; ``mjd_start``
            for cube N is computed as ``mjd_at_specnum_0 +
            (specnum_start * t_int_fast_us / 1e6 / 86400)``. v1
            placeholder is 0.0; production must override.
        fine_dm_pc_cc_full: full fine-DM grid (``DmPlan.fine_dm``);
            required if ``clusterer_config`` is set so the per-cube
            slice can be built. ``None`` defers to a synthetic linear
            grid (test convenience).
        clusterer_config / cube_dump_writer_config /
        bright_pulse_predicate_config / udp_trigger_listener_config /
        cands_logger_config: M6 chunk 1-4 sub-system configs.
        cluster_result_timeout_s: max seconds to block on the cluster
            future per cube. Default 1.0 s (well above the chunk-6
            bench's p99 = 50 ms gate).
    """

    pipeline: CubePipelineConfig
    n_fdm: int
    detector_threshold_sigma: float = 8.0
    detector_dtype: torch.dtype = torch.float16
    detector_device: str = "cpu"
    detector_version: str = "v1.M5"
    detector_streaming: bool = True
    detector_streaming_tile_size: int = 64
    detector_streaming_decoder_n_top: int = 64
    detector_image_tokens: tuple[str, ...] = ("unit",)
    detector_dm_tokens: tuple[str, ...] = ("d1",)
    detector_time_tokens: tuple[str, ...] = (
        "b1", "b2", "b4", "b8", "b16", "b32", "b64",
    )
    detector_boxcar_accum_dtype: Optional[torch.dtype] = None
    detector_layer2_max_samples: Optional[int] = 100_000
    # M7.4 hardening: Layer-2 σ_k EMA knobs exposed for tuning. The
    # detector defaults are sensible at production geometry, but
    # cube_cadence_s + tau_s + n_burnin all interact via
    # ``gamma = 1 - exp(-cadence/tau)`` and the 250924mptq replay
    # showed σ_k can be depressed by sparse-valid cubes (one node
    # snowballed to SNR ~ 4.7e4). ``layer2_sigma_floor`` clamps σ_k
    # from below; ``layer2_valid_min_fraction`` relaxes the strict
    # 100% validity gate so σ_k can keep learning when 1–2% of
    # UV cells are flagged.
    detector_cube_cadence_s: Optional[float] = None  # None ⇒ detector default
    detector_layer2_tau_s: Optional[float] = None
    detector_layer2_n_burnin: Optional[int] = None
    detector_layer2_sigma_floor: float = 0.0
    detector_layer2_valid_min_fraction: float = 1.0
    pipeline_overlap: bool = False
    search_node_id: int = 1
    gpu_half: int = 1
    layer1_n_burnin_cubes: int = 5
    layer1_n_sigma: float = 3.0
    layer1_n_iterations: int = 3
    # M7.2 perf gate: Layer-1 σ-clip cap dropped 100K → 10K. Std-error
    # on σ̂ scales as σ/√(2N): 10K samples ⇒ 7e-3 σ, well below the
    # cube-to-cube EMA noise floor (≈ a few percent). Trade saved ~50 ms
    # of `torch.nanmedian` per cube at production geometry.
    layer1_max_samples: Optional[int] = 10_000
    # M7.4: σ floor for Layer-1 to suppress the static-sky-EMA-warmup
    # transient (cubes 1-2 of a fresh start can drop σ ~200× below
    # steady state, producing massive false-positive SNRs). 0.0 ⇒
    # disabled (legacy default). Production replay default 5e-3 sits
    # between the warmup-transient σ (~10⁻⁴) and the steady-state
    # noise floor (~2×10⁻²) and was empirically validated against
    # the 250924mptq burst.
    layer1_sigma_floor: float = 0.0

    # --- M6 chunk 5: cube-geometry hyperparameters ---------------------
    cube_cell_l_rad: float = 1.5e-4
    cube_cell_m_rad: float = 1.5e-4
    cube_l0_rad: float = 0.0
    cube_m0_rad: float = 0.0
    cube_sample_period_us: float = 131.072
    cube_sample_period_specnum: int = 16
    mjd_at_specnum_0: float = 0.0
    fine_dm_pc_cc_full: Optional[np.ndarray] = None

    # --- M6 chunk 5: sub-system configs (all optional) ----------------
    clusterer_config: Optional[ClustererConfig] = None
    cube_dump_writer_config: Optional[CubeDumpWriterConfig] = None
    bright_pulse_predicate_config: Optional[BrightPulsePredicateConfig] = None
    udp_trigger_listener_config: Optional[UdpTriggerListenerConfig] = None
    cands_logger_config: Optional[CandsLoggerConfig] = None

    # --- M6 chunk 5: orchestration knobs ------------------------------
    cluster_result_timeout_s: float = 1.0

    # --- M7.4 C1 stage ------------------------------------------------
    # ``enable_legacy_clusterer`` gates the legacy DBSCAN/HDBSCAN
    # clusterer + cands_logger + BrightPulsePredicate path. Default
    # False per M7.4 design (C1 emit replaces per-node clustering); set
    # True in offline tools / replay benches that still want T1/T2 rows.
    enable_legacy_clusterer: bool = False
    # C1 cross-kernel merger geometry (see MergerConfig defaults).
    merger_config: Optional[MergerConfig] = None
    # Single SNR knob; when set, overrides ``detector_threshold_sigma``
    # and is also enforced defensively at the C1 emit boundary.
    c1_snr_min: Optional[float] = None
    # Depth of the pinned-host cube retention ring (one slot per cube;
    # 1 slot ≈ 3.2 GiB at the production t_det=192 / n_fdm=34 /
    # n_grid=256 / fp16 geometry, so 8 slots ≈ 26 GiB per gpu_half).
    cube_ring_depth: int = 8
    # C1 → C2 TCP emitter. None disables the emitter (tests + offline
    # benches that don't need it).
    c1_emit_config: Optional[C1EmitConfig] = None
    # C2 → C1 UDP trigger listener. None disables the listener.
    c2_trigger_listener_config: Optional[C2TriggerListenerConfig] = None
    # Cube dump root for the C2-triggered dumps (per-event subdir is
    # built by the listener). When ``c2_trigger_listener_config`` is
    # set this MUST match its ``dump_root`` so the listener and the
    # writer agree on where files land.
    c1_dump_root: Optional[Path] = None

    # --- M7.4 cube uploader (search-node → h23 rsync) -----------------
    # When both ``cube_upload_dest_host`` and ``cube_upload_dest_root``
    # are set the ``CubeDumpWriter`` fires an ``rsync`` after every
    # successful per-event NPZ write. ``None`` (default) disables the
    # uploader (legacy / bench paths). Production wires it from
    # ``c1.uploader.remote_root`` in ``dsart_search_rt.yaml``.
    cube_upload_dest_host: Optional[str] = None
    cube_upload_dest_root: Optional[str] = None
    cube_upload_bandwidth_limit_kbps: int = 0


# ---------------------------------------------------------------------------
# SearchComputeService
# ---------------------------------------------------------------------------


class SearchComputeService:
    """Asyncio orchestrator. One service instance per (search-node, GPU-half).

    Args:
        config: ``SearchComputeConfig`` resolved from yaml + CLI.
        source: any ``RxRingSource``; M4a in production, synthetic in
            benches.
        detector: optional pre-constructed ``DeterministicDetector``
            (lets benches inject a seeded detector). Default: build
            one from ``config``.
        layer1_state: optional pre-constructed ``Layer1State`` (same).
    """

    def __init__(
        self,
        config: SearchComputeConfig,
        source: RxRingSource,
        *,
        detector: Optional[DeterministicDetector] = None,
        layer1_state: Optional[Layer1State] = None,
    ) -> None:
        self._config = config
        self._source = source
        self._detector = detector or self._build_detector(config)
        self._layer1_state = layer1_state or Layer1State(
            n_fdm=config.n_fdm,
            n_burnin_cubes=config.layer1_n_burnin_cubes,
            n_sigma=config.layer1_n_sigma,
            n_iterations=config.layer1_n_iterations,
            max_samples=config.layer1_max_samples,
            sigma_floor=config.layer1_sigma_floor,
        )
        # M7.4: thread the fine-DM grid (pc/cc) into the pipeline so
        # the decoder writes the physical DM (pc/cc) into
        # ``Candidate.dm_fine`` instead of falling back to the fdm INDEX.
        # The C1 emitter ships ``dm_pc_cc=cand.dm_fine`` downstream, so
        # this is what makes the C2 trigger criteria (``dm_median_min_pc_cc``
        # etc.) compare apples to apples.
        fine_dm_tensor: Optional[torch.Tensor] = None
        if config.fine_dm_pc_cc_full is not None:
            fine_dm_tensor = torch.as_tensor(
                config.fine_dm_pc_cc_full,
                dtype=torch.float32,
                device=torch.device(config.pipeline.device),
            )
        self._pipeline = CubePipeline(
            config=config.pipeline,
            detector=self._detector,
            layer1_state=self._layer1_state,
            fine_dm_pc_cm3=fine_dm_tensor,
        )
        self._stopping = asyncio.Event()
        self._cubes_processed = 0
        self._candidates_emitted = 0
        self._clusters_emitted = 0
        self._auto_dumps_dispatched = 0
        self._udp_dumps_dispatched = 0
        self._cluster_timeouts = 0
        # M7.4 RT-perf debug: per-stage wall accumulator (ns); reset by
        # the cube_progress logger so the printed mean is over the
        # last block.
        self._stage_ns_accum: dict[str, int] = {
            "build_cube": 0, "layer1_norm": 0,
            "detector_forward": 0, "total": 0,
        }
        self._stage_ns_count: int = 0

        # Sub-systems — built in start() if their config is non-None.
        self._clusterer: Optional[ClustererService] = None
        self._predicate: Optional[BrightPulsePredicate] = None
        self._cube_dump: Optional[CubeDumpWriter] = None
        self._udp_listener: Optional[UdpTriggerListener] = None
        self._cands_logger: Optional[CandsLogger] = None
        # M7.4 C1 stage.
        self._cube_ring: Optional[CubeRetentionRing] = None
        self._c1_emit: Optional[C1TcpEmitter] = None
        self._c1_emit_task: Optional[asyncio.Task] = None
        self._c2_trigger: Optional[C2TriggerListener] = None
        self._c1_batches_submitted = 0
        self._c1_batches_dropped = 0
        # M7.6: cumulative candidates dropped pre-transmit by the C1→C2
        # width cap (c1.max_c1c2_width_samples).
        self._c1_cands_dropped_width = 0
        # 2026-05-30: cumulative candidates dropped pre-transmit by the
        # DM-smearing-floor filter (c1.dm_width_floor_frac) — unphysically
        # narrow high-DM detections (impulsive RFI on a high-DM trial).
        self._c1_cands_dropped_dmfloor = 0
        # M7.6: C1→C2 metering (cap candidates/block, narrow-then-bright).
        # ``_c1_cands_dropped_meter`` is cumulative; the ``_meter_*`` window
        # accumulators roll up every ``_METER_WINDOW_BLOCKS`` cubes into a
        # single low-rate etcd publish (see ``_publish_c1_metering``).
        self._c1_cands_dropped_meter = 0
        self._meter_window_blocks = 0
        self._meter_window_metered_blocks = 0
        self._meter_window_dropped_sum = 0
        self._meter_window_dropped_max = 0
        self._meter_window_cands_sum = 0
        self._compute_mon: Optional[SearchComputeMonPublisher] = (
            SearchComputeMonPublisher(
                search_node_id=int(config.search_node_id),
                gpu_half=int(config.gpu_half),
            )
        )
        # M7.4 Phase 6c: publish cube_ring window to etcd so the dsa_monitor
        # "Dump Now" button can pick an event_specnum that lands inside the
        # search-side retention window (corr_fast's block_specnum_start is in
        # a different domain — see ``search_ring_mon`` docstring).
        self._ring_mon: Optional[SearchRingMonPublisher] = (
            SearchRingMonPublisher(
                search_node_id=int(config.search_node_id),
                gpu_half=int(config.gpu_half),
            )
        )

        # M7.4 fix (2026-05-27): when ``config.mjd_at_specnum_0`` is the
        # placeholder default (0.0), every restart of search_compute
        # would tag cubes with ``mjd_start ≈ 0`` (= MJD 0 = year 1858),
        # which collapses every restart's batches into the same
        # signal-time window on the C2 side and creates an immortal
        # cluster. ``_geom_from_slot`` checks this override first; the
        # value is latched once on the very first cube using the
        # wall-clock at that moment minus the first cube's specnum
        # offset (good to a few ms — well under C2's window of 5 s).
        # An operator that sets ``mjd_at_specnum_0`` explicitly (i.e.
        # not 0.0) wins; the override is only filled when the cfg is
        # at the placeholder.
        self._mjd_at_specnum_0_override: Optional[float] = None

    @staticmethod
    def _build_cube_uploader_callback(
        config: SearchComputeConfig,
    ):
        """Return a ``CubeDumpWriter`` post-write callback that fires
        ``rsync`` to the configured h23 destination, or ``None`` if
        the uploader is disabled.

        The callback signature is ``(path, manifest)``. The event name
        is extracted from ``path.parent.name`` — by contract the C2
        trigger listener composes
        ``${c1.dump_root}/<event_name>/cube_s<sid>_g<g>_<spec>.npz``
        so the parent directory IS the event archive name. If the
        path's parent IS the writer's flat ``dump_root`` (legacy M6
        auto / udp paths with no event subdir), the upload is skipped
        — no event archive to populate on h23.
        """
        dest_host = config.cube_upload_dest_host
        dest_root = config.cube_upload_dest_root
        if not dest_host or not dest_root:
            return None
        # Local import keeps the dump-writer module decoupled from the
        # coinc package; the import happens once at service start.
        from ..coinc.cube_uploader import upload_event_cubes

        # Resolve the flat ``dump_root`` once so the callback can skip
        # legacy non-per-event writes without re-resolving each call.
        flat_dump_root: Optional[Path] = None
        if config.cube_dump_writer_config is not None:
            try:
                flat_dump_root = Path(
                    config.cube_dump_writer_config.dump_root
                ).resolve()
            except (OSError, RuntimeError):
                flat_dump_root = Path(
                    config.cube_dump_writer_config.dump_root
                )
        bwlimit = int(config.cube_upload_bandwidth_limit_kbps or 0)

        def _on_dump_complete(path: Path, manifest) -> None:  # noqa: ANN001
            event_dir = Path(path).parent
            try:
                resolved = event_dir.resolve()
            except (OSError, RuntimeError):
                resolved = event_dir
            if flat_dump_root is not None and resolved == flat_dump_root:
                # Legacy flat write (no event subdir) — nothing to upload.
                return
            event_name = event_dir.name
            if not event_name:
                _LOG.warning(
                    "cube_uploader: skipping empty event_name for path=%s",
                    path,
                )
                return
            upload_event_cubes(
                event_name=event_name,
                src_dir=event_dir,
                dest_host=dest_host,
                dest_root=dest_root,
                bandwidth_limit_kbps=bwlimit,
            )

        return _on_dump_complete

    @staticmethod
    def _build_detector(config: SearchComputeConfig) -> DeterministicDetector:
        # Pass ``device=`` to the constructor (NOT ``.to(device)`` after-
        # the-fact) so the internal ``Layer2State._s_k`` is allocated
        # directly on the target device (see M5 chunk 6 lock).
        bank = build_kernel_bank(
            image_tokens=config.detector_image_tokens,
            dm_tokens=config.detector_dm_tokens,
            time_tokens=config.detector_time_tokens,
            dtype=config.detector_dtype,
        )
        # Layer-2 EMA knobs: forward only if the yaml supplied an
        # override so the detector's compile-time defaults remain the
        # source of truth for tests + benches that don't set them.
        l2_kwargs: dict = {}
        if config.detector_cube_cadence_s is not None:
            l2_kwargs["cube_cadence_s"] = float(config.detector_cube_cadence_s)
        if config.detector_layer2_tau_s is not None:
            l2_kwargs["layer2_tau_s"] = float(config.detector_layer2_tau_s)
        if config.detector_layer2_n_burnin is not None:
            l2_kwargs["layer2_n_burnin"] = int(config.detector_layer2_n_burnin)

        return DeterministicDetector(
            kernel_bank=bank,
            threshold_sigma=config.detector_threshold_sigma,
            detector_version=config.detector_version,
            search_node_id=config.search_node_id,
            gpu_half=config.gpu_half,
            dtype=config.detector_dtype,
            device=torch.device(config.detector_device),
            streaming=config.detector_streaming,
            streaming_tile_size=config.detector_streaming_tile_size,
            streaming_decoder_n_top=config.detector_streaming_decoder_n_top,
            boxcar_accum_dtype=config.detector_boxcar_accum_dtype,
            layer2_sigma_max_samples=config.detector_layer2_max_samples,
            layer2_sigma_floor=float(config.detector_layer2_sigma_floor),
            layer2_valid_min_fraction=float(
                config.detector_layer2_valid_min_fraction
            ),
            merger_config=config.merger_config,
            c1_snr_min=config.c1_snr_min,
            **l2_kwargs,
        )

    @property
    def detector(self) -> DeterministicDetector:
        return self._detector

    @property
    def pipeline(self) -> CubePipeline:
        return self._pipeline

    @property
    def cubes_processed(self) -> int:
        return self._cubes_processed

    @property
    def candidates_emitted(self) -> int:
        return self._candidates_emitted

    @property
    def clusters_emitted(self) -> int:
        return self._clusters_emitted

    @property
    def auto_dumps_dispatched(self) -> int:
        return self._auto_dumps_dispatched

    @property
    def udp_dumps_dispatched(self) -> int:
        return self._udp_dumps_dispatched

    @property
    def cluster_timeouts(self) -> int:
        return self._cluster_timeouts

    @property
    def clusterer(self) -> Optional[ClustererService]:
        return self._clusterer

    @property
    def cube_dump(self) -> Optional[CubeDumpWriter]:
        return self._cube_dump

    @property
    def udp_listener(self) -> Optional[UdpTriggerListener]:
        return self._udp_listener

    @property
    def cands_logger(self) -> Optional[CandsLogger]:
        return self._cands_logger

    @property
    def cube_ring(self) -> Optional[CubeRetentionRing]:
        return self._cube_ring

    @property
    def c1_emit(self) -> Optional[C1TcpEmitter]:
        return self._c1_emit

    @property
    def c2_trigger(self) -> Optional[C2TriggerListener]:
        return self._c2_trigger

    @property
    def c1_batches_submitted(self) -> int:
        return self._c1_batches_submitted

    @property
    def c1_batches_dropped(self) -> int:
        return self._c1_batches_dropped

    def mon_snapshot(self) -> dict:
        """Combined per-service mon-points snapshot. Includes the C1
        emitter + C2 trigger listener counters under nested keys."""
        snap = {
            "cubes_processed": int(self._cubes_processed),
            "candidates_emitted": int(self._candidates_emitted),
            "clusters_emitted": int(self._clusters_emitted),
            "auto_dumps_dispatched": int(self._auto_dumps_dispatched),
            "udp_dumps_dispatched": int(self._udp_dumps_dispatched),
            "cluster_timeouts": int(self._cluster_timeouts),
            "c1_batches_submitted": int(self._c1_batches_submitted),
            "c1_batches_dropped": int(self._c1_batches_dropped),
            "c1_cands_dropped_width": int(self._c1_cands_dropped_width),
            "c1_cands_dropped_dmfloor": int(self._c1_cands_dropped_dmfloor),
            "c1_cands_dropped_meter": int(self._c1_cands_dropped_meter),
            "search_node_id": int(self._config.search_node_id),
            "gpu_half": int(self._config.gpu_half),
        }
        if self._c1_emit is not None:
            snap["c1_emit"] = dict(self._c1_emit.mon)
        if self._c2_trigger is not None:
            snap["c2_trigger"] = dict(self._c2_trigger.mon)
        if self._cube_ring is not None:
            snap["cube_ring"] = {
                "depth": int(self._cube_ring.depth),
                "n_committed": int(self._cube_ring.n_committed),
            }
        return snap

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    async def start(self) -> None:
        """Bring up the RX-ring source + all configured sub-systems."""
        await self._source.start()
        cfg = self._config
        # M7.4 C1 stage: per-half cube retention ring. Built before the
        # cube driver starts so the first cube can stage into slot 0.
        # The ring's geometry is taken from the first slot lazily in
        # ``_process_one_cube``; here we just defer to the per-cube
        # path to populate.
        if cfg.cube_ring_depth > 0:
            self._cube_ring = None  # actual ring built lazily on first cube
        if cfg.enable_legacy_clusterer and cfg.clusterer_config is not None:
            self._clusterer = ClustererService(config=cfg.clusterer_config)
            self._clusterer.start()
        if cfg.enable_legacy_clusterer and cfg.bright_pulse_predicate_config is not None:
            self._predicate = BrightPulsePredicate(
                config=cfg.bright_pulse_predicate_config
            )
        if cfg.cube_dump_writer_config is not None:
            # M6 D7 dump_root may not exist yet on first run (production:
            # /home/ubuntu/data/m6/cube_dump; tests: tmp_path/dumps).
            # The writer thread's np.savez does not auto-mkdir, so make
            # sure the dir is in place before any dispatch happens.
            Path(cfg.cube_dump_writer_config.dump_root).mkdir(
                parents=True, exist_ok=True
            )
            # M7.4 cube uploader (search-node → h23 rsync). When both
            # ``cube_upload_dest_host`` and ``cube_upload_dest_root``
            # are set in the service config (default-wired from
            # ``c1.uploader.remote_root`` in ``dsart_search_rt.yaml``),
            # bind a per-write callback that spawns a detached rsync
            # next to each NPZ. The callback is a closure capturing
            # the destination config so the writer module stays free
            # of any uploader knowledge.
            on_dump_complete = self._build_cube_uploader_callback(cfg)
            self._cube_dump = CubeDumpWriter(
                config=cfg.cube_dump_writer_config,
                on_dump_complete=on_dump_complete,
            )
            self._cube_dump.start()
        if cfg.udp_trigger_listener_config is not None:
            # Legacy "dump next cube" listener; kept for the M6 path so
            # operator scripts can still arm a one-shot dump while we
            # roll out C2-triggered dumps.
            self._udp_listener = UdpTriggerListener(
                config=cfg.udp_trigger_listener_config
            )
            await self._udp_listener.start()
        if cfg.enable_legacy_clusterer and cfg.cands_logger_config is not None:
            self._cands_logger = CandsLogger(config=cfg.cands_logger_config)
        # M7.4 C1 emitter (TCP → h23).
        if cfg.c1_emit_config is not None:
            self._c1_emit = C1TcpEmitter(config=cfg.c1_emit_config)
            self._c1_emit_task = asyncio.create_task(self._c1_emit.run())
        # M7.4 C2 trigger listener (UDP from h23). Requires the ring
        # to be reachable; we defer ring construction to the first
        # cube but build the listener now bound to a placeholder ring
        # that gets swapped in when the ring exists.
        if cfg.c2_trigger_listener_config is not None:
            # Build a 0-cube ring placeholder until the first cube
            # comes through and we know the geometry. The listener's
            # ``find_cube_for_specnum`` walks the ring snapshot at
            # call time so swapping the ring is safe.
            if self._cube_ring is None:
                placeholder_ring = CubeRetentionRing(
                    depth=max(1, int(cfg.cube_ring_depth)),
                    t_det=1, n_fdm=1, n_grid=1,  # tiny; never written
                    pinned=False,
                )
                self._cube_ring = placeholder_ring
            self._c2_trigger = C2TriggerListener(
                config=cfg.c2_trigger_listener_config,
                ring=self._cube_ring,
                cube_dump=self._cube_dump,
            )
            await self._c2_trigger.start()
        _LOG.info(
            "SearchComputeService up "
            "(sid=%d, gpu_half=%d, cluster=%s, dump=%s, udp=%s, log=%s, "
            "c1_emit=%s, c2_trigger=%s, ring_depth=%d)",
            cfg.search_node_id,
            cfg.gpu_half,
            self._clusterer is not None,
            self._cube_dump is not None,
            self._udp_listener is not None,
            self._cands_logger is not None,
            self._c1_emit is not None,
            self._c2_trigger is not None,
            int(cfg.cube_ring_depth),
        )

    async def stop(self) -> None:
        self._stopping.set()
        if self._c2_trigger is not None:
            await self._c2_trigger.stop()
        if self._c1_emit is not None:
            await self._c1_emit.stop()
            if self._c1_emit_task is not None:
                try:
                    await asyncio.wait_for(self._c1_emit_task, timeout=5.0)
                except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                    self._c1_emit_task.cancel()
                    try:
                        await self._c1_emit_task
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        pass
        if self._udp_listener is not None:
            await self._udp_listener.stop()
        if self._cube_dump is not None:
            self._cube_dump.stop()
        if self._clusterer is not None:
            self._clusterer.shutdown(wait=True)
        if self._cands_logger is not None:
            self._cands_logger.close()
        await self._source.stop()
        _LOG.info(
            "SearchComputeService stopped: cubes=%d cands=%d clusters=%d "
            "auto_dumps=%d udp_dumps=%d cluster_timeouts=%d "
            "c1_batches_submitted=%d c1_batches_dropped=%d",
            self._cubes_processed,
            self._candidates_emitted,
            self._clusters_emitted,
            self._auto_dumps_dispatched,
            self._udp_dumps_dispatched,
            self._cluster_timeouts,
            self._c1_batches_submitted,
            self._c1_batches_dropped,
        )

    # -----------------------------------------------------------------
    # Per-cube driver
    # -----------------------------------------------------------------

    def _geom_from_slot(self, slot: CubeRingSlot) -> CubeGeometry:
        """Build a ``CubeGeometry`` from the slot + the static config.

        Production is expected to provide ``fine_dm_pc_cc_full`` covering
        the full fine-DM grid; the per-cube slice is taken from the
        first ``slot.n_fdm_in_cube`` entries (v1: M5 emits cubes covering
        the same DM trial range each call). Bench fallback: synthesise a
        linear grid ``np.linspace(50, 800, n_fdm_in_cube)``.
        """
        cfg = self._config
        if cfg.fine_dm_pc_cc_full is not None:
            full = np.asarray(cfg.fine_dm_pc_cc_full, dtype=np.float64)
            if full.shape[0] < slot.n_fdm_in_cube:
                raise ValueError(
                    f"fine_dm_pc_cc_full has {full.shape[0]} entries but "
                    f"slot covers {slot.n_fdm_in_cube} fine-DM trials"
                )
            fine_dm = full[: slot.n_fdm_in_cube].astype(np.float64, copy=True)
        else:
            fine_dm = np.linspace(
                50.0, 800.0, slot.n_fdm_in_cube, dtype=np.float64
            )
        # mjd_start = mjd_at_specnum_0 + specnum_start * t_int_fast_us / 1e6 / 86400.
        # t_int_fast_us = sample_period_us / sample_period_specnum.
        t_int_fast_us = cfg.cube_sample_period_us / cfg.cube_sample_period_specnum
        # Latch the per-run wall-clock anchor on the first cube when the
        # operator didn't pin ``mjd_at_specnum_0`` explicitly. The shift
        # by ``-specnum_start * t_int_fast_us`` puts the anchor at the
        # MJD that specnum 0 would have hit IF the run had started at
        # specnum 0 at the same wall-clock cadence. UNIX epoch 1970-01-01
        # = MJD 40587.0 exactly.
        if self._mjd_at_specnum_0_override is None and cfg.mjd_at_specnum_0 == 0.0:
            wall_mjd_now = 40587.0 + time.time() / 86400.0
            self._mjd_at_specnum_0_override = float(
                wall_mjd_now
                - slot.specnum_start * t_int_fast_us * 1e-6 / 86400.0
            )
            _LOG.info(
                "mjd_at_specnum_0 wall-clock latch: %.9f "
                "(slot.specnum_start=%d, t_int_fast_us=%.6f)",
                self._mjd_at_specnum_0_override,
                int(slot.specnum_start),
                t_int_fast_us,
            )
        mjd_at_specnum_0 = (
            self._mjd_at_specnum_0_override
            if self._mjd_at_specnum_0_override is not None
            else cfg.mjd_at_specnum_0
        )
        mjd_start = mjd_at_specnum_0 + (
            slot.specnum_start * t_int_fast_us * 1e-6 / 86400.0
        )
        return CubeGeometry(
            cube_id=slot.cube_id,
            specnum_start=slot.specnum_start,
            sample_period_specnum=cfg.cube_sample_period_specnum,
            t_det=slot.t_det,
            n_grid=slot.n_grid,
            n_fdm_in_cube=slot.n_fdm_in_cube,
            sample_period_us=cfg.cube_sample_period_us,
            cell_l_rad=cfg.cube_cell_l_rad,
            cell_m_rad=cfg.cube_cell_m_rad,
            l0_rad=cfg.cube_l0_rad,
            m0_rad=cfg.cube_m0_rad,
            fine_dm_pc_cc=fine_dm,
            mjd_start=mjd_start,
        )

    async def _process_one_cube(
        self,
        slot: CubeRingSlot,
        *,
        prefetched: Optional[PrefetchedCube] = None,
    ) -> List[Candidate]:
        cfg = self._config

        # Step 1 — legacy UDP arm check (M6 D9): a UDP datagram
        # arriving any time before this cube was dequeued arms a
        # single dump-next for THIS cube. The M7.4 C2 trigger path is
        # listener-driven (dumps from the ring), not a per-cube arm.
        udp_armed = False
        if self._udp_listener is not None:
            udp_armed = self._udp_listener.consume_dump_next_cube_flag()

        # Step 2 — pipeline (M5).
        if prefetched is not None:
            result = self._pipeline.process_prefetched(prefetched)
        else:
            result = self._pipeline.process(slot)
        self._cubes_processed += 1
        self._candidates_emitted += len(result.candidates)
        # M7.4 RT-perf debug (2026-05-27): accumulate per-stage wall
        # timings so the cube_progress log can surface mean build /
        # layer1 / detector / total wall per cube alongside cubes/s.
        # Resets every progress block.
        for _k in ("build_cube", "layer1_norm", "detector_forward", "total"):
            self._stage_ns_accum[_k] += int(result.stage_timings_ns.get(_k, 0))
        self._stage_ns_count += 1

        # Build geometry once for both clusterer + dumps + emitter.
        geom = self._geom_from_slot(slot)

        # Step 3 — UDP dump (this cube). Submit BEFORE clustering so the
        # writer-thread hand-off doesn't compete with clustering for CPU.
        if udp_armed and self._cube_dump is not None:
            udp_manifest = CubeDumpManifest(
                cube_id=slot.cube_id,
                event_specnum_start=slot.specnum_start,
                mjd_start=geom.mjd_start,
                t_det=slot.t_det,
                n_fdm_in_cube=slot.n_fdm_in_cube,
                n_grid=slot.n_grid,
                trigger_source="udp",
                cluster_record=None,
                npz_path=self._cube_dump_path("udp", slot.specnum_start),
                search_node_id=cfg.search_node_id,
                gpu_half=cfg.gpu_half,
            )
            accepted = self._cube_dump.submit(
                cube=result.cube, manifest=udp_manifest
            )
            if accepted:
                self._udp_dumps_dispatched += 1

        # Step 4a — M7.4 C1 stage: stage the cube into the retention
        # ring + emit candidates to C2. We stage AFTER the detector
        # ran (the GPU pipeline is done with the cube tensor) so the
        # ring copy doesn't compete for the compute stream.
        self._maybe_stage_cube_in_ring(slot, result.cube, geom)
        # C1 emit. Always invoked when the emitter is configured —
        # even on empty cubes, so the heartbeat / connectivity check
        # keeps flowing.
        if self._c1_emit is not None:
            self._submit_c1_batch(slot, geom, result.candidates)

        # Step 4b — legacy clusterer (M6 chunk 1) — disabled by default
        # under the M7.4 C1 stage; kept for offline benches.
        if cfg.enable_legacy_clusterer and self._clusterer is not None:
            future = self._clusterer.submit(result.candidates, geom)
            try:
                cluster_result = future.result(
                    timeout=cfg.cluster_result_timeout_s
                )
            except Exception:  # noqa: BLE001
                _LOG.warning(
                    "ClustererService timed out / failed on cube_id=%d "
                    "(n_cands=%d); skipping cube's records",
                    slot.cube_id,
                    len(result.candidates),
                )
                self._cluster_timeouts += 1
                return result.candidates
            self._clusters_emitted += len(cluster_result.records)

            # Step 5 — auto dumps (M6 D8 predicate). Legacy path; the
            # M7.4 C2 trigger is the canonical dump path.
            triggered_ids: set[int] = set()
            for record in cluster_result.records:
                if self._predicate is not None and self._predicate(record):
                    triggered_ids.add(record.cluster_id)
                    if self._cube_dump is not None:
                        auto_manifest = CubeDumpManifest(
                            cube_id=slot.cube_id,
                            event_specnum_start=slot.specnum_start,
                            mjd_start=geom.mjd_start,
                            t_det=slot.t_det,
                            n_fdm_in_cube=slot.n_fdm_in_cube,
                            n_grid=slot.n_grid,
                            trigger_source="auto",
                            cluster_record=record,
                            npz_path=self._cube_dump_path(
                                "auto", record.event_specnum
                            ),
                            search_node_id=cfg.search_node_id,
                            gpu_half=cfg.gpu_half,
                        )
                        accepted = self._cube_dump.submit(
                            cube=result.cube, manifest=auto_manifest
                        )
                        if accepted:
                            self._auto_dumps_dispatched += 1

            # Step 6 — T1/T2 ASCII rows (M6 D1).
            if self._cands_logger is not None:
                self._cands_logger.write_cube(
                    cands=result.candidates,
                    cluster_labels=cluster_result.labels,
                    cluster_records=cluster_result.records,
                    geom=geom,
                    triggered_cluster_ids=triggered_ids,
                )

        return result.candidates

    def _maybe_stage_cube_in_ring(
        self,
        slot: CubeRingSlot,
        cube_tensor: Any,
        geom: CubeGeometry,
    ) -> None:
        """Stage the post-detector cube into the C1 retention ring.

        Lazily builds the ring on the first cube once we know the
        actual geometry. Skips silently when neither the ring nor a
        C2 trigger listener is configured (saves the ~3 GiB copy on
        legacy-only deployments)."""
        cfg = self._config
        # depth<=0 means the operator has explicitly disabled cube
        # retention -- treat as a hard skip even when the C2 trigger
        # listener is wired, so we can run hot perf benches without
        # the 3.2 GiB DtoH per cube. (Burst dumps still require depth>=1.)
        if cfg.cube_ring_depth <= 0:
            return
        wants_ring = (
            cfg.c2_trigger_listener_config is not None
            or cfg.cube_ring_depth > 0
            and self._c2_trigger is not None
        )
        if not wants_ring:
            return
        need_rebuild = (
            self._cube_ring is None
            or self._cube_ring.depth != cfg.cube_ring_depth
            or self._cube_ring.t_det != slot.t_det
            or self._cube_ring.n_fdm != slot.n_fdm_in_cube
            or self._cube_ring.n_grid != slot.n_grid
        )
        if need_rebuild:
            # Pinned host destination is REQUIRED for the production
            # GPU→CPU stage_cube copy to use cudaMemcpyAsync over a
            # DMA-able buffer. With pinned=False the buffer is plain
            # np.empty (pageable), forcing a slow kernel-bounce DMA
            # that BLOCKS the main thread for ~700 ms per 815 MiB cube
            # — measured 1.0 cubes/s vs. the 7.45 cubes/s target on
            # n01 (M7.4 C1 deploy, 2026-05-26). The CubeRetentionRing
            # lazily allocates the ring slots, so the pin cost is paid
            # ``depth`` times at warmup and amortised forever after.
            self._cube_ring = CubeRetentionRing(
                depth=int(cfg.cube_ring_depth),
                t_det=int(slot.t_det),
                n_fdm=int(slot.n_fdm_in_cube),
                n_grid=int(slot.n_grid),
                pinned=True,
            )
            # Re-point the C2 trigger listener at the rebuilt ring.
            if self._c2_trigger is not None:
                self._c2_trigger.set_ring(self._cube_ring)
        try:
            self._cube_ring.stage_cube(
                cube_id=int(slot.cube_id),
                event_specnum_start=int(slot.specnum_start),
                mjd_start=float(geom.mjd_start),
                sample_period_specnum=int(geom.sample_period_specnum),
                sample_period_us=float(geom.sample_period_us),
                cube_tensor=cube_tensor,
            )
        except (ValueError, TypeError) as exc:
            _LOG.warning(
                "cube_ring stage failed (cube_id=%d): %r",
                int(slot.cube_id), exc,
            )

    def _submit_c1_batch(
        self,
        slot: CubeRingSlot,
        geom: CubeGeometry,
        candidates: List[Candidate],
    ) -> None:
        """Project ``candidates`` into the C1 row schema + push onto
        the emitter's outbound queue. Drops on queue-full are surfaced
        via the emitter's mon-points + the service's own counters."""
        assert self._c1_emit is not None
        cfg = self._config
        # 2026-05-30: drop candidates far narrower than the DM-smearing
        # floor permits BEFORE the width cap. These are impulsive RFI
        # mis-assigned to a high-DM trial (cannot be genuine dispersed
        # signals); rejecting them at the source stops the C2 dump-storm
        # failure mode without touching real (low-DM-narrow or high-DM-
        # smeared) events. Cube dump + retention run off the full list
        # upstream and are unaffected.
        dm_floor_frac = (
            cfg.c1_emit_config.dm_width_floor_frac
            if cfg.c1_emit_config is not None
            else None
        )
        if dm_floor_frac is not None and candidates:
            candidates, n_dm_dropped = filter_unphysical_narrow(
                candidates, dm_floor_frac,
                t_search_us=float(geom.sample_period_us),
            )
            if n_dm_dropped:
                self._c1_cands_dropped_dmfloor += n_dm_dropped
        # M7.6: drop candidates wider than the configured C1→C2 width cap
        # BEFORE they are transmitted. Wide boxcars (≥32) are the on-sky
        # false-positive floor; capping at ``max_width_samples`` keeps the
        # coincidence peak_event_specnum on fresh cubes (fixes too_late
        # dump misses) and unloads the C1→C2 path. Cube dump + retention
        # are unaffected (this only gates what we ship to C2).
        max_w = (
            cfg.c1_emit_config.max_width_samples
            if cfg.c1_emit_config is not None
            else None
        )
        if max_w is not None and candidates:
            kept = [c for c in candidates if int(c.width_samples) <= int(max_w)]
            n_dropped = len(candidates) - len(kept)
            if n_dropped:
                self._c1_cands_dropped_width += n_dropped
            candidates = kept
        # M7.6 C1→C2 metering: cap candidates/block, narrow-first
        # (width asc) then bright-first (snr desc). RT-safe — we only pay
        # the selection cost when the cap actually bites, and
        # ``heapq.nsmallest`` is O(k log N) (N = cap ≪ k during floods).
        cap = (
            cfg.c1_emit_config.max_candidates_per_block
            if cfg.c1_emit_config is not None
            else None
        )
        n_cands_pre_meter = len(candidates)
        candidates, n_metered = meter_candidates(candidates, cap)
        if n_metered:
            self._c1_cands_dropped_meter += n_metered
        # Roll the metering state up over a window of blocks so the etcd
        # publish (→ influx → grafana) stays at a low rate.
        self._meter_window_blocks += 1
        self._meter_window_cands_sum += n_cands_pre_meter
        if n_metered:
            self._meter_window_metered_blocks += 1
            self._meter_window_dropped_sum += n_metered
            if n_metered > self._meter_window_dropped_max:
                self._meter_window_dropped_max = n_metered
        if self._meter_window_blocks >= _METER_WINDOW_BLOCKS:
            self._publish_c1_metering(int(cap or 0))
        rows = tuple(
            candidate_to_c1_row(c, geom=geom) for c in candidates
        )
        header = build_header(
            cube_id=int(slot.cube_id),
            event_specnum_start=int(slot.specnum_start),
            mjd_start=float(geom.mjd_start),
            sample_period_specnum=int(geom.sample_period_specnum),
            sample_period_us=float(geom.sample_period_us),
            n_grid=int(slot.n_grid),
            n_fdm_in_cube=int(slot.n_fdm_in_cube),
            search_node_id=int(cfg.search_node_id),
            gpu_half=int(cfg.gpu_half),
            n_candidates=len(rows),
        )
        accepted = self._c1_emit.submit(header, candidates, rows=rows)
        if accepted:
            self._c1_batches_submitted += 1
        else:
            self._c1_batches_dropped += 1

    def _publish_c1_metering(self, cap: int) -> None:
        """Flush the C1→C2 metering window to etcd and reset accumulators.

        Best-effort: a publish failure (etcd hiccup, dsautils missing) is
        swallowed by the publisher so the search hot loop never blocks."""
        pub = self._compute_mon
        if pub is not None:
            try:
                pub.publish_metering(
                    n_blocks=self._meter_window_blocks,
                    n_metered_blocks=self._meter_window_metered_blocks,
                    dropped_sum=self._meter_window_dropped_sum,
                    dropped_max=self._meter_window_dropped_max,
                    cands_sum=self._meter_window_cands_sum,
                    cap=int(cap),
                )
            except Exception:  # noqa: BLE001 — mon must never sink the pipe
                LOG.warning("C1 metering publish failed", exc_info=True)
        self._meter_window_blocks = 0
        self._meter_window_metered_blocks = 0
        self._meter_window_dropped_sum = 0
        self._meter_window_dropped_max = 0
        self._meter_window_cands_sum = 0

    def _cube_dump_path(self, source: str, key_specnum: int) -> str:
        """Build the canonical per-cube NPZ path.

        Uses the configured ``cube_dump_writer_config.dump_root``. The
        file name template per M6 D7 is
        ``cube_s${sid}_g${g}_${event_specnum}.npz`` for auto dumps; UDP
        dumps follow the same template using the cube's
        ``specnum_start`` as the event_specnum.

        Note: the writer thread builds the *actual* on-disk path from
        the manifest's ``event_specnum_start`` (per ``CubeDumpWriter``
        contract — the manifest's ``npz_path`` field is informational
        and intentionally not used by the writer). This helper produces
        a value that *matches* the writer's canonical layout so
        consumers reading the manifest can find the file.
        """
        cfg = self._config
        assert cfg.cube_dump_writer_config is not None
        root = cfg.cube_dump_writer_config.dump_root
        return str(
            Path(root)
            / f"cube_s{cfg.search_node_id}_g{cfg.gpu_half}_{int(key_specnum)}.npz"
        )

    def _log_cube_progress(
        self,
        *,
        now_loop_start: float,
        last_log_mono: float,
        last_cubes_count: int,
    ) -> None:
        """Emit one ``cube_progress`` log line with rich timing info.

        Reads + RESETS the rolling accumulators
        (``_stage_ns_accum``/``_stage_ns_count`` and the source's
        scatter timing) so each line reports per-block means rather
        than running averages. Centralised here (was duplicated across
        the pipeline-overlap and serial branches of :meth:`run`) so the
        timing-instrumentation surface only lives in one place.
        """
        now = time.monotonic()
        d_cubes = self._cubes_processed - last_cubes_count
        dt = max(now - last_log_mono, 1e-9)
        _scatter_t = (
            self._source.get_scatter_timing_and_reset()
            if hasattr(self._source, "get_scatter_timing_and_reset")
            else {"mean_us": 0.0, "max_us": 0.0, "count": 0}
        )
        _build_t = (
            self._pipeline.get_build_event_timing_and_reset()
            if hasattr(self._pipeline, "get_build_event_timing_and_reset")
            else {"h2d_ms": 0.0, "imager_ms": 0.0, "valid_ms": 0.0, "n": 0}
        )
        n_stage = max(int(self._stage_ns_count), 1)
        st = self._stage_ns_accum
        build_ms = (st["build_cube"] / n_stage) / 1_000_000.0
        layer1_ms = (st["layer1_norm"] / n_stage) / 1_000_000.0
        det_ms = (st["detector_forward"] / n_stage) / 1_000_000.0
        total_ms = (st["total"] / n_stage) / 1_000_000.0
        # Reset accumulators after read.
        for _k in self._stage_ns_accum:
            self._stage_ns_accum[_k] = 0
        self._stage_ns_count = 0
        _LOG.info(
            "cube_progress: cubes=%d cands=%d clusters=%d "
            "(%.2f cubes/s last %.1fs; %.2f cubes/s overall) "
            "stage_ms[build/l1/det/total]=%.1f/%.1f/%.1f/%.1f "
            "build_ms[h2d/imager/valid]=%.1f/%.1f/%.1f(n=%d) "
            "scatter=%.1f/%.1f us(mean/max,n=%d) src=%s",
            self._cubes_processed,
            self._candidates_emitted,
            self._clusters_emitted,
            d_cubes / dt,
            dt,
            self._cubes_processed / max(now - now_loop_start, 1e-9),
            build_ms, layer1_ms, det_ms, total_ms,
            _build_t["h2d_ms"], _build_t["imager_ms"], _build_t["valid_ms"],
            _build_t["n"],
            _scatter_t["mean_us"], _scatter_t["max_us"],
            _scatter_t["count"],
            getattr(self._source, "stats", {}),
        )
        # Phase 6c: best-effort publish of the cube_ring window to
        # ``/mon/search/<sid>/<g>/ring`` so the Control-tab "Dump Now"
        # button can pick an event_specnum that's guaranteed to land
        # inside this half's retention window. Failures are silent
        # past the first warning so etcd hiccups never block the loop.
        if self._ring_mon is not None and self._cube_ring is not None:
            try:
                self._ring_mon.publish_from_ring(self._cube_ring)
            except Exception:                                  # noqa: BLE001
                # publish_from_ring is itself best-effort, but
                # belt-and-braces here so no exception escapes into
                # the progress-logging path.
                _LOG.exception(
                    "SearchRingMonPublisher.publish_from_ring failed "
                    "(swallowed; will retry next cycle)"
                )

    async def run(self) -> None:
        """Main loop. Iterates over RX-ring slots until source exhausts
        or ``stop()`` is called.

        Emits one INFO line per ``_status_every_cubes`` cubes (default
        10) so the M7.2 / production soaks have a visible progress
        signal in the orchestrator logs without having to wait for a
        SIGTERM-triggered final stats line.
        """
        status_every = 10
        next_status = status_every
        loop_start = time.monotonic()
        last_log = loop_start
        last_cubes = 0
        if (
            self._config.pipeline.image_backend == "gpu"
            and self._config.pipeline_overlap
        ):
            aiter = self._source.__aiter__()
            try:
                slot = await aiter.__anext__()
            except StopAsyncIteration:
                slot = None
            pending = (
                self._pipeline.prefetch_build(slot)
                if slot is not None
                else None
            )
            while slot is not None and pending is not None:
                if self._stopping.is_set():
                    break
                try:
                    next_slot = await aiter.__anext__()
                except StopAsyncIteration:
                    next_slot = None
                next_pending = (
                    self._pipeline.prefetch_build(next_slot)
                    if next_slot is not None
                    else None
                )
                await self._process_one_cube(slot, prefetched=pending)
                await self._source.release(slot.cube_id)
                # Cooperative yield: _process_one_cube + release run to
                # completion without ever suspending the event loop when
                # the RX ring has data ready, so the co-resident C1 emitter
                # task (c1_emit.run drain loop) would otherwise be starved
                # under heavy candidate load -- it could not drain its queue,
                # send heartbeats, or reconnect, which dropped that half's
                # C1->C2 connection (observed 2026-05-29 as the recurring
                # n01 gpu_half=0 7/8). sleep(0) forces one scheduler cycle
                # per cube so the emitter always makes progress.
                await asyncio.sleep(0)
                slot, pending = next_slot, next_pending
                if self._cubes_processed >= next_status:
                    self._log_cube_progress(now_loop_start=loop_start,
                                            last_log_mono=last_log,
                                            last_cubes_count=last_cubes)
                    last_log = time.monotonic()
                    last_cubes = self._cubes_processed
                    next_status += status_every
        else:
            async for slot in self._source:
                if self._stopping.is_set():
                    break
                await self._process_one_cube(slot)
                await self._source.release(slot.cube_id)
                # Cooperative yield so the co-resident C1 emitter task is
                # not starved under heavy candidate load (see overlap branch
                # above; the recurring n01 gpu_half=0 7/8 root cause).
                await asyncio.sleep(0)
                if self._cubes_processed >= next_status:
                    self._log_cube_progress(now_loop_start=loop_start,
                                            last_log_mono=last_log,
                                            last_cubes_count=last_cubes)
                    last_log = time.monotonic()
                    last_cubes = self._cubes_processed
                    next_status += status_every


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CLI helpers (M7.2)
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    """Minimal YAML loader (lazy import; sub-system configs only)."""
    import yaml  # local import; pyyaml is a runtime dep already
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _dm_grids_from_npz(
    dm_plan_path: Path, n_coarse: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load DM grids from a DmPlan NPZ. Falls back to a synthetic linear
    grid if the file is unreadable. The NPZ layout (set by the corr-side
    ``dm_plan`` tooling): ``coarse_dm`` (or ``coarse_dm_pc_cm3``),
    ``fine_dm`` (or ``fine_dm_pc_cm3``), ``fine_to_coarse``."""
    with np.load(dm_plan_path) as npz:
        coarse_key = next(
            (k for k in ("coarse_dm_pc_cm3", "coarse_dm") if k in npz),
            None,
        )
        fine_key = next(
            (k for k in ("fine_dm_pc_cm3", "fine_dm") if k in npz),
            None,
        )
        if coarse_key is None or fine_key is None:
            raise KeyError(
                f"DM plan {dm_plan_path} missing 'coarse_dm'/'fine_dm' keys; "
                f"found {list(npz.keys())}"
            )
        coarse_dm = np.asarray(npz[coarse_key], dtype=np.float64)
        fine_dm = np.asarray(npz[fine_key], dtype=np.float64)
        if "fine_to_coarse" in npz:
            f2c = np.asarray(npz["fine_to_coarse"], dtype=np.int32)
        else:
            # Default: every fine maps to coarse cell 0 (safe; matches
            # the M4a unit-test convention).
            f2c = np.zeros(len(fine_dm), dtype=np.int32)
    # Trim coarse to ring dims if on-disk plan is bigger. Fine/f2c are
    # intentionally left full-size so the caller can apply the explicit
    # per-half owner selection before trimming.
    coarse_dm = coarse_dm[:n_coarse]
    if len(f2c) != len(fine_dm):
        raise ValueError(
            f"DM plan {dm_plan_path}: fine_to_coarse length "
            f"{len(f2c)} != fine_dm length {len(fine_dm)}"
        )
    return coarse_dm, fine_dm, f2c


def _select_dm_owner_half(
    *,
    coarse_dm: np.ndarray,
    fine_dm: np.ndarray,
    fine_to_coarse: np.ndarray,
    owner_coarse_idx: int,
    expected_n_fdm: int,
    gpu_half: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select one coarse-DM owner for this half and remap to local indices.

    M7.2 v2 ownership model: each GPU-half owns exactly one coarse DM index.
    We therefore:
      - keep only fine trials whose global fine_to_coarse == owner_coarse_idx
      - keep coarse_dm[owner_coarse_idx] as a length-1 local coarse table
      - remap fine_to_coarse to all zeros (local coarse index 0)
    """
    if not 0 <= owner_coarse_idx < len(coarse_dm):
        raise ValueError(
            f"gpu_half={gpu_half}: owner coarse index {owner_coarse_idx} "
            f"outside [0, {len(coarse_dm) - 1}]"
        )
    sel = np.nonzero(fine_to_coarse == owner_coarse_idx)[0]
    if sel.size == 0:
        raise ValueError(
            f"gpu_half={gpu_half}: no fine DM rows map to coarse index "
            f"{owner_coarse_idx}"
        )
    fine_local = fine_dm[sel].astype(np.float64, copy=False)
    coarse_local = np.asarray(
        [coarse_dm[owner_coarse_idx]], dtype=np.float64
    )
    f2c_local = np.zeros(fine_local.shape[0], dtype=np.int32)
    if expected_n_fdm > 0 and fine_local.shape[0] != expected_n_fdm:
        raise ValueError(
            f"gpu_half={gpu_half}: owner coarse index {owner_coarse_idx} "
            f"yields n_fdm={fine_local.shape[0]}, but --n-fdm={expected_n_fdm}. "
            f"Use matching n_fdm (M7.2 v2 expects K = N_fine/8)."
        )
    return coarse_local, fine_local, f2c_local


def _synthetic_dm_grids(
    n_coarse: int, n_fdm: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bench / smoke DM grids: all fine DMs map to coarse cell 0 so
    ``compute_time_shift_search`` produces non-negative shifts."""
    coarse_dm = np.linspace(0.0, 300.0, n_coarse, dtype=np.float64)
    fine_dm = np.linspace(0.0, 100.0, n_fdm, dtype=np.float64)
    f2c = np.zeros(n_fdm, dtype=np.int32)
    return coarse_dm, fine_dm, f2c


def _parse_kernel_tokens(csv: str, *, allowed: tuple[str, ...], field: str) -> tuple[str, ...]:
    toks = tuple(t.strip() for t in str(csv).split(",") if t.strip())
    if not toks:
        raise ValueError(f"{field}: empty token list")
    bad = [t for t in toks if t not in allowed]
    if bad:
        raise ValueError(
            f"{field}: unknown tokens {bad}; allowed={list(allowed)}"
        )
    return toks


def _build_search_config_from_yaml(
    yaml_doc: dict,
    *,
    n_grid: int,
    n_fdm: int,
    gpu_half: int,
    search_node_id: int,
    image_backend: str,
    device: str,
    enable_clusterer: bool,
    enable_cube_dump: bool,
    enable_udp_listener: bool,
    enable_cands_logger: bool,
    detector_streaming_tile_size: int,
    detector_streaming_decoder_n_top: int,
    pipeline_overlap: bool,
    detector_k_img_csv: str,
    detector_k_dm_csv: str,
    detector_k_time_csv: str,
    detector_boxcar_accum_dtype: str,
    detector_layer2_max_samples: Optional[int],
    layer1_max_samples: Optional[int],
    layer1_sigma_floor: float,
    fine_dm_pc_cc_full: Optional[np.ndarray],
    enable_c1: bool = True,
    c1_bind_host_override: Optional[str] = None,
) -> SearchComputeConfig:
    """Translate ``configs/config_compute_search.yaml`` into
    ``SearchComputeConfig`` with optional M6 sub-systems gated on the
    per-subsystem ``enable_*`` flags. This is the M7.2 inline shim; the
    full ``config_loader`` integration is the M7.2.5 follow-up."""
    det = yaml_doc.get("detector", {}) or {}
    noise = yaml_doc.get("noise", {}) or {}
    pipe_cfg = CubePipelineConfig(
        n_grid=int(n_grid),
        image_backend=image_backend,   # "cpu" or "gpu"
        device=device,
    )

    clusterer_cfg: Optional[ClustererConfig] = None
    if enable_clusterer:
        cl = yaml_doc.get("clusterer", {}) or {}
        kwargs: dict[str, Any] = {}
        if "backend" in cl:
            kwargs["backend"] = str(cl["backend"])
        if "feature_mode" in cl:
            kwargs["feature_mode"] = str(cl["feature_mode"])
        if "weights" in cl:
            kwargs["weights"] = tuple(float(w) for w in cl["weights"])
        if "min_cluster_size" in cl:
            kwargs["min_cluster_size"] = int(cl["min_cluster_size"])
        if "min_samples" in cl:
            kwargs["min_samples"] = int(cl["min_samples"])
        if "cluster_selection_epsilon" in cl:
            kwargs["cluster_selection_epsilon"] = float(
                cl["cluster_selection_epsilon"]
            )
        if "dbscan_eps" in cl:
            kwargs["dbscan_eps"] = float(cl["dbscan_eps"])
        if "dbscan_min_samples" in cl:
            kwargs["dbscan_min_samples"] = int(cl["dbscan_min_samples"])
        if "metric" in cl:
            kwargs["metric"] = str(cl["metric"])
        clusterer_cfg = ClustererConfig(**kwargs)

    cube_dump_cfg: Optional[CubeDumpWriterConfig] = None
    if enable_cube_dump:
        cd = yaml_doc.get("cube_dump", {}) or {}
        # M7.4: the C2 trigger listener composes its per-event manifest
        # ``npz_path`` under ``c1.dump_root``; the writer's
        # ``_resolve_path()`` only honors that override when the listener
        # path is a *subdir* of the writer's own ``dump_root``. So the
        # writer ``dump_root`` MUST equal (or be an ancestor of)
        # ``c1.dump_root`` or every C2-triggered dump falls back to the
        # writer's canonical layout (no per-event subdir, file lands
        # outside the candidate archive). Default the writer to
        # ``c1.dump_root`` here so the single yaml knob controls both
        # sides; an explicit ``cube_dump.dump_root`` still wins (e.g. for
        # legacy bench paths).
        c1_yaml = yaml_doc.get("c1", {}) or {}
        default_dump_root = c1_yaml.get(
            "dump_root", "/tmp/dsart-cube-dump"
        )
        cube_dump_cfg = CubeDumpWriterConfig(
            dump_root=Path(cd.get("dump_root", default_dump_root)),
            search_node_id=int(cd.get("search_node_id", search_node_id)),
            gpu_half=int(cd.get("gpu_half", gpu_half)),
            queue_maxsize=int(cd.get("queue_maxsize", 4)),
        )

    udp_listener_cfg: Optional[UdpTriggerListenerConfig] = None
    if enable_udp_listener:
        ul = yaml_doc.get("udp_trigger_listener", {}) or {}
        udp_listener_cfg = UdpTriggerListenerConfig(
            host=str(ul.get("host", "127.0.0.1")),
            port=int(ul.get("port", 11227)),
        )

    cands_logger_cfg: Optional[CandsLoggerConfig] = None
    if enable_cands_logger:
        cl = yaml_doc.get("cands_logger", {}) or {}
        cands_logger_cfg = CandsLoggerConfig(
            log_root=Path(cl.get("log_root", "/tmp/dsart-cands-log")),
            search_node_id=int(cl.get("search_node_id", search_node_id)),
            gpu_half=int(cl.get("gpu_half", gpu_half)),
        )

    predicate_cfg: Optional[BrightPulsePredicateConfig] = None
    if enable_cube_dump:
        bp = yaml_doc.get("bright_pulse_predicate", {}) or {}
        kwargs2: dict[str, Any] = {}
        for k in ("min_snr", "dm_fine_min_pc_cc", "dm_fine_max_pc_cc",
                  "width_samples_max", "min_cntc", "holdoff_ms"):
            if k in bp:
                kwargs2[k] = bp[k]
        if kwargs2:
            predicate_cfg = BrightPulsePredicateConfig(**kwargs2)
        else:
            predicate_cfg = BrightPulsePredicateConfig()

    if detector_boxcar_accum_dtype == "default":
        boxcar_accum_dtype: Optional[torch.dtype] = None
    elif detector_boxcar_accum_dtype == "fp32":
        boxcar_accum_dtype = torch.float32
    elif detector_boxcar_accum_dtype == "fp16":
        boxcar_accum_dtype = torch.float16
    elif detector_boxcar_accum_dtype == "bf16":
        boxcar_accum_dtype = torch.bfloat16
    else:
        raise ValueError(
            "detector_boxcar_accum_dtype must be one of "
            "default/fp32/fp16/bf16"
        )

    det_img_tokens = _parse_kernel_tokens(
        detector_k_img_csv,
        allowed=DETECTOR_IMAGE_KERNELS,
        field="--detector-k-img",
    )
    det_dm_tokens = _parse_kernel_tokens(
        detector_k_dm_csv,
        allowed=DETECTOR_DM_KERNELS,
        field="--detector-k-dm",
    )
    det_time_tokens = _parse_kernel_tokens(
        detector_k_time_csv,
        allowed=DETECTOR_TIME_KERNELS,
        field="--detector-k-time",
    )

    # -----------------------------------------------------------------
    # M7.4 C1 stage wiring (single source of truth for SNR / merger /
    # ring / emit / trigger listener / dump root).
    # -----------------------------------------------------------------
    c1 = (yaml_doc.get("c1", {}) or {}) if enable_c1 else {}
    c1_snr_min: Optional[float] = (
        float(c1.get("snr_min")) if "snr_min" in c1 else None
    )
    merger_yaml = c1.get("merger", {}) or {}
    merger_cfg: Optional[MergerConfig] = None
    if c1 or merger_yaml:
        merger_cfg = MergerConfig(
            lm_max_cells=int(merger_yaml.get("lm_max_cells", 3)),
            dm_max_trials=int(merger_yaml.get("dm_max_trials", 2)),
            t_frac=float(merger_yaml.get("t_frac", 1.0)),
            sample_period_specnum=int(
                merger_yaml.get(
                    "sample_period_specnum",
                    yaml_doc.get("cube", {}).get("sample_period_specnum", 16),
                )
            ),
        )
    cube_ring_depth_yaml = int(c1.get("cube_ring_depth", 8))
    c2_endpoint = c1.get("c2_endpoint", {}) or {}
    c1_emit_cfg: Optional[C1EmitConfig] = None
    if enable_c1 and c2_endpoint:
        _max_c1c2_width = c1.get("max_c1c2_width_samples", None)
        _max_cands_block = c1.get("max_candidates_per_block", None)
        _dm_width_floor = c1.get("dm_width_floor_frac", None)
        c1_emit_cfg = C1EmitConfig(
            host=str(c2_endpoint.get("host", "h23")),
            port=int(c2_endpoint.get("port", 11500)),
            search_node_id=int(search_node_id),
            gpu_half=int(gpu_half),
            queue_depth=int(c1.get("emit_queue_depth", 16)),
            max_width_samples=(
                int(_max_c1c2_width)
                if _max_c1c2_width is not None and int(_max_c1c2_width) > 0
                else None
            ),
            max_candidates_per_block=(
                int(_max_cands_block)
                if _max_cands_block is not None and int(_max_cands_block) > 0
                else None
            ),
            dm_width_floor_frac=(
                float(_dm_width_floor)
                if _dm_width_floor is not None and float(_dm_width_floor) > 0.0
                else None
            ),
        )
    dump_listener_yaml = c1.get("dump_listener", {}) or {}
    c1_dump_root = (
        Path(c1["dump_root"]) if "dump_root" in c1 else None
    )
    # M7.4 cube uploader (search-node → h23 rsync). Parsed from
    # ``c1.uploader.remote_root`` (rsync ``user@host:/path`` shape)
    # plus an optional ``bandwidth_limit_kbps``. When unset, the
    # uploader stays disabled (the writer's post-write hook is None).
    uploader_yaml = c1.get("uploader", {}) or {}
    cube_upload_dest_host: Optional[str] = None
    cube_upload_dest_root: Optional[str] = None
    cube_upload_bwlimit_kbps = int(
        uploader_yaml.get("bandwidth_limit_kbps", 0) or 0
    )
    remote_root_raw = uploader_yaml.get("remote_root", "")
    if remote_root_raw:
        from ..coinc.cube_uploader import parse_remote_root
        cube_upload_dest_host, cube_upload_dest_root = parse_remote_root(
            str(remote_root_raw)
        )
    c2_listener_cfg: Optional[C2TriggerListenerConfig] = None
    if enable_c1 and dump_listener_yaml:
        bind_host = c1_bind_host_override or str(
            dump_listener_yaml.get("bind_host", "")
        ).strip()
        # Empty bind_host defers to per-host CLI override; if no
        # override was provided we fall back to 0.0.0.0 (binds all
        # interfaces). Production wires the per-host search-net IP via
        # the orchestrator's hostargs.
        if not bind_host:
            bind_host = "0.0.0.0"
        listener_dump_root = c1_dump_root or Path(
            c1.get("dump_root", "/home/ubuntu/data/c2/cube_dump")
        )
        c2_listener_cfg = C2TriggerListenerConfig(
            bind_host=bind_host,
            base_port=int(dump_listener_yaml.get("base_port", 11227)),
            gpu_half=int(gpu_half),
            search_node_id=int(search_node_id),
            dump_root=listener_dump_root,
        )

    # Honor a top-level YAML ``enable_legacy_clusterer`` knob as the
    # source of truth when present; falls back to the CLI flag when
    # absent (preserves legacy bench behavior).
    legacy_enabled = bool(
        yaml_doc.get("enable_legacy_clusterer", bool(enable_clusterer))
    )

    # Layer-2 σ_k EMA yaml knobs (M7.4 hardening). All optional —
    # absence preserves detector defaults.
    detector_cube_cadence_s_yaml = det.get("cube_cadence_s", None)
    detector_layer2_tau_s_yaml = det.get("layer2_tau_s", None)
    detector_layer2_n_burnin_yaml = det.get("layer2_n_burnin", None)
    detector_layer2_sigma_floor_yaml = float(det.get("layer2_sigma_floor", 0.0))
    detector_layer2_valid_min_fraction_yaml = float(
        det.get("layer2_valid_min_fraction", 1.0)
    )

    return SearchComputeConfig(
        pipeline=pipe_cfg,
        n_fdm=int(n_fdm),
        detector_threshold_sigma=float(det.get("threshold_sigma", 8.0)),
        detector_streaming_tile_size=int(detector_streaming_tile_size),
        detector_streaming_decoder_n_top=int(detector_streaming_decoder_n_top),
        pipeline_overlap=bool(pipeline_overlap),
        detector_image_tokens=det_img_tokens,
        detector_dm_tokens=det_dm_tokens,
        detector_time_tokens=det_time_tokens,
        detector_boxcar_accum_dtype=boxcar_accum_dtype,
        detector_layer2_max_samples=(
            int(detector_layer2_max_samples)
            if detector_layer2_max_samples is not None
            and int(detector_layer2_max_samples) > 0
            else None
        ),
        detector_cube_cadence_s=(
            float(detector_cube_cadence_s_yaml)
            if detector_cube_cadence_s_yaml is not None else None
        ),
        detector_layer2_tau_s=(
            float(detector_layer2_tau_s_yaml)
            if detector_layer2_tau_s_yaml is not None else None
        ),
        detector_layer2_n_burnin=(
            int(detector_layer2_n_burnin_yaml)
            if detector_layer2_n_burnin_yaml is not None else None
        ),
        detector_layer2_sigma_floor=detector_layer2_sigma_floor_yaml,
        detector_layer2_valid_min_fraction=detector_layer2_valid_min_fraction_yaml,
        detector_device=device,
        search_node_id=int(search_node_id),
        gpu_half=int(gpu_half),
        layer1_n_burnin_cubes=int(noise.get("layer1_n_burnin_cubes", 5)),
        layer1_max_samples=(
            int(layer1_max_samples)
            if layer1_max_samples is not None
            else None
        ),
        # CLI flag --layer1-sigma-floor (>0.0) overrides the YAML
        # default; legacy YAML value is used when the CLI flag is 0.0
        # (its argparse default), so existing callers see no change.
        layer1_sigma_floor=(
            float(layer1_sigma_floor)
            if float(layer1_sigma_floor) > 0.0
            else float(noise.get("layer1_sigma_floor", 0.0))
        ),
        fine_dm_pc_cc_full=fine_dm_pc_cc_full,
        clusterer_config=clusterer_cfg,
        cube_dump_writer_config=cube_dump_cfg,
        bright_pulse_predicate_config=predicate_cfg,
        udp_trigger_listener_config=udp_listener_cfg,
        cands_logger_config=cands_logger_cfg,
        enable_legacy_clusterer=legacy_enabled,
        merger_config=merger_cfg,
        c1_snr_min=c1_snr_min,
        cube_ring_depth=cube_ring_depth_yaml,
        c1_emit_config=c1_emit_cfg,
        c2_trigger_listener_config=c2_listener_cfg,
        c1_dump_root=c1_dump_root,
        cube_upload_dest_host=cube_upload_dest_host,
        cube_upload_dest_root=cube_upload_dest_root,
        cube_upload_bandwidth_limit_kbps=cube_upload_bwlimit_kbps,
    )


def _install_signal_handlers(loop: asyncio.AbstractEventLoop,
                             stop_event: asyncio.Event) -> None:
    """Wire SIGTERM/SIGINT to set ``stop_event`` so the main loop
    exits cleanly. Mirrors the dsart_rt orchestrator's contract."""
    def _handle(signum: int) -> None:
        _LOG.info("received signal %d; stopping…", signum)
        stop_event.set()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle, sig)
        except (NotImplementedError, RuntimeError):
            # Not all platforms / event loops support add_signal_handler;
            # fall back to signal.signal which is good enough for the
            # orchestrator-launched case.
            signal.signal(sig, lambda s, _f: _handle(s))


async def _run_async(args: argparse.Namespace) -> int:
    # Lazy import so the CLI parser still works without the C extension
    # when --help is invoked on a dev box.
    from ..transport.production_rx_ring import ProductionRxRingSource
    from ..transport.recv_ring import (
        BYTES_CFP16_COMPLEX,
        BYTES_CINT8_COMPLEX,
        RxRingDims,
    )

    bpc = (BYTES_CINT8_COMPLEX if args.bytes_per_cell == 2
           else BYTES_CFP16_COMPLEX)
    dims = RxRingDims(
        n_corr=args.n_corr,
        n_coarse_dm=args.n_coarse_dm,
        t_buf_samples=args.t_buf_samples,
        n_filled_per_corr=args.n_filled,
        bytes_per_cell=bpc,
    )

    if args.dm_plan_path is not None and args.dm_plan_path.exists():
        coarse_dm, fine_dm, f2c = _dm_grids_from_npz(
            args.dm_plan_path, args.n_coarse_dm
        )
        _LOG.info(
            "DM grids loaded from %s: n_coarse=%d n_fdm_full=%d",
            args.dm_plan_path, len(coarse_dm), len(fine_dm),
        )
    else:
        coarse_dm, fine_dm, f2c = _synthetic_dm_grids(
            args.n_coarse_dm, args.n_fdm
        )
        _LOG.warning(
            "DM plan path %s unavailable; using synthetic grids "
            "(coarse=%d fine=%d) — this is OK for M7.2 fake-data soak "
            "but PRODUCTION must point --dm-plan-path at the corr-side "
            "DmPlan NPZ.",
            args.dm_plan_path, args.n_coarse_dm, args.n_fdm,
        )

    # Optional M7.2 explicit half-owner mapping: each half owns exactly
    # one coarse DM index and all K fine rows around it.
    owner_idx: Optional[int] = None
    if args.gpu_half == 0 and args.coarse_dm_owners_half_0 is not None:
        owner_idx = int(args.coarse_dm_owners_half_0)
    if args.gpu_half == 1 and args.coarse_dm_owners_half_1 is not None:
        owner_idx = int(args.coarse_dm_owners_half_1)
    if owner_idx is not None:
        coarse_dm, fine_dm, f2c = _select_dm_owner_half(
            coarse_dm=coarse_dm,
            fine_dm=fine_dm,
            fine_to_coarse=f2c,
            owner_coarse_idx=owner_idx,
            expected_n_fdm=int(args.n_fdm),
            gpu_half=int(args.gpu_half),
        )
        _LOG.info(
            "M7.2 owner map: gpu_half=%d owns coarse_dm[%d]=%.3f; "
            "n_fdm_local=%d",
            args.gpu_half,
            owner_idx,
            float(coarse_dm[0]),
            int(fine_dm.shape[0]),
        )
    elif fine_dm.shape[0] > args.n_fdm:
        # Legacy fallback (no explicit owner map): trim fine rows so
        # benches can still request small synthetic cubes.
        fine_dm = fine_dm[: args.n_fdm]
        f2c = f2c[: args.n_fdm]

    # M7.5 imager-data-path activation: build the per-chgroup sparsity
    # patterns on the search side so the dense scatter C helper can
    # run. We mirror corr_fast_compute exactly:
    #   - antpos + core-baseline mask loaded from the SAME cal blob the
    #     corr side used for beamformer_weights (the bf header carries
    #     antpos; the sibling yaml carries antenna_order for the
    #     core/outrigger split). This guarantees ``antpos_hash`` agrees.
    #   - dec_deg + n_grid + kernel_support + chan_sum_factor + cell_
    #     lambda_mode all match the corr-side launcher knobs.
    # When --cal-blob-path or --obs-dec-deg is missing we fall back to
    # the M7.2 stub path; downstream cube data will be zeros (sentinel
    # for an unactivated pipeline).
    linear_lut_per_corr: Optional[np.ndarray] = None
    n_filled_per_corr_arr: Optional[np.ndarray] = None
    if args.cal_blob_path is not None and args.obs_dec_deg is not None:
        try:
            from ..grid.sparsity_pattern import (
                build_pattern,
                compute_top_of_band_cell_lambda,
            )
            from ..services.corr_fast_integration import (
                load_antpos_from_cal_blob,
            )
            antpos_e, antpos_n, is_core_mask = load_antpos_from_cal_blob(
                args.cal_blob_path,
            )
            if args.cell_lambda_mode == "common":
                cell_lambda_used = compute_top_of_band_cell_lambda(
                    antpos_e, antpos_n,
                    n_grid=int(args.n_grid),
                    is_core_baseline_mask=is_core_mask,
                )
            else:
                cell_lambda_used = None
            n_corr_local = int(args.n_corr)
            patterns = []
            for c in range(n_corr_local):
                pat = build_pattern(
                    antpos_e, antpos_n,
                    chgroup=c,
                    dec_deg=float(args.obs_dec_deg),
                    n_grid=int(args.n_grid),
                    kernel_support=int(args.kernel_support),
                    chan_sum_factor=int(args.chan_sum_factor),
                    cell_lambda=cell_lambda_used,
                    is_core_baseline_mask=is_core_mask,
                )
                patterns.append(pat)
            n_filled_max = max(int(p.n_filled) for p in patterns)
            ring_n_filled = int(dims.n_filled_per_corr)
            if n_filled_max > ring_n_filled:
                raise ValueError(
                    f"M7.5 LUT: max n_filled across chgroups = "
                    f"{n_filled_max} > ring n_filled_per_corr = "
                    f"{ring_n_filled}. Increase --n-filled to "
                    f"≥ {n_filled_max} or check pattern build inputs."
                )
            # LUT stride = ring slot's n_filled (so the C helper's
            # ``slot_base + 2*k`` cint8 walk + ``lut_c[k]`` LUT walk are
            # zipped on the same k axis the wire ships).
            linear_lut_per_corr = np.zeros(
                (n_corr_local, ring_n_filled), dtype=np.int32,
            )
            n_filled_per_corr_arr = np.zeros(n_corr_local, dtype=np.int32)
            for c, pat in enumerate(patterns):
                n = int(pat.n_filled)
                # ``ix_row * n_grid + ix_col`` — row-major scatter into the
                # dense [n_grid, n_grid] plane; matches the C helper's
                # ``re_plane[lin] = src[2*k]`` line + the test fixture's
                # ``lut[c, k] = ix * n_grid + iy`` convention.
                lin = (
                    pat.ix_row.astype(np.int32) * int(args.n_grid)
                    + pat.ix_col.astype(np.int32)
                )
                linear_lut_per_corr[c, :n] = lin
                n_filled_per_corr_arr[c] = n
            _LOG.info(
                "M7.5 sparsity LUTs built: n_corr=%d n_grid=%d "
                "kernel_support=%d chan_sum=%d cell_lambda_mode=%s "
                "n_filled_per_corr=%s pattern_ids=%s",
                n_corr_local, int(args.n_grid), int(args.kernel_support),
                int(args.chan_sum_factor), args.cell_lambda_mode,
                n_filled_per_corr_arr.tolist(),
                [f"0x{int(p.pattern_id):016x}" for p in patterns],
            )
        except Exception as exc:
            _LOG.exception(
                "M7.5 LUT build failed (%s); falling back to M7.2 "
                "zero-stub path. n_cands will be 0.",
                exc,
            )
            linear_lut_per_corr = None
            n_filled_per_corr_arr = None
    else:
        _LOG.warning(
            "M7.5 activation skipped: cal_blob_path=%s obs_dec_deg=%s. "
            "Falling back to M7.2 zero-stub assembly (cubes will be "
            "zeros; expect n_cands=0).",
            args.cal_blob_path, args.obs_dec_deg,
        )

    source = ProductionRxRingSource(
        shm_name=args.shm_name,
        ring_dims=dims,
        n_fdm_in_cube=int(fine_dm.shape[0]),
        t_det=args.t_det,
        coarse_dm_pc_cm3=coarse_dm,
        fine_dm_pc_cm3=fine_dm,
        fine_to_coarse=f2c,
        compute_half=args.gpu_half,
        t_int_search_us=args.t_int_search_us,
        cube_cadence_samples=args.cube_cadence_samples,
        n_grid=args.n_grid,
        enable_cuda_register=args.cuda_register,
        poll_interval_s=args.poll_interval_s,
        max_cubes=args.max_cubes if args.max_cubes > 0 else None,
        fan_in_min_corrs=args.fan_in_min_corrs,
        attach_timeout_s=args.attach_timeout_s,
        n_active_dms_per_corr=args.n_active_dms_per_corr,
        # M7.6: keep this half within ~real time of the producer so its
        # candidate event_specnums land inside C2's coincidence window
        # (fixes n01 gpu_half=0 stale-lag → C1→C2 drops + too_late dumps).
        max_realtime_lag_samples=(
            int(args.max_realtime_lag_cubes) * int(args.cube_cadence_samples)
            if int(args.max_realtime_lag_cubes) > 0
            else None
        ),
        # M7.4 fix: in the M7.2 fallback path (no scatter wiring) the
        # validity walk in production_rx_ring uses this to know which
        # coarse_dm to expect data in. Without it the walker marks
        # every t invalid because the non-owned-dm slots are never
        # written by the partitioned TX workers.
        owned_coarse_dm=owner_idx,
        # M7.5 scatter activation — when both linear_lut + n_filled are
        # set the source switches to assemble_dense_block (real cint8
        # scatter); otherwise the M7.2 zero-stub path is used.
        # ``DSART_DISABLE_SCATTER=1`` is a debug-only env to force the
        # M7.2 zero-stub path (used by the M7.3 baseline re-bench
        # 2026-05-27 to confirm the non-scatter pipeline still hits
        # 7.45 cubes/s with the M7.4 C1/C2/Option-A bolt-ons present).
        linear_lut_per_corr=(
            None
            if os.environ.get("DSART_DISABLE_SCATTER", "0") == "1"
            else linear_lut_per_corr
        ),
        n_filled_per_corr=(
            None
            if os.environ.get("DSART_DISABLE_SCATTER", "0") == "1"
            else n_filled_per_corr_arr
        ),
        # M7.4 stage-2-absent escape hatch: bake the per-coarse-DM
        # inter-chgroup ν_bot_proc alignment into the search-side
        # shifts. Mandatory for the M7.4 250924mptq replay until the
        # corr-side stage-2 application is wired in.
        include_coarse_offset_in_search_shifts=bool(
            args.include_coarse_offset_in_search_shifts
        ),
    )

    if args.config_yaml is not None and args.config_yaml.exists():
        yaml_doc = _load_yaml(args.config_yaml)
        _LOG.info("loaded search-compute yaml from %s", args.config_yaml)
    else:
        yaml_doc = {}
        if args.config_yaml is not None:
            _LOG.warning(
                "config_yaml=%s not found; using empty doc (all "
                "M6 sub-systems disabled by default).",
                args.config_yaml,
            )

    cfg = _build_search_config_from_yaml(
        yaml_doc,
        n_grid=args.n_grid,
        n_fdm=int(fine_dm.shape[0]),
        gpu_half=args.gpu_half,
        search_node_id=args.search_node_id,
        image_backend=args.image_backend,
        device=args.device,
        enable_clusterer=args.enable_clusterer,
        enable_cube_dump=args.enable_cube_dump,
        enable_udp_listener=args.enable_udp_listener,
        enable_cands_logger=args.enable_cands_logger,
        detector_streaming_tile_size=args.detector_streaming_tile_size,
        detector_streaming_decoder_n_top=args.detector_streaming_decoder_n_top,
        pipeline_overlap=args.pipeline_overlap,
        detector_k_img_csv=args.detector_k_img,
        detector_k_dm_csv=args.detector_k_dm,
        detector_k_time_csv=args.detector_k_time,
        detector_boxcar_accum_dtype=args.detector_boxcar_accum_dtype,
        detector_layer2_max_samples=args.detector_layer2_max_samples,
        layer1_max_samples=args.layer1_max_samples,
        layer1_sigma_floor=args.layer1_sigma_floor,
        fine_dm_pc_cc_full=fine_dm,
        enable_c1=not args.disable_c1,
        c1_bind_host_override=args.c1_bind_host,
    )

    service = SearchComputeService(config=cfg, source=source)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop, stop_event)

    async def _signal_watcher() -> None:
        await stop_event.wait()
        _LOG.info("stop_event fired; signalling service.stop()…")
        await service.stop()

    await service.start()
    watcher_task = asyncio.create_task(_signal_watcher())
    rc = 0
    try:
        await service.run()
    except Exception:  # noqa: BLE001
        _LOG.exception("search_compute service crashed")
        rc = 1
    finally:
        stop_event.set()
        await watcher_task
        # service.stop() may have been invoked by the watcher already;
        # call it again so the source/sub-systems are definitely torn
        # down on the normal exit path.
        await service.stop()
    _LOG.info(
        "search_compute exit: cubes=%d cands=%d clusters=%d rc=%d",
        service.cubes_processed, service.candidates_emitted,
        service.clusters_emitted, rc,
    )
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- Ring binding (M7.2.3 / Phase B contract) ---------------------
    p.add_argument("--shm-name", required=True,
                   help="POSIX-shm name to attach (created by the "
                        "search-rx process, e.g. /dsart-rxring-n01).")
    p.add_argument("--n-corr", type=int, default=16,
                   help="ring dim: number of correlator groups (default: 16)")
    p.add_argument("--n-coarse-dm", type=int, default=5,
                   help="ring dim: number of coarse DM slabs (default: 5 = "
                        "M7.1 op-point)")
    p.add_argument("--t-buf-samples", type=int, default=4096,
                   help="ring dim: time-axis depth in search-cadence samples "
                        "(default 4096 ≈ 16 cube cadences of headroom).")
    p.add_argument("--n-filled", type=int, default=5000,
                   help="ring dim: cells per (corr, dm) slot")
    p.add_argument("--bytes-per-cell", type=int, default=2, choices=(2, 4),
                   help="ring dim: 2=cint8 complex (prod default), "
                        "4=cfp16 complex (debug)")

    # --- Cube assembler tuning ----------------------------------------
    p.add_argument("--n-fdm", type=int, default=16,
                   help="fine-DM trials per cube for THIS half "
                        "(default: 16). Production: 32 per half = 64 total.")
    p.add_argument("--t-det", type=int, default=512,
                   help="detector window length in samples (default: 512 = "
                        "2 × block_samples_search). Must be ≥ cube_cadence "
                        "for the validity_mask to cover every t.")
    p.add_argument("--cube-cadence-samples", type=int,
                   default=CUBE_CADENCE_SAMPLES_DEFAULT,
                   help=f"slots per cube (default: {CUBE_CADENCE_SAMPLES_DEFAULT})")
    p.add_argument("--t-int-search-us", type=float,
                   default=T_INT_SEARCH_US_DEFAULT,
                   help=f"search-cadence sample period in µs "
                        f"(default: {T_INT_SEARCH_US_DEFAULT})")
    p.add_argument("--n-grid", type=int, default=256,
                   help="uv-grid side length (default: 256)")
    p.add_argument("--cuda-register", action="store_true",
                   help="attempt cudaHostRegister on the ring (deferred "
                        "D-item D2; safe to leave off until the C API "
                        "exposes rx_ring_get_base_ptr).")
    p.add_argument("--poll-interval-s", type=float, default=0.001,
                   help="how long the assembler sleeps between write_seq "
                        "polls (default 1ms; production-rate-safe).")
    p.add_argument("--n-active-dms-per-corr", type=int, default=1,
                   help="number of coarse-DM tiles the upstream "
                        "producer ships per (corr, cube) — equals "
                        "popcount(corr_fast coarse_dm_mask). The "
                        "C-side RxRing write_seq_per_corr counter "
                        "increments per (corr, dm, sample) slot "
                        "write, so the consumer-side ``_iter`` must "
                        "scale ``target_seq`` by this factor to wait "
                        "for the full detector window in EVERY active "
                        "dm plane (not just one). Default 1 preserves "
                        "single-dm benchmark / smoke compatibility; "
                        "production 16x1 / 16x4 with "
                        "coarse_dm_mask=0x03 passes 2.")
    p.add_argument("--fan-in-min-corrs", type=int, default=16,
                   help="minimum number of chgroups that must have "
                        "advanced past the next cube boundary before "
                        "we emit a cube (default 16 = production "
                        "strict all-chgroups-required; M7.2 smoke "
                        "should pass 1 to allow partial fan-in).")
    p.add_argument("--max-realtime-lag-cubes", type=int, default=20,
                   help="M7.6: re-seek this half to the live edge "
                        "whenever it lags the producer by more than this "
                        "many cubes, so every gpu_half stays within C2's "
                        "coincidence window (5 s ≈ 37 cubes). Fixes the "
                        "persistent cold-start anchor lag (n01 gpu_half=0 "
                        "~14 s behind) that pushed candidate event_specnums "
                        "outside the C2 window and caused too_late dump "
                        "misses. 0 disables (overrun-only legacy seek).")
    p.add_argument("--attach-timeout-s", type=float, default=30.0,
                   help="wait up to this long for search_rx to create "
                        "the shm ring before giving up (default 30s; "
                        "covers the search_rx 16-port bind + ring "
                        "init lag when both routines are fork-execed "
                        "by dsart_rt in the same verb dispatch).")

    # --- M7.5 imager-data-path activation: per-chgroup sparsity-pattern
    # LUTs so the dense scatter helper in production_rx_ring can run.
    # If --cal-blob-path + --obs-dec-deg are both set, search_compute
    # builds the (n_corr, n_filled_max) linear LUT at startup by calling
    # dsart.grid.sparsity_pattern.build_pattern(...) for every chgroup —
    # mirroring corr_fast_compute (same antpos + dec + n_grid + K_support
    # + chan_sum_factor + cell_lambda_mode), producing bit-identical
    # patterns per the Option-C contract. Without these args the source
    # falls back to the M7.2 zero-stub path (sentinel cubes; n_cands=0).
    p.add_argument("--cal-blob-path", type=Path, default=None,
                   help="path to any beamformer_weights_*.dat blob; used "
                        "to load (antpos_e, antpos_n, is_core_baseline_"
                        "mask) consistent with the corr-side cal. When "
                        "set together with --obs-dec-deg, search_compute "
                        "builds the per-corr SparsityPattern LUTs and "
                        "enables the M7.4 dense scatter.")
    p.add_argument("--obs-dec-deg", type=float, default=None,
                   help="observation declination in degrees. MUST match "
                        "the corr-side --obs-dec-deg (within the "
                        "0.25° quantisation of pattern_id) for the "
                        "per-packet pattern_id check to pass.")
    p.add_argument("--kernel-support", type=int, default=1,
                   help="gridding kernel support in cells (default: 1, "
                        "matching corr_fast_integration.FastIntegration"
                        "Config.kernel_support default; pillbox).")
    p.add_argument("--chan-sum-factor", type=int, default=8,
                   help="F33: collapse this many fine channels per "
                        "chgroup before gridding (must match the "
                        "corr-side --chan-sum-factor; default 8 to "
                        "match the M7.4 launcher).")
    p.add_argument("--cell-lambda-mode", default="common",
                   choices=("common", "per_chgroup"),
                   help="F28 cell-lambda mode (must match corr; default "
                        "'common' = all chgroups share one image-pixel "
                        "grid via compute_top_of_band_cell_lambda).")

    # --- SearchComputeService identity --------------------------------
    p.add_argument("--gpu-half", type=int, default=0, choices=(0, 1),
                   help="which compute half this process serves (0 or 1)")
    p.add_argument("--coarse-dm-owners-half-0", type=int, default=None,
                   help=("M7.2 explicit owner map: global coarse-DM index "
                         "owned by gpu-half=0 on this search node. "
                         "When set, half-0 selects only fine DM rows that map "
                         "to this coarse index and remaps them to local "
                         "coarse index 0."))
    p.add_argument("--coarse-dm-owners-half-1", type=int, default=None,
                   help=("M7.2 explicit owner map: global coarse-DM index "
                         "owned by gpu-half=1 on this search node. "
                         "When set, half-1 selects only fine DM rows that map "
                         "to this coarse index and remaps them to local "
                         "coarse index 0."))
    p.add_argument("--search-node-id", type=int, default=1,
                   help="search node id (1..4 in production)")
    p.add_argument("--device", default="cpu",
                   help="torch device for detector + cube tensor "
                        "(cpu / cuda / cuda:N). Default: cpu")
    p.add_argument("--image-backend", default="cpu",
                   choices=("cpu", "gpu"),
                   help="CubePipeline image backend (cpu / gpu). Default: cpu "
                        "for M7.2 bring-up; flip to gpu once a real GPU is "
                        "available on the search nodes.")
    p.add_argument("--detector-streaming-tile-size", type=int, default=256,
                   help="Streaming detector W-tile size (default 256 for "
                        "commissioning time-only bank on 2080 Ti).")
    p.add_argument("--detector-streaming-decoder-n-top", type=int, default=64,
                   help="Per-kernel top-k budget used by the streaming "
                        "decoder (default 64).")
    p.add_argument("--pipeline-overlap", action="store_true",
                   help="Enable one-cube lookahead overlap: prebuild cube "
                        "N+1 while processing cube N (GPU backend only).")
    p.add_argument("--detector-k-img", type=str, default="unit",
                   help="Comma-separated detector image kernels "
                        f"(subset of {list(DETECTOR_IMAGE_KERNELS)}). "
                        "Commissioning default: unit.")
    p.add_argument("--detector-k-dm", type=str, default="d1",
                   help="Comma-separated detector DM kernels "
                        f"(subset of {list(DETECTOR_DM_KERNELS)}). "
                        "Commissioning default: d1.")
    p.add_argument("--detector-k-time", type=str,
                   default="b1,b2,b4,b8,b16,b32,b64",
                   help="Comma-separated detector time kernels "
                        f"(subset of {list(DETECTOR_TIME_KERNELS)}). "
                        "Commissioning default excludes b128.")
    p.add_argument("--detector-boxcar-accum-dtype", type=str,
                   default="fp16", choices=("default", "fp32", "fp16", "bf16"),
                   help="Cumsum accumulation dtype for streaming detector "
                        "amortised boxcar path. Commissioning default: fp16.")
    p.add_argument("--detector-layer2-max-samples", type=int, default=100000,
                   help="Per-kernel sample cap for Layer-2 interior sigma "
                        "clipping. Set <=0 to disable subsampling. "
                        "Commissioning default: 100000.")
    p.add_argument("--layer1-max-samples", type=int, default=10_000,
                   help="Per-fdm sample cap for Layer-1 sigma-clipped std. "
                        "Commissioning default 100k (vs 1M bench legacy) for "
                        "lower latency with still-sub-percent sigma error.")
    p.add_argument("--layer1-sigma-floor", type=float, default=0.0,
                   help="Lower-bound clamp on the per-fdm Layer-1 sigma, "
                        "applied uniformly across burnin-median and "
                        "post-burnin paths. Suppresses the static-sky-EMA "
                        "warmup transient (cubes 1-2 of a fresh start can "
                        "drop sigma ~200x below steady state). 0.0 (default) "
                        "disables. M7.4 burst-replay recommended: 5e-3.")
    p.add_argument("--include-coarse-offset-in-search-shifts",
                   action="store_true", default=False,
                   help="M7.4 stage-2-absent escape hatch: bake the "
                        "per-coarse-DM inter-chgroup ν_bot_proc alignment "
                        "into the search-side time_shift_search table. "
                        "Required while the corr-side stage-2 application "
                        "is not yet wired (Stage2FIFO is just a ring "
                        "buffer; no per-(g, c) time-shift application "
                        "exists anywhere in transport/). T_stream grows "
                        "from t_det + ~76 to t_det + ~210 samples; the "
                        "RX cint8 history window grows correspondingly. "
                        "Default False once corr-side stage-2 lands.")

    # --- DM plan source -----------------------------------------------
    p.add_argument("--dm-plan-path", type=Path, default=None,
                   help="path to a DmPlan NPZ "
                        "(keys: coarse_dm_pc_cm3, fine_dm_pc_cm3, "
                        "fine_to_coarse). Falls back to a synthetic linear "
                        "grid if omitted/unreadable.")

    # --- M6 sub-system gating + config --------------------------------
    p.add_argument("--config-yaml", type=Path, default=None,
                   help="optional configs/config_compute_search.yaml. "
                        "Only consulted for the sub-system blocks gated "
                        "by --enable-* flags below.")
    p.add_argument("--enable-clusterer", action="store_true")
    p.add_argument("--enable-cube-dump", action="store_true")
    p.add_argument("--enable-udp-listener", action="store_true")
    p.add_argument("--enable-cands-logger", action="store_true")
    p.add_argument("--enable-all", action="store_true",
                   help="shortcut: enable all M6 sub-systems above")
    # --- M7.4 C1 stage gating (defaults ON; off-switch lets benches /
    # --- legacy soaks skip the C1 wiring entirely) --------------------
    p.add_argument("--disable-c1", action="store_true",
                   help="skip M7.4 C1 wiring (merger / SNR / cube "
                        "ring / emit / trigger listener). Default ON; "
                        "use this only for legacy benches.")
    p.add_argument("--c1-bind-host", default=None,
                   help="override c1.dump_listener.bind_host (the "
                        "search-net IP for the C2 trigger listener). "
                        "Production: the orchestrator's hostargs.")

    # --- Lifecycle ----------------------------------------------------
    p.add_argument("--max-cubes", type=int, default=0,
                   help="stop after this many cubes (default 0 = unlimited; "
                        "production: 0, run until SIGTERM).")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))

    args = p.parse_args(argv)
    if args.enable_all:
        args.enable_clusterer = True
        args.enable_cube_dump = True
        args.enable_udp_listener = True
        args.enable_cands_logger = True

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    return asyncio.run(_run_async(args))


if __name__ == "__main__":
    sys.exit(main())
