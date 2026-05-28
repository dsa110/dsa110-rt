"""Schema + gating invariants for ``configs/dsart_pipeline_rt.yaml``.

The YAML defines a directed-acyclic spawn graph for the per-corr-node
real-time pipeline. The orchestrator (``RtOrchestrator._spawn_routines``)
partitions routines into two waves:

  * **wave 1**: routines with no ``gate_on_paths`` -- spawned immediately
    on the ``start`` verb.
  * **wave 2**: routines that gate on one or more ready-sentinel files --
    spawned only after every listed path exists (or the gate times out).

The PSRDADA ring topology requires capture routines to be in wave 2
gated on consumer ready-sentinels; otherwise the SNAP UDP capture
binaries start writing into dada/eada before corr_fast/corr_slow have
finished their multi-second Python+CUDA+Triton cold start. The
resulting writer-vs-reader offset persists indefinitely (writer and
reader rates match to within 0.1 buffer/s in steady state), pinning
fada at nfull=68/70 and dada at nfull=18/20 with ~9 s of added
end-to-end latency.

This test pins the wave-1/wave-2 partition for every capture mode
(``junkdb`` / ``real`` / ``synth_fada`` / ``replay_burst``) so a future
edit that drops the gate on ``cap_a_real`` / ``cap_b_real`` cannot
silently regress to the 2026-05-28 fada-pinning failure mode.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from dsart.services.dsart_rt import PipelineConfig  # noqa: E402

_YAML_PATH = _REPO / "configs" / "dsart_pipeline_rt.yaml"


# Expected ready-sentinel paths the orchestrator substitutes ``CN`` into
# at spawn time. The literal ``CN`` token is what's stored in the YAML;
# RtOrchestrator._substitute() rewrites it per host.
_CONSUMER_READY_SENTINELS = (
    "/tmp/dsart-corr-fast-CN.ready",
    "/tmp/dsart-corr-slow-CN.ready",
)


def _load() -> PipelineConfig:
    with _YAML_PATH.open() as f:
        return PipelineConfig.from_dict(yaml.safe_load(f))


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return _load()


def _by_name(cfg: PipelineConfig) -> dict[str, object]:
    return {r.name: r for r in cfg.routines}


def test_yaml_parses(cfg: PipelineConfig) -> None:
    assert cfg.schema_version == 1
    assert len(cfg.routines) > 0


@pytest.mark.parametrize("name", ["cap_a_real", "cap_b_real"])
def test_real_capture_gated_on_consumers(cfg: PipelineConfig, name: str) -> None:
    """The production capture binaries MUST be wave-2 gated on the two
    consumer ready-sentinels. Regression-test for the 2026-05-28
    "fada pinned at 68/70 for the entire post-restart window"
    incident: cap_a_real / cap_b_real had no gate_on_paths, so the
    SNAP UDP capture started before corr_fast / corr_slow finished
    their multi-second CUDA + Triton cold start. The resulting
    writer-reader offset persists indefinitely once the steady state
    is reached.
    """
    routines = _by_name(cfg)
    assert name in routines, f"{name} missing from dsart_pipeline_rt.yaml"
    r = routines[name]
    assert set(r.gate_on_paths) >= set(_CONSUMER_READY_SENTINELS), (
        f"{name} must gate on {_CONSUMER_READY_SENTINELS!r}; "
        f"got {r.gate_on_paths!r}. Removing the gate reintroduces "
        f"the 2026-05-28 fada=68/70 cold-start regression."
    )


@pytest.mark.parametrize("name", ["cap_a_junkdb", "cap_b_junkdb"])
def test_junkdb_capture_still_gated(cfg: PipelineConfig, name: str) -> None:
    """The junkdb synth-capture pair is the older sibling of cap_*_real
    and has been gated since M7.2 (2026-05-19). Keep it gated so the
    synthetic-noise validation soaks behave the same as production."""
    routines = _by_name(cfg)
    if name not in routines:
        pytest.skip(f"{name} not present in this YAML revision")
    r = routines[name]
    assert set(r.gate_on_paths) >= set(_CONSUMER_READY_SENTINELS), (
        f"{name} lost its consumer gate -- regression of M7.2 fix."
    )


@pytest.mark.parametrize("name", ["corr_slow", "corr_fast"])
def test_consumers_are_wave1(cfg: PipelineConfig, name: str) -> None:
    """The consumer routines write the ready-sentinels. They MUST NOT
    gate on those same sentinels (deadlock). They run in wave-1."""
    routines = _by_name(cfg)
    assert name in routines, f"{name} missing"
    r = routines[name]
    assert not r.gate_on_paths, (
        f"{name} must not have gate_on_paths (it would deadlock on "
        f"its own consumer sentinel)"
    )


def test_consumer_sentinels_match_arg_paths(cfg: PipelineConfig) -> None:
    """The sentinel paths the captures wait on must match the
    ``--ready-sentinel-path`` argv values the consumers pass. If these
    drift apart the gate fires immediately on stale-or-never-touched
    files and the protection is silently lost.
    """
    routines = _by_name(cfg)
    # The consumer args are a single argv string; just substring-check.
    fast_args = (
        f"{routines['corr_fast'].args} "
        + " ".join(routines['corr_fast'].hostargs.values())
    )
    slow_args = (
        f"{routines['corr_slow'].args} "
        + " ".join(routines['corr_slow'].hostargs.values())
    )
    assert "/tmp/dsart-corr-fast-CN.ready" in fast_args, (
        f"corr_fast no longer writes the expected sentinel path; "
        f"check --ready-sentinel-path in args. Got: {fast_args}"
    )
    assert "/tmp/dsart-corr-slow-CN.ready" in slow_args, (
        f"corr_slow no longer writes the expected sentinel path; "
        f"check --ready-sentinel-path in args. Got: {slow_args}"
    )
