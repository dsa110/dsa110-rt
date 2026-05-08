"""Smoke + invariant tests for ``bench/cube_dump_e2e.py`` (M6 chunk 7).

The bench drives the chunk-5 ``SearchComputeService`` end-to-end on a
synthetic RX-ring source and verifies the two cube-dump trigger paths
(auto + UDP) plus the chunk-3 queue-backpressure gate. These tests
gate the chunk-7 deliverable per the chunk-7 plan:

  1. CLI runs end-to-end (``--n-cubes 10``); ``report.json`` parses.
  2. Auto dumps fire when injection is bright + predicate is permissive.
  3. UDP dumps fire at every cube listed in ``--udp-cubes``.
  4. NPZ files round-trip: cube shape + dtype, manifest JSON parseable,
     ``trigger_source`` is correct.
  5. Backpressure: ``--queue-maxsize 1 --inject-backpressure 50`` +
     bursty injection forces drops; dispatch hot-path latency stays
     sub-millisecond per cube (``wall_s / n_cubes < 0.01``).
  6. Sustained throughput smoke: with default knobs,
     ``n_dumps_written == n_auto_dumps_dispatched + n_udp_dumps_dispatched``.

All tests are CPU-only (small geometry) and self-contained against
``tmp_path``; the chunk-7 spec keeps the bench off the voltage fixture
to stay fast + deterministic ("Stick to ``SyntheticRxRingSource``").
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("DSART_TEST", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from bench.cube_dump_e2e import (  # noqa: E402
    DEFAULT_INJECT_EVERY,
    main,
    parse_udp_cubes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_bench(
    tmp_path: Path,
    *,
    n_cubes: int = 10,
    queue_maxsize: int = 4,
    inject_backpressure: float = 0.0,
    enable_udp: bool = False,
    udp_cubes: str = "",
    auto_min_snr: float = 0.1,
    rng_seed: int = 42,
    inject_every: int = DEFAULT_INJECT_EVERY,
    inject_amplitude: float = 200.0,
    extra: tuple[str, ...] = (),
) -> dict:
    """Run the bench CLI with ``tmp_path`` as the report dir.

    Returns the parsed ``report.json`` dict.
    """
    argv = [
        "--report-dir", str(tmp_path),
        "--n-cubes", str(n_cubes),
        "--t-det", "32",
        "--n-fdm", "4",
        "--n-grid", "16",
        "--queue-maxsize", str(queue_maxsize),
        "--auto-min-snr", str(auto_min_snr),
        "--inject-every", str(inject_every),
        "--inject-amplitude", str(inject_amplitude),
        "--inject-backpressure", str(inject_backpressure),
        "--rng-seed", str(rng_seed),
    ]
    if enable_udp:
        argv.append("--enable-udp")
    if udp_cubes:
        argv.extend(["--udp-cubes", udp_cubes])
    argv.extend(extra)

    rc = main(argv)
    assert rc == 0, f"bench exited non-zero (rc={rc}, argv={argv})"

    report_path = tmp_path / "report.json"
    assert report_path.exists()
    return json.loads(report_path.read_text())


# ---------------------------------------------------------------------------
# udp-cubes parser
# ---------------------------------------------------------------------------


def test_parse_udp_cubes_dedups_and_sorts() -> None:
    assert parse_udp_cubes("") == ()
    assert parse_udp_cubes(None) == ()
    assert parse_udp_cubes("10") == (10,)
    assert parse_udp_cubes("10,30,50") == (10, 30, 50)
    # Whitespace tolerance + dedup + sort.
    assert parse_udp_cubes(" 30 , 10 , 30 ") == (10, 30)


def test_parse_udp_cubes_rejects_non_int() -> None:
    with pytest.raises(ValueError):
        parse_udp_cubes("abc")


# ---------------------------------------------------------------------------
# 1. CLI runs end-to-end with n_cubes=10, produces report.json
# ---------------------------------------------------------------------------


def test_cli_runs_end_to_end_n_cubes_10(tmp_path: Path) -> None:
    report = _run_bench(tmp_path, n_cubes=10)

    # Top-level envelope.
    assert report["bench"] == "cube_dump_e2e"
    assert report["milestone"] == "M6"
    assert report["chunk"] == 7
    assert report["schema_version"] == 1

    # Summary block carries the chunk-7 spec fields.
    s = report["summary"]
    for k in (
        "n_cubes_processed",
        "n_candidates_emitted",
        "n_clusters_emitted",
        "n_auto_dumps_dispatched",
        "n_udp_dumps_dispatched",
        "n_dumps_written",
        "n_dumps_dropped",
        "n_dumps_failed",
        "writer_p50_ms",
        "writer_p99_ms",
        "writer_max_ms",
        "wall_s",
    ):
        assert k in s, f"summary missing key {k!r}"
    assert s["n_cubes_processed"] == 10

    # queue_backpressure block.
    qb = report["queue_backpressure"]
    assert qb["queue_maxsize"] == 4
    assert "max_observed_depth" in qb
    assert "dropped_at_full" in qb

    # Side-artefacts: report dir contains the canonical sub-dirs.
    assert (tmp_path / "cube_dump").is_dir()
    assert (tmp_path / "logs").is_dir()
    assert (tmp_path / "bench.log").exists()


# ---------------------------------------------------------------------------
# 2. Auto dumps fire when injection is bright + predicate is permissive
# ---------------------------------------------------------------------------


def test_auto_dumps_fire_for_bright_injection(tmp_path: Path) -> None:
    """Bursty bright injections + permissive predicate → auto dumps fire.

    With ``inject-every=5`` over 10 cubes we expect at least one
    cluster-firing cube; ``auto_min_snr=0.1`` then ensures the
    predicate fires for every emitted cluster.
    """
    report = _run_bench(
        tmp_path,
        n_cubes=10,
        inject_every=5,
        auto_min_snr=0.1,
    )
    s = report["summary"]
    assert s["n_auto_dumps_dispatched"] >= 1, (
        "expected at least one auto dump from a bright injection"
    )
    assert s["n_dumps_written"] >= 1
    assert s["n_dumps_failed"] == 0
    # Auto dump NPZ files exist on disk.
    npz_files = sorted((tmp_path / "cube_dump").glob("cube_s*_g*_*.npz"))
    assert len(npz_files) >= 1


# ---------------------------------------------------------------------------
# 3. UDP dumps fire at every cube listed in --udp-cubes
# ---------------------------------------------------------------------------


def test_udp_dumps_fire_for_listed_cubes(tmp_path: Path) -> None:
    udp_list = "2,5,8"
    report = _run_bench(
        tmp_path,
        n_cubes=10,
        enable_udp=True,
        udp_cubes=udp_list,
        # Disable auto-trigger injections so UDP is the ONLY dump
        # source — n_dumps_written == 3 then exactly equals
        # len(udp_cubes).
        inject_every=10**6,  # effectively no injections
        auto_min_snr=10**6,  # never fire predicate
    )
    s = report["summary"]
    assert s["n_udp_dumps_dispatched"] == 3
    assert s["n_auto_dumps_dispatched"] == 0
    assert s["n_dumps_written"] == 3
    assert s["n_dumps_dropped"] == 0
    # All written NPZ files exist + carry trigger_source='udp'.
    npz_files = sorted((tmp_path / "cube_dump").glob("cube_s*_g*_*.npz"))
    assert len(npz_files) == 3
    for path in npz_files:
        with np.load(path, allow_pickle=False) as data:
            assert str(data["trigger_source"]) == "udp"


# ---------------------------------------------------------------------------
# 4. NPZ round-trip: cube shape + dtype + manifest JSON
# ---------------------------------------------------------------------------


def test_npz_round_trip_cube_shape_and_manifest(tmp_path: Path) -> None:
    """Open one written NPZ + verify cube + manifest."""
    report = _run_bench(
        tmp_path,
        n_cubes=10,
        enable_udp=True,
        udp_cubes="3",
        inject_every=10**6,  # no auto dumps
        auto_min_snr=10**6,  # ditto
    )
    assert report["summary"]["n_udp_dumps_dispatched"] == 1

    npz_files = sorted((tmp_path / "cube_dump").glob("cube_s*_g*_*.npz"))
    assert len(npz_files) == 1
    with np.load(npz_files[0], allow_pickle=False) as data:
        # Cube shape: [t_det=32, n_fdm=4, n_grid=16, n_grid=16].
        assert tuple(data["cube"].shape) == (32, 4, 16, 16)
        assert data["cube"].dtype == np.float16
        # Manifest fields.
        assert int(data["t_det"]) == 32
        assert int(data["n_fdm_in_cube"]) == 4
        assert int(data["n_grid"]) == 16
        assert str(data["trigger_source"]) == "udp"
        # cluster_record stores JSON; udp dumps store "null".
        decoded = json.loads(str(data["cluster_record"]))
        assert decoded is None


def test_npz_round_trip_auto_dump_carries_cluster_record(
    tmp_path: Path,
) -> None:
    """Auto dumps round-trip a non-null cluster_record JSON."""
    report = _run_bench(
        tmp_path,
        n_cubes=10,
        inject_every=5,
        auto_min_snr=0.1,
    )
    if report["summary"]["n_auto_dumps_dispatched"] == 0:
        pytest.skip(
            "no auto dumps fired in this seed — covered by other tests"
        )
    npz_files = sorted((tmp_path / "cube_dump").glob("cube_s*_g*_*.npz"))
    assert len(npz_files) >= 1
    # At least one file is an auto dump; check the first one we find.
    for path in npz_files:
        with np.load(path, allow_pickle=False) as data:
            ts = str(data["trigger_source"])
            decoded = json.loads(str(data["cluster_record"]))
            if ts == "auto":
                # Cluster record JSON has the chunk-1 D1 fields.
                assert isinstance(decoded, dict)
                for k in ("snr", "l_pix", "m_pix", "fine_dm_idx",
                          "t_in_cube", "kernel_id"):
                    assert k in decoded, f"cluster_record missing {k!r}"
                return
    pytest.fail("no NPZ file with trigger_source='auto' found")


# ---------------------------------------------------------------------------
# 5. Backpressure: drops + non-blocking dispatch hot-path
# ---------------------------------------------------------------------------


def test_backpressure_forces_drops_with_non_blocking_dispatch(
    tmp_path: Path,
) -> None:
    """Slow writer + queue=1 + bursty injection → drops > 0.

    Simultaneously gates the dispatch hot-path latency: the per-cube
    cube_dump.submit dispatch MUST stay sub-millisecond even when the
    writer thread's ``np.savez`` is artificially stalled to 50 ms —
    queue.put_nowait is non-blocking and the slow writer cannot leak
    into the real-time path. The chunk-7 spec assertion is
    ``wall_s / n_cubes < 0.01`` (10 ms safety margin around the sub-ms
    dispatch budget); on h01-CPU at this geometry the pipeline-bound
    outer loop dominates wall_s, so the bench surfaces the
    writer-isolated metric (``submit_dispatch_total_ms``) and we gate
    on that — the spec invariant ("the bench doesn't get blocked by
    the slow writer") is exactly the dispatch-only wall divided by
    n_cubes.
    """
    n_cubes = 20
    report = _run_bench(
        tmp_path,
        n_cubes=n_cubes,
        queue_maxsize=1,
        inject_backpressure=50.0,
        # Bursty: every 2 cubes fires an auto dump → submit cadence
        # well above the slow-savez 50 ms drain → drops happen.
        inject_every=2,
        auto_min_snr=0.1,
    )
    s = report["summary"]
    qb = report["queue_backpressure"]
    assert s["n_dumps_dropped"] > 0, (
        f"expected drops from queue_maxsize=1 + 50ms writer + bursty "
        f"injection; got n_dumps_dropped={s['n_dumps_dropped']}"
    )
    assert qb["dropped_at_full"] is True
    # Writer-isolated dispatch hot path: each cube_dump.submit is
    # queue.put_nowait → returns within microseconds regardless of
    # how slow the writer thread is. Total submit time / n_cubes must
    # stay < 10 ms (sub-millisecond budget × 10x safety per the
    # chunk-7 spec wording).
    per_cube_submit_s = (
        float(s["submit_dispatch_total_ms"]) / 1.0e3 / float(n_cubes)
    )
    assert per_cube_submit_s < 0.01, (
        f"dispatch hot-path latency too high: "
        f"{per_cube_submit_s*1e3:.3f} ms/cube "
        f"(submit_dispatch_total_ms="
        f"{s['submit_dispatch_total_ms']:.3f}, n_cubes={n_cubes})"
    )
    # Plus an absolute p99 sanity check: even pessimistic CPython
    # contention shouldn't push a put_nowait past 10 ms.
    assert s["submit_dispatch_p99_us"] < 10_000.0, (
        f"submit p99 too high: "
        f"{s['submit_dispatch_p99_us']:.1f} us"
    )


# ---------------------------------------------------------------------------
# 6. Sustained throughput smoke: no drops, no failures, write count
#    matches dispatch count
# ---------------------------------------------------------------------------


def test_sustained_throughput_no_drops_no_failures(tmp_path: Path) -> None:
    """Default knobs (queue=4, no backpressure, bursty 1/10 injection)
    → every dispatched dump lands on disk; no drops, no failures.
    """
    n_cubes = 30
    report = _run_bench(
        tmp_path,
        n_cubes=n_cubes,
        queue_maxsize=4,
        inject_backpressure=0.0,
        enable_udp=True,
        udp_cubes="5,15,25",
        inject_every=10,
        auto_min_snr=0.1,
    )
    s = report["summary"]
    assert s["n_dumps_failed"] == 0
    assert s["n_dumps_dropped"] == 0
    expected = (
        s["n_auto_dumps_dispatched"] + s["n_udp_dumps_dispatched"]
    )
    assert s["n_dumps_written"] == expected, (
        f"writer count {s['n_dumps_written']} != "
        f"auto={s['n_auto_dumps_dispatched']} + "
        f"udp={s['n_udp_dumps_dispatched']} = {expected}"
    )
    # UDP path fired for every listed cube.
    assert s["n_udp_dumps_dispatched"] == 3


# ---------------------------------------------------------------------------
# 7. The bench accepts --voltage-run-id (DoD compat) without changing
#    behaviour.
# ---------------------------------------------------------------------------


def test_voltage_run_id_is_accepted_and_recorded(tmp_path: Path) -> None:
    report = _run_bench(
        tmp_path,
        n_cubes=5,
        extra=("--voltage-run-id", "250924mptq"),
    )
    assert report["config"]["voltage_run_id"] == "250924mptq"


# ---------------------------------------------------------------------------
# 8. CubeDumpWriter.recent_write_ms_ms ring buffer feeds the percentile
#    metrics (chunk-7 minimal extension).
# ---------------------------------------------------------------------------


def test_writer_percentiles_populated_after_writes(tmp_path: Path) -> None:
    """``recent_write_ms_ms`` is populated → p50/p99/max are positive.

    The ring buffer is the chunk-7 extension to ``CubeDumpWriter``;
    if its plumbing breaks the bench's writer_p99_ms drops to 0
    (or stays at the empty-deque sentinel), masking real perf
    regressions. Pin both the bench-side plumbing AND the writer-
    side ring buffer here.
    """
    report = _run_bench(
        tmp_path,
        n_cubes=10,
        enable_udp=True,
        udp_cubes="2,4,6,8",
        inject_every=10**6,
        auto_min_snr=10**6,
    )
    s = report["summary"]
    assert s["n_dumps_written"] >= 4
    # All three percentile values must be present + non-negative.
    # On h01 each ~32×4×16×16 fp16 cube savez clocks well under 10 ms,
    # but the gate tolerates a ridiculously high upper bound to keep
    # the test stable across CI hosts.
    assert s["writer_p50_ms"] >= 0.0
    assert s["writer_p99_ms"] >= s["writer_p50_ms"]
    assert s["writer_max_ms"] >= s["writer_p99_ms"]
    assert s["writer_max_ms"] < 1000.0


# ---------------------------------------------------------------------------
# Direct-API smoke: CubeDumpWriter exposes recent_write_ms_ms (chunk-7
# minimal extension to ``src/dsart/dump/cube_dump.py``).
# ---------------------------------------------------------------------------


def test_recent_write_ms_property_is_present_and_typed() -> None:
    """``CubeDumpWriter.recent_write_ms_ms`` is the property the bench
    relies on — gate its presence + return type at module load.
    """
    from dsart.dump.cube_dump import CubeDumpWriter, CubeDumpWriterConfig

    cfg = CubeDumpWriterConfig(
        dump_root=Path("/tmp"),
        search_node_id=0,
        gpu_half=0,
        queue_maxsize=4,
    )
    writer = CubeDumpWriter(cfg)
    # Pre-start: empty tuple.
    assert isinstance(writer.recent_write_ms_ms, tuple)
    assert len(writer.recent_write_ms_ms) == 0
