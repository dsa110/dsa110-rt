"""Layer-3 voltage replay (§4.7). M2 Chunk 6 lands the full PSRDADA writer (D7).

Two modes:

  1. **Manifest mode** (operator-supplied continuum/burst fixtures, plan §3.3):
       python -m bench.replay_voltage_dump --run-id <id> --chgroups 0 --rate native

     Reads ``$DSART_VOLTAGE_FIXTURE_ROOT/<run-id>/manifest.yaml`` (validated
     against ``tests/fixtures/voltage_fixture_manifest.schema.json``) and
     ``fl_*_chgroup<g>.out``; writes one block at a time to the fada
     PSRDADA buffer at the requested cadence.

  2. **Synthetic mode** (no fixture; CI / Chunk-6 smoke harness / replay
     M-tests without operator data):
       python -m bench.replay_voltage_dump --synthesize \\
           --synth-thermal-sigma 1.5 --synth-source 0.05,0,4 \\
           --n-blocks 15 --rate native --seed 12345

     Generates deterministic synthetic voltage blocks in-memory using a
     synthetic E-W linear array + DSA band channel grid. Each block is a
     fresh thermal-noise realization with the same continuum sources
     baked in (constant in time, planar wave per channel). Bytes hit
     ``FADA_BYTES_PER_BLOCK`` exactly via the same int4 packing
     ``slow_corr_kernel.unpack_int4_split`` consumes.

Both paths emit the same fada header (synthesized after the legacy
``correlator_header_dsaX.txt`` template — meridian_fringestop's run path
does not read headers, but ``dsamfs.utils.read_header`` requires
``MJD_START`` + ``TSAMP`` so we populate them for forward compat). Per
F8 in M2_PLAN_FIXES.md: the M0 stub ``NotImplementedError("M3 owns the
PSRDADA writer")`` is retired here.

Pacing notes:
  * ``--rate native`` honours the 134.218 ms native cadence between
    block writes via a sleep-to-target schedule.
  * ``--rate fast`` writes as fast as the buffer allows (still throttled
    by consumer drain rate).
  * ``--rate N×`` runs N times faster than native.
  * Backpressure is handled by psrdada-python's ``Writer.__iter__``
    (blocks until a slot is free in the ring).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import concurrent.futures
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple

import numpy as np
import yaml

try:
    from jsonschema import Draft202012Validator
except ImportError as exc:  # pragma: no cover
    raise SystemExit("jsonschema is required for replay_voltage_dump") from exc

# --- repo-local imports (constants live in dsart.common) -------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.common.constants import (  # noqa: E402
    BLOCK_SAMPLES_SPECNUM,
    FADA_BYTES_PER_BLOCK,
    NANTS,
    NCHAN_PER_CHGROUP,
    NPOL,
)

# --- constants -------------------------------------------------------------
BLOCK_MS_NATIVE = 134.218
FIXTURE_ROOT_DEFAULT = "/home/ubuntu/data/voltage_fixtures"

NPACKETS_PER_BLOCK = BLOCK_SAMPLES_SPECNUM
NTIMES_PER_PACKET = 2
_FADA_VOLT_SHAPE: tuple[int, ...] = (
    NPACKETS_PER_BLOCK, NANTS, NCHAN_PER_CHGROUP, NTIMES_PER_PACKET, NPOL,
)
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


# --- IPC keys + buffer config ---------------------------------------------


def _ipc_key(name: str) -> int:
    """Map common 4-char buffer names to PSRDADA hex keys."""
    table = {"dada": 0xDADA, "dadc": 0xDADC, "eada": 0xEADA,
             "fada": 0xFADA, "bada": 0xBADA}
    if name in table:
        return table[name]
    if len(name) == 4 and re.fullmatch(r"[0-9a-fA-F]{4}", name):
        return int(name, 16)
    raise ValueError(f"unsupported buffer key {name!r}")


# --- CLI argument parsing helpers -----------------------------------------


def _parse_chgroups(spec: str) -> list[int]:
    s = spec.strip()
    if ".." in s:
        a, _, b = s.partition("..")
        lo, hi = int(a), int(b)
        return list(range(lo, hi + 1))
    if re.fullmatch(r"\d+-\d+", s):
        lo, _, hi = s.partition("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_rate(arg: str) -> tuple[str, float]:
    a = arg.strip().lower().replace("×", "x")
    if a == "native":
        return "native", BLOCK_MS_NATIVE
    if a == "fast":
        return "fast", 0.0
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*x", a)
    if m:
        mult = float(m.group(1))
        if mult <= 0:
            raise ValueError("rate multiplier must be > 0")
        return arg, BLOCK_MS_NATIVE / mult
    raise ValueError(f"unsupported rate {arg!r}; expected native|fast|N×")


def _parse_source_spec(spec: str) -> tuple[float, float, float]:
    """``l,m,amp`` → (l, m, amp_pre_fluff). Raises ValueError on malformed input."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--synth-source must be 'l,m,amp' (got {spec!r})")
    return float(parts[0]), float(parts[1]), float(parts[2])


# --- header synthesis ------------------------------------------------------


def build_fada_header(
    *,
    utc_start_iso: str | None = None,
    mjd_start: float | None = None,
    source_label: str = "SYNTH",
    dec_deg: float = 0.0,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a sane fada PSRDADA header dict (ASCII-only string values).

    Mirrors the legacy ``correlator_header_dsaX.txt`` template so M2
    replays carry the same provenance keys as on-sky data. ``MJD_START``
    + ``TSAMP`` are populated even though ``run_fringestopping`` does
    not read them — ``dsamfs.utils.read_header`` does, and it's good
    forward compat (and free).
    """
    if utc_start_iso is None:
        utc_start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H:%M:%S")
    if mjd_start is None:
        unix_s = datetime.now(timezone.utc).timestamp()
        mjd_start = unix_s / 86400.0 + 40587.0

    # Per-sample tsamp = block_duration / N_TIME_SAMPLES = 134.218 ms / 4096
    tsamp_s = (BLOCK_MS_NATIVE * 1e-3) / (NPACKETS_PER_BLOCK * NTIMES_PER_PACKET)

    hdr: dict[str, str] = {
        # PSRDADA / DSAX housekeeping
        "HDR_VERSION": "1.0",
        "HDR_SIZE": "4096",
        "RESOLUTION": "4096",
        "INSTRUMENT": "DSA-RT",
        "MODE": "RAW",
        "TELESCOPE": "DSA-110",
        "OBSERVER": "M2_REPLAY",
        "RECEIVER": "SANDY",
        "PID": "M2",
        # Timing
        "UTC_START": utc_start_iso,
        "MJD_START": f"{mjd_start:.9f}",
        "TSAMP": f"{tsamp_s:.12g}",
        # Sky / source provenance
        "SOURCE": source_label,
        "RA": "00:00:00.000",
        "DEC": f"{dec_deg:+.4f}",
        "CFREQ": "1473.75",
        "FREQ": "1473.750",
        "BANDWIDTH": "187.5",
        "BW": "187.5",
        # Voltage layout (FADA)
        "NBIT": "4",
        "NDIM": "2",
        "NPOL": str(NPOL),
        "NCHAN": str(NCHAN_PER_CHGROUP),
        "NANT": str(NANTS),
        "ANTENNAS": "1-96",
        # File / transfer bookkeeping
        "FILE_NUMBER": "0",
        "FILE_SIZE": str(FADA_BYTES_PER_BLOCK),
        "TRANSFER_SIZE": str(FADA_BYTES_PER_BLOCK),
        "ACC_LEN": "1",
        "FSCRUNCH": "1",
        "TSCRUNCH": "1",
        "NBEAM": "1",
        "N_PROD": "1",
        "CHAN_AV": "0",
        "DSB": "0",
        "OBS_OFFSET": "0",
        "OBS_UNIT": "SECONDS",
        "OBS_VAL": "0000.0000",
        # M2 provenance
        "DSART_PRODUCER": "replay_voltage_dump",
    }
    if extra:
        hdr.update({k: str(v) for k, v in extra.items()})
    return hdr


# --- synthesis helpers -----------------------------------------------------


def _quantize_to_int4(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.round(arr), -8, 7).astype(np.int8)


def _pack_int4_bytes(real_q: np.ndarray, imag_q: np.ndarray) -> np.ndarray:
    """Pack real/imag int4 nibbles into uint8 bytes (low=real, high=imag)."""
    real_u4 = (real_q.astype(np.int8) & 0x0F).astype(np.uint8)
    imag_u4 = (imag_q.astype(np.int8) & 0x0F).astype(np.uint8)
    return ((imag_u4 & 0x0F) << 4) | real_u4


# ---- parallel noise-only synthesis (used by the M7.4 synth_fada test) -----
#
# numpy's PRNG is single-threaded for ``rng.normal`` / ``standard_normal``,
# so a (2048, 96, 384, 2, 2) draw takes ~6-15 s/block on the corr nodes.
# We shard along the packet axis across a persistent ProcessPoolExecutor.
# Each worker draws its slice with an independent ``SeedSequence``-derived
# Generator, quantises to int4, and packs to uint8 — returning bytes only,
# so the parent never holds the float32 intermediate (avoids 5 GB peak).

_NOISE_POOL: Optional["concurrent.futures.ProcessPoolExecutor"] = None
_NOISE_POOL_WORKERS: int = 0


def _noise_chunk_worker(args: Tuple[int, int, float, Tuple[int, ...]]) -> np.ndarray:
    """ProcessPool worker: draw + quantise + pack one packet-axis shard.

    Returns packed uint8 bytes of shape ``(n_packets, NANTS, NCHAN, 2t,
    NPOL)``. The seed is derived from (block_idx, worker_idx) so each
    block is reproducible-but-distinct, and across workers within a
    block the streams are independent (SeedSequence.spawn semantics).
    """
    block_idx, worker_idx, sigma, shape = args
    ss = np.random.SeedSequence([0x5713_5A09, int(block_idx), int(worker_idx)])
    rng = np.random.default_rng(ss)
    real_f = rng.standard_normal(size=shape, dtype=np.float32) * np.float32(sigma)
    imag_f = rng.standard_normal(size=shape, dtype=np.float32) * np.float32(sigma)
    real_q = _quantize_to_int4(real_f)
    imag_q = _quantize_to_int4(imag_f)
    return _pack_int4_bytes(real_q, imag_q)


def _get_noise_pool(workers: int) -> "concurrent.futures.ProcessPoolExecutor":
    """Lazily create / reuse a process pool sized to ``workers``."""
    global _NOISE_POOL, _NOISE_POOL_WORKERS
    if _NOISE_POOL is None or _NOISE_POOL_WORKERS != workers:
        import concurrent.futures
        if _NOISE_POOL is not None:
            _NOISE_POOL.shutdown(wait=False, cancel_futures=True)
        _NOISE_POOL = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        _NOISE_POOL_WORKERS = workers
    return _NOISE_POOL


def _synthesize_block_noise_parallel(
    sigma: float,
    block_idx: int,
    workers: int = 8,
) -> np.ndarray:
    """Generate ``_FADA_VOLT_SHAPE`` of int4-packed gaussian noise in parallel.

    Shards along the leading NPACKETS axis (2048). ``workers`` must
    divide NPACKETS; default 8 gives 256 packets/worker, which fits in
    ~1 GB of float32 + int8 in each worker before packing.
    """
    npackets = _FADA_VOLT_SHAPE[0]
    if npackets % workers != 0:
        raise ValueError(
            f"workers={workers} must divide NPACKETS={npackets}"
        )
    per = npackets // workers
    chunk_shape = (per,) + _FADA_VOLT_SHAPE[1:]
    pool = _get_noise_pool(workers)
    args = [(block_idx, w, sigma, chunk_shape) for w in range(workers)]
    chunks = list(pool.map(_noise_chunk_worker, args))
    out = np.concatenate(chunks, axis=0).reshape(-1)
    if out.nbytes != FADA_BYTES_PER_BLOCK:
        raise RuntimeError(
            f"parallel noise block size {out.nbytes} != {FADA_BYTES_PER_BLOCK}"
        )
    return out


def antpos_synth_2d_grid(
    nants: int = NANTS, n_x: int = 12, n_y: int = 8,
    spacing_m: float = 0.5,
) -> np.ndarray:
    """Synthetic 2D antenna grid for smoke-test geometry.

    Default ``96 ants = 12 × 8`` with 0.5 m spacing in both x (east) and
    y (north). This gives a 5.5 m × 3.5 m physical aperture → 2D uv
    coverage, so dirty images aren't degenerate along one axis. (Real
    DSA-110 layout is 1D-dominated but still has slight N-S extent;
    this test array is more isotropic to exercise the gridder.)
    """
    if n_x * n_y != nants:
        raise ValueError(f"{n_x} × {n_y} = {n_x * n_y} != nants={nants}")
    pos = np.zeros((nants, 3), dtype=np.float64)
    for k in range(nants):
        ix = k % n_x
        iy = k // n_x
        pos[k, 0] = spacing_m * ix          # east
        pos[k, 1] = spacing_m * iy          # north
    return pos


# Kept for backward compatibility; superseded by antpos_synth_2d_grid.
antpos_linear_ew = antpos_synth_2d_grid


def channel_freqs_hz(
    nchan: int = NCHAN_PER_CHGROUP,
    nu_top_GHz: float = 1.5,
    nu_bot_GHz: float = 1.45,
) -> np.ndarray:
    """Decreasing per dsa convention."""
    return np.linspace(nu_top_GHz, nu_bot_GHz, nchan) * 1e9


def synthesize_block(
    *,
    block_idx: int,
    rng: np.random.Generator,
    thermal_sigma_pre_fluff: float = 0.0,
    continuum_sources: Iterable[tuple[float, float, float]] = (),
    antenna_pos_m: np.ndarray | None = None,
    nu_Hz: np.ndarray | None = None,
) -> np.ndarray:
    """Generate one fada block of synthetic voltage data (302 MB packed bytes).

    Parameters
    ----------
    block_idx : int
        Used as part of the per-block thermal-noise stream (each block
        gets its own RNG draws so consecutive blocks are independent).
    rng : np.random.Generator
        Pre-seeded RNG (reused for all blocks; sequential calls ok).
    thermal_sigma_pre_fluff : float
        Per-component Gaussian std in pre-fluff (int4) units.
        0 = no thermal (pure source).
    continuum_sources : iterable of (l, m, amp_pre_fluff)
        Each entry adds a planar wave at direction cosines (l, m) with
        pre-fluff complex amplitude `amp`. Source is constant in time
        (point continuum source over the block duration).
    antenna_pos_m, nu_Hz : optional
        Override the synthetic E-W array / DSA-band channel grid.

    Returns
    -------
    raw_bytes : ndarray of uint8, length FADA_BYTES_PER_BLOCK
        Packed int4 complex voltages, C-order under
        `_FADA_VOLT_SHAPE = (NPACKETS=2048, NANTS=96, NCHAN=384, 2t, NPOL=2)`.
    """
    if antenna_pos_m is None:
        antenna_pos_m = antpos_synth_2d_grid()
    if nu_Hz is None:
        nu_Hz = channel_freqs_hz()

    sources_list = list(continuum_sources)
    # Fast path (no continuum sources): generate float32 real + imag noise
    # in parallel across worker processes (each draws 1/8 of the block)
    # and pack to int4 inside the workers, so this function returns
    # already-packed uint8 bytes ready for fada. Without parallelism the
    # original path needed ~14-17 s/block on the corr nodes
    # (single-threaded ``rng.normal`` of a (2048, 96, 384, 2, 2)
    # complex128 buffer ≈ 5 GB); the persistent ProcessPoolExecutor pool
    # drops that to ~2.5 s/block with 8 workers — close enough to native
    # cadence (134 ms) that the producer/consumer rate mismatch in the
    # search-side rx ring stops dropping cubes.
    if not sources_list and thermal_sigma_pre_fluff > 0:
        return _synthesize_block_noise_parallel(
            sigma=float(thermal_sigma_pre_fluff),
            block_idx=block_idx,
        )

    if thermal_sigma_pre_fluff > 0:
        E = (
            rng.normal(0, thermal_sigma_pre_fluff, size=_FADA_VOLT_SHAPE)
            + 1j * rng.normal(0, thermal_sigma_pre_fluff, size=_FADA_VOLT_SHAPE)
        )
    else:
        E = np.zeros(_FADA_VOLT_SHAPE, dtype=np.complex128)

    for l, m, amp in sources_list:
        n_dir = float(np.sqrt(max(0.0, 1.0 - l * l - m * m)))
        s_hat = np.array([l, m, n_dir], dtype=np.float64)
        bdotS = antenna_pos_m @ s_hat                                # (NANTS,)
        phase = 2 * np.pi * bdotS[:, None] * nu_Hz[None, :] / SPEED_OF_LIGHT_M_PER_S
        E_ant_ch = amp * np.exp(1j * phase)                          # (NANTS, NCHAN)
        # Constant in time + identical across both pols (unpolarized source).
        E += E_ant_ch[None, :, :, None, None]

    real_q = _quantize_to_int4(E.real)
    imag_q = _quantize_to_int4(E.imag)
    block_bytes = _pack_int4_bytes(real_q, imag_q).reshape(-1)
    if block_bytes.nbytes != FADA_BYTES_PER_BLOCK:
        raise RuntimeError(
            f"synth block size {block_bytes.nbytes} != {FADA_BYTES_PER_BLOCK}"
        )
    # Free intermediates eagerly (302 MB each).
    del E, real_q, imag_q
    return block_bytes


# --- writer driver ---------------------------------------------------------


def write_blocks_to_fada(
    block_iter: Iterator[np.ndarray],
    *,
    fada_key: int,
    header: dict[str, str],
    n_blocks: int,
    pace_ms: float,
    log: Callable[[str], None] = print,
    mark_eod: bool = True,
) -> dict[str, Any]:
    """Connect to fada writer, push n_blocks at the requested pacing.

    `block_iter` yields one ndarray (uint8, length FADA_BYTES_PER_BLOCK)
    per block. Pacing: `pace_ms` controls minimum inter-block spacing
    (0 = unlimited). Returns timing summary.
    """
    try:
        from psrdada import Writer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("psrdada-python is required for live replay") from exc

    writer = Writer(fada_key)
    try:
        writer.setHeader(header)
        log(f"set fada header: {len(header)} keys")

        t0 = time.monotonic()
        per_block_ms: list[float] = []
        n_written = 0
        # markEndOfData() in psrdada-python must be called from INSIDE the
        # writer iteration (the "writing" state ends as soon as the for loop
        # exits naturally; after that the call raises "not writing"). So we
        # iterate one extra page at the end to mark EOD if requested.
        for page in writer:
            if n_written >= n_blocks:
                if mark_eod:
                    writer.markEndOfData()
                break

            try:
                block_bytes = next(block_iter)
            except StopIteration:
                log(f"block iterator exhausted at bi={n_written}; stopping")
                if mark_eod:
                    writer.markEndOfData()
                break
            if block_bytes.nbytes != FADA_BYTES_PER_BLOCK:
                raise RuntimeError(
                    f"block #{n_written} size {block_bytes.nbytes} != "
                    f"{FADA_BYTES_PER_BLOCK}"
                )

            page_arr = np.asarray(page)
            if page_arr.nbytes != FADA_BYTES_PER_BLOCK:
                raise RuntimeError(
                    f"fada page size {page_arr.nbytes} != "
                    f"FADA_BYTES_PER_BLOCK ({FADA_BYTES_PER_BLOCK}) — "
                    f"buffer mis-sized; check `dada_db -b`."
                )

            t_block_start = time.monotonic()
            page_arr[:] = block_bytes
            writer.markFilled()
            inner_ms = (time.monotonic() - t_block_start) * 1000.0
            per_block_ms.append(inner_ms)
            n_written += 1

            if pace_ms > 0:
                target_elapsed_s = n_written * pace_ms * 1e-3
                actual_elapsed_s = time.monotonic() - t0
                sleep_for = target_elapsed_s - actual_elapsed_s
                if sleep_for > 0:
                    time.sleep(sleep_for)

            log(f"wrote block {n_written}/{n_blocks} ({inner_ms:.1f} ms inner)")

        elapsed = time.monotonic() - t0
        log(f"wrote {len(per_block_ms)} blocks in {elapsed:.1f}s "
            f"(pace {pace_ms} ms/block)")
        return {
            "n_blocks_written": len(per_block_ms),
            "elapsed_s": elapsed,
            "ms_per_block_p50": float(np.median(per_block_ms)) if per_block_ms else float("nan"),
            "ms_per_block_p99": float(np.percentile(per_block_ms, 99)) if per_block_ms else float("nan"),
        }
    finally:
        try:
            writer.disconnect()
        except Exception:
            pass


# --- manifest mode ---------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _schema_path() -> Path:
    return _repo_root() / "tests" / "fixtures" / "voltage_fixture_manifest.schema.json"


def _pick_voltage_file(fixture_dir: Path, chgroup: int) -> Path:
    pat = f"fl_*_chgroup{chgroup}.out"
    matches = sorted(fixture_dir.glob(pat))
    if not matches:
        raise FileNotFoundError(f"{fixture_dir}: no files matching {pat}")
    return matches[0]


def _probe_binary_header(path: Path, nbytes: int = 8192) -> dict[str, object]:
    raw = path.open("rb").read(nbytes)
    out: dict[str, object] = {
        "probe_bytes": min(nbytes, len(raw)),
        "sha256_prefix_hex": None,
    }
    if len(raw) >= 32:
        import hashlib
        out["sha256_prefix_hex"] = hashlib.sha256(raw[:1024]).hexdigest()[:16]
    try:
        txt = raw.decode("ascii", errors="ignore")
        if "HDR_VERSION" in txt or "DADA" in txt:
            out["ascii_probe_has_dada_keywords"] = True
    except Exception:  # pragma: no cover
        pass
    return out


def _dry_run_report_manifest(
    *,
    run_id: str,
    chgroups: list[int],
    rate_label: str,
    pace_ms: float,
    manifest: dict[str, object],
    vol_path: Path,
    inject_noise: bool,
    probe: dict[str, object],
) -> str:
    fk = manifest.get("fixture_kind")
    nb = manifest.get("n_blocks")
    sz = vol_path.stat().st_size
    inferred = sz // nb if (isinstance(nb, int) and nb > 0 and sz % nb == 0) else None
    pace_note = (
        f"pace_target_ms={pace_ms:.6g}" if pace_ms > 0
        else "pace_target=unlimited_fast"
    )
    inf_note = (
        f"inferred_bytes_per_block={inferred}" if inferred is not None
        else "inferred_bytes_per_block=non_integer_division"
    )
    return (
        f"replay_voltage_dump dry-run (manifest): fixture_kind={fk!s} "
        f"run_id={run_id!r} chgroups={chgroups} rate={rate_label!r} "
        f"native_block_ms={BLOCK_MS_NATIVE} {pace_note} "
        f"voltage_file={vol_path.name} file_bytes={sz} manifest_n_blocks={nb} "
        f"{inf_note} inject_noise={inject_noise} header_probe={probe}"
    )


def _load_manifest(run_id: str, root: Path) -> tuple[dict[str, object], Path]:
    fixture_dir = root / run_id
    if not fixture_dir.is_dir():
        raise SystemExit(f"missing fixture dir {fixture_dir}")

    schema = json.loads(_schema_path().read_text())
    Draft202012Validator.check_schema(schema)

    manifest_path = fixture_dir / "manifest.yaml"
    if not manifest_path.is_file():
        raise SystemExit(f"missing {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise SystemExit("manifest.yaml root must be a mapping")
    Draft202012Validator(schema).validate(manifest)
    return manifest, fixture_dir


def _iter_fixture_blocks(vol_path: Path) -> Iterator[np.ndarray]:
    """Yield FADA_BYTES_PER_BLOCK-sized uint8 arrays from the fixture file."""
    with vol_path.open("rb") as f:
        while True:
            chunk = f.read(FADA_BYTES_PER_BLOCK)
            if not chunk:
                break
            if len(chunk) != FADA_BYTES_PER_BLOCK:
                raise RuntimeError(
                    f"truncated block at end of {vol_path.name}: "
                    f"got {len(chunk)} bytes, expected {FADA_BYTES_PER_BLOCK}"
                )
            yield np.frombuffer(chunk, dtype=np.uint8)


def _run_manifest(ns: argparse.Namespace, chgroups: list[int],
                  rate_label: str, pace_ms: float) -> int:
    """Manifest mode: dry-run report OR live PSRDADA write of fixture blocks."""
    root = Path(os.environ.get("DSART_VOLTAGE_FIXTURE_ROOT", FIXTURE_ROOT_DEFAULT))
    manifest, fixture_dir = _load_manifest(ns.run_id, root)
    vol_path = _pick_voltage_file(fixture_dir, chgroups[0])

    if ns.dry_run:
        probe = _probe_binary_header(vol_path)
        print(_dry_run_report_manifest(
            run_id=ns.run_id, chgroups=chgroups, rate_label=rate_label,
            pace_ms=pace_ms, manifest=manifest, vol_path=vol_path,
            inject_noise=ns.inject_noise, probe=probe,
        ))
        return 0

    n_blocks_manifest = int(manifest.get("n_blocks", 15))
    n_blocks = ns.n_blocks if ns.n_blocks > 0 else n_blocks_manifest
    if n_blocks > n_blocks_manifest:
        print(
            f"WARNING: --n-blocks {n_blocks} > manifest n_blocks "
            f"{n_blocks_manifest}; capping",
            file=sys.stderr,
        )
        n_blocks = n_blocks_manifest

    fixture_kind = str(manifest.get("fixture_kind", "?"))
    dec_deg = float(manifest.get("dec_deg", 0.0))
    utc_start = str(manifest.get("utc_start_iso", ""))
    extra = {
        "FIXTURE_KIND": fixture_kind,
        "FIXTURE_RUN_ID": ns.run_id,
        "FIXTURE_CHGROUP": str(chgroups[0]),
    }
    if utc_start:
        extra["UTC_START"] = utc_start
    header = build_fada_header(
        utc_start_iso=utc_start or None,
        source_label=fixture_kind.upper(),
        dec_deg=dec_deg,
        extra=extra,
    )

    fada_key = _ipc_key(ns.fada_key)
    print(f"manifest replay: run_id={ns.run_id} chgroup={chgroups[0]} "
          f"file={vol_path.name} ({vol_path.stat().st_size} B) "
          f"n_blocks={n_blocks} pace_ms={pace_ms}")
    summary = write_blocks_to_fada(
        _iter_fixture_blocks(vol_path),
        fada_key=fada_key, header=header, n_blocks=n_blocks, pace_ms=pace_ms,
    )
    print(f"manifest replay summary: {summary}")
    return 0


# --- synthesize mode -------------------------------------------------------


def _synth_block_iter(
    *, n_blocks: int, seed: int, thermal_sigma: float,
    continuum_sources: list[tuple[float, float, float]],
) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(seed)
    for bi in range(n_blocks):
        yield synthesize_block(
            block_idx=bi, rng=rng,
            thermal_sigma_pre_fluff=thermal_sigma,
            continuum_sources=continuum_sources,
        )


def _run_synthesize(ns: argparse.Namespace, rate_label: str, pace_ms: float) -> int:
    """Synthetic mode: dry-run summary OR live PSRDADA write of synth blocks."""
    sources = [_parse_source_spec(s) for s in ns.synth_source]
    spec_msg = (
        f"synth: n_blocks={ns.n_blocks} thermal_sigma={ns.synth_thermal_sigma} "
        f"sources={sources} seed={ns.seed}"
    )
    if ns.dry_run:
        print(f"replay_voltage_dump dry-run (synthesize): {spec_msg} "
              f"rate={rate_label!r} pace_ms={pace_ms} "
              f"bytes_per_block={FADA_BYTES_PER_BLOCK}")
        return 0

    extra = {
        "SYNTH_THERMAL_SIGMA_PRE_FLUFF": str(ns.synth_thermal_sigma),
        "SYNTH_SEED": str(ns.seed),
        "SYNTH_N_BLOCKS": str(ns.n_blocks),
        "SYNTH_SOURCES": json.dumps(sources),
    }
    header = build_fada_header(source_label="SYNTH", extra=extra)

    fada_key = _ipc_key(ns.fada_key)
    print(spec_msg + f" fada_key=0x{fada_key:04x}")
    summary = write_blocks_to_fada(
        _synth_block_iter(
            n_blocks=ns.n_blocks, seed=ns.seed,
            thermal_sigma=ns.synth_thermal_sigma,
            continuum_sources=sources,
        ),
        fada_key=fada_key, header=header, n_blocks=ns.n_blocks, pace_ms=pace_ms,
    )
    print(f"synth replay summary: {summary}")
    return 0


# --- main ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Mode selection (mutually exclusive)
    ap.add_argument("--run-id", help="manifest fixture run id (manifest mode)")
    ap.add_argument("--synthesize", action="store_true",
                    help="generate synthetic blocks in-memory (no fixture)")

    ap.add_argument("--chgroups", default="0", help='e.g. "0", "0,1", "0..15"')
    ap.add_argument("--rate", default="native", help="native | fast | N×")
    ap.add_argument("--inject-noise", action="store_true",
                    help="(reserved) annotate dry-run report only")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fada-key", default="fada", help="4-char fada IPC key")
    ap.add_argument("--n-blocks", type=int, default=15,
                    help="number of blocks to write (defaults to manifest "
                         "value in manifest mode)")
    ap.add_argument("--seed", type=int, default=12345,
                    help="RNG seed (synth mode)")
    # Synthesis spec
    ap.add_argument("--synth-thermal-sigma", type=float, default=0.0,
                    help="per-component Gaussian std in pre-fluff units "
                         "(synth mode)")
    ap.add_argument("--synth-source", action="append", default=[],
                    help="add continuum source 'l,m,amp_pre_fluff' "
                         "(repeatable; synth mode)")

    ns = ap.parse_args(argv)

    if ns.synthesize and ns.run_id:
        print("ERROR: --synthesize and --run-id are mutually exclusive",
              file=sys.stderr)
        return 2
    if not ns.synthesize and not ns.run_id:
        print("ERROR: must specify either --run-id or --synthesize",
              file=sys.stderr)
        return 2

    rate_label, pace_ms = _parse_rate(ns.rate)
    chgroups = _parse_chgroups(ns.chgroups)

    if ns.synthesize:
        return _run_synthesize(ns, rate_label, pace_ms)
    return _run_manifest(ns, chgroups, rate_label, pace_ms)


if __name__ == "__main__":
    raise SystemExit(main())
