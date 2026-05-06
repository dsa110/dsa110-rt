"""Fast correlator service entry (M3 chunk 2b; plan §4.2 + §8 line 2262).

**Spine version** of `corr_fast_compute`: reads `fada` PSRDADA blocks,
unpacks int4 voltages (M2 D15 / D16), applies the F21 DEC-phased cal
(M3 chunk 1), runs the FastCorrKernel (M3 chunk 2a), pol-sums to
Stokes I, writes the per-block fast-vis tensor to disk for inspection.

The full corr-side pipeline (RFI flagger + cal + GEMM + static-sky +
coarse-DM + gridder + transport-TX) gets wired up in chunk 4
(`corr_fast_integration`); this service is the chunks 1-2a integration
proof — it runs end-to-end against synthetic + real fada voltages and
validates that the cal-apply + GEMM compose correctly. Until chunks
3a / 3b / 3c land, the on-disk output is the full
``(n_fast_vis_per_block, NBASE, NCHAN) cfp32`` Stokes-I tensor (or a
single-tile slice via ``--blocks-output-mode first_tile_only`` for
debug runs on disk-constrained hosts).

Per PARALLEL_AGENTS.md §4: this service runs on h01 GPU 0 with
``DSART_BUFFER_KEY_PREFIX=m3`` (which maps the legacy ``fada`` key to
``fa3a`` per the dsart buffer-key prefix rule) so it can run alongside
the dsart-search-compute@01-1 (M5) instance on GPU 1 without
PSRDADA-buffer or GPU contention.

CLI:

    python -m dsart.services.corr_fast_compute \\
        [--fada-key fada]
        [--device auto]
        [--max-blocks N]
        [--t-int-fast-native 8 (default 262.144 µs cadence)]
        [--obs-dec-deg 53.848986]
        [--apply-cal /path/to/beamformer_weights_*.dat]
        [--cal-mode phase_only|full]
        [--cal-pol-swap]
        [--output-dir /tmp/dsart-fast-vis]
        [--blocks-output-mode full|first_tile_only]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dsart.cal.cal_loader import (
    CalMode,
    FastCorrCalTensors,
    load_cal_with_dec_phase,
)
from dsart.common.config_loader import load
from dsart.common.constants import (
    FADA_BYTES_PER_BLOCK,
    NATIVE_SAMPLE_US,
    PHI_LAT_OVRO_DEG,
    T_INT_FAST_NATIVE,
)
from dsart.services.corr_fast_kernel import (
    FastCorrKernel,
    stokes_i_pol_sum,
)
from dsart.services.slow_corr_kernel import (
    apply_cal_split,
    unpack_int4_split,
)


LOG = logging.getLogger("corr_fast_compute")

DEFAULT_CONFIG_PATH = Path("/home/ubuntu/proj/dsa110-rt/configs/config_corr.yaml")


# ---------------------------------------------------------------------------
# Helpers (mirror corr_slow_compute.py — same semantics, fast-corr inputs)
# ---------------------------------------------------------------------------


def _key_to_int(key_str: str) -> int:
    """4-char buffer key string → DADA hex int (mirror dsamfs.routines)."""
    if len(key_str) != 4:
        raise ValueError(f"buffer key must be 4 chars, got {key_str!r}")
    return int(f"0x{key_str}", 16)


def _pick_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _load_fada_buffer_size(cfg_path: Path) -> int:
    """Return ``fada bytes_per_block`` from config_corr.yaml."""
    cfg = load(cfg_path)
    return int(cfg["buffers"]["fada"]["bytes_per_block"])


class _StopRequested(Exception):
    """Raised by SIGTERM / SIGINT handler to break the main loop."""


def _install_signals(state: dict[str, Any]) -> None:
    def handle(signum, _frame):
        LOG.info("received signal %d, requesting stop", signum)
        state["stop"] = True
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)


def _build_cal_tensors_with_f21(
    cal_path: Path,
    *,
    chgroup: int,
    obs_dec_rad: float,
    cal_mode: str,
    cal_pol_swap: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> FastCorrCalTensors:
    """Thin wrapper around :func:`load_cal_with_dec_phase` with logging.

    Logs cal-blob provenance + F21 DEC-phase parameters so post-mortem
    on the on-disk per-block JSON matches what the service actually
    used at runtime.
    """
    out = load_cal_with_dec_phase(
        cal_path,
        chgroup=chgroup,
        obs_dec_rad=obs_dec_rad,
        cal_mode=cal_mode,
        pol_swap=cal_pol_swap,
        device=device,
        dtype=dtype,
    )
    LOG.info(
        "cal: %s mode=%s pol_swap=%s n_flagged=%d mag_p50=%.3g mag_p99=%.3g",
        out.info["cal_path"], out.info["cal_mode"], out.info["pol_swap"],
        out.info["n_flagged"],
        out.info.get("cal_mag_p50", float("nan")),
        out.info.get("cal_mag_p99", float("nan")),
    )
    LOG.info(
        "F21 DEC-phase: chgroup=%d obs_dec_deg=%.4f phi_lat_deg=%.3f "
        "(delta_dec_deg=%+.4f)",
        chgroup, out.info["obs_dec_deg"], PHI_LAT_OVRO_DEG,
        out.info["obs_dec_deg"] - PHI_LAT_OVRO_DEG,
    )
    return out


# ---------------------------------------------------------------------------
# Per-block compute (the "spine"): unpack → cal → GEMM → Stokes-I
# ---------------------------------------------------------------------------


def compute_block(
    raw: np.ndarray,
    *,
    kernel: FastCorrKernel,
    cal: FastCorrCalTensors | None,
    voltage_dtype: torch.dtype,
) -> torch.Tensor:
    """One block: int4 raw bytes → (n_fast_vis, NBASE, NCHAN) cfp32 Stokes I.

    Pipeline:
      1. ``unpack_int4_split(raw, device, out_dtype=voltage_dtype)``
         → (real_v, imag_v) fp16 in M2's GEMM layout.
      2. ``apply_cal_split(real_v, imag_v, cal_real, cal_imag)``
         (only if `cal is not None`) — folds the cal blob × F21 DEC-phase
         into the voltages per the chunk-1 design.
      3. ``kernel.compute_split(real_v, imag_v)``
         → (n_fast_vis, NBASE, NCHAN, BADA_NPOL=2) cfp32.
      4. ``stokes_i_pol_sum(vis)`` → (n_fast_vis, NBASE, NCHAN) cfp32.

    Returns
    -------
    torch.Tensor
        Stokes-I fast vis on `kernel.device`.
    """
    real_v, imag_v = unpack_int4_split(
        raw, device=kernel.device, out_dtype=voltage_dtype,
    )
    if cal is not None:
        real_v_c, imag_v_c = apply_cal_split(
            real_v, imag_v, cal.cal_real, cal.cal_imag,
        )
        del real_v, imag_v
        real_v, imag_v = real_v_c, imag_v_c
    vis_2pol = kernel.compute_split(real_v, imag_v)         # (fv, NBASE, NCHAN, 2) cfp32
    del real_v, imag_v
    return stokes_i_pol_sum(vis_2pol)                       # (fv, NBASE, NCHAN) cfp32


# ---------------------------------------------------------------------------
# Service shell (PSRDADA fada → in-memory pipeline → on-disk torch.save)
# ---------------------------------------------------------------------------


def run(
    fada_key: int,
    output_dir: Path,
    device: torch.device,
    *,
    t_int_fast_native: int = T_INT_FAST_NATIVE,
    obs_dec_rad: float = 0.0,
    cal_path: Path | None = None,
    cal_mode: str = CalMode.PHASE_ONLY,
    cal_pol_swap: bool = False,
    chgroup: int = 0,
    max_blocks: int | None = None,
    expected_fada_bytes: int = FADA_BYTES_PER_BLOCK,
    blocks_output_mode: str = "full",
) -> dict[str, Any]:
    """Connect to PSRDADA fada, run the spine pipeline, write per-block
    Stokes-I fast-vis tensors to ``output_dir``.

    Returns a summary dict with at minimum:
        {n_blocks_in, n_blocks_processed, n_dropped, elapsed_s,
         ms_per_block_p50, ms_per_block_p99, output_dir,
         t_int_fast_native, n_fast_vis_per_block}

    On-disk layout (per processed block N):
        <output_dir>/block_<N:06d>/fast_vis.pt   (torch.save'd tensor)
        <output_dir>/block_<N:06d>/meta.json     (block metadata)

    With ``blocks_output_mode='first_tile_only'``, only fast-vis tile 0
    is written (~14 MB at default config) instead of the full
    (n_fast_vis, NBASE, NCHAN) cfp32 tensor (~1.4 GB at the 4× burst-test
    cadence; ~7.3 GB at the production cadence). Use 'full' on disks
    with > 10 GB free; 'first_tile_only' for smoke runs.
    """
    from psrdada import Reader  # imported lazily so module loads in
                                 # environments without psrdada (CI, h23).

    if blocks_output_mode not in ("full", "first_tile_only"):
        raise ValueError(
            f"blocks_output_mode must be 'full' or 'first_tile_only', "
            f"got {blocks_output_mode!r}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {"stop": False}
    _install_signals(state)

    LOG.info("connecting fada=0x%x device=%s", fada_key, device)
    if cal_path is not None:
        LOG.info("cal: path=%s mode=%s pol_swap=%s",
                 cal_path, cal_mode, cal_pol_swap)
    LOG.info("F21: chgroup=%d obs_dec_deg=%.4f", chgroup, math.degrees(obs_dec_rad))
    LOG.info("output: %s mode=%s", output_dir, blocks_output_mode)
    reader = Reader(fada_key)

    n_in = 0
    n_processed = 0
    n_drop = 0

    try:
        # 1. Header pass-through (R12: ONCE, log only).
        fada_header = reader.getHeader()
        LOG.info("fada header: %d keys (UTC_START=%s)",
                 len(fada_header), fada_header.get("UTC_START", "?"))

        # 2. Construct kernel + cal tensors after first header (so device is committed).
        kernel = FastCorrKernel(
            device=device,
            t_int_fast_native=t_int_fast_native,
        )
        n_fast_vis_per_block = kernel.n_fast_vis_per_full_block
        LOG.info(
            "FastCorrKernel ready: t_int_fast_native=%d (%.3f µs cadence) "
            "→ n_fast_vis_per_block=%d",
            t_int_fast_native, t_int_fast_native * NATIVE_SAMPLE_US,
            n_fast_vis_per_block,
        )

        # 2b. Pick voltage dtype based on cal mode (mirrors slow-corr D17).
        if cal_path is not None and cal_mode == CalMode.FULL:
            voltage_dtype: torch.dtype = torch.float32
            LOG.info("cal-mode=full → routing voltages + GEMM through fp32")
        else:
            voltage_dtype = torch.float16

        cal: FastCorrCalTensors | None = None
        if cal_path is not None:
            cal = _build_cal_tensors_with_f21(
                cal_path,
                chgroup=chgroup,
                obs_dec_rad=obs_dec_rad,
                cal_mode=cal_mode,
                cal_pol_swap=cal_pol_swap,
                device=kernel.device,
                dtype=voltage_dtype,
            )

        # 3. Main loop.
        per_block_ms: list[float] = []
        t_start = time.monotonic()

        while not state["stop"]:
            try:
                page = reader.getNextPage()
            except StopIteration:
                LOG.info("fada reader StopIteration (EOD)")
                break
            if reader.isEndOfData:
                LOG.info("fada EOD flag set; draining final block")
            n_in += 1

            t_block_start = time.monotonic()

            page_arr = np.asarray(page)
            if page_arr.nbytes != expected_fada_bytes:
                LOG.error(
                    "fada block #%d wrong size: got=%d expected=%d; skipping",
                    n_in, page_arr.nbytes, expected_fada_bytes,
                )
                reader.markCleared()
                n_drop += 1
                if max_blocks is not None and n_in >= max_blocks:
                    break
                continue

            # --- COMPUTE ---
            vis_stokes_i = compute_block(
                page_arr,
                kernel=kernel,
                cal=cal,
                voltage_dtype=voltage_dtype,
            )                                                # (fv, NBASE, NCHAN) cfp32

            # --- WRITE ---
            block_dir = output_dir / f"block_{n_in:06d}"
            block_dir.mkdir(exist_ok=True)
            if blocks_output_mode == "first_tile_only":
                out_tensor = vis_stokes_i[0:1].cpu().contiguous()
            else:
                out_tensor = vis_stokes_i.cpu().contiguous()
            torch.save(out_tensor, block_dir / "fast_vis.pt")

            meta = {
                "block_n": n_in,
                "t_int_fast_native": t_int_fast_native,
                "t_int_fast_us": t_int_fast_native * NATIVE_SAMPLE_US,
                "n_fast_vis_per_block": n_fast_vis_per_block,
                "n_fast_vis_written": int(out_tensor.shape[0]),
                "vis_shape": list(out_tensor.shape),
                "vis_dtype": str(out_tensor.dtype),
                "obs_dec_deg": math.degrees(obs_dec_rad),
                "chgroup": chgroup,
                "cal_path": str(cal_path) if cal_path else None,
                "cal_mode": cal_mode if cal_path else None,
                "cal_pol_swap": cal_pol_swap if cal_path else None,
                "fada_key": f"0x{fada_key:x}",
                "blocks_output_mode": blocks_output_mode,
                "voltage_dtype": str(voltage_dtype),
                "device": str(device),
                "utc_start": fada_header.get("UTC_START"),
            }
            (block_dir / "meta.json").write_text(json.dumps(meta, indent=2))

            n_processed += 1
            reader.markCleared()
            del vis_stokes_i, out_tensor

            t_block_end = time.monotonic()
            per_block_ms.append((t_block_end - t_block_start) * 1000.0)

            if n_in % 16 == 0:
                LOG.info(
                    "processed n_in=%d n_processed=%d n_drop=%d "
                    "last_block=%.1fms",
                    n_in, n_processed, n_drop, per_block_ms[-1],
                )

            if max_blocks is not None and n_in >= max_blocks:
                LOG.info("hit --max-blocks=%d; stopping", max_blocks)
                break

            if reader.isEndOfData:
                LOG.info("fada EOD; loop done")
                break

        elapsed = time.monotonic() - t_start
        ms_p50 = float(np.median(per_block_ms)) if per_block_ms else float("nan")
        ms_p99 = float(np.percentile(per_block_ms, 99)) if per_block_ms else float("nan")
        LOG.info(
            "summary: n_in=%d n_processed=%d n_drop=%d elapsed=%.1fs "
            "p50=%.1fms p99=%.1fms",
            n_in, n_processed, n_drop, elapsed, ms_p50, ms_p99,
        )
        return {
            "n_blocks_in": n_in,
            "n_blocks_processed": n_processed,
            "n_dropped": n_drop,
            "elapsed_s": elapsed,
            "ms_per_block_p50": ms_p50,
            "ms_per_block_p99": ms_p99,
            "output_dir": str(output_dir),
            "t_int_fast_native": t_int_fast_native,
            "n_fast_vis_per_block": n_fast_vis_per_block,
        }
    finally:
        try:
            reader.disconnect()
        except Exception:
            LOG.exception("reader.disconnect failed (non-fatal)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--fada-key", default="fada",
                   help="4-char fada buffer key (default: fada). "
                        "PARALLEL_AGENTS.md §4.1 maps via "
                        "$DSART_BUFFER_KEY_PREFIX so e.g. m3 → 'fa3a'.")
    p.add_argument("--device", default="auto",
                   help="auto / cuda / cuda:N / cpu (default: auto). "
                        "PARALLEL_AGENTS.md §4.2 pins M3 to GPU 0 on h01.")
    p.add_argument("--config", type=Path, default=Path("configs/config_corr.yaml"),
                   help="path to config_corr.yaml for fada buffer-size validation")
    p.add_argument("--max-blocks", type=int, default=1,
                   help="stop after N fada blocks (default: 1 for "
                        "smoke runs; production: omit for unlimited)")
    p.add_argument("--t-int-fast-native", type=int, default=T_INT_FAST_NATIVE,
                   help="fast-corr integration depth in NATIVE samples per "
                        "fast-vis tile. Default: %d (= %d µs cadence). "
                        "Burst-test override: 32 (= 1048.576 µs, 4× cadence)."
                        % (T_INT_FAST_NATIVE, int(T_INT_FAST_NATIVE * NATIVE_SAMPLE_US)))
    p.add_argument("--obs-dec-deg", type=float, default=0.0,
                   help="observing source declination in degrees, for the "
                        "F21 cal DEC-phase fold. Required if --apply-cal is set.")
    p.add_argument("--chgroup", type=int, default=0,
                   help="corr-node chgroup index 0..15 (for F21 frequencies)")
    p.add_argument("--apply-cal", type=Path, default=None,
                   help="path to legacy beamformer_weights_*.dat blob. "
                        "Required for production fast-corr (the F21 fold "
                        "lives in the cal pipeline; see chunk 1 docstring).")
    p.add_argument("--cal-mode", default=CalMode.PHASE_ONLY,
                   choices=(CalMode.PHASE_ONLY, CalMode.FULL),
                   help="phase_only = divide by |G| first (default; matches "
                        "bfCorr `wnorm`; stays in fp16). "
                        "full = preserve gain magnitude (routes through fp32).")
    p.add_argument("--cal-pol-swap", action="store_true",
                   help="swap cal pol axis (use if voltage is [A,B] and "
                        "cal is [B,A]; default: assume both [B,A]).")
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/dsart-fast-vis"),
                   help="per-block fast-vis output dir (default: %(default)s)")
    p.add_argument("--blocks-output-mode", default="full",
                   choices=("full", "first_tile_only"),
                   help="full: write the entire (n_fast_vis, NBASE, NCHAN) "
                        "tensor per block (~1-7 GB depending on cadence). "
                        "first_tile_only: write only fast-vis tile 0 "
                        "(~14 MB) for disk-constrained smoke runs.")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.config.exists():
        fada_b = _load_fada_buffer_size(args.config)
        if fada_b != FADA_BYTES_PER_BLOCK:
            LOG.error("config %s fada bytes_per_block=%d != %d",
                      args.config, fada_b, FADA_BYTES_PER_BLOCK)
            return 2
    else:
        LOG.warning("config %s not found; skipping buffer-size validation", args.config)

    fada_int = _key_to_int(args.fada_key)
    device = _pick_device(args.device)

    if args.apply_cal is not None and not args.apply_cal.is_file():
        LOG.error("--apply-cal path %s not found", args.apply_cal)
        return 2

    obs_dec_rad = math.radians(args.obs_dec_deg)

    try:
        run(
            fada_int,
            args.output_dir,
            device,
            t_int_fast_native=args.t_int_fast_native,
            obs_dec_rad=obs_dec_rad,
            cal_path=args.apply_cal,
            cal_mode=args.cal_mode,
            cal_pol_swap=args.cal_pol_swap,
            chgroup=args.chgroup,
            max_blocks=args.max_blocks,
            blocks_output_mode=args.blocks_output_mode,
        )
    except _StopRequested:
        LOG.info("clean stop")
    except KeyboardInterrupt:
        LOG.info("KeyboardInterrupt; clean stop")
    except Exception:
        LOG.exception("fatal error in run()")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
