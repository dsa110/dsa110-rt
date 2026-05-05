"""Slow correlator service entry (M2; plan §8 lines 2161-2177).

Reads `fada` PSRDADA blocks, runs the slow correlator kernel, writes
`bada` PSRDADA blocks. Designed to run as `dsart-corr-slow@<host>`.

Per D2 in M2_PLAN_FIXES.md (user constraint 2026-05-04): NO calibration
is applied — the user derives cal solutions from these visibilities
downstream in their own code (out-of-scope for this revamp).

Per D13: this module runs in the `dsa110-rt` conda env (depends on
`torch` + `psrdada-python`; does NOT depend on `dsacalib` / `dsamfs`
/ `casacore`).

Per R12 mitigation (Subagent B): writes the `bada` header ONCE at
service startup (mirrored from the first `fada` header) rather than
per-block, to avoid filling the bada header ring if downstream
`meridian_fringestop` does not drain headers from the bada ring. The
data-block timing is propagated implicitly via `markFilled()` ordering.

CLI:

    python -m dsart.services.corr_slow_compute \\
        [--fada-key fada] [--bada-key bada] [--device auto] [--max-blocks N]

`--max-blocks` is for benching / smoke-tests; production uses the
default of unlimited (run until fada EOD).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dsart.common.config_loader import load
from dsart.common.constants import (
    BADA_BYTES_PER_INTEGRATION,
    FADA_BYTES_PER_BLOCK,
)
from dsart.services.slow_corr_kernel import (
    SlowCorrKernel,
    pack_bada_block,
    unpack_int4_complex,
)

LOG = logging.getLogger("corr_slow_compute")

DEFAULT_CONFIG_PATH = Path("/home/ubuntu/proj/dsa110-rt/configs/config_corr.yaml")


def _key_to_int(key_str: str) -> int:
    """4-char buffer key string → DADA hex int (mirror dsamfs.routines)."""
    if len(key_str) != 4:
        raise ValueError(f"buffer key must be 4 chars, got {key_str!r}")
    return int(f"0x{key_str}", 16)


def _pick_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _load_buffer_sizes(cfg_path: Path) -> tuple[int, int]:
    """Return (fada_bytes_per_block, bada_bytes_per_block) from yaml."""
    cfg = load(cfg_path)
    bufs = cfg.get("buffers", {})
    return (
        int(bufs["fada"]["bytes_per_block"]),
        int(bufs["bada"]["bytes_per_block"]),
    )


class _StopRequested(Exception):
    """Raised by SIGTERM / SIGINT handler to break the main loop."""


def _install_signals(state: dict[str, Any]) -> None:
    def handle(signum, _frame):
        LOG.info("received signal %d, requesting stop", signum)
        state["stop"] = True
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)


def run(
    fada_key: int,
    bada_key: int,
    device: torch.device,
    *,
    max_blocks: int | None = None,
    expected_fada_bytes: int = FADA_BYTES_PER_BLOCK,
    expected_bada_bytes: int = BADA_BYTES_PER_INTEGRATION,
) -> dict[str, Any]:
    """Connect to PSRDADA, run the corr loop, return summary stats.

    Returns a dict with at minimum:
        {n_blocks_in, n_blocks_out, n_dropped, elapsed_s, ms_per_block_p50}
    """
    from psrdada import Reader, Writer  # imported lazily so the module loads
                                        # in environments without psrdada.

    state: dict[str, Any] = {"stop": False}
    _install_signals(state)

    LOG.info("connecting fada=0x%x bada=0x%x device=%s", fada_key, bada_key, device)
    reader = Reader(fada_key)
    writer = Writer(bada_key)

    try:
        # 1. Header pass-through (R12: ONCE, not per-block).
        fada_header = reader.getHeader()
        LOG.info("fada header: %d keys (UTC_START=%s)",
                 len(fada_header), fada_header.get("UTC_START", "?"))
        bada_header = {k: v for k, v in fada_header.items()
                       if k not in ("__RAW_HEADER__",)}
        # Stamp one M2-specific provenance key so downstream tooling can
        # tell M2 corr_slow_compute apart from legacy dsaX_bfCorr.
        bada_header["DSART_PRODUCER"] = "corr_slow_compute"
        writer.setHeader(bada_header)
        LOG.info("bada header: %d keys written", len(bada_header))

        # 2. Construct kernel after first header (so device is committed).
        kernel = SlowCorrKernel(device=device)
        LOG.info("kernel ready: nants=%d nchan=%d nbase=%d nbada_pol=%d",
                 kernel.nants, kernel.nchan, kernel._nbase, kernel.nbada_pol)

        # 3. Main loop.
        n_in = 0
        n_out = 0
        n_drop = 0
        per_block_ms: list[float] = []
        t_start = time.monotonic()

        while not state["stop"]:
            # --- READ ---
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
                LOG.error("fada block #%d wrong size: got=%d expected=%d; skipping",
                          n_in, page_arr.nbytes, expected_fada_bytes)
                reader.markCleared()
                n_drop += 1
                if max_blocks is not None and n_in >= max_blocks:
                    break
                continue

            # --- COMPUTE ---
            voltages = unpack_int4_complex(page_arr, device=device)
            vis = kernel.compute(voltages)

            # --- WRITE ---
            out_bytes = pack_bada_block(vis)            # uint8 view, length BADA_BYTES_PER_INTEGRATION
            if out_bytes.nbytes != expected_bada_bytes:
                LOG.error("packed bada size %d != expected %d",
                          out_bytes.nbytes, expected_bada_bytes)
                reader.markCleared()
                n_drop += 1
                continue

            out_page = writer.getNextPage()
            out_page_arr = np.asarray(out_page)
            if out_page_arr.nbytes != expected_bada_bytes:
                # Buffer config mismatch — fail-fast per Chunk-0 hardening.
                raise RuntimeError(
                    f"bada page size {out_page_arr.nbytes} != "
                    f"expected {expected_bada_bytes} (config drift)"
                )
            out_page_arr[:] = out_bytes
            writer.markFilled()
            n_out += 1

            reader.markCleared()

            t_block_end = time.monotonic()
            per_block_ms.append((t_block_end - t_block_start) * 1000.0)

            if n_in % 64 == 0:
                LOG.info("processed n_in=%d n_out=%d n_drop=%d "
                         "last_block=%.1fms",
                         n_in, n_out, n_drop, per_block_ms[-1])

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
            "summary: n_in=%d n_out=%d n_drop=%d elapsed=%.1fs "
            "p50=%.1fms p99=%.1fms",
            n_in, n_out, n_drop, elapsed, ms_p50, ms_p99,
        )
        return {
            "n_blocks_in": n_in,
            "n_blocks_out": n_out,
            "n_dropped": n_drop,
            "elapsed_s": elapsed,
            "ms_per_block_p50": ms_p50,
            "ms_per_block_p99": ms_p99,
        }
    finally:
        try:
            reader.disconnect()
        except Exception:
            LOG.exception("reader.disconnect failed (non-fatal)")
        try:
            writer.disconnect()
        except Exception:
            LOG.exception("writer.disconnect failed (non-fatal)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fada-key", default="fada",
                   help="4-char fada buffer key (default: fada)")
    p.add_argument("--bada-key", default="bada",
                   help="4-char bada buffer key (default: bada)")
    p.add_argument("--device", default="auto",
                   help="auto / cuda / cuda:N / cpu (default: auto)")
    p.add_argument("--config", type=Path, default=Path("configs/config_corr.yaml"),
                   help="path to config_corr.yaml for buffer-size validation")
    p.add_argument("--max-blocks", type=int, default=None,
                   help="stop after N fada blocks (smoke-test / bench)")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.config.exists():
        fada_b, bada_b = _load_buffer_sizes(args.config)
        if fada_b != FADA_BYTES_PER_BLOCK:
            LOG.error("config %s fada bytes_per_block=%d != %d",
                      args.config, fada_b, FADA_BYTES_PER_BLOCK)
            return 2
        if bada_b != BADA_BYTES_PER_INTEGRATION:
            LOG.error("config %s bada bytes_per_block=%d != %d",
                      args.config, bada_b, BADA_BYTES_PER_INTEGRATION)
            return 2
    else:
        LOG.warning("config %s not found; skipping buffer-size validation", args.config)

    fada_int = _key_to_int(args.fada_key)
    bada_int = _key_to_int(args.bada_key)
    device = _pick_device(args.device)

    try:
        run(fada_int, bada_int, device, max_blocks=args.max_blocks)
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
