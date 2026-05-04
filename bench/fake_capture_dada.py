"""Synthetic capture-side noise feed into `dada` at native block cadence (§6.1)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

try:
    from psrdada import Writer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "psrdada-python must be installed (tools/ops/install_psrdada.sh on h01)."
    ) from exc

BLOCK_S = 134.218e-3


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ipc_key(name: str) -> int:
    m = {
        "dada": 0xDADA,
        "dadc": 0xDADC,
        "eada": 0xEADA,
        "fada": 0xFADA,
        "bada": 0xBADA,
    }
    if name not in m:
        raise ValueError(f"unsupported buffer key {name!r}")
    return m[name]


def _load_buffer_cfg(buf_name: str) -> tuple[int, int]:
    cfg_path = _repo_root() / "configs" / "config_corr.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    b = cfg["buffers"][buf_name]
    return int(b["bytes_per_block"]), int(b["num_blocks"])


def _write_dataset(writer: Writer, rng: np.random.Generator, npages: int, idx: int) -> None:
    writer.setHeader({"BLOCK_IDX": str(idx), "SOURCE": "bench.fake_capture_dada"})
    n = 0
    for page in writer:
        data = np.asarray(page)
        noise = rng.standard_normal(data.shape)
        data[:] = np.asarray(noise * 0.05, dtype=data.dtype)
        n += 1
        if n >= npages:
            writer.markEndOfData()
            break


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rate", default="native", choices=("native", "fast"))
    ap.add_argument("--secs", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dada-key",
        default="dada",
        help="PSRDADA buffer key (default dada; M0 plumbing uses dadc on a dedicated ring).",
    )
    ns = ap.parse_args(argv)

    bpp, ring_blocks = _load_buffer_cfg("dada")
    rng = np.random.default_rng(ns.seed)
    writer = Writer(_ipc_key(ns.dada_key))

    target = max(1, int(ns.secs / BLOCK_S))
    t0 = time.monotonic()
    for bi in range(target):
        _write_dataset(writer, rng, npages=ring_blocks, idx=bi)
        if ns.rate == "native":
            elapsed = time.monotonic() - t0
            sleep_for = (bi + 1) * BLOCK_S - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    writer.disconnect()
    print(
        f"fake_capture_dada: wrote ~{target} block(s) to {ns.dada_key} "
        f"({bpp} B/page × {ring_blocks} pages)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
