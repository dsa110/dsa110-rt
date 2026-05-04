"""Minimal Gaussian noise writer for SNAP capture rings (§6.1 contract). M0: one dada + one eada transfer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

try:
    from psrdada import Writer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "psrdada-python must be installed (tools/ops/install_psrdada.sh on h01)."
    ) from exc


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ipc_key(name: str) -> int:
    m = {"dada": 0xDADA, "eada": 0xEADA, "fada": 0xFADA, "bada": 0xBADA}
    if name not in m:
        raise ValueError(f"unsupported buffer key {name!r}")
    return m[name]


def _load_buffer_cfg(buf_name: str) -> tuple[int, int]:
    cfg_path = _repo_root() / "configs" / "config_corr.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    b = cfg["buffers"][buf_name]
    return int(b["bytes_per_block"]), int(b["num_blocks"])


def _write_once(writer: Writer, rng: np.random.Generator, npages: int, label: str) -> None:
    writer.setHeader({"DATASET": label, "SOURCE": "bench.fake_snap_replay"})
    n = 0
    for page in writer:
        data = np.asarray(page)
        noise = rng.standard_normal(data.shape)
        scaled = np.asarray(noise * 0.05, dtype=data.dtype)
        data[:] = scaled
        n += 1
        if n >= npages:
            writer.markEndOfData()
            break
    writer.disconnect()


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rate", default="native", help="native pacing hint (M0 writes once)")
    ap.add_argument("--secs", type=float, default=60.0, help="ignored in M0 minimal mode")
    ap.add_argument("--seed", type=int, default=0)
    ns = ap.parse_args(argv)
    _ = ns.rate, ns.secs  # full pacing deferred to later milestones

    rng = np.random.default_rng(ns.seed)
    for buf_name in ("dada", "eada"):
        _bpp, _nb = _load_buffer_cfg(buf_name)
        writer = Writer(_ipc_key(buf_name))
        _write_once(writer, rng, npages=1, label=f"snap_{buf_name}")
    print("fake_snap_replay: wrote 1 page to dada and 1 page to eada")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
