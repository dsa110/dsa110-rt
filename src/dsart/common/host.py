"""Resolve host phase / role from configs/host_phase.yaml.

Pure Python; requires PyYAML. CONFIG_DIR from DSART_CONFIG_DIR or repo configs/.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

import yaml

HOST_ID: str = socket.gethostname().split(".")[0]


def _default_config_dir() -> Path:
    explicit = os.environ.get("DSART_CONFIG_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    here = Path(__file__).resolve().parent
    for base in [here, *here.parents]:
        cand = base / "configs"
        if cand.is_dir():
            return cand.resolve()
    raise RuntimeError(
        "Cannot locate configs/: set DSART_CONFIG_DIR to the directory containing host_phase.yaml"
    )


CONFIG_DIR: Path = _default_config_dir()


def _load_host_phase() -> dict[str, Any]:
    path = CONFIG_DIR / "host_phase.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing host_phase.yaml at {path}")
    # Explicit utf-8: this module is imported at production-node startup under
    # systemd / non-interactive ssh, where locale.getpreferredencoding() can
    # be ASCII. host_phase.yaml's header comment includes "§" (non-ASCII).
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    hosts = data.get("hosts")
    if not isinstance(hosts, dict):
        raise ValueError("host_phase.yaml must contain a mapping 'hosts:'")
    return hosts


def _normalize_phase(raw: str) -> str:
    if raw in ("a", "b"):
        return raw
    if isinstance(raw, str) and raw.startswith("phase_") and len(raw) >= len("phase_a"):
        suffix = raw.rsplit("_", 1)[-1]
        if len(suffix) == 1 and suffix in "ab":
            return suffix
    raise ValueError(f"Invalid phase value {raw!r}; expected 'phase_a', 'phase_b', 'a', or 'b'")


_hosts = _load_host_phase()
_entry = _hosts.get(HOST_ID)
if _entry is None:
    raise KeyError(
        f"hostname {HOST_ID!r} not found under host_phase.yaml::hosts. "
        f"Known hosts: {sorted(_hosts)!r}. Set DSART_CONFIG_DIR if needed."
    )

PHASE: str = _normalize_phase(str(_entry["phase"]))
ROLE: str = str(_entry.get("role", "")).strip()
CO_RESIDENT_CORR_SEARCH: bool = bool(_entry.get("co_resident_corr_search", False))
