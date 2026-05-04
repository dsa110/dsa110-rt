"""YAML config loader with optional JSON Schema validation (placeholders in M0)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover
    Draft202012Validator = None  # type: ignore[misc,assignment]

DSART_TEST: bool = os.environ.get("DSART_TEST") == "1"

_SCHEMAS: dict[str, dict[str, Any]] = {}

CONFIG_FILENAMES_M0 = (
    "ant_groups.yaml",
    "cal_paths.yaml",
    "chgroup_assignments.yaml",
    "config_compute_corr.yaml",
    "config_compute_search.yaml",
    "config_corr.yaml",
    "config_search.yaml",
    "dm_ranges.yaml",
    "etcd_endpoints.yaml",
    "host_phase.yaml",
    "numa_topology.yaml",
    "operating_points.yaml",
)


def register_schema(filename: str, schema: dict[str, Any]) -> None:
    """Register a JSON Schema (draft 2020-12 compatible dict) for configs/<filename>."""
    _SCHEMAS[filename] = schema


def _validate(instance: Any, schema: dict[str, Any], label: str) -> None:
    if Draft202012Validator is None:
        raise RuntimeError(
            f"jsonschema is required to validate {label}; install deps or fix environment."
        )
    Draft202012Validator(schema).validate(instance)


def load(path: Path | str) -> dict[str, Any]:
    """Load a YAML config file; validate against a registered schema when present."""
    p = Path(path)
    text = p.read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{p}: root must be a mapping, got {type(data).__name__}")
    name = p.name
    schema = _SCHEMAS.get(name)
    if schema is not None:
        _validate(data, schema, str(p))
    return data


def _register_placeholder_schemas() -> None:
    placeholder: dict[str, Any] = {"type": "object"}
    for fn in CONFIG_FILENAMES_M0:
        if fn not in _SCHEMAS:
            _SCHEMAS[fn] = placeholder


_register_placeholder_schemas()


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print("usage: python -m dsart.common.config_loader <path-to.yaml>", file=sys.stderr)
        return 2
    path = Path(argv[0])
    try:
        load(path)
    except Exception as exc:
        print(f"config_loader: FAILED {path}: {exc}", file=sys.stderr)
        return 1
    print(f"config_loader: OK {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
