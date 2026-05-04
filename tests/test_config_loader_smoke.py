"""Smoke-load repo configs and validate voltage manifest templates against §3.3 schema."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from dsart.common import config_loader

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_configs_yaml_load_via_config_loader() -> None:
    cfg_dir = REPO_ROOT / "configs"
    paths = sorted(cfg_dir.glob("*.yaml"))
    assert paths, "expected configs/*.yaml"
    for p in paths:
        config_loader.load(p)


def test_voltage_manifest_templates_match_schema() -> None:
    schema_path = REPO_ROOT / "tests" / "fixtures" / "voltage_fixture_manifest.schema.json"
    schema = json.loads(schema_path.read_text())
    validator = Draft202012Validator(schema)
    for name in ("manifest.template.continuum.yaml", "manifest.template.burst.yaml"):
        doc_path = REPO_ROOT / "voltage_fixtures" / name
        doc = yaml.safe_load(doc_path.read_text())
        validator.validate(doc)
