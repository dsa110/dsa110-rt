"""Layer-3 voltage replay (§4.7). M3 implements PSRDADA writes; M0 ships --dry-run only."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

try:
    from jsonschema import Draft202012Validator
except ImportError as exc:  # pragma: no cover
    raise SystemExit("jsonschema is required for replay_voltage_dump") from exc

BLOCK_MS_NATIVE = 134.218
FIXTURE_ROOT_DEFAULT = "/home/ubuntu/data/voltage_fixtures"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _schema_path() -> Path:
    return _repo_root() / "tests" / "fixtures" / "voltage_fixture_manifest.schema.json"


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


def _pick_voltage_file(fixture_dir: Path, chgroup: int) -> Path:
    pat = f"fl_*_chgroup{chgroup}.out"
    matches = sorted(fixture_dir.glob(pat))
    if not matches:
        raise FileNotFoundError(f"{fixture_dir}: no files matching {pat}")
    return matches[0]


def _probe_binary_header(path: Path, nbytes: int = 8192) -> dict[str, object]:
    raw = path.open("rb").read(nbytes)
    out: dict[str, object] = {"probe_bytes": min(nbytes, len(raw)), "sha256_prefix_hex": None}
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


def _dry_run_report(
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
    inferred = None
    if isinstance(nb, int) and nb > 0 and sz % nb == 0:
        inferred = sz // nb
    pace_note = f"pace_target_ms={pace_ms:.6g}" if pace_ms > 0 else "pace_target=unlimited_fast"
    inf_note = f"inferred_bytes_per_block={inferred}" if inferred is not None else "inferred_bytes_per_block=non_integer_division"
    noise_note = f"inject_noise={inject_noise}"
    return (
        f"replay_voltage_dump dry-run: fixture_kind={fk!s} run_id={run_id!r} chgroups={chgroups} "
        f"rate={rate_label!r} native_block_ms={BLOCK_MS_NATIVE} {pace_note} "
        f"voltage_file={vol_path.name} file_bytes={sz} manifest_n_blocks={nb} {inf_note} "
        f"{noise_note} header_probe={probe}"
    )


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--chgroups", required=True, help='e.g. "0", "0,1", "0..15"')
    ap.add_argument("--rate", required=True, help="native | fast | N×")
    ap.add_argument("--inject-noise", action="store_true", help="annotate dry-run report only in M0")
    ap.add_argument("--dry-run", action="store_true")
    ns = ap.parse_args(argv)

    if not ns.dry_run:
        print(
            "NotImplementedError: M3 owns the PSRDADA writer path",
            file=sys.stderr,
        )
        return 2

    root = Path(os.environ.get("DSART_VOLTAGE_FIXTURE_ROOT", FIXTURE_ROOT_DEFAULT))
    fixture_dir = root / ns.run_id
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

    chgroups = _parse_chgroups(ns.chgroups)
    rate_label, pace_ms = _parse_rate(ns.rate)

    vol_path = _pick_voltage_file(fixture_dir, chgroups[0])
    probe = _probe_binary_header(vol_path)
    print(
        _dry_run_report(
            run_id=ns.run_id,
            chgroups=chgroups,
            rate_label=rate_label,
            pace_ms=pace_ms,
            manifest=manifest,
            vol_path=vol_path,
            inject_noise=ns.inject_noise,
            probe=probe,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
