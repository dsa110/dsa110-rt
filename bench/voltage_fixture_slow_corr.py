#!/usr/bin/env python3
"""bench/voltage_fixture_slow_corr.py — M2 Chunk 6 voltage-fixture orchestration.

Plan §8 M2 DoD line 2172 ("voltage-fixture sub-DoD — operator sign-off
gate"). Brings up the dada_db buffers, spawns the slow correlator
service, replays voltage data into ``fada``, captures the resulting
``bada`` (optionally via ``meridian_fringestop`` to produce UVH5), then
tears the rings down.

Two replay modes (mirrors ``bench/replay_voltage_dump.py``):

  1. **Manifest mode** (operator-supplied continuum/burst fixtures):
       python -m bench.voltage_fixture_slow_corr --run-id <id> [--rate native]

  2. **Synthetic mode** (CI / hermetic Chunk-6 acceptance, no fixture):
       python -m bench.voltage_fixture_slow_corr --synthesize \\
           --synth-thermal-sigma 1.5 --synth-source 0.05,0,4 \\
           --n-blocks 15 [--rate native] [--skip-meridian]

Output paths:
  * ``--bada-capture-out PATH`` writes the raw bada blocks (every per-block
    bytes appended) for offline inspection.
  * ``--uvh5-out PATH`` writes the partial UVH5 file produced by
    meridian_fringestop (when not --skip-meridian).

Exit code 0 = pipeline ran end-to-end and corr_slow_compute reported
``n_blocks_out >= n_blocks_in`` (no drops).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.common.constants import (  # noqa: E402
    BADA_BYTES_PER_INTEGRATION,
    FADA_BYTES_PER_BLOCK,
)

DEFAULT_FADA_KEY = "fada"
DEFAULT_BADA_KEY = "bada"
DEFAULT_FADA_NUM_BLOCKS = 4
DEFAULT_BADA_NUM_BLOCKS = 8
DADA_HDR_SIZE = 4096
DADA_HDR_NHDRS = 8


# ---- dada_db lifecycle --------------------------------------------------


def _dada_db_create(key: str, bytes_per_block: int, num_blocks: int,
                    *, log: subprocess.PIPE | None = None) -> None:
    """Run dada_db -k <key> -b <bytes> -n <nblocks> -a <hdrsize> -p."""
    cmd = [
        "dada_db", "-k", key,
        "-b", str(bytes_per_block),
        "-n", str(num_blocks),
        "-a", str(DADA_HDR_SIZE),
        "-p",
    ]
    print(f"[orchestrator] dada_db create: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"dada_db create failed (rc={proc.returncode}): "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )


def _dada_db_destroy(key: str, *, ignore_missing: bool = True) -> None:
    """Best-effort teardown."""
    proc = subprocess.run(
        ["dada_db", "-d", "-k", key],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 and not ignore_missing:
        raise RuntimeError(
            f"dada_db destroy {key} failed (rc={proc.returncode}): "
            f"{proc.stderr!r}"
        )


@contextmanager
def dada_buffers(*, fada_key: str, bada_key: str,
                 fada_blocks: int, bada_blocks: int) -> Iterator[None]:
    """Context manager: create fada+bada dada_db rings, tear down on exit."""
    # Best-effort cleanup of any stale rings under these keys.
    _dada_db_destroy(fada_key)
    _dada_db_destroy(bada_key)

    _dada_db_create(fada_key, FADA_BYTES_PER_BLOCK, fada_blocks)
    try:
        _dada_db_create(bada_key, BADA_BYTES_PER_INTEGRATION, bada_blocks)
        try:
            yield
        finally:
            _dada_db_destroy(bada_key)
    finally:
        _dada_db_destroy(fada_key)


# ---- subprocess management ---------------------------------------------


@contextmanager
def background_proc(cmd: list[str], *, label: str, env: dict[str, str] | None = None,
                    stdout_path: Path | None = None) -> Iterator[subprocess.Popen]:
    """Spawn a subprocess in the background, terminate on context exit."""
    print(f"[orchestrator] starting {label}: {' '.join(cmd)}", flush=True)
    out_fd = None
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        out_fd = open(stdout_path, "w")
    p = subprocess.Popen(
        cmd,
        stdout=out_fd if out_fd else subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        # Put in own process group so SIGTERM doesn't propagate to parent.
        preexec_fn=os.setsid,
    )
    try:
        yield p
    finally:
        if p.poll() is None:
            print(f"[orchestrator] terminating {label} pid={p.pid}", flush=True)
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"[orchestrator] hard-killing {label} pid={p.pid}", flush=True)
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait(timeout=2)
        if out_fd is not None:
            out_fd.close()


def _wait_with_timeout(proc: subprocess.Popen, timeout_s: float, label: str) -> int:
    """Wait for a subprocess with a timeout. Returns exit code (or -1 on timeout)."""
    try:
        rc = proc.wait(timeout=timeout_s)
        print(f"[orchestrator] {label} exited rc={rc}", flush=True)
        return rc
    except subprocess.TimeoutExpired:
        print(f"[orchestrator] {label} timed out after {timeout_s}s", flush=True)
        return -1


# ---- bada capture (simple psrdada Reader → file) -----------------------


def _bada_capture_subprocess_cmd(
    bada_key: str, n_blocks: int, out_path: Path, dsart_python: str,
) -> list[str]:
    """Build a python -c command that reads N bada blocks and writes them to out_path.

    Inlined to avoid a dependency on a separate reader module.
    """
    snippet = f"""
import sys
import numpy as np
from psrdada import Reader

reader = Reader(0x{int(bada_key, 16):04x})
hdr = reader.getHeader()
print(f'[bada-capture] header keys: {{len(hdr)}} (DSART_PRODUCER={{hdr.get(\"DSART_PRODUCER\", \"?\")}})', flush=True)

n = 0
N_TARGET = {n_blocks}
EXPECTED_SIZE = {BADA_BYTES_PER_INTEGRATION}

with open(r'{out_path}', 'wb') as f:
    for page in reader:
        arr = np.asarray(page)
        if arr.nbytes != EXPECTED_SIZE:
            print(f'[bada-capture] WRONG SIZE: got {{arr.nbytes}} expected {{EXPECTED_SIZE}}', flush=True)
            reader.markCleared()
            continue
        f.write(arr.tobytes())
        f.flush()
        reader.markCleared()
        n += 1
        print(f'[bada-capture] captured block {{n}}/{{N_TARGET}}', flush=True)
        if n >= N_TARGET:
            break
        if reader.isEndOfData:
            print('[bada-capture] EOD; stopping', flush=True)
            break

reader.disconnect()
print(f'[bada-capture] DONE: {{n}} blocks captured to {out_path}', flush=True)
sys.exit(0)
"""
    return [dsart_python, "-c", snippet]


# ---- mode dispatch -----------------------------------------------------


def build_replay_cmd(
    args: argparse.Namespace, dsart_python: str, fada_key: str,
) -> list[str]:
    """Build the bench.replay_voltage_dump command line."""
    cmd = [
        dsart_python, "-m", "bench.replay_voltage_dump",
        "--rate", args.rate,
        "--fada-key", fada_key,
        "--n-blocks", str(args.n_blocks),
    ]
    if args.synthesize:
        cmd.append("--synthesize")
        cmd.extend(["--seed", str(args.seed)])
        cmd.extend(["--synth-thermal-sigma", str(args.synth_thermal_sigma)])
        for s in args.synth_source:
            cmd.extend(["--synth-source", s])
    else:
        cmd.extend(["--run-id", args.run_id, "--chgroups", args.chgroups])
    return cmd


def build_corr_cmd(
    args: argparse.Namespace, dsart_python: str, fada_key: str, bada_key: str,
) -> list[str]:
    cmd = [
        dsart_python, "-m", "dsart.services.corr_slow_compute",
        "--fada-key", fada_key,
        "--bada-key", bada_key,
        "--device", args.device,
        "--max-blocks", str(args.n_blocks),
        "--config", str(REPO_ROOT / "configs" / "config_corr.yaml"),
        "--log-level", args.corr_log_level,
    ]
    if args.apply_cal:
        cmd.extend(["--apply-cal", str(args.apply_cal),
                    "--cal-mode", args.cal_mode])
        if args.cal_pol_swap:
            cmd.append("--cal-pol-swap")
    return cmd


# ---- meridian_fringestop integration -----------------------------------


def _meridian_fringestop_cmd(args: argparse.Namespace) -> list[str] | None:
    """Build the meridian_fringestop command line, or None if --skip-meridian.

    Two modes:
      * default — invokes ``dsamfs.meridian_fringestop`` directly. This
        relies on ``socket.gethostname()`` matching a key in the dsamfs
        param file's ``ch0:`` dict and on etcd serving the correct
        ``/mon/array/dec`` value.
      * ``--meridian-param <path>`` — invokes the casa38 wrapper at
        ``bench/casa38_meridian_wrapper.py`` which patches
        ``dsamfs.utils.get_pointing_declination`` in-process to honour
        ``--meridian-pt-dec-deg`` (D17 wrapper, no casa38 source mods).
    """
    if args.skip_meridian:
        return None
    if shutil.which(args.casa38_python) is None:
        print(
            f"[orchestrator] casa38 python {args.casa38_python} not found; "
            f"skipping meridian_fringestop. Pass --skip-meridian to silence "
            f"this warning.",
            file=sys.stderr,
        )
        return None
    if not args.uvh5_out:
        raise SystemExit(
            "--uvh5-out is required when running meridian_fringestop "
            "(or pass --skip-meridian to bypass)"
        )
    out_dir = Path(args.uvh5_out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.meridian_param:
        # D17 wrapper: monkey-patch get_pointing_declination in-process.
        cmd = [
            args.casa38_python,
            str(REPO_ROOT / "bench" / "casa38_meridian_wrapper.py"),
            "--param-file", str(args.meridian_param),
            "--out-dir", str(out_dir),
            "--working-dir", str(out_dir),
        ]
        if args.meridian_pt_dec_deg is not None:
            cmd.extend(["--pt-dec-deg", str(args.meridian_pt_dec_deg)])
        return cmd

    # Default: invoke dsamfs.meridian_fringestop directly (positional args).
    # See dsamfs/meridian_fringestop.py: argv[1]=OUTDIR, argv[2]=WORKING_DIR.
    cmd = [
        args.casa38_python, "-m", "dsamfs.meridian_fringestop",
        str(out_dir),
        str(out_dir),
    ]
    return cmd


# ---- main pipeline -----------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> int:
    """Bring up buffers, start corr+replay+capture, tear down."""
    fada_key = args.fada_key
    bada_key = args.bada_key
    n_blocks = args.n_blocks

    bada_capture_path: Path | None = None
    if args.bada_capture_out:
        bada_capture_path = Path(args.bada_capture_out)
        bada_capture_path.parent.mkdir(parents=True, exist_ok=True)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    corr_log = work_dir / "corr_slow_compute.log"
    replay_log = work_dir / "replay_voltage_dump.log"
    capture_log = work_dir / "bada_capture.log"
    summary_path = work_dir / "summary.json"

    summary: dict[str, object] = {
        "n_blocks_target": n_blocks,
        "synthesize": args.synthesize,
        "run_id": args.run_id if not args.synthesize else None,
        "rate": args.rate,
        "fada_key": fada_key,
        "bada_key": bada_key,
        "logs": {
            "corr_slow_compute": str(corr_log),
            "replay_voltage_dump": str(replay_log),
            "bada_capture": str(capture_log),
        },
    }
    if bada_capture_path is not None:
        summary["bada_capture"] = str(bada_capture_path)

    overall_rc = 0
    try:
        with dada_buffers(
            fada_key=fada_key, bada_key=bada_key,
            fada_blocks=DEFAULT_FADA_NUM_BLOCKS,
            bada_blocks=DEFAULT_BADA_NUM_BLOCKS,
        ):
            # 1. Start corr_slow_compute (bada producer + fada consumer).
            corr_cmd = build_corr_cmd(args, args.dsart_python, fada_key, bada_key)
            with background_proc(corr_cmd, label="corr_slow_compute",
                                  stdout_path=corr_log) as corr_p:
                # Brief pause so corr is in reader-blocking state before
                # we start writing to fada.
                time.sleep(2.0)
                if corr_p.poll() is not None:
                    print(f"[orchestrator] corr_slow_compute died early "
                          f"rc={corr_p.returncode}; aborting", file=sys.stderr)
                    return 2

                # 2. Start bada capture (or meridian_fringestop) in background.
                meridian_cmd = _meridian_fringestop_cmd(args)
                if meridian_cmd is not None:
                    consumer_label = "meridian_fringestop"
                    consumer_cmd = meridian_cmd
                    consumer_log = work_dir / "meridian_fringestop.log"
                else:
                    if bada_capture_path is None:
                        bada_capture_path = work_dir / "bada_capture.bin"
                        summary["bada_capture"] = str(bada_capture_path)
                    consumer_label = "bada_capture"
                    consumer_cmd = _bada_capture_subprocess_cmd(
                        bada_key, n_blocks, bada_capture_path, args.dsart_python,
                    )
                    consumer_log = capture_log

                with background_proc(consumer_cmd, label=consumer_label,
                                      stdout_path=consumer_log) as cons_p:
                    time.sleep(1.0)
                    if cons_p.poll() is not None:
                        print(f"[orchestrator] {consumer_label} died early "
                              f"rc={cons_p.returncode}; aborting",
                              file=sys.stderr)
                        return 3

                    # 3. Run replay (foreground; the producer driver).
                    replay_cmd = build_replay_cmd(args, args.dsart_python, fada_key)
                    with background_proc(replay_cmd, label="replay_voltage_dump",
                                          stdout_path=replay_log) as rep_p:
                        rep_rc = _wait_with_timeout(
                            rep_p, args.timeout_s, "replay_voltage_dump",
                        )
                        summary["replay_rc"] = rep_rc
                        if rep_rc != 0:
                            overall_rc = max(overall_rc, 4)

                    # 4. Wait for corr_slow_compute to finish processing the
                    # blocks we just sent (it stops at --max-blocks).
                    corr_rc = _wait_with_timeout(
                        corr_p, args.timeout_s, "corr_slow_compute",
                    )
                    summary["corr_rc"] = corr_rc
                    if corr_rc != 0:
                        overall_rc = max(overall_rc, 5)

                    # 5. Wait for the consumer to finish reading the
                    # corresponding bada blocks.
                    cons_rc = _wait_with_timeout(
                        cons_p, args.timeout_s, consumer_label,
                    )
                    summary[f"{consumer_label}_rc"] = cons_rc
                    if cons_rc != 0:
                        overall_rc = max(overall_rc, 6)

    except Exception as e:
        print(f"[orchestrator] FAILED: {e}", file=sys.stderr)
        summary["error"] = str(e)
        overall_rc = max(overall_rc, 1)

    summary["overall_rc"] = overall_rc
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[orchestrator] summary written to {summary_path}", flush=True)
    print(json.dumps(summary, indent=2, default=str))
    return overall_rc


# ---- main ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Mode (one of)
    ap.add_argument("--run-id", help="manifest fixture run id (manifest mode)")
    ap.add_argument("--synthesize", action="store_true",
                    help="generate synthetic blocks in-memory (no fixture)")

    # Replay common args
    ap.add_argument("--chgroups", default="0", help='e.g. "0", "0,1", "0..15"')
    ap.add_argument("--rate", default="native", help="native | fast | N×")
    ap.add_argument("--n-blocks", type=int, default=15,
                    help="number of blocks to replay/correlate")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--synth-thermal-sigma", type=float, default=0.0)
    ap.add_argument("--synth-source", action="append", default=[],
                    help="synth: 'l,m,amp_pre_fluff' (repeatable)")

    # Buffer keys
    ap.add_argument("--fada-key", default=DEFAULT_FADA_KEY)
    ap.add_argument("--bada-key", default=DEFAULT_BADA_KEY)

    # Corr args
    ap.add_argument("--device", default="auto",
                    help="corr_slow_compute device (auto/cuda/cpu)")
    ap.add_argument("--corr-log-level", default="INFO",
                    choices=("DEBUG", "INFO", "WARNING", "ERROR"))

    # meridian_fringestop / bada-capture mode
    ap.add_argument("--skip-meridian", action="store_true",
                    help="don't run meridian_fringestop; just capture bada to file")
    ap.add_argument("--uvh5-out", help="output path for meridian_fringestop UVH5 "
                                       "(required unless --skip-meridian)")
    ap.add_argument("--bada-capture-out",
                    help="(when --skip-meridian) raw bada bytes output path "
                         "(default: <work-dir>/bada_capture.bin)")
    ap.add_argument("--meridian-param", type=Path, default=None,
                    help="custom dsamfs param yaml (D17 wrapper mode); "
                         "if set, invokes bench/casa38_meridian_wrapper.py "
                         "which monkey-patches get_pointing_declination")
    ap.add_argument("--meridian-pt-dec-deg", type=float, default=None,
                    help="(wrapper mode only) override pt_dec in degrees "
                         "(bypasses etcd/array/dec)")

    # Cal application (D17 — passed through to corr_slow_compute)
    ap.add_argument("--apply-cal", type=Path, default=None,
                    help="path to legacy beamformer_weights_*.dat blob "
                         "(D17 test-only)")
    ap.add_argument("--cal-mode", default="full", choices=("full", "phase"),
                    help="full = preserve gain magnitude; phase = divide by |G| first")
    ap.add_argument("--cal-pol-swap", action="store_true",
                    help="swap cal pol axis (use if voltage and cal pol orders differ)")

    # Python interpreters
    ap.add_argument("--dsart-python",
                    default="/home/ubuntu/miniforge3/envs/dsa110-rt/bin/python",
                    help="python in the dsa110-rt conda env (D13)")
    ap.add_argument("--casa38-python",
                    default="/home/ubuntu/anaconda3/envs/casa38/bin/python",
                    help="python in the casa38 conda env for meridian_fringestop")

    # Bookkeeping
    ap.add_argument("--work-dir", default="/tmp/voltage_fixture_slow_corr",
                    help="directory for logs + summary.json")
    ap.add_argument("--timeout-s", type=float, default=300.0,
                    help="per-subprocess wait timeout")

    args = ap.parse_args(argv)

    if args.synthesize and args.run_id:
        print("ERROR: --synthesize and --run-id are mutually exclusive",
              file=sys.stderr)
        return 2
    if not args.synthesize and not args.run_id:
        print("ERROR: must specify either --run-id or --synthesize",
              file=sys.stderr)
        return 2
    if not args.skip_meridian and not args.uvh5_out:
        # Default to skip_meridian for now (full meridian path is exercised
        # when operator supplies --uvh5-out explicitly).
        args.skip_meridian = True

    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
