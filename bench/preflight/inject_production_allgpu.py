#!/usr/bin/env python3
"""Production all-GPU injection cubes.

For each (DM, width) injection, inject ONE realistic (~target-sigma) burst at
the TRUE DM into pure-noise voltages, run corr-fast ONCE, then image+detect it
through the EXACT fp16/complex32 production search path for EVERY coarse-DM
owner (all search GPUs). Produces one notebook-viewable NPZ cube per owner, so
you can see what each production GPU would output for the same burst.

Fluence is calibrated per width (at a reference DM, with the fp32 audit imager
used ONLY as a non-overflowing SNR meter) so the owning-GPU detector
matched-filter SNR is ~target sigma (default 30) -- realistic, non-clipping.

CUDA isolation: the orchestrator process touches NO CUDA. Each calibration
probe and each all-owners injection runs in its own child process (fresh CUDA
context, fully released on exit) so corr-fast subprocesses never contend with a
resident imager.

Run on a CUDA host (e.g. n01):

    CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      PYTHONPATH=src:bench:. python -u -m bench.preflight.inject_production_allgpu \\
        --out-dir /tmp/inject_allgpu --target-snr 30
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

LOG = logging.getLogger("inject_production_allgpu")

DEFAULT_DMS = (150.0, 1200.0, 2400.0)
DEFAULT_WIDTHS_MS = (1.0, 16.0)
CALIB_DM = 1200.0                       # reference DM for fluence calibration
# Guesses in the faint/no-clip linear regime (a point source's coherent fp16
# FFT peak saturates well above ~70 sigma; cint8 clips above ~noise_max).
GUESS_FLUENCE = {1.0: 4.0, 16.0: 16.0}  # Jy*ms initial guesses per width
# cint8 noise target: the combine sums 16 chgroups BEFORE the fp16 ifft2, so
# the unnormalised coherent peak ~ N_filled * 16 * source_cint8 must stay below
# the fp16 max (65504). target_max=20 verified safe+focused at ~30 sigma
# (owning cube peak ~23, no overflow); 12 adds headroom for brighter
# (lower-DM) owning cells. Layer-1 makes the recovered SNR scale-invariant, so
# lowering target_max does NOT change the ~30 sigma.
QUANT_TARGET_MAX = 12


def _n_blocks_for_dm(dm: float) -> int:
    return 8 if dm <= 700 else (12 if dm <= 1400 else 20)


# ===========================================================================
# Worker entry points (run in isolated child processes)
# ===========================================================================

def _worker_calib(a: argparse.Namespace) -> int:
    """Child: fp32-audit SNR probe at one (dm,width,fluence); prints PROBE_SNR."""
    from bench.preflight._inject_search_driver import run_search_driver
    from dsart.services.search_compute import _dm_grids_from_npz
    import numpy as np
    coarse, _, _ = _dm_grids_from_npz(Path(a.dm_plan_path), n_coarse=8)
    owner = int(np.argmin(np.abs(coarse - a.dm)))
    res = run_search_driver(
        owner_idx=owner, dm_pc_cm3=a.dm, dm_target=a.dm,
        l_rad=a.l_rad, m_rad=a.m_rad, width_ms=a.width, fluence_jy_ms=a.fluence,
        t_det=a.t_det, n_grid=a.n_grid, n_fdm=None,
        n_blocks=_n_blocks_for_dm(a.dm), chan_sum_factor=a.chan_sum_factor,
        dm_plan_path=a.dm_plan_path, device=a.device, n_burnin=a.n_burnin,
        threshold_sigma=a.threshold_sigma, out_dir=a.out_dir,
        run_noise_only=False, reuse_corr=False, audit_fp32=True,
        zero_dm_filter=True, quant_target_max=QUANT_TARGET_MAX,
        corr_save_all_owners=False, verbose=a.verbose,
    )
    cands = sorted(res["candidates"], key=lambda c: -c["snr"])
    # Prefer the detector matched-filter SNR; if below the 8-sigma trigger
    # (undetected), fall back to the injection cube's peak sigma as a linear
    # SNR proxy so we can still scale fluence from a faint probe.
    det_snr = float(cands[0]["snr"]) if cands else 0.0
    cube_max = float(res.get("inj_cube_max", 0.0))
    snr = det_snr if det_snr > 0 else cube_max
    print(f"PROBE_DET_SNR={det_snr} PROBE_CUBE_MAX={cube_max}", flush=True)
    print(f"PROBE_SNR={snr}", flush=True)
    return 0


def _worker_owner(a: argparse.Namespace) -> int:
    """Child: image+detect ONE coarse-DM owner (exact fp16) from the shared
    corr-fast scratch. Owner 0 runs corr (saving all owners); owners 1-7 reuse
    it. Isolating each owner in its own process avoids cross-owner CUDA-context
    degradation (repeated imager build/teardown -> H2D 'invalid argument')."""
    import json as _json
    from bench.preflight._inject_search_driver import run_search_driver
    res = run_search_driver(
        owner_idx=a.owner_idx, dm_pc_cm3=a.dm, dm_target=a.dm,
        l_rad=a.l_rad, m_rad=a.m_rad, width_ms=a.width, fluence_jy_ms=a.fluence,
        t_det=a.t_det, n_grid=a.n_grid, n_fdm=None,
        n_blocks=int(a.n_blocks), chan_sum_factor=a.chan_sum_factor,
        dm_plan_path=a.dm_plan_path, device=a.device, n_burnin=a.n_burnin,
        threshold_sigma=a.threshold_sigma,
        search_node_id=(a.owner_idx // 2), gpu_half=(a.owner_idx % 2),
        out_dir=a.out_dir,
        run_noise_only=bool(a.run_noise_only),
        reuse_corr=bool(a.reuse_corr), audit_fp32=False, zero_dm_filter=True,
        corr_work_dir=a.corr_dir, corr_save_all_owners=True,
        quant_target_max=QUANT_TARGET_MAX, verbose=a.verbose,
    )
    cands = sorted(res["candidates"], key=lambda c: -c["snr"])
    top = cands[0] if cands else None
    summ = {
        "owner_idx": a.owner_idx, "npz_path": res["npz_path"],
        "candidates_csv": res["candidates_csv"], "n_candidates": len(cands),
        "inj_cube_max": res.get("inj_cube_max"),
        "noise_only_fp_count": res.get("noise_only_fp_count"),
        "top_snr": (float(top["snr"]) if top else None),
        "top_dm": (float(top["dm_pc_cc"]) if top else None),
        "top_fdm": (int(top["fine_dm_idx"]) if top else None),
        "top_l": (int(top["l_pix"]) if top else None),
        "top_m": (int(top["m_pix"]) if top else None),
        "top_box": (int(top["width_samples"]) if top else None),
        "top_t_in_cube": (int(top["t_in_cube"]) if top else None),
    }
    Path(a.out_dir, "owner_result.json").write_text(_json.dumps(summ, indent=2))
    return 0


def _spawn(args_list: list[str], log_path: Path) -> int:
    """Run a child worker, teeing output to a log file. Settle briefly first so
    the device is fully released by any previous child before the corr-fast
    CUDA graph initialises."""
    import time
    time.sleep(6.0)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", "-m", "bench.preflight.inject_production_allgpu", *args_list]
    with open(log_path, "w") as fh:
        proc = subprocess.run(cmd, env=dict(os.environ), stdout=fh,
                              stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def _read_probe_snr(log_path: Path) -> float:
    for line in reversed(log_path.read_text().splitlines()):
        if line.startswith("PROBE_SNR="):
            return float(line.split("=", 1)[1])
    return 0.0


# ===========================================================================
# Orchestrator (no CUDA in this process)
# ===========================================================================

def _common_worker_args(a: argparse.Namespace) -> list[str]:
    out = [
        "--l-rad", str(a.l_rad), "--m-rad", str(a.m_rad),
        "--t-det", str(a.t_det), "--n-grid", str(a.n_grid),
        "--chan-sum-factor", str(a.chan_sum_factor), "--n-burnin", str(a.n_burnin),
        "--threshold-sigma", str(a.threshold_sigma), "--device", str(a.device),
        "--dm-plan-path", str(a.dm_plan_path),
    ]
    if not a.verbose:
        out.append("--quiet")
    return out


def calibrate_fluence(width_ms: float, target_snr: float, a: argparse.Namespace) -> float:
    guess = float(GUESS_FLUENCE.get(width_ms, 0.1))
    probe_dir = Path(a.out_dir) / f"_calib_w{width_ms:g}"
    log = Path(a.out_dir) / f"_calib_w{width_ms:g}.log"
    LOG.info("calibrating w=%.1fms: probe @ DM=%.0f fluence=%.4f (fp32 audit, isolated)",
             width_ms, CALIB_DM, guess)
    rc = _spawn(["--worker", "calib", "--dm", str(CALIB_DM), "--width", str(width_ms),
                 "--fluence", str(guess), "--out-dir", str(probe_dir),
                 *_common_worker_args(a)], log)
    snr = _read_probe_snr(log) if rc == 0 else 0.0
    if snr <= 0:
        LOG.warning("w=%.1fms probe undetected/failed (rc=%d); using guess fluence",
                    width_ms, rc)
        return guess
    fluence = guess * (target_snr / snr)
    LOG.info("w=%.1fms: probe snr=%.1f @ %.4f -> fluence=%.4f for ~%.0f sigma",
             width_ms, snr, guess, fluence, target_snr)
    return float(fluence)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr,
    )
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="/tmp/inject_allgpu")
    p.add_argument("--dms", type=float, nargs="*", default=list(DEFAULT_DMS))
    p.add_argument("--widths-ms", type=float, nargs="*", default=list(DEFAULT_WIDTHS_MS))
    p.add_argument("--target-snr", type=float, default=30.0)
    p.add_argument("--fluence-w1", type=float, default=None,
                   help="skip w=1ms calibration; use this fluence (Jy*ms)")
    p.add_argument("--fluence-w16", type=float, default=None,
                   help="skip w=16ms calibration; use this fluence (Jy*ms)")
    p.add_argument("--l-rad", type=float, default=0.004)
    p.add_argument("--m-rad", type=float, default=-0.002)
    p.add_argument("--t-det", type=int, default=192)
    p.add_argument("--n-grid", type=int, default=256)
    p.add_argument("--chan-sum-factor", type=int, default=8)
    p.add_argument("--n-burnin", type=int, default=8)
    p.add_argument("--threshold-sigma", type=float, default=8.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dm-plan-path", default=None)
    p.add_argument("--quiet", dest="verbose", action="store_false")
    # worker-mode args
    p.add_argument("--worker", choices=["calib", "owner"], default=None)
    p.add_argument("--dm", type=float, default=None)
    p.add_argument("--width", type=float, default=None)
    p.add_argument("--fluence", type=float, default=None)
    p.add_argument("--owner-idx", type=int, default=0)
    p.add_argument("--n-blocks", type=int, default=20)
    p.add_argument("--corr-dir", default=None)
    p.add_argument("--reuse-corr", action="store_true")
    p.add_argument("--run-noise-only", action="store_true")
    p.set_defaults(verbose=True)
    args = p.parse_args(argv)

    from bench.preflight._inject_search_driver import DEFAULT_DM_PLAN_PATH
    if args.dm_plan_path is None:
        args.dm_plan_path = str(DEFAULT_DM_PLAN_PATH)

    # ---- worker dispatch (child process) ----
    if args.worker == "calib":
        return _worker_calib(args)
    if args.worker == "owner":
        return _worker_owner(args)

    # ---- orchestrator (no CUDA here) ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = {1.0: args.fluence_w1, 16.0: args.fluence_w16}
    fluence_by_w = {}
    for w in args.widths_ms:
        ov = overrides.get(w)
        if ov is not None:
            LOG.info("w=%.1fms: using provided fluence=%.4f (skip calibration)", w, ov)
            fluence_by_w[w] = float(ov)
        else:
            fluence_by_w[w] = calibrate_fluence(w, args.target_snr, args)
    LOG.info("calibrated fluences: %s", fluence_by_w)

    import numpy as np
    from dsart.services.search_compute import _dm_grids_from_npz
    coarse, _, _ = _dm_grids_from_npz(Path(args.dm_plan_path), n_coarse=8)
    n_owners = int(coarse.shape[0])
    # All owners share ONE corr pass, so the resolved window must fit the
    # LARGEST coarse-DM owner's dedispersion sweep (not just the injection DM).
    # +4 headroom: the binding case is a LOW-DM injection (window centered late)
    # processed by the HIGHEST-DM owner (largest sweep) -> needs > diagonal n_blocks.
    n_blocks_all = _n_blocks_for_dm(float(coarse.max())) + 4
    LOG.info("n_blocks=%d (max coarse DM=%.0f + 4 headroom); %d owners",
             n_blocks_all, float(coarse.max()), n_owners)

    index = []
    for dm in args.dms:
        for w in args.widths_ms:
            tag = f"dm{int(dm)}_w{w:g}ms"
            tag_dir = out_dir / tag
            corr_dir = tag_dir / "corr_shared"
            owning = int(np.argmin(np.abs(coarse - dm)))
            fluence = fluence_by_w[w]
            LOG.info("=== injection %s (fluence=%.4f, owning GPU %d) -> %d owner children ===",
                     tag, fluence, owning, n_owners)
            owners = []
            owner_wargs: dict[int, list] = {}
            owner_rc: dict[int, int] = {}

            def _read_ores(o: int, rc: int) -> dict:
                rp = (tag_dir / f"owner{o}") / "owner_result.json"
                ores = json.loads(rp.read_text()) if rp.exists() else {"owner_idx": o, "rc_fail": rc}
                ores["coarse_dm"] = float(coarse[o])
                ores["is_owning"] = (o == owning)
                return ores

            for o in range(n_owners):
                owner_dir = tag_dir / f"owner{o}"
                log = tag_dir / f"owner{o}.log"
                wargs = ["--worker", "owner", "--dm", str(dm), "--width", str(w),
                         "--fluence", str(fluence), "--owner-idx", str(o),
                         "--n-blocks", str(n_blocks_all),
                         "--out-dir", str(owner_dir), "--corr-dir", str(corr_dir),
                         *_common_worker_args(args)]
                if o != 0:
                    wargs.append("--reuse-corr")
                if o == owning:
                    wargs.append("--run-noise-only")
                owner_wargs[o] = wargs
                rc = _spawn(wargs, log)
                _attempt = 1
                while rc != 0 and _attempt < 3:
                    LOG.warning("  %s owner %d rc=%d -> retry %d/2 (fresh process)",
                                tag, o, rc, _attempt)
                    time.sleep(10.0)
                    rc = _spawn(wargs, log)
                    _attempt += 1
                owner_rc[o] = rc
                ores = _read_ores(o, rc)
                owners.append(ores)
                LOG.info("  %s owner %d/%d: rc=%d snr=%s dm=%s b%s pix=(%s,%s) cubemax=%s%s",
                         tag, o, n_owners - 1, rc, ores.get("top_snr"), ores.get("top_dm"),
                         ores.get("top_box"), ores.get("top_l"), ores.get("top_m"),
                         ores.get("inj_cube_max"), " [OWNING]" if o == owning else "")

            # ----- refill pass -----
            # Owners that ran the flaky corr-unpack BEFORE any owner had built
            # the shared corr can fail transiently (CUDA "invalid argument").
            # Once a later owner successfully builds corr_shared, the failed
            # owners can be re-run with --reuse-corr: they load the cached
            # all-owners corr and slice their own stream, skipping the unpack
            # entirely. Only attempt while the cached corr still exists.
            cached_corr = bool(list(corr_dir.glob("corr_out_g*.npz"))) if corr_dir.exists() else False
            failed = [o for o in range(n_owners) if owner_rc.get(o, 1) != 0]
            if failed and cached_corr:
                LOG.info("  %s refill pass: owners %s reuse shared corr", tag, failed)
                for o in failed:
                    rargs = list(owner_wargs[o])
                    if "--reuse-corr" not in rargs:
                        rargs.append("--reuse-corr")
                    log = tag_dir / f"owner{o}.log"
                    time.sleep(5.0)
                    rc = _spawn(rargs, log)
                    _attempt = 1
                    while rc != 0 and _attempt < 3:
                        time.sleep(8.0)
                        rc = _spawn(rargs, log)
                        _attempt += 1
                    owner_rc[o] = rc
                    owners[o] = _read_ores(o, rc)
                    LOG.info("  %s refill owner %d: rc=%d snr=%s cubemax=%s", tag, o, rc,
                             owners[o].get("top_snr"), owners[o].get("inj_cube_max"))

            summ = {"dm_pc_cm3": dm, "width_ms": w, "fluence_jy_ms": fluence,
                    "owning_owner": owning, "n_owners": n_owners, "owners": owners}
            (tag_dir / "allgpu_summary.json").write_text(json.dumps(summ, indent=2))
            # free corr scratch (~7-12 GB) before the next injection
            import shutil
            shutil.rmtree(corr_dir, ignore_errors=True)
            ow = owners[owning]
            LOG.info("%s DONE: owning GPU %d snr=%s dm=%s pix=(%s,%s) noiseFP=%s",
                     tag, owning, ow.get("top_snr"), ow.get("top_dm"),
                     ow.get("top_l"), ow.get("top_m"), ow.get("noise_only_fp_count"))
            index.append({"tag": tag, "dm": dm, "width_ms": w,
                          "fluence_jy_ms": fluence, "owning_owner": owning})

    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    LOG.info("DONE: %d injections; cubes under %s/<tag>/owner<o>/", len(index), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
