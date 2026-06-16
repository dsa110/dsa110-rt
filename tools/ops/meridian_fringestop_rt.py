#!/usr/bin/env python3
"""Production wrapper that runs the legacy ``meridian_fringestop`` slow-vis
fringe-stopper against the M7 real-time correlator's ``bada`` PSRDADA ring,
**without modifying the casa38 / dsamfs install** (hard constraint D14).

Run-environment: **casa38** (``/home/ubuntu/anaconda3/envs/casa38/bin/python``,
python 3.8). This module is launched by the ``dsart_rt`` orchestrator as a
normal routine (see ``configs/dsart_pipeline_rt.yaml`` -> routine
``meridian_fringestop``), one process per corr node, replacing the
``bada_drain`` stand-in as the single ``bada`` reader.

What it does, in order:

1. Resolve the pointing declination (``--pt-dec-deg`` literal, or ``auto`` ->
   read ``/mon/array/dec``) and **snap it to the fring-table cache grid**
   (default 0.25 deg). ``meridian_fringestop`` only accepts a cached
   ``fringestopping_table`` whose ``dec_rad`` matches ``pt_dec`` to 1e-6 rad
   (``dsamfs.utils.load_visibility_model``), so the cache is only usable at the
   grid points; we snap and then *pin* the dec to the grid value.
2. Assert this host is present in ``/cnf/corr.ch0`` (so dsamfs tags the right
   frequency subband). dsamfs *silently* falls back to ``subband=0`` /
   ``ch0=3400`` on an unknown host, which mis-tags the band on every node; we
   fail loudly instead.
3. Resolve the per-(dec) cache file under ``--fstable-cache`` and stage it into
   ``--working-dir`` under the **legacy table name** dsamfs looks up
   (``fringestopping_table_dec{:.1f}deg_{nant}ant.npz``) via a symlink. The
   cached ``.npz`` holds only the geometric w-term (``bw``, in metres); the
   per-channel phasor is built at load time from each node's ``fobs`` -- so one
   table per dec is correct for all 16 subbands.
4. Monkeypatch ``dsamfs.utils.get_pointing_declination`` (in *our* process only)
   to return the snapped grid dec, so ``parse_params`` and the table-name
   builder both use the exact dec the cache was built at.
5. Publish a best-effort liveness heartbeat to
   ``/mon/corr_rt/<cn>/meridian_ready`` (mirrors ``corr_fast_ready``) so the
   dashboard can see meridian is alive and which file it last wrote. dsart_rt
   does NOT auto-restart routines and this is the sole ``bada`` reader, so a
   meridian crash deadlocks ``corr_slow`` within ~40 s (the 300-buffer ring) --
   the heartbeat is how an operator notices.
6. Call ``dsamfs.routines.run_fringestopping(param_file=None, ...)`` which reads
   ``/cnf/corr`` + ``/cnf/fringe`` from etcd, opens ``bada``, fringe-stops, and
   writes ``<UTC>_sb<NN>.hdf5`` to ``--working-dir`` (posting ``/cmd/cal`` per
   file). It blocks until ``bada`` EOD (corr_slow marks EOD on stop).

Nothing here is imported by the dsart package; the only casa38-side imports
(``dsamfs``, ``astropy``, ``dsautils``) are deferred into functions so the pure
helpers below can be unit-tested from the dsa110-rt env (numpy + stdlib only).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

LOG = logging.getLogger("meridian_fringestop_rt")

#: Default fringe-table cache directory (built by tools/build_fstable_cache.py).
DEFAULT_FSTABLE_CACHE = Path("/home/ubuntu/data/fstables")

#: Default working/output dir. dsamfs writes ``<UTC>_sb<NN>.hdf5`` here AND the
#: ``/cmd/cal`` notification path uses ``--output-dir``; keep them equal so the
#: downstream consumer finds the file where it was written.
DEFAULT_DATA_DIR = Path("/home/ubuntu/data")

#: Cache grid step (deg). Must match tools/build_fstable_cache.py --dec-step.
DEFAULT_DEC_GRID_STEP = 0.25

#: Number of antennas in the correlator (DSA-110). nbls = nant*(nant+1)/2 = 4656
#: with autocorrelations, matching the bada contract.
DEFAULT_NANT = 96


# ---------------------------------------------------------------------------
# Pure helpers (stdlib + numpy only; no dsamfs/astropy/etcd) -> unit-testable
# from the dsa110-rt env.
# ---------------------------------------------------------------------------


def snap_dec_to_grid(dec_deg: float, step_deg: float = DEFAULT_DEC_GRID_STEP) -> float:
    """Snap a pointing declination to the nearest cache grid point.

    The fringe-table cache is built on a fixed ``step_deg`` grid (default
    0.25 deg). ``load_visibility_model`` asserts the table's ``dec_rad``
    matches ``pt_dec`` to 1e-6 rad, so we must pin the dec to a grid value to
    get a cache hit. Returns a value that is an exact integer multiple of
    ``step_deg`` (0.25 is exact in binary, so the round-trip is clean)."""
    if step_deg <= 0.0:
        raise ValueError(f"step_deg must be > 0, got {step_deg!r}")
    return round(dec_deg / step_deg) * step_deg


def cache_table_filename(dec_deg_grid: float, nant: int, refmjd: float) -> str:
    """Cache file name as written by tools/build_fstable_cache.py (D6 scheme):
    4-decimal sign-aware dec + refmjd suffix so the 0.25 deg grid doesn't
    collide and refmjd drift is a glob filter."""
    return (
        f"fringestopping_table_dec_{dec_deg_grid:+08.4f}deg_"
        f"{int(nant)}ant_refmjd{float(refmjd):.6f}.npz"
    )


def legacy_table_name(dec_deg_grid: float, nant: int) -> str:
    """The name ``dsamfs.routines.run_fringestopping`` builds and looks up in
    ``working_dir``: ``fringestopping_table_dec{:.1f}deg_{nant}ant.npz`` where
    the dec is ``(pt_dec*u.rad).to_value(u.deg)``. Because we pin ``pt_dec`` to
    the snapped grid dec, the ``:.1f`` here reproduces that name exactly.

    Only one observation's table is ever staged in ``working_dir`` at a time,
    so the legacy ``:.1f`` rounding (which collides adjacent 0.25 deg points in
    a full bank) is not a problem here."""
    return f"fringestopping_table_dec{dec_deg_grid:.1f}deg_{int(nant)}ant.npz"


def resolve_subband(ch0_map: dict, hostname: str) -> int:
    """Return the subband index dsamfs assigns this host: the position of
    ``hostname`` in ``ch0``'s key order (``parse_params``:
    ``list(corr_cnf['ch0'].keys()).index(hname)``). Raises ``KeyError`` if the
    host is absent -- the caller must fail loudly rather than let dsamfs
    silently fall back to subband 0."""
    keys = list(ch0_map.keys())
    if hostname not in keys:
        raise KeyError(
            f"host {hostname!r} not in /cnf/corr.ch0 (keys={keys}); dsamfs would "
            f"silently fall back to subband 0 / ch0=3400 and mis-tag the band"
        )
    return keys.index(hostname)


def validate_cache_table(
    path: Path,
    *,
    nbls: int,
    nint: int,
    dec_deg_grid: float,
    tsamp: float,
    refmjd: float,
) -> tuple[bool, str]:
    """Cheap pre-flight that mirrors the cheap half of
    ``dsamfs.utils.load_visibility_model``'s asserts so we can give a clear
    error before handing off (dsamfs would otherwise silently regenerate the
    table at runtime). Returns ``(ok, reason)``. numpy-only."""
    import numpy as np

    if not path.exists():
        return False, f"cache file does not exist: {path}"
    try:
        d = np.load(path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        return False, f"cache file failed to load: {exc!r}"
    try:
        bw = d["bw"]
        if tuple(bw.shape) != (int(nint), int(nbls)):
            return False, (
                f"bw shape {tuple(bw.shape)} != expected ({nint}, {nbls}) "
                f"-- nint/nant mismatch with /cnf"
            )
        dec_rad_expected = math.radians(dec_deg_grid)
        if abs(float(d["dec_rad"]) - dec_rad_expected) >= 1e-6:
            return False, (
                f"dec_rad {float(d['dec_rad']):.9f} != snapped "
                f"{dec_rad_expected:.9f} (>=1e-6 rad)"
            )
        if abs(float(d["tsamp_s"]) - float(tsamp)) >= 1e-6:
            return False, f"tsamp_s {float(d['tsamp_s'])} != /cnf tsamp {tsamp}"
        if abs(float(d["refmjd"]) - float(refmjd)) >= 1e-6:
            return False, f"refmjd {float(d['refmjd'])} != /cnf refmjd {refmjd}"
    except KeyError as exc:
        return False, f"cache file missing key {exc!r}"
    return True, "ok"


def stage_legacy_symlink(
    cache_file: Path, working_dir: Path, dec_deg_grid: float, nant: int
) -> Path:
    """Symlink ``cache_file`` into ``working_dir`` under the legacy table name
    dsamfs looks up. Replaces any stale symlink/file at that path. Returns the
    staged path. (If dsamfs ever decides to regenerate -- e.g. an assert we did
    not pre-check fails -- it ``os.unlink``s this symlink and writes a real file
    in its place, which is still safe because ``working_dir`` is writable.)"""
    working_dir.mkdir(parents=True, exist_ok=True)
    dest = working_dir / legacy_table_name(dec_deg_grid, nant)
    try:
        if dest.is_symlink() or dest.exists():
            dest.unlink()
    except OSError as exc:
        raise RuntimeError(f"could not clear stale table at {dest}: {exc}") from exc
    os.symlink(os.fspath(cache_file), os.fspath(dest))
    return dest


# ---------------------------------------------------------------------------
# etcd-backed config reads (casa38 side; dsautils available there)
# ---------------------------------------------------------------------------


def _store() -> Any:
    from dsautils.dsa_store import DsaStore

    return DsaStore()


def read_pt_dec_deg(store: Any) -> float:
    """Read the pointing declination (deg) from ``/mon/array/dec``."""
    d = store.get_dict("/mon/array/dec")
    if not isinstance(d, dict) or "dec_deg" not in d:
        raise RuntimeError(f"/mon/array/dec missing or malformed: {d!r}")
    return float(d["dec_deg"])


def read_corr_fringe_cnf(store: Any) -> dict[str, Any]:
    """Pull the fields meridian needs from ``/cnf/corr`` + ``/cnf/fringe`` so we
    can resolve the cache file + assert the subband BEFORE handing off to
    dsamfs (which re-reads the same keys itself)."""
    corr = store.get_dict("/cnf/corr")
    fringe = store.get_dict("/cnf/fringe")
    if not isinstance(corr, dict) or not isinstance(fringe, dict):
        raise RuntimeError("/cnf/corr or /cnf/fringe missing/empty in etcd")
    ant_od = corr["antenna_order"]
    antenna_order = [int(v) for v in list(ant_od.values())]
    nant = int(corr.get("nant", len(antenna_order)))
    return {
        "ch0": dict(corr["ch0"]),
        "nant": nant,
        "nbls": (nant * (nant + 1)) // 2,
        "tsamp": float(corr["tsamp"]),
        "nint": int(fringe["nint"]),
        "refmjd": float(fringe["refmjd"]),
    }


# ---------------------------------------------------------------------------
# Liveness heartbeat
# ---------------------------------------------------------------------------


def _now_mjd() -> float:
    return time.time() / 86400.0 + 40587.0


class Heartbeat:
    """Best-effort liveness publisher to ``/mon/corr_rt/<cn>/meridian_ready``.
    Never raises into the run loop; a publish failure is logged and ignored."""

    def __init__(self, cn_id: int, working_dir: Path, subband: int, dec_deg: float,
                 interval_s: float = 10.0, spl: bool = False) -> None:
        # SPL gets its OWN heartbeat key so the dashboard can show the
        # two bada readers independently (a dead SPL fringe-stopper is
        # just as fatal to corr_slow as a dead production one).
        self._spl = bool(spl)
        suffix = "meridian_spl_ready" if self._spl else "meridian_ready"
        self._key = f"/mon/corr_rt/{int(cn_id)}/{suffix}"
        self._working_dir = working_dir
        self._subband = int(subband)
        self._dec_deg = float(dec_deg)
        self._interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="mfs-heartbeat", daemon=True)
        self._store: Any = None

    def _newest_hdf5(self) -> tuple[Optional[str], int]:
        # SPL writes *_sb<NN>_spl.hdf5; production writes *_sb<NN>.hdf5.
        # Match the right family so the file count is meaningful per
        # reader (both can co-exist in different working dirs).
        pat = (f"*_sb{self._subband:02d}_spl*.hdf5" if self._spl
               else f"*_sb{self._subband:02d}.hdf5")
        try:
            files = sorted(self._working_dir.glob(pat))
        except OSError:
            return None, 0
        return (files[-1].name if files else None), len(files)

    def _publish(self, ready: bool) -> None:
        try:
            if self._store is None:
                self._store = _store()
            last, n = self._newest_hdf5()
            self._store.put_dict(self._key, {
                "ready": bool(ready),
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "subband": self._subband,
                "dec_deg": self._dec_deg,
                "ts_wall_unix": time.time(),
                "time_mjd": _now_mjd(),
                "last_hdf5": last,
                "n_hdf5": n,
            })
        except Exception as exc:  # noqa: BLE001
            LOG.warning("heartbeat publish failed (continuing): %s", exc)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._publish(ready=True)
            self._stop.wait(self._interval_s)

    def start(self) -> None:
        self._publish(ready=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._publish(ready=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Do all the etcd reads + cache staging + monkeypatch. Returns a small
    context dict. Separated from the dsamfs run so it can be exercised on-node
    with ``--prepare-only`` (stages the symlink, prints the plan, exits)."""
    store = _store()
    cnf = read_corr_fringe_cnf(store)
    hostname = socket.gethostname()

    # Subband assertion (fail loud; do not let dsamfs fall back to 0).
    subband = resolve_subband(cnf["ch0"], hostname)

    # Resolve + snap dec.
    if args.pt_dec_deg.strip().lower() in ("auto", "", "none", "customdec"):
        # 'CUSTOMDEC' reaches us literally only if dsart_rt failed to
        # substitute it (val was None and dec couldn't be resolved); fall back
        # to a direct /mon/array/dec read so we still launch.
        dec_raw = read_pt_dec_deg(store)
    else:
        dec_raw = float(args.pt_dec_deg)
    dec_grid = snap_dec_to_grid(dec_raw, args.dec_grid_step)

    # Resolve the SPL nint/nfreq_int overrides (if any). The fringe-
    # stopping table's bw array is shaped (nint, nbls), so the cache
    # must be validated against the EFFECTIVE nint we will actually run
    # at -- which for SPL may differ from the production /cnf/fringe
    # nint. nint precedence: --nint-spl > --integration-s (converted via
    # tsamp) > /cnf/fringe (left to dsamfs). nfreq_int: --nfreq-int-spl
    # > /cnf/fringe.
    override_nint: Optional[int] = None
    override_nfreq_int: Optional[int] = None
    if args.spl:
        if args.nint_spl is not None:
            override_nint = int(args.nint_spl)
        elif args.integration_s is not None:
            override_nint = max(1, int(round(float(args.integration_s) / cnf["tsamp"])))
        if args.nfreq_int_spl is not None:
            override_nfreq_int = int(args.nfreq_int_spl)
    eff_nint = override_nint if override_nint is not None else cnf["nint"]

    cache_file = Path(args.fstable_cache) / cache_table_filename(
        dec_grid, cnf["nant"], cnf["refmjd"]
    )
    ok, reason = validate_cache_table(
        cache_file, nbls=cnf["nbls"], nint=eff_nint, dec_deg_grid=dec_grid,
        tsamp=cnf["tsamp"], refmjd=cnf["refmjd"],
    )
    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    if ok:
        staged = stage_legacy_symlink(cache_file, working_dir, dec_grid, cnf["nant"])
        LOG.info("staged fstable cache %s -> %s", cache_file.name, staged)
    else:
        if not args.allow_regenerate:
            raise RuntimeError(
                f"fringe-table cache miss/mismatch ({reason}). Build it in casa38:\n"
                f"  {sys.executable} tools/build_fstable_cache.py --from-etcd "
                f"--dec-min {dec_grid} --dec-max {dec_grid} --output-dir "
                f"{args.fstable_cache}\n"
                f"or pass --allow-regenerate to let dsamfs regenerate at runtime "
                f"(~30 s, needs casatools)."
            )
        # Ensure no stale legacy-named file blocks dsamfs's own regeneration.
        stale = working_dir / legacy_table_name(dec_grid, cnf["nant"])
        if stale.is_symlink() or stale.exists():
            stale.unlink()
        LOG.warning(
            "fstable cache unusable (%s); --allow-regenerate set, dsamfs will "
            "regenerate at dec=%+.4f in %s (slow)", reason, dec_grid, working_dir,
        )

    # Pin dec to the grid value in *our* process so parse_params and the
    # table-name builder both use the exact dec the cache was built at.
    import astropy.units as u  # noqa: WPS433 (casa38 side)
    import dsamfs.utils as dsamfs_utils

    _grid_q = dec_grid * u.deg

    def _patched_get_pointing_declination():  # noqa: WPS430
        return _grid_q

    dsamfs_utils.get_pointing_declination = _patched_get_pointing_declination

    # SPL nint/nfreq_int overrides: dsamfs reads these from the
    # /cnf/fringe nint_spl/nfreq_int_spl per-host maps inside
    # parse_params. To drive them from the Control-tab panel WITHOUT
    # editing /cnf/fringe (and without touching the casa38 dsamfs
    # install -- constraint D14), we wrap parse_params in OUR process so
    # the returned nint (tuple index 8) / nfreq_int (index 9) reflect
    # the overrides. routines.run_fringestopping calls pu.parse_params
    # where pu IS dsamfs.utils, so patching the module attr is enough.
    if override_nint is not None or override_nfreq_int is not None:
        _orig_parse = dsamfs_utils.parse_params

        def _patched_parse_params(param_file=None, nsfrb=False, spl=False):  # noqa: WPS430
            res = list(_orig_parse(param_file=param_file, nsfrb=nsfrb, spl=spl))
            if override_nint is not None:
                res[8] = int(override_nint)
            if override_nfreq_int is not None:
                res[9] = int(override_nfreq_int)
            return tuple(res)

        dsamfs_utils.parse_params = _patched_parse_params

    LOG.info(
        "host=%s subband=%d dec_raw=%+.4f -> grid=%+.4f (step=%.4f) nant=%d "
        "nint=%d (eff=%d) nfreq_int_override=%s spl=%s refmjd=%.6f",
        hostname, subband, dec_raw, dec_grid, args.dec_grid_step, cnf["nant"],
        cnf["nint"], eff_nint, override_nfreq_int, bool(args.spl), cnf["refmjd"],
    )
    return {
        "subband": subband, "dec_grid": dec_grid, "working_dir": working_dir,
        "cache_ok": ok, "eff_nint": eff_nint,
        "override_nfreq_int": override_nfreq_int,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cn-id", type=int, required=True, help="corr cn_id (for the heartbeat key)")
    p.add_argument("--pt-dec-deg", default="auto",
                   help="pointing dec in deg, or 'auto'/'CUSTOMDEC' to read /mon/array/dec")
    p.add_argument("--fstable-cache", type=Path, default=DEFAULT_FSTABLE_CACHE)
    p.add_argument("--working-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help="where dsamfs writes <UTC>_sbNN.hdf5 + the staged fstable")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="dir used in the /cmd/cal notification path; defaults to --working-dir "
                        "(MUST equal working-dir so the consumer finds the file)")
    p.add_argument("--dec-grid-step", type=float, default=DEFAULT_DEC_GRID_STEP)
    p.add_argument("--spl", action="store_true",
                   help="spectral-line mode: write a second *_sb<NN>_spl.hdf5 product "
                        "with finer channelisation. Without the override flags below "
                        "the per-host nfreq_int_spl/nint_spl maps in /cnf/fringe are "
                        "used (legacy behaviour).")
    p.add_argument("--integration-s", type=float, default=None,
                   help="SPL integration time in seconds; converted to nint via the "
                        "/cnf/corr tsamp (overrides /cnf/fringe nint_spl). Ignored "
                        "unless --spl. --nint-spl takes precedence if both are set.")
    p.add_argument("--nint-spl", type=int, default=None,
                   help="SPL nint override (number of slow-vis frames to integrate). "
                        "Overrides both --integration-s and /cnf/fringe nint_spl. "
                        "Ignored unless --spl.")
    p.add_argument("--nfreq-int-spl", type=int, default=None,
                   help="SPL nfreq_int override (channels to average; must divide the "
                        "384-channel sub-band). Overrides /cnf/fringe nfreq_int_spl. "
                        "Ignored unless --spl.")
    p.add_argument("--allow-regenerate", action="store_true",
                   help="on cache miss, let dsamfs regenerate the table at runtime "
                        "(slow) instead of failing")
    p.add_argument("--heartbeat-interval-s", type=float, default=10.0)
    p.add_argument("--prepare-only", action="store_true",
                   help="do the etcd reads + cache staging + print the plan, then exit "
                        "(no bada read); for on-node validation")
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.output_dir is None:
        args.output_dir = args.working_dir
    if Path(args.output_dir) != Path(args.working_dir):
        LOG.warning(
            "output-dir (%s) != working-dir (%s): dsamfs writes hdf5 to working-dir "
            "but advertises output-dir in /cmd/cal -- the consumer may not find it",
            args.output_dir, args.working_dir,
        )

    ctx = _prepare(args)

    if args.prepare_only:
        LOG.info("--prepare-only: staged and validated; exiting before bada read. ctx=%s", ctx)
        return 0

    hb = Heartbeat(
        cn_id=args.cn_id, working_dir=Path(args.working_dir),
        subband=ctx["subband"], dec_deg=ctx["dec_grid"],
        interval_s=args.heartbeat_interval_s, spl=bool(args.spl),
    )
    hb.start()
    try:
        from dsamfs.routines import run_fringestopping

        LOG.info("starting run_fringestopping (sole bada reader); blocks until EOD")
        run_fringestopping(
            param_file=None,
            header_file=None,
            output_dir=os.fspath(args.output_dir),
            working_dir=os.fspath(args.working_dir),
            nsfrb=False,
            spl=bool(args.spl),
        )
        LOG.info("run_fringestopping returned (bada EOD)")
        return 0
    finally:
        hb.stop()


if __name__ == "__main__":
    raise SystemExit(main())
