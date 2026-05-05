#!/usr/bin/env python3
"""bench/casa38_meridian_wrapper.py — D17 wrapper around dsamfs.run_fringestopping.

Lets the M2 voltage-fixture pipeline run ``meridian_fringestop`` against
a custom param file and a non-shared-state pointing declination, WITHOUT
modifying any installed casa38 sources (per user constraint 2026-05-04).

Two in-process monkey-patches are applied to ``dsamfs.utils`` *before*
:func:`dsamfs.routines.run_fringestopping` is called:

  * ``get_pointing_declination``: returns ``--pt-dec-deg * u.deg`` instead
    of querying ``/mon/array/dec`` from etcd. Default: read from etcd
    (legacy behaviour).
  * ``put_outrigger_delays`` and ``put_refmjd``: become no-ops if
    ``--no-etcd-write`` is passed, so a test run on a workstation that
    happens to be plumbed to etcd doesn't pollute production state.
    Default: keep legacy etcd writes.

The script imports dsamfs lazily so it can be inspected (--help) outside
the casa38 env. It must be RUN inside casa38 (where dsamfs / dsacalib /
psrdada-python live).

Invocation::

    /home/ubuntu/anaconda3/envs/casa38/bin/python \\
        bench/casa38_meridian_wrapper.py \\
        --param-file /tmp/0319_sb00.yaml \\
        --out-dir   /home/ubuntu/data/vikram/0319_uvh5/ \\
        --working-dir /home/ubuntu/data/vikram/0319_uvh5/ \\
        --pt-dec-deg 41.51169444 \\
        --no-etcd-write
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--param-file", required=True,
                   help="path to dsamfs YAML param file")
    p.add_argument("--out-dir", required=True,
                   help="output dir for UVH5")
    p.add_argument("--working-dir", required=True,
                   help="working dir (fringestopping table cache lives here)")
    p.add_argument("--pt-dec-deg", type=float, default=None,
                   help="override pointing declination in degrees "
                        "(default: read from /mon/array/dec via etcd)")
    p.add_argument("--no-etcd-write", action="store_true",
                   help="stub put_outrigger_delays + put_refmjd to no-ops "
                        "so this test run doesn't touch shared etcd state")
    p.add_argument("--header-file", default=None,
                   help="optional dada header file (passed through)")
    p.add_argument("--spl", action="store_true",
                   help="forward to dsamfs (split-pol output)")
    p.add_argument("--nsfrb", action="store_true",
                   help="forward to dsamfs (NSFRB mode)")
    args = p.parse_args(argv)

    # --- Late imports so --help works outside casa38 ---
    import numpy as np
    import astropy.units as u
    from dsamfs import utils as dsamfs_utils
    from dsamfs.routines import run_fringestopping

    # --- Patch 1: pointing declination override ---
    if args.pt_dec_deg is not None:
        target_dec = args.pt_dec_deg
        print(f"[wrapper] monkey-patching dsamfs.utils.get_pointing_declination "
              f"-> {target_dec} deg (was etcd-based)", flush=True)

        def _patched_dec():
            return target_dec * u.deg

        dsamfs_utils.get_pointing_declination = _patched_dec
    else:
        print("[wrapper] using etcd-backed get_pointing_declination "
              "(no --pt-dec-deg)", flush=True)

    # --- Patch 2: stub etcd writes if requested ---
    if args.no_etcd_write:
        print("[wrapper] stubbing put_outrigger_delays + put_refmjd "
              "(no-ops; --no-etcd-write)", flush=True)
        dsamfs_utils.put_outrigger_delays = lambda _delays: None
        dsamfs_utils.put_refmjd = lambda _refmjd: None

    # --- Patch 3: clean end-of-stream handling for read_buffer.
    #
    # dsamfs/io.py:343 has a latent shape bug on the SECOND iteration of its
    # outer `while not nans:` loop: after a successful first-iter freq-int
    # reshape, it count_nonzero's the previous iteration's already-averaged
    # `data` and reshapes the result with the un-averaged nfreq_int factor,
    # crashing with "cannot reshape array of size 96 into shape (1, 48, 48, 8, 2)".
    #
    # We can't (and won't) edit dsamfs sources. Instead we patch
    # `dsamfs.utils.read_buffer` to raise SystemExit (a BaseException, NOT
    # caught by `except (AssertionError, ValueError, PSRDadaError)` in
    # dada_to_uvh5) once we're past the first integration. SystemExit
    # unwinds the inner+outer while loops cleanly without ever entering the
    # buggy reshape branch. The first integration's hdf5 (the one we
    # actually want) is renamed by `os.rename` BEFORE this exception fires,
    # so it survives intact.
    #
    # Three independent termination conditions ensure iter-2 never runs:
    #   1. Empty page from psrdada (the EOD marker has size 0).
    #   2. `reader.isEndOfData` flag flipped after markCleared.
    #   3. A hard cap on `nint * samples_per_frame_out` successful reads,
    #      derived from the dsamfs param yaml. (Defense in depth: even if
    #      the writer never marks EOD and stale pages keep coming through,
    #      we still exit after exactly one integration's worth of data.)
    import yaml as _yaml
    with open(args.param_file, encoding="utf-8") as _f:
        _param = _yaml.safe_load(_f)
    _max_reads = int(_param["nint"]) * int(_param["samples_per_frame_out"])
    print(f"[wrapper] patching dsamfs.utils.read_buffer for clean EOD "
          f"handling (max_reads={_max_reads})", flush=True)

    _reads_done = [0]

    def _patched_read_buffer(reader, nbls, nchan, npol):
        if _reads_done[0] >= _max_reads:
            raise SystemExit(
                f"dsamfs: reached max_reads={_max_reads} (one integration "
                f"complete) — terminating to dodge dsamfs/io.py:343 bug")
        page = reader.getNextPage()
        reader.markCleared()
        if reader.isEndOfData:
            raise SystemExit("dsamfs: bada stream EOD (reader flag set)")
        data = np.asarray(page)
        if data.size == 0:
            raise SystemExit("dsamfs: bada stream ended (empty page)")
        data = data.view(np.float32)
        data = data.reshape(-1, 2).view(np.complex64).squeeze(axis=-1)
        try:
            data = data.reshape(-1, nbls, nchan, npol)
        except ValueError:
            full = nbls * nchan * npol
            n_full = data.shape[0] // full
            if n_full == 0:
                raise SystemExit(
                    f"dsamfs: short bada page ({data.shape[0]} < "
                    f"{full} samples) — treating as EOD")
            data = data[:n_full * full].reshape(-1, nbls, nchan, npol)
        _reads_done[0] += 1
        return data

    dsamfs_utils.read_buffer = _patched_read_buffer
    # `dsamfs.io` did `import dsamfs.utils as pu` at module-load time, so
    # reassigning `dsamfs_utils.read_buffer` is sufficient — `pu.read_buffer`
    # in io.py looks up the attribute at call time on the same module object.

    # --- Run ---
    print(f"[wrapper] run_fringestopping(param_file={args.param_file!r}, "
          f"out_dir={args.out_dir!r}, working_dir={args.working_dir!r}, "
          f"nsfrb={args.nsfrb}, spl={args.spl})", flush=True)
    run_fringestopping(
        param_file=args.param_file,
        header_file=args.header_file,
        output_dir=args.out_dir,
        working_dir=args.working_dir,
        nsfrb=args.nsfrb,
        spl=args.spl,
    )
    print("[wrapper] run_fringestopping returned cleanly", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
