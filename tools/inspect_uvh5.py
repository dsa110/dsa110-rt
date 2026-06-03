#!/usr/bin/env python3
"""Print salient header info from one or more DSA-110 UVH5 files.

Usage:
    python tools/inspect_uvh5.py <file.hdf5> [<file.hdf5> ...]

Reads with raw h5py (no pyuvdata dependency) so it works in any env
with h5py installed. Times are converted from JD to ISO UTC.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
from astropy.time import Time


def _scalar(v):
    """Decode bytes and unwrap 0-d numpy arrays."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray) and v.shape == ():
        return _scalar(v.item())
    return v


def inspect(path: Path) -> None:
    with h5py.File(path, "r") as f:
        h = f["Header"]
        ek = h["extra_keywords"]

        t_jd = h["time_array"][:]
        t0 = Time(t_jd[0], format="jd", scale="utc")
        t1 = Time(t_jd[-1], format="jd", scale="utc")
        dt_s = (t1 - t0).sec

        freq = h["freq_array"][:]  # (Nspws, Nfreqs) Hz
        f_lo, f_hi = freq.min() / 1e9, freq.max() / 1e9
        chan_w_hz = float(_scalar(h["channel_width"][()]))

        dec_rad = float(_scalar(h["phase_center_app_dec"][()]))
        dec_deg_extra = ek.get("phase_center_dec")
        dec_extra_str = (
            f"{float(_scalar(dec_deg_extra[()])):.4f} deg"
            if dec_deg_extra is not None
            else "(missing)"
        )

        ha_rad = ek.get("ha_phase_center")
        ha_str = (
            f"{float(_scalar(ha_rad[()])):.4f} rad"
            if ha_rad is not None
            else "(missing)"
        )

        epoch = ek.get("phase_center_epoch")
        epoch_str = _scalar(epoch[()]) if epoch is not None else "(missing)"

        int_time = h["integration_time"][:]
        int_med = float(np.median(int_time))

        print(f"== {path.name} ==")
        print(f"  telescope/instr : {_scalar(h['telescope_name'][()])} / "
              f"{_scalar(h['instrument'][()])}")
        print(f"  site (lat/lon/alt): "
              f"{float(_scalar(h['latitude'][()])):.4f} deg, "
              f"{float(_scalar(h['longitude'][()])):.4f} deg, "
              f"{float(_scalar(h['altitude'][()])):.1f} m")
        print(f"  object/phase    : {_scalar(h['object_name'][()])} "
              f"({_scalar(h['phase_type'][()])})")
        print(f"  phase-cen DEC   : {np.rad2deg(dec_rad):.4f} deg  "
              f"({dec_rad:.6f} rad)   extra={dec_extra_str}")
        print(f"  phase-cen HA    : {ha_str}")
        print(f"  epoch           : {epoch_str}")
        print(f"  obs t0 (UTC)    : {t0.isot}  (JD {t_jd[0]:.6f})")
        print(f"  obs t1 (UTC)    : {t1.isot}  (JD {t_jd[-1]:.6f})")
        print(f"  obs duration    : {dt_s:.3f} s  ({len(t_jd)} times, "
              f"int={int_med:.4f}s)")
        print(f"  freqs           : {f_lo:.6f} -- {f_hi:.6f} GHz   "
              f"chan_w={chan_w_hz / 1e3:.3f} kHz   nchan={freq.shape[1]}")
        print(f"  Nants / Nbls    : {int(_scalar(h['Nants_data'][()]))} ants, "
              f"{int(_scalar(h['Nbls'][()]))} baselines")
        print(f"  Nblts / Ntimes  : {int(_scalar(h['Nblts'][()]))} / "
              f"{int(_scalar(h['Ntimes'][()]))}")
        print(f"  pols            : {h['polarization_array'][:].tolist()}")
        print(f"  visdata shape   : {f['Data']['visdata'].shape}  "
              f"dtype={f['Data']['visdata'].dtype}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    for arg in argv[1:]:
        try:
            inspect(Path(arg))
        except Exception as exc:
            print(f"!! {arg}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
