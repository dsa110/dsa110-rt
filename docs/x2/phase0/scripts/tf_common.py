"""Shared setup for the time-filter study: binds the model copy, then the snr2x scheme definitions
(read-only import from docs/timeres_study/snr2x; its own `snr_model` import is the unchanged original,
whose Scheme methods are identical to the copy's)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tf_snr_model as M  # noqa: E402  (binds dsart to this worktree first)

SNR2X = os.path.abspath(os.path.join(HERE, "..", "..", "..", "timeres_study", "snr2x"))
sys.path.append(SNR2X)
from schemes import T4, proposed  # noqa: E402

RES = os.path.abspath(os.path.join(HERE, "..", "results"))
WIDTHS_MS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]       # run_snr2x.WIDTHS_MS
SHAPES = ("gauss", "tophat", "exp")
BANK_KEYS = ("boxcar", "b2", "b3", "gauss")


def scheme():
    return proposed(q1=T4, n_sub=2)


__all__ = ["M", "T4", "proposed", "RES", "WIDTHS_MS", "SHAPES", "BANK_KEYS", "scheme"]
