"""Cost of one DM point of the grid (all banks, shapes, widths) - sizing check for tf_run.py."""
import time

import numpy as np

from tf_common import WIDTHS_MS, M, scheme

S = scheme()
np.set_printoptions(precision=4, suppress=True)
for dm in (812.3, 2210.3):
    t = time.time()
    r = M.evaluate_banks(S, dm, [w * 1e3 for w in WIDTHS_MS])
    print(f"DM {dm}: {time.time() - t:.1f} s")
    print(r)
