#!/usr/bin/env python3
"""Where does AsyncTransportTx.transmit spend its pipeline-thread time?

Splits the hand-off into (a) the pinned D2H of the whole cube and
(b) the per-worker np.copyto into the shm ring slots, and times (b)
against plain anonymous memory as a control. Cube sizes are the
production n_sub=1 (8 x 128 x 3322) and n_sub=4 (8 x 128 x 8575).
"""
from __future__ import annotations

import time
from multiprocessing import shared_memory

import numpy as np
import torch


def t_ms(fn, reps=10):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main():
    dev = torch.device("cuda:0")
    for label, cells in (("n_sub=1", 3322), ("n_sub=4", 8575)):
        shape = (8, 128, cells)
        nbytes = int(np.prod(shape)) * 8
        cube = torch.randn(shape, dtype=torch.complex64, device=dev)
        pinned = torch.empty(shape, dtype=torch.complex64, pin_memory=True)
        pageable = torch.empty(shape, dtype=torch.complex64)

        d2h_pin = t_ms(lambda: pinned.copy_(cube))
        d2h_page = t_ms(lambda: pageable.copy_(cube))

        host = pinned.numpy()
        worker_shape = (2, 128, cells)
        anon = [np.empty(worker_shape, np.complex64) for _ in range(4)]
        shms = [shared_memory.SharedMemory(create=True, size=nbytes // 4)
                for _ in range(4)]
        shm_views = [np.ndarray(worker_shape, np.complex64, buffer=s.buf)
                     for s in shms]

        def copy_into(dsts):
            for w in range(4):
                np.copyto(dsts[w], host[2 * w:2 * w + 2])

        c_anon = t_ms(lambda: copy_into(anon))
        c_shm = t_ms(lambda: copy_into(shm_views))
        gbs = lambda ms: nbytes / ms / 1e6  # noqa: E731
        print("%s  cube %.1f MB" % (label, nbytes / 1e6))
        print("   D2H pinned   %6.2f ms (%5.1f GB/s)   pageable %6.2f ms"
              % (d2h_pin, gbs(d2h_pin), d2h_page))
        print("   copy->anon   %6.2f ms (%5.1f GB/s)" % (c_anon, gbs(c_anon)))
        print("   copy->shm    %6.2f ms (%5.1f GB/s)" % (c_shm, gbs(c_shm)))
        for v in shm_views:
            del v
        for s in shms:
            s.close()
            s.unlink()


if __name__ == "__main__":
    main()
