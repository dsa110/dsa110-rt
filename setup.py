"""setuptools build script for dsart C extensions and binaries.

* ``_recv_ring`` (M4a chunk 4) — POSIX-shm SPMC sparse receive ring.
  Source: ``src/dsart/transport/recv_ring.c``.
* ``_recv_epoll`` (M4a chunk 6) — C epoll receive loop that drains UDP
  via ``recvmmsg``, parses the 72-byte ProdFrame header, runs the
  per-(corr, dm) reorder window in C, and exposes atomic counters to
  Python. Replaces the Python ``_RxLoop`` + ``TransportRxProd.ingest_datagram``
  hot path at production rates.
  Source: ``src/dsart/transport/recv_epoll.c``.
* ``dsart_capture_manythread`` (M7.5) — SNAP-UDP -> PSRDADA capture
  binary, vendored from dsa110-xengine with dsart improvements
  (recvmmsg batch receive, deterministic arming, explicit SO_RCVBUF,
  POSIX-shm mon publisher). NOT a Python extension; a standalone
  binary built by a Makefile but invoked by the same `build_ext`
  step so `_sync_fleet.sh` rebuilds it across the fleet in one go.
  Source: ``src/dsart/capture/Makefile``.

The .so extensions and the capture binary all land alongside the
package as ``src/dsart/<subpkg>/<artifact>``. The Python wrappers
glob for the .so names so both editable (`pip install -e .`) and
in-place (`python setup.py build_ext --inplace`) builds work.

Why a setup.py alongside pyproject.toml?
    pyproject.toml [project] declares metadata + pure-Python packages.
    setuptools ext_modules cannot be declared in pyproject.toml's
    [tool.setuptools] section as of setuptools<64. D-item D3 in
    M4a_PLAN_FIXES.md locks this decision.
"""

import os
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

_recv_ring = Extension(
    name="dsart.transport._recv_ring",
    sources=["src/dsart/transport/recv_ring.c"],
    extra_compile_args=[
        "-O2",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-pthread",
    ],
    extra_link_args=[
        "-lrt",
        "-pthread",
    ],
    language="c",
)

# NOTE: _recv_epoll links a private copy of recv_ring.c so that the M7.2
# Phase B ring-publish path can call rx_ring_write_slot without a
# cross-extension ctypes hop on the hot path. _recv_ring.so still
# ships its own copy for the Python-driven ring create/attach/read
# path; the two .so files cooperate via the POSIX shm name (each has
# its own static copy of the ring functions but they operate on the
# same kernel-shared mmap). See recv_ring.h.
_recv_epoll = Extension(
    name="dsart.transport._recv_epoll",
    sources=[
        "src/dsart/transport/recv_epoll.c",
        "src/dsart/transport/recv_ring.c",
    ],
    include_dirs=["src/dsart/transport"],
    extra_compile_args=[
        "-O3",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-pthread",
    ],
    extra_link_args=[
        "-lrt",
        "-pthread",
    ],
    language="c",
)

_CAPTURE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src", "dsart", "capture",
)


class build_ext_with_capture_binary(build_ext):
    """Extend `build_ext` so it also builds the capture binary.

    After the normal `.so` extensions land, we shell out to `make`
    inside `src/dsart/capture/` to produce `dsart_capture_manythread`.

    Skipping the binary build is non-fatal: the C extensions still
    install, and the orchestrator's `cap_a_real` / `cap_b_real`
    routines simply fail at spawn time if the binary is missing
    (the operator sees a clear `cmd not found` log line). This is the
    right behaviour for dev environments without libpsrdada
    installed -- we don't want the entire `pip install -e .` step
    to fail because libpsrdada isn't on the developer's box.

    Suppress with `DSART_SKIP_CAPTURE_BINARY=1` for explicit
    "don't even try" behaviour.
    """

    def run(self) -> None:
        super().run()
        if os.environ.get("DSART_SKIP_CAPTURE_BINARY") == "1":
            sys.stderr.write(
                "[setup.py] DSART_SKIP_CAPTURE_BINARY=1; "
                "not building dsart_capture_manythread\n"
            )
            return
        if not shutil.which("make"):
            sys.stderr.write(
                "[setup.py] `make` not found on PATH; "
                "skipping dsart_capture_manythread build\n"
            )
            return
        sys.stderr.write(
            "[setup.py] building dsart_capture_manythread in %s\n"
            % _CAPTURE_DIR
        )
        try:
            subprocess.check_call(
                ["make", "-C", _CAPTURE_DIR, "all"],
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as exc:
            # libpsrdada missing on a dev box should not be fatal;
            # only the fleet nodes (n01..n22) have libpsrdada and
            # need the binary for the M7.5+ on-sky stages.
            sys.stderr.write(
                "[setup.py] dsart_capture_manythread build FAILED "
                "(rc=%d). This is expected on dev boxes without "
                "libpsrdada; on fleet nodes (n01..n22) this means "
                "the build is broken and cap_a_real/cap_b_real will "
                "fail to spawn.\n" % exc.returncode
            )
            return
        sys.stderr.write(
            "[setup.py] dsart_capture_manythread build OK\n"
        )


setup(
    ext_modules=[_recv_ring, _recv_epoll],
    cmdclass={"build_ext": build_ext_with_capture_binary},
)
