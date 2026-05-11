"""setuptools build script for dsart C extensions.

* ``_recv_ring`` (M4a chunk 4) — POSIX-shm SPMC sparse receive ring.
  Source: ``src/dsart/transport/recv_ring.c``.
* ``_recv_epoll`` (M4a chunk 6) — C epoll receive loop that drains UDP
  via ``recvmmsg``, parses the 72-byte ProdFrame header, runs the
  per-(corr, dm) reorder window in C, and exposes atomic counters to
  Python. Replaces the Python ``_RxLoop`` + ``TransportRxProd.ingest_datagram``
  hot path at production rates.
  Source: ``src/dsart/transport/recv_epoll.c``.

Both extensions land alongside the package as
``_recv_ring.cpython-*.so`` / ``_recv_epoll.cpython-*.so``. The Python
wrappers glob for these names so both editable (`pip install -e .`)
and in-place (`python setup.py build_ext --inplace`) builds work.

Why a setup.py alongside pyproject.toml?
    pyproject.toml [project] declares metadata + pure-Python packages.
    setuptools ext_modules cannot be declared in pyproject.toml's
    [tool.setuptools] section as of setuptools<64. D-item D3 in
    M4a_PLAN_FIXES.md locks this decision.
"""

from setuptools import Extension, setup

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

_recv_epoll = Extension(
    name="dsart.transport._recv_epoll",
    sources=["src/dsart/transport/recv_epoll.c"],
    extra_compile_args=[
        "-O3",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-pthread",
    ],
    extra_link_args=[
        "-pthread",
    ],
    language="c",
)

setup(
    ext_modules=[_recv_ring, _recv_epoll],
)
