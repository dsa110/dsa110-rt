"""setuptools build script for dsart (M4a chunk 4: _recv_ring C extension).

Why a setup.py alongside pyproject.toml?
    pyproject.toml [project] declares metadata + pure-Python packages.
    setuptools ext_modules cannot be declared in pyproject.toml's
    [tool.setuptools] section as of setuptools<64. D-item D3 in
    M4a_PLAN_FIXES.md locks this decision.

The _recv_ring extension is compiled from
    src/dsart/transport/recv_ring.c
and installed as
    dsart.transport._recv_ring

Build commands:
    pip install -e .                        # editable; rebuilds on pip install
    python setup.py build_ext --inplace     # inplace; .so next to recv_ring.c

The .so name follows PEP 3149 (e.g.
    _recv_ring.cpython-311-x86_64-linux-gnu.so);
recv_ring.py globs for _recv_ring*.so so both inplace and installed paths work.
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
        "-lrt",          # shm_open / shm_unlink on Linux
        "-pthread",
    ],
    language="c",
)

setup(
    ext_modules=[_recv_ring],
)
