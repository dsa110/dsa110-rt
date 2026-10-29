"""Repository-wide pytest configuration.

Registers the custom marks and turns on strict markers, so a mistyped mark
(e.g. ``x2_gpus``) is a collection ERROR instead of a mark that ``-m
x2_gpu`` / ``-m 'not x2_gpu'`` silently ignores. Lives here (not in
pyproject.toml) because the node test harness overlays ``tests/`` onto
the node's checkout but not ``pyproject.toml``.
"""

MARKS = {
    "x2_gpu": "needs a GPU; CP/G0 checkpoint runs only (tools/x2/node_cpu_tests.sh -g)",
    "x2_bench": "timing benchmark; checkpoint runs only",
    "x2_ckpt": "checkpoint acceptance test (CP runner)",
    "slow": "long-running test",
}


def pytest_configure(config):
    for name, doc in MARKS.items():
        config.addinivalue_line("markers", f"{name}: {doc}")
    # strict markers, version-independent: an unknown mark's warning is an error
    config.addinivalue_line("filterwarnings", "error::pytest.PytestUnknownMarkWarning")
