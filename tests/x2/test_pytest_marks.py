"""tests/conftest.py registers x2_gpu / x2_bench / x2_ckpt / slow and makes
marks strict: a mistyped GPU mark is a collection error, not a test that
``-m x2_gpu`` silently skips."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

CONF = Path(__file__).resolve().parents[1] / "conftest.py"


def _run(tmp_path, body: str, *args):
    shutil.copy(CONF, tmp_path / "conftest.py")
    (tmp_path / "test_m.py").write_text("import pytest\n\n" + body)
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           *args, str(tmp_path / "test_m.py")], cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=120)


def test_registered_marks_select_and_do_not_warn(tmp_path):
    body = ("@pytest.mark.x2_gpu\ndef test_a():\n    pass\n\n"
            "@pytest.mark.x2_bench\n@pytest.mark.x2_ckpt\n@pytest.mark.slow\n"
            "def test_b():\n    pass\n")
    p = _run(tmp_path, body, "-m", "x2_gpu")
    assert p.returncode == 0 and "1 passed, 1 deselected" in p.stdout, p.stdout + p.stderr
    assert "PytestUnknownMarkWarning" not in p.stdout + p.stderr


def test_mistyped_mark_is_an_error(tmp_path):
    p = _run(tmp_path, "@pytest.mark.x2_gpus\ndef test_a():\n    pass\n", "-m", "not x2_gpu")
    assert p.returncode != 0 and "x2_gpus" in p.stdout + p.stderr, p.stdout + p.stderr
