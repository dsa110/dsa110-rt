"""Smoke tests for the dsa_monitor SEFD routes.

These tests boot the Flask app via ``app.test_client()`` against a
tmp_path scanner sandbox (the SEFD scanner is mocked: we just lay
down a state.json + a results/ tree) and verify that:

* ``GET /sefds`` renders the native summary template with one row per
  observation date in the lookback window,
* ``GET /sefds/source/<name>`` renders the per-source detail page,
* ``GET /sefds/day/<date>`` renders the per-day detail page,
* ``GET /sefds/results/<path>`` serves a real PNG (and 404s on
  path-traversal attempts and missing files),
* ``GET /api/sefd_status`` returns the JSON freshness payload.

We monkeypatch the module-level :data:`SEFD_STATE_FILE` /
:data:`SEFD_RESULTS_DIR` and rebuild the :data:`sefd_view` singleton
so the routes pick up the sandbox.  The RFI poller side of the app
is heavy at import-time (it spawns threads + tries to contact every
corr node); the suite-wide patch in ``conftest.py`` keeps that side
stubbed.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest


HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, ".."))
DSA_MONITOR_DIR = os.path.normpath(os.path.join(
    REPO_ROOT, "tools", "dashboard", "dsa_monitor",
))
DSART_SRC = os.path.join(REPO_ROOT, "src")
for p in (DSART_SRC, DSA_MONITOR_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Heavyweight import: stub the RFI poller before importing app, so we
# don't open 16 HTTP connections at module-import time.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app_module():
    with mock.patch("rfi_store.RFIPoller.start", return_value=None):
        # First import.  The poller's start() is patched so the
        # background threads never spin up.
        import app  # noqa: F401
        yield app


# ---------------------------------------------------------------------------
# Per-test scanner sandbox
# ---------------------------------------------------------------------------


def _iso_n(n):
    return (
        datetime.now(timezone.utc) - timedelta(days=n)
    ).date().isoformat()


@pytest.fixture()
def scanner_sandbox(tmp_path, app_module):
    """Plant a tmp_path scanner sandbox and rebind the app's
    :data:`sefd_view` to look at it."""
    state_file = tmp_path / "state.json"
    results = tmp_path / "results"
    results.mkdir()

    today = _iso_n(0)
    yest = _iso_n(1)
    state = {
        f"{today}_0318+164": {
            "status": "complete",
            "updated": datetime.now(timezone.utc).isoformat(),
            "date": today, "source": "0318+164",
            "metrics": {
                "median_amplitude": 0.04, "median_noise": 0.011,
                "median_coherence": 0.92, "median_hi_peak_db": 1.6,
            },
            "full_metrics": {"median_sefd": 6000.0, "n_baselines": 4163},
        },
        f"{yest}_2253+161": {
            "status": "light_done",
            "updated": datetime.now(timezone.utc).isoformat(),
            "date": yest, "source": "2253+161",
            "metrics": {"median_amplitude": 0.08},
        },
    }
    state_file.write_text(json.dumps(state))

    (results / "0318+164" / today).mkdir(parents=True)
    (results / "0318+164" / today / "amp_vs_baseline.png").write_bytes(
        # Valid 1x1 PNG so send_file's mimetype + Content-Length are
        # well-defined.  (The smoke test only checks the response
        # status + content-type; we don't decode the image.)
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010806"
            "0000001f15c4890000000d49444154789c63f8cf00000003"
            "00010000c6a05a8a0000000049454e44ae426082"
        ),
    )

    from sefd_view import SefdView
    new_view = SefdView(
        state_file=str(state_file), results_dir=str(results),
    )
    monkey = mock.patch.object(app_module, "sefd_view", new_view)
    monkey.start()
    monkey2 = mock.patch.object(app_module, "SEFD_STATE_FILE", str(state_file))
    monkey2.start()
    monkey3 = mock.patch.object(
        app_module, "SEFD_RESULTS_DIR", str(results),
    )
    monkey3.start()

    client = app_module.app.test_client()
    yield client, str(state_file), str(results), today, yest

    monkey.stop()
    monkey2.stop()
    monkey3.stop()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


class TestSefdSummary:
    def test_summary_renders(self, scanner_sandbox):
        client, _, _, today, yest = scanner_sandbox
        r = client.get("/sefds")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        assert today in body
        assert yest in body
        # Per-source headers link to source pages.
        assert "/sefds/source/0318+164" in body
        assert "/sefds/source/2253+161" in body
        # Per-day links go to day pages.
        assert f"/sefds/day/{today}" in body
        # No iframe leftovers from the old implementation.
        assert "<iframe" not in body
        assert "sefd_url" not in body

    def test_summary_lookback_clamped(self, scanner_sandbox):
        client, *_ = scanner_sandbox
        # Bogus value falls back to the default; route never 500s.
        r = client.get("/sefds?days=abc")
        assert r.status_code == 200

    def test_summary_shows_scanner_freshness(self, scanner_sandbox):
        client, *_ = scanner_sandbox
        r = client.get("/sefds")
        body = r.get_data(as_text=True)
        assert "scanner" in body.lower()


class TestSefdSource:
    def test_source_renders(self, scanner_sandbox):
        client, *_, today, _ = scanner_sandbox
        r = client.get("/sefds/source/0318+164")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        assert "0318+164" in body
        assert today in body

    def test_unknown_source_404(self, scanner_sandbox):
        client, *_ = scanner_sandbox
        r = client.get("/sefds/source/not_a_source")
        assert r.status_code == 404

    def test_empty_window_renders_empty_banner(self, scanner_sandbox):
        client, *_ = scanner_sandbox
        # 0521+166 has no entries in the sandbox.
        r = client.get("/sefds/source/0521+166")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        assert "No processed observations" in body


class TestSefdDay:
    def test_day_renders(self, scanner_sandbox):
        client, *_, today, _ = scanner_sandbox
        r = client.get(f"/sefds/day/{today}")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        assert today in body
        assert "0318+164" in body
        # Per-source links present.
        assert "/sefds/source/0318+164" in body

    def test_day_missing_renders_empty_banner(self, scanner_sandbox):
        client, *_ = scanner_sandbox
        r = client.get("/sefds/day/2099-01-01")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        assert "No observations processed" in body


class TestSefdResults:
    def test_real_png_served(self, scanner_sandbox):
        client, _, _, today, _ = scanner_sandbox
        r = client.get(f"/sefds/results/0318+164/{today}/amp_vs_baseline.png")
        assert r.status_code == 200
        assert r.mimetype == "image/png"
        assert len(r.data) > 0

    def test_traversal_denied(self, scanner_sandbox):
        client, *_ = scanner_sandbox
        # `..` segments resolved BEFORE the route is dispatched by
        # Werkzeug's URL normaliser, so we hit `resolve_plot_path`
        # with something starting with `/` and it must say no.
        for path in [
            "/sefds/results/..%2F..%2Fetc%2Fpasswd",
            "/sefds/results/../state.json",
        ]:
            r = client.get(path)
            # Werkzeug returns 404 for both the parsed-out-of-tree
            # case and the explicit missing-file case.
            assert r.status_code == 404

    def test_missing_file_404(self, scanner_sandbox):
        client, *_ = scanner_sandbox
        r = client.get("/sefds/results/0318+164/2099-01-01/nope.png")
        assert r.status_code == 404


class TestSefdApi:
    def test_status_json(self, scanner_sandbox):
        client, *_ = scanner_sandbox
        r = client.get("/api/sefd_status")
        assert r.status_code == 200
        body = r.get_json()
        assert body["state_path"]
        assert body["state_mtime_unix"] is not None
        assert body["scanner_alive"] is True
        assert body["state_error"] is None
