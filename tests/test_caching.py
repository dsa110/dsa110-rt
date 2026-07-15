"""Browser-caching header tests for the dsa_monitor dashboard.

Five resource classes, five policies:

* event plot PNGs           — max-age=604800 + ETag, 304 on match
* /static JS                — 30-day max-age; templates use ?v=<mtime>
* HTML pages                — Cache-Control: no-cache + ETag, 304 on match
* JSON APIs (+ POSTs)       — Cache-Control: no-store
* live RFI plots (/plot/*)  — never long-cached (no-store)

Uses the Flask test client with the same import guard as
``test_annotations.py`` (skips if the app needs live resources).
"""

from __future__ import annotations

import os
import sys
import time
from unittest import mock

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, ".."))
DSA_MONITOR_DIR = os.path.normpath(
    os.path.join(REPO_ROOT, "tools", "dashboard", "dsa_monitor")
)
DSART_SRC = os.path.join(REPO_ROOT, "src")
for _p in (DSART_SRC, DSA_MONITOR_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cands_panel_funcs as cpf  # noqa: E402

# Valid 1x1 PNG (same bytes the SEFD route tests use).
_PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806"
    "0000001f15c4890000000d49444154789c63f8cf00000003"
    "00010000c6a05a8a0000000049454e44ae426082"
)


@pytest.fixture(scope="module")
def app_module(tmp_path_factory):
    dbfile = tmp_path_factory.mktemp("annot_cache") / "annot.db"
    os.environ["DSA_MONITOR_ANNOT_DB"] = str(dbfile)
    try:
        with mock.patch("rfi_store.RFIPoller.start", return_value=None):
            import app  # noqa: F401
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"app import needs live resources: {exc!r}")
    app.app.config.update(TESTING=True)
    return app


@pytest.fixture()
def client(app_module):
    return app_module.app.test_client()


def _ev(name, *, c3_action="KEEP", mtime_unix=None):
    return cpf.EventSummary(
        name=name, mtime_unix=mtime_unix or time.time(), mjd_peak=None,
        trigger_class=None, n_events=None, snr_max=None, dm_median=None,
        l_median=None, m_median=None, n_cubes=0, n_plots=0,
        c3_action=c3_action,
    )


# ---------------------------------------------------------------------------
# Event plot PNGs: 7-day cache + ETag + conditional 304
# ---------------------------------------------------------------------------


def test_event_plot_long_cache_and_304(client, app_module, tmp_path):
    png = tmp_path / "dm_time_x.png"
    png.write_bytes(_PNG_1PX)
    with mock.patch.object(app_module.cands_browser, "plot_path",
                           return_value=png):
        r = client.get("/bursts/260714aaaa/plot/dm_time_x.png")
        assert r.status_code == 200
        assert "max-age=604800" in r.headers.get("Cache-Control", "")
        # NOT immutable: plotter.py can rewrite the same path in place.
        assert "immutable" not in r.headers.get("Cache-Control", "")
        etag = r.headers.get("ETag")
        assert etag
        r2 = client.get(
            "/bursts/260714aaaa/plot/dm_time_x.png",
            headers={"If-None-Match": etag},
        )
        assert r2.status_code == 304
        assert r2.get_data() == b""


def test_event_fil_plot_long_cache(client, app_module, tmp_path):
    png = tmp_path / "fil_x.png"
    png.write_bytes(_PNG_1PX)
    with mock.patch.object(app_module.cands_browser, "fil_plot_path",
                           return_value=png):
        r = client.get("/bursts/260714aaaa/filplot/fil_x.png")
        assert r.status_code == 200
        assert "max-age=604800" in r.headers.get("Cache-Control", "")
        assert r.headers.get("ETag")


# ---------------------------------------------------------------------------
# Static JS: 30-day max-age; templates carry ?v=<mtime>
# ---------------------------------------------------------------------------


def test_static_js_long_cache(client):
    r = client.get("/static/annotations.js")
    assert r.status_code == 200
    assert "max-age=2592000" in r.headers.get("Cache-Control", "")


def test_templates_version_stamp_static(client, app_module):
    with mock.patch.object(app_module.cands_browser, "list_events",
                           return_value=[]):
        html = client.get("/bursts").get_data(as_text=True)
    assert "/static/typeahead.js?v=" in html


# ---------------------------------------------------------------------------
# HTML pages: no-cache + ETag revalidation (304, empty body)
# ---------------------------------------------------------------------------


def test_bursts_html_etag_304(client, app_module):
    events = [_ev("260714aaaa")]
    with mock.patch.object(app_module.cands_browser, "list_events",
                           return_value=events):
        r = client.get("/bursts")
        assert r.status_code == 200
        assert r.headers.get("Cache-Control") == "no-cache"
        etag = r.headers.get("ETag")
        assert etag
        r2 = client.get("/bursts", headers={"If-None-Match": etag})
        assert r2.status_code == 304
        assert r2.get_data() == b""
        # Changed content -> full 200 again (stale ETag must not 304).
        events2 = [_ev("260714bbbb")]
    with mock.patch.object(app_module.cands_browser, "list_events",
                           return_value=events2):
        r3 = client.get("/bursts", headers={"If-None-Match": etag})
        assert r3.status_code == 200


def test_burst_event_html_etag_304(client, app_module):
    from pathlib import Path
    fake = cpf.EventDetail(
        name="260714aaaa", archive_dir=Path("/tmp/260714aaaa"),
        metadata={"x": 1}, plots=(), cubes=(),
        has_c2_csv=False, has_c1_csv=False,
    )
    with mock.patch.object(app_module.cands_browser, "event_detail",
                           return_value=fake), \
         mock.patch.object(app_module.cands_browser, "list_events",
                           return_value=[]):
        r = client.get("/bursts/260714aaaa")
        assert r.status_code == 200
        assert r.headers.get("Cache-Control") == "no-cache"
        etag = r.headers.get("ETag")
        assert etag
        r2 = client.get("/bursts/260714aaaa",
                        headers={"If-None-Match": etag})
        assert r2.status_code == 304


# ---------------------------------------------------------------------------
# JSON APIs: no-store (GET + POST bodies)
# ---------------------------------------------------------------------------


def test_api_annotations_no_store(client):
    r = client.get("/api/annotations")
    assert r.status_code == 200
    assert r.headers.get("Cache-Control") == "no-store"
    r = client.get("/api/annotations/vocab")
    assert r.headers.get("Cache-Control") == "no-store"


def test_api_status_no_store(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    assert r.headers.get("Cache-Control") == "no-store"


def test_post_endpoints_no_store(client):
    r = client.post("/annotations/user", json={"name": "cachetester"})
    assert r.status_code == 200
    assert r.headers.get("Cache-Control") == "no-store"
    r = client.post(
        "/annotations/classify",
        json={"event": "EVC", "user": "cachetester", "label": "FRB"},
    )
    assert r.status_code == 200
    assert r.headers.get("Cache-Control") == "no-store"


# ---------------------------------------------------------------------------
# Live RFI plots: never long-cached
# ---------------------------------------------------------------------------


def test_live_rfi_plot_not_cached(client):
    # Renders from the (empty, poller-stubbed) in-memory store; if the
    # renderer can't handle an empty snapshot in this env, skip rather
    # than fake it — the header contract is what matters here.
    try:
        r = client.get("/plot/bandpass.png")
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"live plot render needs data: {exc!r}")
    if r.status_code != 200:  # pragma: no cover - env-dependent
        pytest.skip(f"live plot render returned {r.status_code}")
    cc = r.headers.get("Cache-Control", "")
    assert "no-store" in cc
    assert "2592000" not in cc and "604800" not in cc
