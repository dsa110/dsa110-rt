"""Unit tests for ``dsart.services.dsart_rt._dada_dbmetric`` (M7.4 Phase 7).

The Phase 7 buffer-health Grafana wiring assumes:

1. ``_dada_dbmetric()`` returns canonical integer fields (``nbufs``,
   ``nfull``, ``nclear``, ``n_written``, ``n_read``, ``free_blocks``,
   ``free``, ``full``) regardless of whether the on-node ``dada_dbmetric``
   binary emits ``key=value`` tokens or positional CSV.
2. On binary-missing / timeout / parse failure it returns a diagnostic
   ``{"_error": str}`` dict so the influx pusher can skip it without
   losing the failure mode in the etcd snapshot.
3. ``_normalise_dbmetric()`` is idempotent and never overwrites existing
   keys (so a future binary that emits ``free_blocks`` directly doesn't
   collide with our derived value).
"""

from __future__ import annotations

import subprocess
from typing import Any
from unittest import mock

import pytest

from dsart.services.dsart_rt import (
    _DBMETRIC_CSV_FIELDS,
    _dada_dbmetric,
    _normalise_dbmetric,
)


# ---------------------------------------------------------------------------
# _normalise_dbmetric
# ---------------------------------------------------------------------------


class TestNormaliseDbmetric:
    def test_empty_dict_passes_through(self):
        assert _normalise_dbmetric({}) == {}

    def test_derives_free_blocks_from_nbufs_minus_nfull(self):
        out = _normalise_dbmetric({"nbufs": 70, "nfull": 12})
        assert out["free_blocks"] == 58
        assert out["free"] == 58
        assert out["full"] == 12

    def test_zero_free_blocks_is_explicit(self):
        out = _normalise_dbmetric({"nbufs": 20, "nfull": 20})
        assert out["free_blocks"] == 0
        assert out["free"] == 0

    def test_does_not_overwrite_existing_free_blocks(self):
        out = _normalise_dbmetric({
            "nbufs": 70, "nfull": 12, "free_blocks": 999,
        })
        assert out["free_blocks"] == 999

    def test_does_not_overwrite_existing_full(self):
        out = _normalise_dbmetric({"nbufs": 20, "nfull": 5, "full": 99})
        assert out["full"] == 99
        assert out["free"] == 15

    def test_missing_nbufs_leaves_aliases_unset(self):
        out = _normalise_dbmetric({"nfull": 12})
        assert "free_blocks" not in out
        assert "free" not in out

    def test_non_integer_values_do_not_crash(self):
        out = _normalise_dbmetric({"nbufs": "lol", "nfull": 12})
        assert "free_blocks" not in out

    def test_csv_field_order_is_stable(self):
        # Phase 7 Grafana panels query ``nbufs / nfull / nclear /
        # n_written / n_read`` by name — if the CSV-to-name map drifts,
        # the dashboard goes dark silently.
        assert _DBMETRIC_CSV_FIELDS == (
            "nbufs", "nfull", "nclear", "n_written", "n_read",
        )


# ---------------------------------------------------------------------------
# _dada_dbmetric — happy-path k=v parsing
# ---------------------------------------------------------------------------


def _mock_run(stdout: str = "", stderr: str = "", returncode: int = 0):
    """Build a mock subprocess.run that returns a CompletedProcess."""
    proc = subprocess.CompletedProcess(
        args=["dada_dbmetric", "-k", "test"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )
    return mock.MagicMock(return_value=proc)


class TestDadaDbmetricKvOutput:
    def test_kv_output_parses_canonical_fields(self):
        stdout = "nbufs=70 nfull=12 nclear=58 n_written=12516 n_read=12513\n"
        with mock.patch.object(subprocess, "run", _mock_run(stdout=stdout)):
            out = _dada_dbmetric("fada")
        assert out["nbufs"] == 70
        assert out["nfull"] == 12
        assert out["nclear"] == 58
        assert out["n_written"] == 12516
        assert out["n_read"] == 12513
        # Derived fields.
        assert out["free_blocks"] == 58
        assert out["free"] == 58
        assert out["full"] == 12

    def test_kv_output_comma_separated_also_parses(self):
        # Some PSRDADA builds emit comma-separated k=v.
        stdout = "nbufs=20,nfull=3,nclear=17,n_written=100,n_read=97\n"
        with mock.patch.object(subprocess, "run", _mock_run(stdout=stdout)):
            out = _dada_dbmetric("dada")
        assert out["nbufs"] == 20
        assert out["nfull"] == 3
        assert out["free_blocks"] == 17

    def test_partial_kv_output_normalises_what_it_can(self):
        # Some builds omit n_written / n_read.
        stdout = "nbufs=20 nfull=5\n"
        with mock.patch.object(subprocess, "run", _mock_run(stdout=stdout)):
            out = _dada_dbmetric("dada")
        assert out["nbufs"] == 20
        assert out["nfull"] == 5
        assert out["free_blocks"] == 15


# ---------------------------------------------------------------------------
# _dada_dbmetric — positional-CSV fallback (legacy PSRDADA builds)
# ---------------------------------------------------------------------------


class TestDadaDbmetricCsvOutput:
    def test_positional_csv_maps_to_canonical_names(self):
        # Legacy CSV: <nbufs>,<nfull>,<nclear>,<n_written>,<n_read>
        stdout = "70,12,58,12516,12513\n"
        with mock.patch.object(subprocess, "run", _mock_run(stdout=stdout)):
            out = _dada_dbmetric("fada")
        assert out["nbufs"] == 70
        assert out["nfull"] == 12
        assert out["nclear"] == 58
        assert out["n_written"] == 12516
        assert out["n_read"] == 12513
        assert out["free_blocks"] == 58
        # Raw CSV is preserved for forensic inspection.
        assert out["raw_csv"] == "70,12,58,12516,12513"

    def test_short_csv_only_parses_available_columns(self):
        stdout = "20,5\n"
        with mock.patch.object(subprocess, "run", _mock_run(stdout=stdout)):
            out = _dada_dbmetric("dada")
        assert out["nbufs"] == 20
        assert out["nfull"] == 5
        assert "nclear" not in out
        assert out["free_blocks"] == 15

    def test_unparseable_csv_emits_error_diag(self):
        stdout = "abc,def,ghi\n"
        with mock.patch.object(subprocess, "run", _mock_run(stdout=stdout)):
            out = _dada_dbmetric("dada")
        assert "_error" in out
        assert "unparseable" in out["_error"]


# ---------------------------------------------------------------------------
# _dada_dbmetric — failure paths
# ---------------------------------------------------------------------------


class TestDadaDbmetricStreamHandling:
    """The dsa110-cluster ``/usr/local/bin/dada_dbmetric`` build writes
    the metric line to **STDERR** (not STDOUT). This is the root cause
    of the M7.4 gate-soak observation that every ``buffers.*.metric``
    was ``{}`` on the live fleet — the parser was only reading stdout.
    Phase 7 fixes that by coalescing both streams.
    """

    def test_metric_line_on_stderr_only_is_parsed(self):
        """Real h23 production behaviour: stdout empty, CSV on stderr."""
        stderr = "4,0,3,9,9,4,0,3,3,3\n"
        with mock.patch.object(
            subprocess, "run",
            _mock_run(stdout="", stderr=stderr, returncode=0),
        ):
            out = _dada_dbmetric("dada")
        assert out["nbufs"] == 4
        assert out["nfull"] == 0
        assert out["nclear"] == 3
        assert out["n_written"] == 9
        assert out["n_read"] == 9
        assert out["free_blocks"] == 4

    def test_ipc_error_chain_filtered_then_surfaced_as_error(self):
        """When the ring isn't created yet, dada_dbmetric emits a 3-line
        ``ipc_alloc / ipcsync_get / ipcbuf_connect`` error chain on
        stderr. The parser must filter these so they don't get misread
        as CSV rows of nonsense, but should still surface the first
        line as ``_error`` so an operator can diagnose "buffer not
        created" vs other failures.
        """
        stderr = (
            "ipc_alloc: shmget (key=dadb, size=528, flag=1b6) "
            "No such file or directory\n"
            "ipcsync_get: ipc_alloc error\n"
            "ipcbuf_connect: ipcsync_get error\n"
        )
        with mock.patch.object(
            subprocess, "run",
            _mock_run(stdout="", stderr=stderr, returncode=1),
        ):
            out = _dada_dbmetric("dada")
        assert "_error" in out
        assert "ipc_alloc" in out["_error"]
        # No numeric fields ⇒ pusher will skip the point.
        assert not any(
            isinstance(v, (int, float)) for v in out.values()
        )

    def test_kv_on_stdout_takes_precedence(self):
        """Newer PSRDADA writes metric to stdout; the same parser path
        handles it.
        """
        with mock.patch.object(
            subprocess, "run",
            _mock_run(stdout="nbufs=20 nfull=3\n", stderr=""),
        ):
            out = _dada_dbmetric("dada")
        assert out["nbufs"] == 20
        assert out["nfull"] == 3
        assert out["free_blocks"] == 17


class TestDadaDbmetricFailures:
    def test_binary_not_found_returns_error_dict(self):
        # Both lookup paths raise FileNotFoundError ⇒ _error.
        def _raise(*a, **kw):
            raise FileNotFoundError(2, "No such file or directory")

        with mock.patch.object(subprocess, "run", side_effect=_raise):
            out = _dada_dbmetric("dada")
        assert "_error" in out
        assert "FileNotFoundError" in out["_error"]
        # No numeric fields ⇒ pusher will skip it.
        assert not any(
            isinstance(v, (int, float)) for v in out.values()
        )

    def test_timeout_returns_error_dict(self):
        def _raise(*a, **kw):
            raise subprocess.TimeoutExpired(cmd=a[0], timeout=2.0)

        with mock.patch.object(subprocess, "run", side_effect=_raise):
            out = _dada_dbmetric("dada")
        assert out == {"_error": "dada_dbmetric timeout after 2.0s"}

    def test_invalid_key_surfaces_diag_line(self):
        # Typo'd key: dada_dbmetric prints its own error line.
        stderr = "dada_dbmetric: could not parse key from xxxx\n"
        with mock.patch.object(
            subprocess, "run",
            _mock_run(stdout="", stderr=stderr, returncode=255),
        ):
            out = _dada_dbmetric("xxxx")
        assert out == {"_error": "dada_dbmetric: could not parse key from xxxx"}

    def test_first_path_falls_through_to_absolute_path(self):
        """If ``dada_dbmetric`` is not on PATH but
        ``/usr/local/bin/dada_dbmetric`` is, the parser must try the
        absolute path before declaring failure.
        """
        calls = []

        def _side_effect(args, *a, **kw):
            calls.append(args[0])
            if args[0] == "dada_dbmetric":
                raise FileNotFoundError(2, "No such file or directory")
            # Second call: the absolute path succeeds.
            return subprocess.CompletedProcess(
                args=args, returncode=0,
                stdout="nbufs=20 nfull=3\n",
                stderr="",
            )

        with mock.patch.object(subprocess, "run", side_effect=_side_effect):
            out = _dada_dbmetric("dada")
        assert calls == ["dada_dbmetric", "/usr/local/bin/dada_dbmetric"]
        assert out["nbufs"] == 20
        assert out["nfull"] == 3
        assert out["free_blocks"] == 17
