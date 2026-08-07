"""C3 KEEP -> Slack notification (best-effort, never affects C3 behaviour).

When C3 KEEPs a real FRB-candidate event (passed the cube-morphology veto),
:class:`SlackNotifier` posts a short summary to the ops Slack channel, in
addition to the existing dashboard/webserver surface. Uses the Slack Web API
directly via ``requests`` (no ``slack_sdk`` dependency — not installed in the
h23 service env).

Every failure mode (missing token, network error, malformed API response,
bad config) is caught, logged at WARNING, and reported back as a small
status dict — it must never raise into the C3 scan loop or affect the KEEP
decision/audit trail in any way.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

LOG = logging.getLogger("dsart.services.slack_notify")

_SLACK_API = "https://slack.com/api"

#: unix epoch expressed as MJD (same constant used elsewhere in c3.py /
#: cal_hdf5_archive.py for mjd -> unix conversions).
_MJD_UNIX_EPOCH = 40587.0

#: how long to poll files.info for the file-share message ts before
#: giving up (the card is still fully posted at that point — we just
#: won't have a ``ts`` to thread the followup onto, so it lands as a
#: fresh top-level message instead; see post_keep_followup).
_FILE_SHARE_POLL_TIMEOUT_S = 15.0
_FILE_SHARE_POLL_INTERVAL_S = 1.0


@dataclass(frozen=True)
class SlackNotifyConfig:
    enabled: bool = False
    channel: str = ""
    token_env: str = "SLACK_TOKEN_DSA"
    #: Path to a file holding the bot token as its only contents (the
    #: legacy T3 "candplotter" bot's token lives at
    #: ``~/.config/slack_api``, the same file
    #: ``dsa110-T3/dsaT3/filplot_funcs.py`` reads — see ``_token()``).
    #: Preferred over ``token_env`` when both are set; empty string
    #: disables the file path entirely (env-only, the pre-v3.6 default).
    token_file: str = ""
    timeout_s: float = 10.0
    upload_plot: bool = True
    card: bool = True
    #: dsa_monitor dashboard base URL. Every KEEP message links to
    #: ``<dashboard_base_url>/bursts/<name>`` (the burst-detail PAGE
    #: route — verified read-only against
    #: ``tools/dashboard/dsa_monitor/app.py:823`` ``@app.route("/bursts/<name>")
    #: def burst_event`` — NOT the ``/bursts/<name>/plot/<plot_name>``
    #: plot-asset route at app.py:907). Default assumes localhost:5778,
    #: i.e. the reader has the team's normal SSH port-forward to h23 open
    #: — this is the intended/expected access pattern, not a bug.
    dashboard_base_url: str = "http://localhost:5778"
    #: sqlite db mapping event -> (channel, ts) of its posted card message,
    #: so the annotation relay (slack_annotation_relay.py) can thread
    #: classification updates onto it. Best-effort: a write failure here
    #: logs a warning and never fails the post. ``~`` is expanded at use
    #: time (see ``SlackNotifier._persist_post_map``).
    post_map_db: str = "~/.dsa_monitor/slack_candidate_posts.db"
    #: Bounded wait for the C2 plot worker's four cube-panel PNGs before
    #: rendering/posting the card. C3's KEEP decision routinely lands
    #: ~1 min BEFORE Level2/plots/ is populated (first live post,
    #: 260723pllz 2026-07-23 23:04 UT, went out all-placeholders: card
    #: posted 23:04:19, plots arrived 23:05:14). Poll every plot_poll_s
    #: until all four exist and have sat unmodified for >=2 s (never read
    #: a half-written PNG); on timeout post with whatever exists (a late
    #: card beats no card). 0 disables the wait. The staged voltages sit
    #: durably on corr NVMe, so delaying _do_keep this long is safe.
    plot_wait_s: float = 300.0
    plot_poll_s: float = 5.0
    #: cube count that makes an event "complete" for the readiness gate
    #: (mirrors c3.expected_cube_count / the dashboard's "cubes 8/8").
    expected_cubes: int = 8

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "SlackNotifyConfig":
        d = d or {}
        return cls(
            enabled=bool(d.get("enabled", False)),
            channel=str(d.get("channel", "")),
            token_env=str(d.get("token_env", "SLACK_TOKEN_DSA")),
            token_file=str(d.get("token_file", "")),
            timeout_s=float(d.get("timeout_s", 10.0)),
            upload_plot=bool(d.get("upload_plot", True)),
            card=bool(d.get("card", True)),
            dashboard_base_url=str(
                d.get("dashboard_base_url", "http://localhost:5778")),
            post_map_db=str(
                d.get("post_map_db", "~/.dsa_monitor/slack_candidate_posts.db")),
            plot_wait_s=float(d.get("plot_wait_s", 300.0)),
            plot_poll_s=float(d.get("plot_poll_s", 5.0)),
            expected_cubes=int(d.get("expected_cubes", 8)),
        )


def _mjd_to_utc_str(mjd: Any) -> Optional[str]:
    try:
        from datetime import datetime, timezone
        unix_s = (float(mjd) - _MJD_UNIX_EPOCH) * 86400.0
        return datetime.fromtimestamp(
            unix_s, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (TypeError, ValueError, OverflowError, OSError):
        return None


class SlackNotifier:
    """Best-effort Slack poster for C3 KEEP events.

    Two-stage flow (v3.7+): the candidate card (cube-only plots, no
    voltages) is posted the moment C3 decides KEEP — before voltage
    collection / filterbank even start — so ops sees a candidate within
    seconds rather than after the whole bbproc pipeline completes.
    :meth:`post_keep_followup` then adds the bbproc inspection plot (or a
    brief failure note) into that message's thread once the filterbank
    stage finishes, which can be minutes later.

    Both calls are independently best-effort: neither raises, and a
    failure in one has no bearing on the other (e.g. a failed card post
    still lets the followup land, just as a fresh top-level message
    instead of a threaded reply — see :meth:`post_keep_followup`).
    """

    def __init__(self, config: SlackNotifyConfig) -> None:
        self._cfg = config
        self._warned_no_token = False

    # ----- public API ----------------------------------------------------

    def post_keep_card(
        self,
        name: str,
        c2row: Mapping[str, Any],
        ev_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Post the candidate card IMMEDIATELY at KEEP time.

        ``keep_report`` (voltage-fragment counts) is not known yet at
        this stage — the card is rendered with ``keep_report=None``
        (``render_card``/the header formatters treat that as "n/a",
        never raise). Returns a status dict; ``status["ts"]`` (when
        present) is the message timestamp to thread
        :meth:`post_keep_followup` onto. Always returns a status dict,
        never raises."""
        if not self._cfg.enabled:
            return {"ok": False, "error": "disabled"}
        try:
            return self._post_keep_card(name, c2row, ev_dir)
        except Exception as exc:  # noqa: BLE001 — must never break C3
            LOG.exception("slack_notify %s: unexpected failure (card)", name)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def post_keep_followup(
        self,
        name: str,
        ev_dir: Optional[Path],
        fb_report: Mapping[str, Any],
        keep_report: Mapping[str, Any],
        thread_ts: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Post the bbproc follow-up once the filterbank stage finishes.

        Uploads BOTH bbproc variants (``filterbank/<name>.png`` unflagged
        and ``filterbank/<name>_rfi.png`` SK RFI-flagged) into the thread
        named by ``thread_ts``, each with its own label. A missing
        variant is skipped silently. If ``thread_ts`` is ``None`` (the
        card post failed or was skipped), each upload lands as a fresh
        top-level message instead — so the plots are never lost just
        because the card didn't post. If BOTH variants are missing:
        posts a brief one-line thread note only when ``fb_report``
        indicates an actual failure (``ok`` is ``False``); silently
        skips when the filterbank stage was disabled or simply produced
        no plot for an ok run. Always returns a status dict, never
        raises."""
        if not self._cfg.enabled:
            return {"ok": False, "error": "disabled"}
        try:
            return self._post_keep_followup(
                name, ev_dir, fb_report, keep_report, thread_ts)
        except Exception as exc:  # noqa: BLE001 — must never break C3
            LOG.exception(
                "slack_notify %s: unexpected failure (followup)", name)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def post_text(
        self,
        text: str,
        *,
        thread_ts: Optional[str] = None,
        reply_broadcast: bool = False,
    ) -> Dict[str, Any]:
        """Post a plain text message to the configured channel.

        General-purpose entry point for non-C3 producers (e.g. the
        injection sentinel). ``thread_ts`` threads the message onto an
        existing one. Returns ``{"ok": True, "ts": <message ts>}`` on
        success, ``{"ok": False, "error": ...}`` otherwise. Always
        returns a status dict, never raises.
        """
        if not self._cfg.enabled:
            return {"ok": False, "error": "disabled"}
        try:
            token = self._token()
            if not token:
                return {"ok": False, "error": "no_token"}
            payload: Dict[str, Any] = {
                "channel": self._cfg.channel,
                "text": text,
            }
            if thread_ts:
                payload["thread_ts"] = thread_ts
                if reply_broadcast:
                    payload["reply_broadcast"] = True
            resp = self._api_post("chat.postMessage", token, payload)
            if not resp or not resp.get("ok"):
                return {
                    "ok": False,
                    "error": (resp or {}).get("error", "request_failed"),
                }
            return {"ok": True, "ts": resp.get("ts")}
        except Exception as exc:  # noqa: BLE001 — never break the caller
            LOG.exception("slack_notify: unexpected failure (post_text)")
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def post_file(
        self,
        path: Path,
        *,
        title: Optional[str] = None,
        initial_comment: Optional[str] = None,
        thread_ts: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload a file (e.g. a summary plot PNG) to the configured
        channel via the external-upload flow, optionally into a thread
        and with a leading comment. Returns ``{"ok": True, "ts": <share
        message ts or None>}`` on success. Always returns a status
        dict, never raises."""
        if not self._cfg.enabled:
            return {"ok": False, "error": "disabled"}
        try:
            token = self._token()
            if not token:
                return {"ok": False, "error": "no_token"}
            p = Path(path)
            if not p.is_file():
                return {"ok": False, "error": f"not_a_file: {p}"}
            ok, doc = self._upload_file(
                token, p,
                title=title,
                channel=self._cfg.channel,
                thread_ts=thread_ts,
                initial_comment=initial_comment,
            )
            if not ok or not doc:
                return {"ok": False, "error": "upload_failed"}
            ts = self._poll_file_share_ts(
                token, doc["file_id"], self._cfg.channel,
            )
            return {"ok": True, "ts": ts, "file_id": doc["file_id"]}
        except Exception as exc:  # noqa: BLE001 — never break the caller
            LOG.exception("slack_notify: unexpected failure (post_file)")
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # ----- internals -------------------------------------------------------

    def _token(self) -> Optional[str]:
        """Resolve the bot token: ``token_file`` (if configured and
        readable) takes priority over ``token_env`` — the legacy T3
        "candplotter" bot's token lives in a file
        (``~/.config/slack_api``, same file ``filplot_funcs.py`` reads),
        not an env var, so this is the primary path in production; the
        env var remains a fallback for envs where the file isn't
        provisioned (e.g. test/dev). The token's contents are NEVER
        logged, in either path — only the fact that a source was
        missing/unreadable."""
        if self._cfg.token_file:
            try:
                tok = Path(self._cfg.token_file).read_text().strip()
            except OSError as exc:
                LOG.warning(
                    "slack_notify: token_file %s unreadable (%s); "
                    "falling back to env var %s",
                    self._cfg.token_file, exc, self._cfg.token_env)
                tok = ""
            if tok:
                return tok

        tok = os.environ.get(self._cfg.token_env)
        if not tok:
            if not self._warned_no_token:
                LOG.warning(
                    "slack_notify: neither token_file (%s) nor env var %s "
                    "yielded a token; C3 KEEP notifications disabled "
                    "(best-effort, not fatal)",
                    self._cfg.token_file or "(unset)", self._cfg.token_env)
                self._warned_no_token = True
            return None
        return tok

    _CUBE_PANELS = ("dm_time", "image_peak", "kernel_snrs", "lightcurve")

    def _wait_for_event_ready(self, name: str, ev_dir: Path) -> bool:
        """Block (bounded by ``plot_wait_s``) until the event is complete
        the same way the dashboard's burst page counts it: all four
        cube-panel PNGs in ``<ev_dir>/Level2/plots/`` (each unmodified
        for >=2 s — never read a PNG the plot worker is still writing)
        AND ``expected_cubes`` npz cubes in ``<ev_dir>/cubes/``. The
        timeout is a fail-open bound, not the mechanism: a wedged plot
        worker must degrade to a placeholder card, never to an unposted
        candidate. Returns True when complete, False on timeout."""
        wait_s = float(self._cfg.plot_wait_s)
        if wait_s <= 0:
            return True
        plots_dir = Path(ev_dir) / "Level2" / "plots"
        paths = [plots_dir / f"{p}_{name}.png" for p in self._CUBE_PANELS]
        cubes_dir = Path(ev_dir) / "cubes"
        want_cubes = int(self._cfg.expected_cubes)
        deadline = time.monotonic() + wait_s
        waited = False
        while True:
            try:
                plots_ready = all(
                    p.is_file() and (time.time() - p.stat().st_mtime) >= 2.0
                    for p in paths
                )
                n_cubes = len(list(cubes_dir.glob("*.npz")))
            except OSError:
                plots_ready, n_cubes = False, 0
            if plots_ready and n_cubes >= want_cubes:
                if waited:
                    LOG.info(
                        "slack_notify %s: event complete after wait "
                        "(plots 4/4, cubes %d/%d)", name, n_cubes, want_cubes)
                return True
            if time.monotonic() >= deadline:
                missing = [p.name for p in paths if not p.is_file()]
                LOG.warning(
                    "slack_notify %s: event incomplete after %.0fs "
                    "(missing plots: %s; cubes %d/%d) — posting card "
                    "with what exists",
                    name, wait_s, ", ".join(missing) or "none (unsettled)",
                    n_cubes, want_cubes)
                return False
            waited = True
            time.sleep(max(0.5, float(self._cfg.plot_poll_s)))

    def _post_keep_card(
        self,
        name: str,
        c2row: Mapping[str, Any],
        ev_dir: Optional[Path],
    ) -> Dict[str, Any]:
        """v3.8+: the card image IS the message — no text-first
        ``chat.postMessage`` before it. Render the candidate card, upload
        it via the external-upload flow with the dashboard-link as the
        upload's ``initial_comment``, then poll ``files.info`` briefly to
        learn the resulting message ``ts`` (so the followup can thread
        onto it) and persist an event->(channel, ts) mapping for the
        annotation relay. If rendering or any upload step fails, falls
        back to a plain ``chat.postMessage`` (header text + dashboard
        link) so the event is never silently unposted.
        """
        token = self._token()
        if not token:
            return {"ok": False, "error": "no token"}

        card_path: Optional[Path] = None
        card_tmp_fd: Optional[int] = None
        try:
            if self._cfg.card and ev_dir is not None:
                # The C2 plot worker populates Level2/plots/ AFTER C3's
                # KEEP decision (typically ~1 min later) — wait until the
                # event is complete as the dashboard counts it (plots
                # 4/4, cubes 8/8), bounded by plot_wait_s (fail-open).
                self._wait_for_event_ready(name, ev_dir)
                # keep_report isn't known yet at card time — render_card
                # treats keep_report=None as "n/a" fragments, never raises.
                card_path, card_tmp_fd = self._render_card_tmp(
                    name, c2row, None, ev_dir)

            if card_path is not None:
                status = self._post_card_as_file(
                    token, name, c2row, card_path, ev_dir)
                if status.get("ok"):
                    return status
                LOG.info(
                    "slack_notify %s: card upload failed (%s); falling "
                    "back to a plain text message", name, status.get("error"))

            return self._post_fallback_message(token, name, c2row, ev_dir)
        finally:
            if card_path is not None:
                self._cleanup_card_tmp(card_tmp_fd, card_path)

    def _post_card_as_file(
        self,
        token: str,
        name: str,
        c2row: Mapping[str, Any],
        card_path: Path,
        ev_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        initial_comment = self._metadata_line(name, c2row, ev_dir)

        ok, doc = self._upload_file(
            token, card_path, title=name, channel=self._cfg.channel,
            initial_comment=initial_comment,
        )
        if not ok or doc is None:
            return {"ok": False, "error": "upload failed"}

        file_id = doc.get("file_id")
        ts = None
        if file_id:
            ts = self._poll_file_share_ts(token, file_id, self._cfg.channel)
        if ts is None:
            LOG.warning(
                "slack_notify %s: card uploaded but no share ts found "
                "within %.0fs (followup will post standalone)",
                name, _FILE_SHARE_POLL_TIMEOUT_S)

        if ts is not None:
            self._persist_post_map(name, ts)

        return {"ok": True, "error": None, "ts": ts, "file_id": file_id}

    def _post_fallback_message(
        self, token: str, name: str, c2row: Mapping[str, Any],
        ev_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        text = self._metadata_line(name, c2row, ev_dir)
        resp = self._api_post(
            "chat.postMessage", token,
            {"channel": self._cfg.channel, "text": text},
        )
        if resp is None or not resp.get("ok"):
            err = (resp or {}).get("error", "postMessage failed") if resp \
                else "postMessage failed"
            LOG.warning("slack_notify %s: chat.postMessage failed: %s",
                        name, err)
            return {"ok": False, "error": str(err)}

        ts = resp.get("ts")
        if ts is not None:
            self._persist_post_map(name, ts)
        return {"ok": True, "error": None, "ts": ts}

    def _metadata_line(
        self, name: str, c2row: Mapping[str, Any], ev_dir: Optional[Path],
    ) -> str:
        """One-line metadata summary used both as the card upload's
        ``initial_comment`` and as the plain-text fallback message when
        the card can't be posted as a file: name, significance, DM (the
        peak C1 detection's DM when resolvable — same value the card
        header quotes, via candidate_card's ``_resolve_detection_dm`` —
        else the c2 cluster-median fallback), UTC time, and the
        dashboard link. Missing values render as "n/a"; never raises."""
        try:
            from .candidate_card import _mjd_to_utc_str as _card_mjd_to_utc_str
            from .candidate_card import _resolve_detection_dm
        except Exception:  # noqa: BLE001 — candidate_card unavailable
            _card_mjd_to_utc_str = _mjd_to_utc_str
            _resolve_detection_dm = None

        snr = c2row.get("snr_max")
        dm = None
        if ev_dir is not None and _resolve_detection_dm is not None:
            try:
                dm = _resolve_detection_dm(ev_dir, name)
            except Exception:  # noqa: BLE001 — must never break the post
                dm = None
        if dm is None:
            dm = c2row.get("dm_median")
        utc = _card_mjd_to_utc_str(c2row.get("t_peak_mjd"))

        def _num(value: Any, fmt: str) -> Optional[str]:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(v):
                return None
            return fmt.format(v)

        snr_txt = _num(snr, "{:.1f}")
        sigma_str = f"{snr_txt}σ" if snr_txt is not None else "n/a"
        dm_txt = _num(dm, "{:.1f}")
        dm_str = f"DM {dm_txt} pc cm⁻³" if dm_txt is not None else "DM n/a"
        utc_str = utc if utc else "n/a"
        dashboard_url = f"{self._cfg.dashboard_base_url}/bursts/{name}"

        return (
            f"*{name}* — {sigma_str}, {dm_str}, {utc_str} | "
            f"<{dashboard_url}|Open in dashboard>"
        )

    @staticmethod
    def _frag_str(keep_report: Optional[Mapping[str, Any]]) -> Optional[str]:
        if not isinstance(keep_report, Mapping):
            return None
        n_present = keep_report.get("n_fragments_present")
        n_total = keep_report.get("n_fragments_total")
        if n_present is not None and n_total is not None:
            return f"{n_present}/{n_total}"
        return None

    #: (variant key, filename template, upload title template, upload
    #: comment) for the two bbproc variants the followup uploads — the
    #: unflagged filterbank and the SK RFI-flagged one. See
    #: post_keep_followup's docstring.
    _FOLLOWUP_VARIANTS: Tuple[Tuple[str, str, str, str], ...] = (
        ("unflagged", "{name}.png", "{name} bbproc - unflagged",
         "bbproc - unflagged"),
        ("rfi_flagged", "{name}_rfi.png", "{name} bbproc - SK RFI-flagged",
         "bbproc - SK RFI-flagged"),
    )

    def _post_keep_followup(
        self,
        name: str,
        ev_dir: Optional[Path],
        fb_report: Mapping[str, Any],
        keep_report: Mapping[str, Any],
        thread_ts: Optional[str],
    ) -> Dict[str, Any]:
        """Upload BOTH bbproc variants (unflagged + SK RFI-flagged) into
        the card's thread. A missing variant is skipped silently; if
        BOTH are missing and ``fb_report["ok"]`` is ``False``, posts a
        brief one-line failure note instead. ``thread_ts=None`` (card
        post failed/skipped) makes each upload land as a fresh
        top-level message rather than a threaded reply.
        """
        token = self._token()
        if not token:
            return {"ok": False, "error": "no token"}

        # Filterbank stage never ran (disabled, or no voltage fragments
        # to beamform) — nothing to add, skip silently.
        if isinstance(fb_report, Mapping) and fb_report.get("skipped"):
            return {"ok": True, "error": None,
                     "skipped": str(fb_report.get("skipped"))}

        if not self._cfg.upload_plot:
            return {"ok": True, "error": None, "skipped": "upload disabled"}

        fb_ok = fb_report.get("ok") if isinstance(fb_report, Mapping) else None
        frag_str = self._frag_str(keep_report)

        uploaded: List[str] = []
        failed: List[str] = []
        if ev_dir is not None:
            for key, fname_tmpl, title_tmpl, comment in self._FOLLOWUP_VARIANTS:
                png = ev_dir / "filterbank" / fname_tmpl.format(name=name)
                if not png.is_file():
                    continue
                ok, _doc = self._upload_file(
                    token, png, title=title_tmpl.format(name=name),
                    channel=self._cfg.channel, thread_ts=thread_ts,
                    initial_comment=comment,
                )
                if ok:
                    uploaded.append(key)
                else:
                    failed.append(key)

        if uploaded:
            return {
                "ok": not failed,
                "error": f"{len(failed)} variant(s) failed to upload"
                if failed else None,
                "uploaded": uploaded, "failed": failed,
            }

        if failed:
            # Every variant present on disk failed to upload.
            return {"ok": False, "error": "all variant uploads failed",
                    "uploaded": uploaded, "failed": failed}

        # Neither variant exists on disk.
        if fb_ok is False:
            # Brief one-line failure note — thread it if we have a
            # parent message, else it lands as a fresh message.
            text = f"{name}: bbproc filterbank failed"
            if frag_str:
                text += f" (voltage fragments {frag_str})"
            payload: Dict[str, Any] = {
                "channel": self._cfg.channel, "text": text,
            }
            if thread_ts:
                payload["thread_ts"] = thread_ts
            resp = self._api_post("chat.postMessage", token, payload)
            if resp is None or not resp.get("ok"):
                err = (resp or {}).get(
                    "error", "postMessage failed") if resp \
                    else "postMessage failed"
                LOG.warning(
                    "slack_notify %s: followup chat.postMessage "
                    "failed: %s", name, err)
                return {"ok": False, "error": str(err)}
            return {"ok": True, "error": None, "ts": resp.get("ts")}
        # Filterbank ok (or unknown) but produced no plot — nothing
        # actionable to post.
        return {"ok": True, "error": None, "skipped": "no plot"}

    def _render_card_tmp(
        self,
        name: str,
        c2row: Mapping[str, Any],
        keep_report: Mapping[str, Any],
        ev_dir: Path,
    ) -> tuple:
        """Best-effort candidate-card render into a system-temp PNG.

        Returns ``(path_or_None, fd_or_None)``. Callers must clean up via
        ``_cleanup_card_tmp`` when ``path`` is not None. Never raises."""
        if not self._cfg.card:
            return None, None
        fd = None
        tmp_path: Optional[Path] = None
        try:
            from .candidate_card import render_card

            fd, tmp_name = tempfile.mkstemp(
                prefix=f"card_{name}_", suffix=".png")
            tmp_path = Path(tmp_name)
            result = render_card(ev_dir, name, c2row, tmp_path, keep_report)
            if not result.get("ok"):
                LOG.info("slack_notify %s: candidate card render failed: %s",
                          name, result.get("error"))
                self._cleanup_card_tmp(fd, tmp_path)
                return None, None
            return tmp_path, fd
        except Exception as exc:  # noqa: BLE001 — fall back to _first_png
            LOG.warning("slack_notify %s: candidate card render raised: %s",
                        name, exc)
            self._cleanup_card_tmp(fd, tmp_path)
            return None, None

    @staticmethod
    def _cleanup_card_tmp(fd: Optional[int], path: Optional[Path]) -> None:
        try:
            if fd is not None:
                os.close(fd)
        except OSError:
            pass
        try:
            if path is not None and path.is_file():
                path.unlink()
        except OSError:
            pass

    # ----- Slack Web API plumbing -----------------------------------------

    def _api_post(
        self, method: str, token: str, payload: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        try:
            import requests
            r = requests.post(
                f"{_SLACK_API}/{method}",
                headers={"Authorization": f"Bearer {token}"},
                json=dict(payload),
                timeout=self._cfg.timeout_s,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("slack_notify: %s request failed: %s", method, exc)
            return None

    def _api_get(
        self, method: str, token: str, params: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        try:
            import requests
            r = requests.get(
                f"{_SLACK_API}/{method}",
                headers={"Authorization": f"Bearer {token}"},
                params=dict(params),
                timeout=self._cfg.timeout_s,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("slack_notify: %s request failed: %s", method, exc)
            return None

    def _upload_file(
        self,
        token: str,
        path: Path,
        *,
        title: Optional[str] = None,
        channel: Optional[str] = None,
        thread_ts: Optional[str] = None,
        initial_comment: Optional[str] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Slack's external-upload flow: getUploadURLExternal -> PUT/POST
        bytes -> completeUploadExternal. Any step failing returns
        ``(False, None)`` — callers decide what (if anything) to fall
        back to. On success, returns ``(True, {"file_id": ..., "response":
        <completeUploadExternal doc>})``."""
        try:
            import requests

            size = path.stat().st_size
            r1 = requests.get(
                f"{_SLACK_API}/files.getUploadURLExternal",
                headers={"Authorization": f"Bearer {token}"},
                params={"filename": path.name, "length": size},
                timeout=self._cfg.timeout_s,
            )
            r1.raise_for_status()
            doc1 = r1.json()
            if not doc1.get("ok"):
                LOG.warning("slack_notify: getUploadURLExternal failed: %s",
                            doc1.get("error"))
                return False, None
            upload_url = doc1["upload_url"]
            file_id = doc1["file_id"]

            with path.open("rb") as fh:
                r2 = requests.post(
                    upload_url, files={"file": fh},
                    timeout=self._cfg.timeout_s,
                )
            r2.raise_for_status()

            complete_payload: Dict[str, Any] = {
                "files": [{"id": file_id, "title": title or path.stem}],
            }
            if channel:
                complete_payload["channel_id"] = channel
            if thread_ts:
                complete_payload["thread_ts"] = thread_ts
            if initial_comment:
                complete_payload["initial_comment"] = initial_comment
            r3 = requests.post(
                f"{_SLACK_API}/files.completeUploadExternal",
                headers={"Authorization": f"Bearer {token}"},
                json=complete_payload,
                timeout=self._cfg.timeout_s,
            )
            r3.raise_for_status()
            doc3 = r3.json()
            if not doc3.get("ok"):
                LOG.warning(
                    "slack_notify: completeUploadExternal failed: %s",
                    doc3.get("error"))
                return False, None
            return True, {"file_id": file_id, "response": doc3}
        except Exception as exc:  # noqa: BLE001
            LOG.warning("slack_notify: file upload failed: %s", exc)
            return False, None

    def _poll_file_share_ts(
        self, token: str, file_id: str, channel: str,
        timeout_s: float = _FILE_SHARE_POLL_TIMEOUT_S,
    ) -> Optional[str]:
        """Poll ``files.info`` until the uploaded file shows up shared
        into ``channel`` (Slack processes the upload->message-share
        asynchronously), returning that message's ``ts`` — or ``None``
        if it hasn't appeared within ``timeout_s``. Never raises."""
        deadline = time.monotonic() + timeout_s
        while True:
            resp = self._api_get("files.info", token, {"file": file_id})
            if resp and resp.get("ok"):
                shares = (resp.get("file") or {}).get("shares") or {}
                for visibility in ("public", "private"):
                    entries = (shares.get(visibility) or {}).get(channel)
                    if entries:
                        ts = entries[0].get("ts")
                        if ts:
                            return ts
            if time.monotonic() >= deadline:
                return None
            time.sleep(_FILE_SHARE_POLL_INTERVAL_S)

    def _persist_post_map(self, name: str, ts: str) -> None:
        """Best-effort event->(channel, ts) mapping so the annotation
        relay (slack_annotation_relay.py) can thread classification
        updates onto this event's card message. Failure logs a warning
        and never propagates — this is a convenience index, not the
        source of truth (the Slack message itself is)."""
        if not self._cfg.post_map_db:
            return
        try:
            db_path = Path(self._cfg.post_map_db).expanduser()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS posts ("
                    "event TEXT PRIMARY KEY, channel TEXT, ts TEXT, "
                    "posted_utc TEXT)"
                )
                conn.execute(
                    "INSERT OR REPLACE INTO posts "
                    "(event, channel, ts, posted_utc) VALUES (?, ?, ?, ?)",
                    (name, self._cfg.channel, ts,
                     datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            LOG.warning(
                "slack_notify %s: failed to persist post map (%s): %s",
                name, self._cfg.post_map_db, exc)
