/* Refined-localization UI for the burst event page (Sky position card).
 *
 * Dependency-free vanilla JS, same conventions as annotations.js: the
 * server embeds window.POS = {event, refined, history}; the acting user
 * is the one selected in the Human classification card (shared
 * localStorage key). All display strings (adaptive sexagesimal,
 * tooltip) come from the server — no coordinate formatting here.
 */
(function () {
  "use strict";

  var POS = window.POS || {};
  var USER_KEY = "dsa_monitor_user";

  var refined = POS.refined || null;
  var history = POS.history || [];

  var $ = function (id) { return document.getElementById(id); };

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function shortTs(ts) {
    if (!ts) return "";
    var m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(ts);
    return m ? (m[2] + "-" + m[3] + " " + m[4] + ":" + m[5] + " UTC") : ts;
  }

  function activeUser() {
    try { return localStorage.getItem(USER_KEY) || null; } catch (e) { return null; }
  }

  function setMsg(text, kind) {
    var el = $("pos-msg");
    el.textContent = text || "";
    el.className = "annot-msg" + (kind ? " " + kind : "");
  }

  // --- renderers -----------------------------------------------------------
  function renderRefined() {
    var host = $("pos-refined");
    if (!refined) { host.innerHTML = ""; return; }
    host.innerHTML =
      '<div class="pos-refined-line">' +
      '<strong>refined</strong><sup class="refined-mark" title="' +
      esc(refined.tooltip) + '">R</sup>&nbsp; ' +
      '<strong>RA</strong>&nbsp; <strong>' + esc(refined.ra_hms) + '</strong>' +
      '&nbsp;<span class="muted">(' + Number(refined.ra_deg).toFixed(5) + '°)</span>' +
      '&nbsp;·&nbsp; <strong>Dec</strong>&nbsp; <strong>' +
      esc(refined.dec_dms) + '</strong>' +
      '&nbsp;<span class="muted">(' +
      (refined.dec_deg >= 0 ? "+" : "") + Number(refined.dec_deg).toFixed(5) +
      '°)</span></div>' +
      '<div class="pos-refined-meta">&sigma; = ' +
      esc(refined.ra_err_arcsec) + '" (RA, on-sky) × ' +
      esc(refined.dec_err_arcsec) + '" (Dec) · method: ' +
      esc(refined.method) + ' · by ' + esc(refined.username) + ' · ' +
      shortTs(refined.created_utc) + '</div>';
  }

  function renderHistory() {
    var host = $("pos-history");
    if (!history.length) {
      host.innerHTML = '<div class="muted">no history.</div>';
      return;
    }
    var rows = history.map(function (h) {
      var what = (h.ra_deg == null)
        ? "(cleared)"
        : (Number(h.ra_deg).toFixed(5) + " " +
           (h.dec_deg >= 0 ? "+" : "") + Number(h.dec_deg).toFixed(5) +
           " deg · ±" + h.ra_err_arcsec + '"/' + h.dec_err_arcsec + '" · ' +
           esc(h.method));
      return "<tr><td>" + shortTs(h.created_utc) + "</td><td>" +
        esc(h.username) + "</td><td>" + what + "</td></tr>";
    }).join("");
    host.innerHTML = "<table>" + rows + "</table>";
  }

  function renderAll() { renderRefined(); renderHistory(); }

  // --- server round-trips ----------------------------------------------------
  function refresh(onDone) {
    fetch("/api/position/" + encodeURIComponent(POS.event))
      .then(function (r) { return r.json(); })
      .then(function (j) {
        if (j && j.ok) {
          refined = j.refined || null;
          history = j.history || [];
          renderAll();
        }
        if (onDone) onDone();
      })
      .catch(function () { if (onDone) onDone(); });
  }

  function post(url, body, onOk) {
    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(function (r) {
      return r.json().then(function (j) { return { ok: r.ok, j: j }; });
    }).then(function (res) {
      if (!res.ok || !res.j.ok) {
        setMsg((res.j && res.j.error) || "request failed", "warn");
        return;
      }
      onOk(res.j);
    }).catch(function (e) {
      setMsg("network error: " + e, "warn");
    });
  }

  // --- actions ---------------------------------------------------------------
  function needUser() {
    if (!activeUser()) {
      setMsg("pick a user in the Human classification card first.", "warn");
      return true;
    }
    return false;
  }

  function doSet() {
    if (needUser()) return;
    post("/api/position/" + encodeURIComponent(POS.event), {
      ra: ($("pos-ra").value || "").trim(),
      dec: ($("pos-dec").value || "").trim(),
      ra_err_arcsec: ($("pos-ra-err").value || "").trim(),
      dec_err_arcsec: ($("pos-dec-err").value || "").trim(),
      method: ($("pos-method").value || "").trim(),
      user: activeUser(),
    }, function (j) {
      refined = j.refined || null;
      renderRefined();
      setMsg("refined position set: " + refined.ra_hms + " " +
             refined.dec_dms, "ok");
      refresh();   // pull the appended audit row into the history table
    });
  }

  function doClear() {
    if (needUser()) return;
    post("/api/position/" + encodeURIComponent(POS.event) + "/clear",
      { user: activeUser() },
      function (j) {
        refined = null;
        renderRefined();
        setMsg(j.cleared ? "refined position cleared"
                         : "no refined position was set", "ok");
        refresh();
      });
  }

  // --- init --------------------------------------------------------------------
  function init() {
    if (!$("pos-set")) return;   // card absent (defensive)
    $("pos-set").onclick = doSet;
    $("pos-clear").onclick = doClear;
    if (refined) {
      $("pos-ra").value = Number(refined.ra_deg).toFixed(5);
      $("pos-dec").value = (refined.dec_deg >= 0 ? "+" : "") +
        Number(refined.dec_deg).toFixed(5);
      $("pos-ra-err").value = refined.ra_err_arcsec;
      $("pos-dec-err").value = refined.dec_err_arcsec;
      $("pos-method").value = refined.method;
    }
    renderAll();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
