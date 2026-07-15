/* Human-classification UI for the burst event page.
 *
 * Dependency-free vanilla JS. Server embeds window.ANNOT = {event, block,
 * vocab, builtins, history, nextUnclassified}. Per-user last-click-wins;
 * the active user is persisted in localStorage so it survives navigation.
 * All mutations POST JSON and re-render in place from the response block.
 */
(function () {
  "use strict";

  var ANNOT = window.ANNOT || {};
  var USER_KEY = "dsa_monitor_user";

  var LABEL_COLORS = {
    FRB: "#00b894",
    RFI: "#d63031",
    NOISE: "#636e72",
    PULSAR: "#0984e3",
    INJECTION: "#e17055",
  };
  function labelColor(l) { return LABEL_COLORS[l] || "#8395a7"; }

  // --- state ---------------------------------------------------------------
  var block = ANNOT.block || { event: ANNOT.event, classifications: [], labels: {}, source_name: null };
  var vocab = ANNOT.vocab || { builtin_labels: [], custom_tags: [], labels: [], users: [], source_names: [] };
  var history = ANNOT.history || [];
  var activeUser = null;

  var $ = function (id) { return document.getElementById(id); };

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function shortTs(ts) {
    if (!ts) return "";
    // ISO-8601 → "MM-DD HH:MM UTC" (stored timestamps are UTC; house
    // rule: every human-visible timestamp is explicitly UTC).
    var m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(ts);
    return m ? (m[2] + "-" + m[3] + " " + m[4] + ":" + m[5] + " UTC") : ts;
  }

  function currentLabelFor(user) {
    if (!user) return null;
    var lc = user.toLowerCase();
    for (var i = 0; i < block.classifications.length; i++) {
      if (block.classifications[i].user.toLowerCase() === lc) {
        return block.classifications[i].label;
      }
    }
    return null;
  }

  // --- messaging -----------------------------------------------------------
  function setMsg(text, kind) {
    var el = $("annot-msg");
    el.textContent = text || "";
    el.className = "annot-msg" + (kind ? " " + kind : "");
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

  // --- renderers -----------------------------------------------------------
  function renderSummary() {
    var parts = [];
    block.classifications.forEach(function (c) {
      parts.push('<span><strong>' + esc(c.user) + '</strong>: ' +
        '<span class="annot-badge" style="background:' + labelColor(c.label) + '">' +
        esc(c.label) + '</span></span>');
    });
    var line = parts.length ? parts.join(" &nbsp;·&nbsp; ") : "no classifications yet";
    if (block.source_name && block.source_name.source_name) {
      line += ' &nbsp;·&nbsp; source: <span class="annot-summary-src">' +
        esc(block.source_name.source_name) + "</span>";
    }
    $("annot-summary").innerHTML = line;
  }

  function renderUsers() {
    var host = $("annot-users");
    host.innerHTML = "";
    (vocab.users || []).forEach(function (u) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "annot-user-btn" + (u === activeUser ? " active" : "");
      b.textContent = u;
      b.onclick = function () { selectUser(u); };
      host.appendChild(b);
      host.appendChild(document.createTextNode(" "));
    });
    if (!(vocab.users || []).length) {
      host.innerHTML = '<span class="muted">no users yet — add yourself →</span> ';
    }
  }

  function renderLabels() {
    var host = $("annot-labels");
    host.innerHTML = "";
    var mine = currentLabelFor(activeUser);
    var all = (ANNOT.builtins || []).concat(vocab.custom_tags || []);
    all.forEach(function (lab) {
      var b = document.createElement("button");
      b.type = "button";
      var on = (lab === mine);
      b.className = "annot-label-btn" + (on ? " active" : "");
      if (on) { b.style.background = labelColor(lab); }
      b.textContent = lab;
      b.onclick = function () { doClassify(lab); };
      host.appendChild(b);
      host.appendChild(document.createTextNode(" "));
    });
  }

  function renderSource() {
    var who = $("annot-source-who");
    if (block.source_name && block.source_name.source_name) {
      $("annot-source").value = block.source_name.source_name;
      who.textContent = "set by " + block.source_name.user + " · " +
        shortTs(block.source_name.ts_utc);
    } else {
      who.textContent = "";
    }
  }

  function renderHistory() {
    var host = $("annot-history");
    if (!history.length) { host.innerHTML = '<div class="muted">no history.</div>'; return; }
    var rows = history.map(function (h) {
      var what = h.kind === "source_name"
        ? ("source → " + (h.source_name == null ? "(cleared)" : esc(h.source_name)))
        : ("label → " + (h.label == null ? "(cleared)" : esc(h.label)));
      return "<tr><td>" + shortTs(h.ts_utc) + "</td><td>" + esc(h.user) +
        "</td><td>" + what + "</td></tr>";
    }).join("");
    host.innerHTML = "<table>" + rows + "</table>";
  }

  function renderAll() {
    renderSummary(); renderUsers(); renderLabels(); renderSource(); renderHistory();
  }

  function applyBlock(j) {
    // classify/source responses carry the updated current block.
    if (j && j.event) {
      block = {
        event: j.event,
        classifications: j.classifications || [],
        labels: j.labels || {},
        source_name: j.source_name || null,
      };
    }
    if (j && j.vocab) { vocab = j.vocab; }
    renderAll();
  }

  // --- actions -------------------------------------------------------------
  function selectUser(u) {
    activeUser = u;
    try { localStorage.setItem(USER_KEY, u); } catch (e) {}
    setMsg("");
    renderUsers(); renderLabels();
  }

  function needUser() {
    if (!activeUser) {
      setMsg("pick or add a user first (click a name above, or type one and press + user).", "warn");
      var box = $("annot-newuser");
      if (box) box.focus();
      return true;
    }
    return false;
  }

  function addUser() {
    var box = $("annot-newuser");
    var name = (box.value || "").trim();
    if (!name) { setMsg("type a name to add.", "warn"); return; }
    post("/annotations/user", { name: name }, function (j) {
      box.value = "";
      if (j.vocab) vocab = j.vocab;
      selectUser(j.user);
      setMsg("added user " + j.user, "ok");
      renderAll();
    });
  }

  function doClassify(label) {
    if (needUser()) return;
    post("/annotations/classify",
      { event: ANNOT.event, user: activeUser, label: label },
      function (j) { applyBlock(j); setMsg(label ? ("you: " + label) : "cleared", "ok"); });
  }

  function doClear() { if (!needUser()) doClassify(null); }

  function addTagAndApply() {
    if (needUser()) return;
    var box = $("annot-newtag");
    var tag = (box.value || "").trim();
    if (!tag) { setMsg("type a tag name.", "warn"); return; }
    post("/annotations/tag", { tag: tag, user: activeUser, event: ANNOT.event },
      function (j) {
        box.value = "";
        if (j.vocab) vocab = j.vocab;
        renderLabels();
        // Apply the freshly-created tag to the active user.
        doClassify(j.tag);
      });
  }

  function setSource() {
    if (needUser()) return;
    var val = ($("annot-source").value || "").trim();
    hideDropdown();
    post("/annotations/source",
      { event: ANNOT.event, user: activeUser, source_name: val },
      function (j) { applyBlock(j); setMsg(val ? ("source set: " + val) : "source cleared", "ok"); });
  }

  function clearSource() {
    if (needUser()) return;
    $("annot-source").value = "";
    $("annot-source-hint").textContent = "";
    hideDropdown();
    post("/annotations/source",
      { event: ANNOT.event, user: activeUser, source_name: null },
      function (j) { applyBlock(j); setMsg("source cleared", "ok"); });
  }

  // --- source typeahead (live substring dropdown) --------------------------
  var ddIndex = -1;
  var ddItems = [];

  function sourceNames() { return vocab.source_names || []; }

  function updateHint() {
    var val = ($("annot-source").value || "").trim();
    var hint = $("annot-source-hint");
    if (!val) { hint.textContent = ""; return; }
    var exact = sourceNames().some(function (s) {
      return s.toLowerCase() === val.toLowerCase();
    });
    hint.textContent = exact ? ""
      : "new source name — check the suggestions above for an existing spelling.";
  }

  function hideDropdown() {
    $("annot-source-dd").style.display = "none";
    ddIndex = -1; ddItems = [];
  }

  function showDropdown() {
    var val = ($("annot-source").value || "").trim().toLowerCase();
    var dd = $("annot-source-dd");
    var matches = sourceNames().filter(function (s) {
      return val === "" ? true : s.toLowerCase().indexOf(val) !== -1;
    }).slice(0, 10);
    ddItems = matches;
    ddIndex = -1;
    if (!matches.length) { hideDropdown(); return; }
    dd.innerHTML = "";
    matches.forEach(function (s, i) {
      var row = document.createElement("div");
      row.textContent = s;
      row.onmousedown = function (e) { e.preventDefault(); pickSuggestion(i); };
      dd.appendChild(row);
    });
    dd.style.display = "block";
  }

  function highlightDd() {
    var dd = $("annot-source-dd");
    var kids = dd.children;
    for (var i = 0; i < kids.length; i++) {
      kids[i].className = (i === ddIndex) ? "active" : "";
    }
  }

  function pickSuggestion(i) {
    if (i < 0 || i >= ddItems.length) return;
    $("annot-source").value = ddItems[i];
    hideDropdown();
    updateHint();
    $("annot-source").focus();
  }

  function wireSource() {
    var inp = $("annot-source");
    inp.addEventListener("input", function () { showDropdown(); updateHint(); });
    inp.addEventListener("focus", function () { showDropdown(); });
    inp.addEventListener("blur", function () { setTimeout(hideDropdown, 150); });
    inp.addEventListener("keydown", function (e) {
      var open = $("annot-source-dd").style.display === "block";
      if (e.key === "ArrowDown") {
        if (!open) { showDropdown(); return; }
        e.preventDefault(); ddIndex = Math.min(ddIndex + 1, ddItems.length - 1); highlightDd();
      } else if (e.key === "ArrowUp") {
        e.preventDefault(); ddIndex = Math.max(ddIndex - 1, 0); highlightDd();
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (open && ddIndex >= 0) { pickSuggestion(ddIndex); }
        else { setSource(); }
      } else if (e.key === "Escape") {
        hideDropdown();
      }
    });
  }

  // --- keyboard shortcuts --------------------------------------------------
  function isTyping(el) {
    if (!el) return false;
    var tag = el.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || el.isContentEditable;
  }

  function wireShortcuts() {
    var map = { "1": "FRB", "2": "RFI", "3": "NOISE", "4": "PULSAR", "5": "INJECTION" };
    document.addEventListener("keydown", function (e) {
      if (isTyping(e.target)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (map[e.key]) { e.preventDefault(); doClassify(map[e.key]); }
      else if (e.key === "c" || e.key === "C") { e.preventDefault(); doClear(); }
      else if (e.key === "n" || e.key === "N") {
        if (ANNOT.nextUnclassified) {
          window.location.href = "/bursts/" + encodeURIComponent(ANNOT.nextUnclassified);
        } else {
          setMsg("no more unclassified events.", "ok");
        }
      }
    });
  }

  // --- init ----------------------------------------------------------------
  function init() {
    try {
      var stored = localStorage.getItem(USER_KEY);
      if (stored && (vocab.users || []).indexOf(stored) !== -1) { activeUser = stored; }
    } catch (e) {}

    $("annot-adduser").onclick = addUser;
    $("annot-newuser").addEventListener("keydown", function (e) {
      if (e.key === "Enter") { e.preventDefault(); addUser(); }
    });
    $("annot-clear").onclick = doClear;
    $("annot-addtag").onclick = addTagAndApply;
    $("annot-newtag").addEventListener("keydown", function (e) {
      if (e.key === "Enter") { e.preventDefault(); addTagAndApply(); }
    });
    $("annot-source-set").onclick = setSource;
    $("annot-source-clear").onclick = clearSource;
    wireSource();
    wireShortcuts();
    renderAll();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
