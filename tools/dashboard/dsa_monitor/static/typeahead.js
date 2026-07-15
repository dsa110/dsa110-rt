/* Shared live type-ahead dropdown for source-name inputs.
 *
 * Used by the burst event page (annotations.js) and the bursts list
 * page. Dependency-free. Case-insensitive substring match anywhere in
 * the name, updates every keystroke, ArrowUp/Down + Enter + Escape +
 * click, closes on blur. Rows use mousedown (with preventDefault) so
 * picking never fights the input's blur.
 *
 * Usage:
 *   var ta = dsaTypeahead({
 *     input:    <input element>,
 *     dd:       <dropdown container element (styled like .annot-dd)>,
 *     getNames: function () { return ["B1913+16", ...]; },
 *     onPick:   function (name) {},     // suggestion chosen (input filled first)
 *     onSubmit: function (text) {},     // Enter with no highlighted row
 *     onInput:  function (text) {},     // optional, every keystroke
 *     rowExtra: function (name, row, api) {},  // optional right-side widget
 *   });
 *   api/ta: {show, hide, refresh}
 */
(function () {
  "use strict";

  window.dsaTypeahead = function (cfg) {
    var input = cfg.input;
    var dd = cfg.dd;
    var idx = -1;
    var items = [];

    function hide() {
      dd.style.display = "none";
      idx = -1;
      items = [];
    }

    function highlight() {
      var kids = dd.children;
      for (var i = 0; i < kids.length; i++) {
        kids[i].classList.toggle("active", i === idx);
      }
    }

    function pick(i) {
      if (i < 0 || i >= items.length) return;
      var name = items[i];
      input.value = name;
      hide();
      if (cfg.onPick) cfg.onPick(name);
      input.focus();
    }

    function show() {
      var val = (input.value || "").trim().toLowerCase();
      var names = (cfg.getNames && cfg.getNames()) || [];
      var matches = names.filter(function (s) {
        return val === "" ? true : s.toLowerCase().indexOf(val) !== -1;
      }).slice(0, 10);
      items = matches;
      idx = -1;
      if (!matches.length) { hide(); return; }
      dd.innerHTML = "";
      matches.forEach(function (s, i) {
        var row = document.createElement("div");
        var label = document.createElement("span");
        label.textContent = s;
        row.appendChild(label);
        row.onmousedown = function (e) { e.preventDefault(); pick(i); };
        if (cfg.rowExtra) cfg.rowExtra(s, row, api);
        dd.appendChild(row);
      });
      dd.style.display = "block";
    }

    input.addEventListener("input", function () {
      show();
      if (cfg.onInput) cfg.onInput(input.value);
    });
    input.addEventListener("focus", function () { show(); });
    input.addEventListener("blur", function () { setTimeout(hide, 150); });
    input.addEventListener("keydown", function (e) {
      var open = dd.style.display === "block";
      if (e.key === "ArrowDown") {
        if (!open) { show(); return; }
        e.preventDefault();
        idx = Math.min(idx + 1, items.length - 1);
        highlight();
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        idx = Math.max(idx - 1, 0);
        highlight();
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (open && idx >= 0) { pick(idx); }
        else if (cfg.onSubmit) { hide(); cfg.onSubmit(input.value); }
      } else if (e.key === "Escape") {
        hide();
      }
    });

    var api = { show: show, hide: hide, refresh: show };
    return api;
  };
})();
