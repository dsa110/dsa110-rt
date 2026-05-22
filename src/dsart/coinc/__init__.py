"""C2 coincidencer service library (M7.4 bring-up).

Modules in this package implement the *fan-in* side of the
C1 → C2 pipeline described in ``docs/c1c2/C1C2_DESIGN.md``:

  * ``wire``          — the C1 → C2 ASCII batch parser + C2 → C1
                        UDP trigger packet encoder.
  * ``window``        — rolling time-window buffer.
  * ``components``    — connected-components on the half-window
                        time edge ``|Δt| ≤ (w_i + w_j) / 2``.
  * ``stats``         — ClusterStats dataclass + computer.
  * ``names``         — event-name allocator (wraps
                        ``event.names.increment_name``).
  * ``criteria``      — YAML-driven trigger-criteria evaluator with
                        SIGHUP-style hot-reload.
  * ``broadcast``     — UDP fan-out to the 8 C1 dump listeners.
  * ``archive``       — per-event archive layout writer
                        (``/dataz/dsa110/candidates/<name>/``).
  * ``csv_rotator``   — hourly-rotated C1 + C2 CSV writers with
                        48-hour retention enforcement.
  * ``plotter``       — 4-panel cube-event PNG generator.
  * ``service``       — async orchestrator tying everything
                        together; lifted into ``dsart.services.coincidencer``
                        for systemd entry.
"""

from __future__ import annotations
