# v1.0 — Baseline: Vikram → Vishnu handover

**Date:** 2026-07-10
**Tag:** `v1.0` (this commit, on `main`)
**Custodian from here on:** Vishnu Balakrishnan (`vishnubk`, vishnubk93@gmail.com)

This tag freezes the state of `dsa110-rt` at handover. The only change on
top of Vikram's last commit (`36313a4`, "dashboard: per-calibrator 'Update
cals' button on the SEFDs page") is the addition of `docs/onboarding/` —
no source, config, or tooling was modified.

## Repo state at handover

- Milestones M0–M8 complete; M8 live end-to-end voltage-dump test PASSED
  2026-07-09 (`scratch/M8_E2E_VOLTAGE_DUMP_TEST_20260709.md` in the
  workspace).
- Open questions at handover are recorded in
  `docs/onboarding/research/03-feature-status.md` § "Open unknowns"
  (M7.6 cutover status, legacy Hella retirement, SPL on-sky status,
  `mjd_target=0.0` fix, `dsart_c3.service` unit mismatch).

### Branch heads at handover (for rollback of the whole workspace)

| Branch | HEAD | Checkout |
| --- | --- | --- |
| `main` | `36313a4` (+ this docs commit) | `dsa110-rt/` |
| `m4a/tx-prod-header` | `581165a` | `dsa110-rt-m4a-tx/` |
| `m5/main` | `8b8b028` | `dsa110-rt-m5/` |
| `m6/main` | `1dfd5a5` | `dsa110-rt-m6/` |
| `operator-integration` | `d99cf35` | `dsa110-rt-operint/` |
| `m3/rfi-flagger` | `25c77ac` | `dsa110-rt-rfi-flagger/` |

(`dsa110-operator`, the LLM telescope controller, is a separate repo and
is intentionally NOT part of this handover — untouched at `95c235e`.)

## What `docs/onboarding/` contains

| Artifact | What it is |
| --- | --- |
| `BEGINNERS_GUIDE.md` | Fact-checked ~7,700-word onboarding guide (image-plane searching, host map, data journey, operations, injections, voltage dumps, feature status, glossary). |
| `research/00–03*.md` | Research packets with file:line citations; "do not invent" gaps flagged. |
| `tools/md2pdf.py` | Markdown → PDF converter (`python3 md2pdf.py FILE.md`; needs `pip install markdown weasyprint==52.5`). |

## Rollback

```bash
git checkout v1.0            # inspect the baseline (detached HEAD)
git revert <bad-commit>      # undo one commit, keep history (preferred)
git reset --hard v1.0        # rewrite local main back to baseline (last resort)
```

Vikram's exact pre-handover state is `36313a4` (= `v1.0~1`).
