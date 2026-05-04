#!/usr/bin/env bash
# Pre-M0 driver bump status gate (runs from operator workstation with SSH to h01).
set -euo pipefail
# 'stage' and 'legacy_smoke' use startswith() so descriptive suffixes
# (e.g. 'PASS (bfCorr against 525)') do not break the gate. The operator
# is encouraged to include test-detail context after the keyword.
ssh h01 'python3 -c "import json; p=\"/home/ubuntu/dsart-driver-bump-status.json\"; s=json.load(open(p)); assert s.get(\"stage\", \"\").startswith(\"complete\"), s; assert s.get(\"legacy_smoke\", \"\").startswith(\"PASS\"), s; print(\"M0_prereq PASS\")"'
