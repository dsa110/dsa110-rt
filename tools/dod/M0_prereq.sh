#!/usr/bin/env bash
# Pre-M0 driver bump status gate (runs from operator workstation with SSH to h01).
set -euo pipefail
ssh h01 'python3 -c "import json; p=\"/home/ubuntu/dsart-driver-bump-status.json\"; s=json.load(open(p)); assert s.get(\"stage\")==\"complete\", s; assert s.get(\"legacy_smoke\")==\"PASS\", s; print(\"M0_prereq PASS\")"'
