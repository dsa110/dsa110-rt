#!/usr/bin/env bash
# Mirror /home/ubuntu/proj/dsa110-shell/dsa110-xengine/utils/antennas.out
# from a source corr node to the four search nodes (n01, n02, n09, n13).
#
# The user's cal pipeline writes antennas.out on every corr node after
# each calibration source pass; this script propagates one of those
# files (any of them; the ANTPOS block is identical across corr nodes,
# and that is the only thing the search side reads from the blob) to
# the search nodes so search_compute_0/1 can resolve --cal-blob-path
# to the same per-node path as corr_fast --apply-cal.
#
# Usage (from any host that has ssh keys to both source + search):
#   tools/ops/sync_antennas_out_to_search.sh [SRC_HOST]
#
# SRC_HOST defaults to n03.pro.pvt. The search-node list is hard-
# coded to match services_inventory._SEARCH_CN_IDS.
#
# Recommended: call this at the end of the cal-update script that
# refreshes antennas.out on the corr nodes.

set -euo pipefail

SRC_HOST="${1:-n03.pro.pvt}"
SRC_PATH="/home/ubuntu/proj/dsa110-shell/dsa110-xengine/utils/antennas.out"
DST_PATH="$SRC_PATH"
SEARCH_HOSTS=(n01.pro.pvt n02.pro.pvt n09.pro.pvt n13.pro.pvt)

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT
LOCAL_COPY="$STAGING/antennas.out"

echo "[$(date -u +%FT%TZ)] pulling $SRC_HOST:$SRC_PATH -> $LOCAL_COPY"
rsync -a --partial -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
    "$SRC_HOST:$SRC_PATH" "$LOCAL_COPY"
SRC_MD5="$(md5sum "$LOCAL_COPY" | awk '{print $1}')"
SRC_SIZE="$(stat -c %s "$LOCAL_COPY")"
echo "  src md5=$SRC_MD5  size=$SRC_SIZE"

FAIL=0
for h in "${SEARCH_HOSTS[@]}"; do
    echo "[$(date -u +%FT%TZ)] pushing -> $h:$DST_PATH"
    if ! ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$h" \
            "mkdir -p $(dirname "$DST_PATH")"; then
        echo "  FAIL: cannot mkdir on $h"
        FAIL=$((FAIL + 1))
        continue
    fi
    if ! rsync -a --partial \
            -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
            "$LOCAL_COPY" "$h:$DST_PATH"; then
        echo "  FAIL: rsync to $h"
        FAIL=$((FAIL + 1))
        continue
    fi
    REMOTE_MD5="$(ssh -o BatchMode=yes "$h" \
        "md5sum $DST_PATH | awk '{print \$1}'")"
    if [[ "$REMOTE_MD5" != "$SRC_MD5" ]]; then
        echo "  FAIL: md5 mismatch on $h: got $REMOTE_MD5 want $SRC_MD5"
        FAIL=$((FAIL + 1))
    else
        echo "  OK ($h md5=$REMOTE_MD5)"
    fi
done

if (( FAIL > 0 )); then
    echo "[$(date -u +%FT%TZ)] DONE with $FAIL failure(s)"
    exit 1
fi
echo "[$(date -u +%FT%TZ)] DONE — all ${#SEARCH_HOSTS[@]} search nodes carry md5=$SRC_MD5"
