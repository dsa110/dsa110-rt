#!/bin/bash
# Back up the Grafana configuration on lxd110h20 to the DR archive.
#
# Why this exists (2026-08-03): DISASTER_RECOVERY.md §7 records that the
# only Grafana recovery path was the 2019 `grafana.db` shipped in the
# lxd110maas tarball. Every dashboard change since then -- Antenna
# Monitor, calibration, DSA110-WX, heatmap, the mp/ant_num row sets and
# the generated dsart-rt board -- existed ONLY in the live sqlite DB on
# one host. A host loss took them all.
#
# Runs on h23, NOT on h20, deliberately:
#   * h23 holds GRAFANA_AUTH (~/.dsart/secrets.env); h20 has no copy and
#     this script does not create one;
#   * /dataz/dsa110/dr_archive is not writable from inside the h20
#     container (LXD uid mapping) but is from h23;
#   * a backup stored on the host being backed up is not a backup.
#
# Produces, per run, in $DEST/<UTC-date>/ :
#   dashboards/<uid>__<slug>.json   one file per dashboard, via the API.
#                                   Re-import with build_dashboard.py's
#                                   POST path or `curl -X POST
#                                   /api/dashboards/db`.
#   datasources.json                datasource definitions (Grafana
#                                   redacts stored passwords; the sql
#                                   dump below carries the encrypted
#                                   originals).
#   grafana_db.sql.gz               logical dump of grafana.db -- users,
#                                   orgs, prefs, alerts, datasource
#                                   secrets. Restore with
#                                   `zcat ... | sqlite3 grafana.db`.
#   MANIFEST.txt                    what was captured, and the versions.
#
# h20 has Python 3.6 and no sqlite3 CLI, so neither `VACUUM INTO`
# (needs SQLite >= 3.27; it has 3.22) nor `sqlite3.Connection.backup()`
# (needs Python >= 3.7) is available. The dump is therefore taken with
# iterdump() inside a read transaction opened on a read-only handle,
# which gives a consistent logical snapshot without ever writing to the
# live database.
#
# Usage:
#   ./backup_grafana.sh                 # normal run
#   KEEP=30 ./backup_grafana.sh         # change retention (default 14)
#   DEST=/tmp/gtest ./backup_grafana.sh # somewhere else, for a dry run
#
# Exit status is non-zero if the dashboard export fails; a failed sql
# dump is reported but does not fail the run, since the dashboards are
# the artifact that matters most.

set -eo pipefail

H20_HOST="${H20_HOST:-lxd110h20.pro.pvt}"
GRAFANA_URL="${GRAFANA_URL:-http://10.42.0.228:3000}"
GRAFANA_DB="${GRAFANA_DB:-/home/ubuntu/proj/grafana-6.2.5/data/grafana.db}"
DEST="${DEST:-/dataz/dsa110/dr_archive/grafana}"
KEEP="${KEEP:-14}"
SECRETS="${SECRETS:-$HOME/.dsart/secrets.env}"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
die() { echo "[FATAL] $*" >&2; exit 1; }

# ---- credentials ----------------------------------------------------------
if [ -z "$GRAFANA_AUTH" ] && [ -r "$SECRETS" ]; then
    # shellcheck disable=SC1090
    . "$SECRETS"
fi
[ -n "$GRAFANA_AUTH" ] || die "GRAFANA_AUTH unset and not found in $SECRETS"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$DEST/$STAMP"
mkdir -p "$OUT/dashboards" || die "cannot create $OUT"
log "writing to $OUT"

# ---- dashboards ----------------------------------------------------------
# api/search lists them; api/dashboards/uid/<uid> returns the full model.
# The wrapper carries a `meta` block we keep -- it records the folder and
# the version, which the import path can use.
LIST="$(curl -sf -u "$GRAFANA_AUTH" \
        "$GRAFANA_URL/api/search?type=dash-db&limit=500")" \
    || die "dashboard search failed against $GRAFANA_URL"

N_DASH=0
for UID_ in $(echo "$LIST" | python3 -c '
import json, sys
for d in json.load(sys.stdin):
    u = d.get("uid")
    if u:
        print(u)
'); do
    BODY="$(curl -sf -u "$GRAFANA_AUTH" \
            "$GRAFANA_URL/api/dashboards/uid/$UID_")" || {
        log "WARNING: could not fetch dashboard $UID_"
        continue
    }
    SLUG="$(echo "$BODY" | python3 -c '
import json, re, sys
t = json.load(sys.stdin)["dashboard"].get("title", "untitled")
print(re.sub(r"[^A-Za-z0-9_.-]+", "_", t)[:60] or "untitled")
')"
    echo "$BODY" | python3 -m json.tool > "$OUT/dashboards/${UID_}__${SLUG}.json"
    N_DASH=$((N_DASH + 1))
done
[ "$N_DASH" -gt 0 ] || die "no dashboards were exported"
log "exported $N_DASH dashboard(s)"

# ---- datasources ---------------------------------------------------------
if curl -sf -u "$GRAFANA_AUTH" "$GRAFANA_URL/api/datasources" \
        | python3 -m json.tool > "$OUT/datasources.json"; then
    log "exported datasources"
else
    log "WARNING: datasource export failed"
    rm -f "$OUT/datasources.json"
fi

# ---- logical dump of grafana.db ------------------------------------------
# Read-only handle + an explicit read transaction, so we never touch the
# live DB and iterdump sees one consistent snapshot.
DUMP_OK=no
if ssh -o BatchMode=yes "$H20_HOST" "python3 - '$GRAFANA_DB'" > "$OUT/grafana_db.sql" <<'PYEOF'
import sqlite3, sys
path = sys.argv[1]
con = sqlite3.connect('file:%s?mode=ro' % path, uri=True)
con.execute('BEGIN')          # pin a consistent read snapshot
out = sys.stdout
for line in con.iterdump():
    out.write(line)
    out.write('\n')
con.rollback()
con.close()
PYEOF
then
    if [ -s "$OUT/grafana_db.sql" ]; then
        gzip -f "$OUT/grafana_db.sql"
        DUMP_OK=yes
        log "dumped grafana.db ($(du -h --apparent-size \
            "$OUT/grafana_db.sql.gz" | cut -f1))"   # --apparent-size:
        # /dataz is compressed ZFS, so plain du reports allocated
        # blocks and understates the artifact by ~50x.
    else
        log "WARNING: grafana.db dump was empty"
        rm -f "$OUT/grafana_db.sql"
    fi
else
    log "WARNING: grafana.db dump failed (dashboards above are still good)"
    rm -f "$OUT/grafana_db.sql"
fi

# ---- manifest ------------------------------------------------------------
{
    echo "Grafana configuration backup"
    echo "taken_utc:      $STAMP"
    echo "taken_by:       $(whoami)@$(hostname) via tools/ops/backup_grafana.sh"
    echo "grafana_url:    $GRAFANA_URL"
    echo "grafana_host:   $H20_HOST"
    echo "grafana_db:     $GRAFANA_DB"
    echo "n_dashboards:   $N_DASH"
    echo "db_dump:        $DUMP_OK"
    echo "grafana_version: $(curl -sf -u "$GRAFANA_AUTH" \
        "$GRAFANA_URL/api/health" 2>/dev/null \
        | python3 -c 'import json,sys; print(json.load(sys.stdin).get("version","?"))' \
        2>/dev/null || echo '?')"
    echo
    echo "Restore a single dashboard:"
    echo "  curl -u \$GRAFANA_AUTH -H 'Content-Type: application/json' \\"
    echo "       -d \"\$(python3 -c 'import json,sys;"
    echo "            d=json.load(open(sys.argv[1]))[\\\"dashboard\\\"];"
    echo "            d.pop(\\\"id\\\",None);"
    echo "            print(json.dumps({\\\"dashboard\\\":d,\\\"overwrite\\\":True}))' FILE.json)\" \\"
    echo "       $GRAFANA_URL/api/dashboards/db"
    echo
    echo "Restore everything (users, datasource secrets, prefs):"
    echo "  systemctl stop grafana && mv grafana.db grafana.db.old && \\"
    echo "    zcat grafana_db.sql.gz | sqlite3 grafana.db && systemctl start grafana"
    echo
    echo "dashboards captured:"
    ls -1 "$OUT/dashboards" | sed 's/^/  /'
} > "$OUT/MANIFEST.txt"

# The sql dump carries datasource secrets and admin password hashes, so
# match the 0700 the rest of dr_archive uses rather than inheriting the
# umask default.
chmod -R go-rwx "$OUT"
chmod go-rwx "$DEST" 2>/dev/null || true

ln -sfn "$STAMP" "$DEST/latest"

# ---- retention -----------------------------------------------------------
# Keep the newest $KEEP dated dirs. `latest` is a symlink so it is never
# a deletion candidate.
mapfile -t OLD < <(find "$DEST" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
                   | sort -r | tail -n +$((KEEP + 1)))
for d in "${OLD[@]}"; do
    [ -n "$d" ] || continue
    rm -rf "$DEST/$d"
    log "pruned old backup $d"
done

log "done: $N_DASH dashboards, db_dump=$DUMP_OK -> $OUT"
