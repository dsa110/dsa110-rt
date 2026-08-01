#!/bin/bash
#
# Provision a DSA-110 corr or search node from a bare MaaS deploy.
#
# Cloud-init handles only what must happen before first boot (packages,
# user, NTP, sysctl, netplan/MTU). Everything else lives here, because a
# cloud-init per-once hook runs exactly once, is awkward to re-run, and
# fails into a log nobody reads. This script is idempotent and can be
# run by hand against an existing node to repair drift.
#
# ALWAYS DRY-RUN FIRST:
#     ./provision_node.sh --role corr --dry-run
#
# Nothing destructive happens without --apply. There is no default-apply
# mode on purpose: this runs as root against a node that may be carrying
# live buffers.
#
# Usage:
#   provision_node.sh --role {corr|search} [--apply] [--dry-run]
#                     [--only STAGE[,STAGE...]] [--skip STAGE[,STAGE...]]
#                     [--list-stages] [--log FILE]
#
set -eo pipefail
# NOTE: deliberately no `set -u`. The conda activation hooks reference
# MKL_INTERFACE_LAYER unset and would abort the run (PARALLEL_AGENTS.md §6).
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU,LP64}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/versions.env"

ROLE=""
APPLY="no"
ONLY=""
SKIP=""
LOG=""

# Order matters: repos before conda (the dsa110-rt env is built from the
# repo's own envs/dsa110-rt.yml), and builds before dsart (dsart's C
# extensions link against the PSRDADA installed by the builds stage).
STAGES=(preflight tuning mellanox cuda repos conda builds dsart role verify)

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --role)        ROLE="$2"; shift 2 ;;
        --apply)       APPLY="yes"; shift ;;
        --dry-run)     APPLY="no"; shift ;;
        --only)        ONLY="$2"; shift 2 ;;
        --skip)        SKIP="$2"; shift 2 ;;
        --log)         LOG="$2"; shift 2 ;;
        --list-stages) printf '%s\n' "${STAGES[@]}"; exit 0 ;;
        -h|--help)     usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

[[ "$ROLE" == "corr" || "$ROLE" == "search" ]] || {
    echo "--role must be 'corr' or 'search'" >&2; exit 2; }

if [[ -n "$LOG" ]]; then exec > >(tee -a "$LOG") 2>&1; fi

# ---------------------------------------------------------------------
# Execution harness. Every mutating action in every lib goes through
# run() / run_sh(), so --dry-run is total rather than best-effort: a
# stage cannot accidentally mutate the node because someone called a
# command directly.
# ---------------------------------------------------------------------
C_OK=$'\033[32m'; C_SKIP=$'\033[90m'; C_ACT=$'\033[36m'; C_WARN=$'\033[33m'
C_ERR=$'\033[31m'; C_OFF=$'\033[0m'
[[ -t 1 ]] || { C_OK=""; C_SKIP=""; C_ACT=""; C_WARN=""; C_ERR=""; C_OFF=""; }

_N_WOULD=0; _N_RAN=0; _N_SKIP=0; _N_WARN=0

log()   { printf '%s\n' "$*"; }
stage() { printf '\n%s=== %s %s%s\n' "$C_ACT" "$1" "$(printf '=%.0s' $(seq 1 $((60-${#1}))))" "$C_OFF"; }
ok()    { printf '  %s[ ok ]%s %s\n' "$C_OK" "$C_OFF" "$*"; _N_SKIP=$((_N_SKIP+1)); }
warn()  { printf '  %s[warn]%s %s\n' "$C_WARN" "$C_OFF" "$*"; _N_WARN=$((_N_WARN+1)); }
die()   { printf '  %s[fail]%s %s\n' "$C_ERR" "$C_OFF" "$*"; exit 1; }

# run <description> -- <command...>
run() {
    local desc="$1"; shift
    [[ "$1" == "--" ]] && shift
    if [[ "$APPLY" == "yes" ]]; then
        printf '  %s[ run]%s %s\n' "$C_ACT" "$C_OFF" "$desc"
        "$@" || die "failed: $desc"
        _N_RAN=$((_N_RAN+1))
    else
        printf '  %s[would]%s %s\n' "$C_SKIP" "$C_OFF" "$desc"
        printf '         %s$ %s%s\n' "$C_SKIP" "$*" "$C_OFF"
        _N_WOULD=$((_N_WOULD+1))
    fi
}

# run_sh <description> <shell-snippet>   (for pipes / redirects / heredocs)
run_sh() {
    local desc="$1" snippet="$2"
    if [[ "$APPLY" == "yes" ]]; then
        printf '  %s[ run]%s %s\n' "$C_ACT" "$C_OFF" "$desc"
        bash -eo pipefail -c "$snippet" || die "failed: $desc"
        _N_RAN=$((_N_RAN+1))
    else
        printf '  %s[would]%s %s\n' "$C_SKIP" "$C_OFF" "$desc"
        printf '%s\n' "$snippet" | sed "s/^/         $C_SKIP| /;s/$/$C_OFF/"
        _N_WOULD=$((_N_WOULD+1))
    fi
}

# Reads are always allowed, in both modes -- a dry run that cannot
# inspect the node cannot tell you what it would skip.
have()      { command -v "$1" >/dev/null 2>&1; }
is_root()   { [[ "$(id -u)" -eq 0 ]]; }

# Which interface carries the 10.41 data fabric. Memoized and lazy so
# that stages invoked on their own via --only still resolve it: it used
# to be exported by preflight, so `--only role` saw an empty value and
# reported a missing interface that was actually there.
data_iface() {
    if [[ -z "$DATA_IFACE" ]]; then
        DATA_IFACE="$(ip -br -4 addr show 2>/dev/null \
                      | awk -v s="$DATA_SUBNET" '$3 ~ s {print $1; exit}')"
        export DATA_IFACE
    fi
    printf '%s' "$DATA_IFACE"
}
want_stage() {
    local s="$1"
    [[ -n "$ONLY" && ",$ONLY," != *",$s,"* ]] && return 1
    [[ -n "$SKIP" && ",$SKIP," == *",$s,"* ]] && return 1
    return 0
}

export -f log ok warn die run run_sh have is_root 2>/dev/null || true

# shellcheck disable=SC1090
for f in "$HERE"/lib/*.sh; do source "$f"; done

# ---------------------------------------------------------------------
log "DSA-110 node provisioning"
log "  role     : $ROLE"
log "  mode     : $([[ "$APPLY" == yes ]] && echo 'APPLY (will modify this node)' || echo 'DRY-RUN (no changes)')"
log "  host     : $(hostname) [$(hostname -f 2>/dev/null || echo '-')]"
log "  maas     : $WEB_MAAS"
[[ -n "$ONLY" ]] && log "  only     : $ONLY"
[[ -n "$SKIP" ]] && log "  skip     : $SKIP"

if [[ "$APPLY" == "yes" ]] && ! is_root; then
    die "--apply needs root (sudo). Dry-run works unprivileged."
fi

for s in "${STAGES[@]}"; do
    want_stage "$s" || continue
    stage "$s"
    "stage_${s}"
done

printf '\n%s=== summary %s%s\n' "$C_ACT" "$(printf '=%.0s' $(seq 1 52))" "$C_OFF"
if [[ "$APPLY" == "yes" ]]; then
    log "  actions run      : $_N_RAN"
else
    log "  actions pending  : $_N_WOULD   (re-run with --apply to execute)"
fi
log "  already correct  : $_N_SKIP"
log "  warnings         : $_N_WARN"
[[ $_N_WARN -gt 0 ]] && log "  ${C_WARN}review the warnings above before --apply${C_OFF}"
exit 0
