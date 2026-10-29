#!/bin/bash
# Production guard for x2 work on a dsart node: run a command and kill its whole process group the
# moment it could hurt production.
#
#   tools/x2/prod_guard.sh <command...>
#
# Two modes, chosen at start:
# - LIVE (a production dsart process runs on the node): tail the live search_compute logs (never
#   re-reading them) and trip on any "lag-recovery seek" or two consecutive cube_progress reports below
#   MIN_RATE cubes/s (cadence 4.97; normal >= 5.0). Refuses (exit 98) if production runs but there are
#   no search_compute logs to watch (corr nodes while observing: never run x2 work there).
# - STOPPED (no production dsart process): run the command; trip as soon as any production process
#   appears (e.g. the user restarts observing), so a restart is never contended (exit 97).
# GUARD_REQUIRE_STOPPED=1 (GPU work) refuses unless the node is in STOPPED mode.
# Production = processes running dsart.services.* or the capture binary whose cwd is NOT a
# /home/ubuntu/x2_scratch copy (x2 tests and benches always run from scratch copies).
#
# Why: 2026-10-22 07:25-07:31 an mmap reader of the production RX ring on n13 (kernel minor faults at
# ~2.4 ms each with numa_balancing=1) made search_compute_0 skip 41.5 s of data in 12 lag-recovery seeks.
MIN_RATE=${MIN_RATE:-4.9}

prod_pids() {
  local pid cwd
  for pid in $(pgrep -f "${GUARD_PROD_PATTERN:-dsart[.]services[.]|dsart_capture_manythread}"); do
    cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null)
    case "$cwd" in /home/ubuntu/x2_scratch*) continue ;; esac
    echo "$pid"
  done
}

if [ -n "$(prod_pids)" ]; then
  MODE=live
  if [ "${GUARD_REQUIRE_STOPPED:-0}" = 1 ]; then
    echo "PRODUCTION_GUARD: production is running on $(hostname); this run needs a stopped node, refusing"
    exit 98
  fi
  LOGS=$(ls ${GUARD_LOG_GLOB:-/tmp/dsart-rt-children/*search_compute_*.log} 2>/dev/null)
  if [ -z "$LOGS" ]; then
    echo "PRODUCTION_GUARD: production runs on $(hostname) but no search_compute logs; refusing to run unguarded"
    exit 98
  fi
else
  MODE=stopped
fi

setsid "$@" &
PID=$!
WPID=
TRIP=$(mktemp)
# The command runs in its own session, so a signal to this script (timeout, ssh hang-up, ^C) would
# otherwise orphan it, still running and no longer guarded: kill its process group first.
# (2026-10-22: a `timeout` around this script left a pytest running on n13 after the timeout fired.)
_stop() {
  kill -9 -- -"$PID" 2>/dev/null
  if [ -n "$WPID" ]; then pkill -P "$WPID" 2>/dev/null; kill "$WPID" 2>/dev/null; fi
  rm -f "$TRIP"
}
trap '_stop; echo "PRODUCTION_GUARD: interrupted on $(hostname) -> test killed"; exit 143' TERM INT HUP
if [ "$MODE" = live ]; then
  (
    tail -n0 -F $LOGS 2>/dev/null | awk -v min="$MIN_RATE" '
      /lag-recovery seek/ { print "lag-recovery seek: " substr($0, 1, 160); fflush(); exit }
      /cubes\/s last/ {
        s = $0; sub(/ cubes\/s last.*/, "", s); sub(/.*\(/, "", s)
        if (s + 0 < min) { low++ } else { low = 0 }
        if (low >= 2) { print "cube rate " s " < " min " twice: " substr($0, 1, 60); fflush(); exit }
      }' > "$TRIP"
    if [ -s "$TRIP" ]; then
      kill -9 -- -"$PID" 2>/dev/null
    fi
  ) &
else
  (
    while kill -0 "$PID" 2>/dev/null; do
      p=$(prod_pids | head -3 | tr '\n' ' ')
      if [ -n "$p" ]; then
        echo "production started (pids $p)" > "$TRIP"
        kill -9 -- -"$PID" 2>/dev/null
        break
      fi
      sleep 2
    done
  ) 2>/dev/null &
fi
WPID=$!
wait "$PID"
rc=$?
pkill -P "$WPID" 2>/dev/null
kill "$WPID" 2>/dev/null
wait "$WPID" 2>/dev/null
if [ -s "$TRIP" ]; then
  echo "PRODUCTION_GUARD: tripped on $(hostname) [$MODE]: $(cat "$TRIP") -> test killed"
  rm -f "$TRIP"
  [ "$MODE" = live ] && exit 99
  exit 97
fi
rm -f "$TRIP"
exit $rc
