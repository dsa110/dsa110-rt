#!/bin/bash
# Production guard for x2 work on a live SEARCH node: run a command and kill its whole process group
# the moment any search half on this node starts to fall behind real time.
#
#   tools/x2/prod_guard.sh <command...>
#
# Trips on (tailing the live search_compute logs, never re-reading them):
#   - any "lag-recovery seek" (the consumer has lagged past the ring horizon: data is being skipped);
#   - two consecutive cube_progress reports below MIN_RATE cubes/s (cadence 4.97; normal >= 5.0).
# Prints "PRODUCTION_GUARD: ..." and exits 99 when it trips. Refuses to run (exit 98) if the node has
# no search_compute logs to watch (do not run x2 work unguarded on corr nodes).
#
# Why: 2026-10-22 07:25-07:31 an mmap reader of the production RX ring on n13 (kernel minor faults at
# ~2.4 ms each with numa_balancing=1) made search_compute_0 skip 41.5 s of data in 12 lag-recovery seeks.
MIN_RATE=${MIN_RATE:-4.9}
LOGS=$(ls ${GUARD_LOG_GLOB:-/tmp/dsart-rt-children/*search_compute_*.log} 2>/dev/null)
if [ -z "$LOGS" ]; then
  echo "PRODUCTION_GUARD: no search_compute logs on $(hostname); refusing to run unguarded"
  exit 98
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
WPID=$!
wait "$PID"
rc=$?
pkill -P "$WPID" 2>/dev/null
kill "$WPID" 2>/dev/null
wait "$WPID" 2>/dev/null
if [ -s "$TRIP" ]; then
  echo "PRODUCTION_GUARD: tripped on $(hostname): $(cat "$TRIP") -> test killed"
  rm -f "$TRIP"
  exit 99
fi
rm -f "$TRIP"
exit $rc
