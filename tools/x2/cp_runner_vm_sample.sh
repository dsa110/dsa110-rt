#!/bin/bash
# READ-ONLY kernel sampler for the x2 checkpoint runner (ledger E5: NUMA balancing vs the ring v4
# mmap readers). Reads /proc/vmstat, /proc/stat and one sysctl; writes nothing on the node.
#
#   ssh <node> 'bash -s <n_samples> <period_s>' < tools/x2/cp_runner_vm_sample.sh > <node>_vm.txt
#
# Output: "# numa_balancing <0|1> host <host>" then one line per sample:
#   <unix> numa_hint_faults numa_hint_faults_local numa_pages_migrated pgfault pgmajfault
#          cpu_user cpu_system cpu_idle   (cumulative counters; /proc/stat jiffies)
N=${1:-60}
P=${2:-5}
echo "# numa_balancing $(cat /proc/sys/kernel/numa_balancing 2>/dev/null) host $(hostname)"
for ((i = 0; i < N; i++)); do
  v=$(awk '$1 == "numa_hint_faults" {a = $2} $1 == "numa_hint_faults_local" {b = $2}
           $1 == "numa_pages_migrated" {c = $2} $1 == "pgfault" {d = $2} $1 == "pgmajfault" {e = $2}
           END {print a+0, b+0, c+0, d+0, e+0}' /proc/vmstat)
  s=$(awk '$1 == "cpu" {print $2 + $3, $4, $5; exit}' /proc/stat)
  echo "$(date +%s.%N) $v $s"
  sleep "$P"
done
