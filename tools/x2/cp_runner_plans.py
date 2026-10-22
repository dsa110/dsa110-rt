"""Checkpoint plans for tools/x2/cp_runner.py (plan rev 4 §CP1; runbook docs/x2/CP1_RUNBOOK.md).

Commands use ``@{var}`` runner variables (cp_runner.build_ctx): repo, tag, scratch (node scratch
copy), state (h23 results dir of this checkpoint tag), py (node env), py_h23, corr_node,
search_node, live_format, format (set at the S1 sign-off), node lists.

Steps that call scripts other workstreams still owe are placeholders (``requires``): the dry run
prints them, a real run refuses them. Their command lines are the CONTRACT those scripts must
implement (subcommands, --out, exit codes; see the cp_runner docstring).
"""

from __future__ import annotations

from cp_runner import CORR_NODES, PROD_PROC_RE, SEARCH_NODES, Command, Plan, Step

#: S1 decision inputs (docs/x2/phase0/SUMMARY.md, b79c5d4): "total vs model" with the S2 choices
#: and the clean ingress per search node needed at <= 80% load. Gridded ns2 cint5 (not a plan
#: row) needs ~305k datagrams/s on the worst node.
FORMAT_TABLE: dict[str, dict] = {
    "2x-gridded-ns2-cint4": {"kind": "gridded", "total": 0.996, "need_gbps": 16.4},
    "2x-gridded-ns4-cint5": {"kind": "gridded", "total": 0.997, "need_gbps": 17.2},
    "2x-gridded-ns4-cint4": {"kind": "gridded", "total": 0.994, "need_gbps": 14.0},
    "2x-baselines-ns4-cint5": {"kind": "baselines", "total": 0.995, "need_gbps": 14.0},
    "2x-baselines-ns4-cint4": {"kind": "baselines", "total": 0.993, "need_gbps": 11.4},
    "2x-baselines-ns2-cint4": {"kind": "baselines", "total": 0.990, "need_gbps": 11.5},
}
#: Capacity sweep rates per search node (Gb/s): through the largest need (17.2) with margin.
SWEEP_RATES_GBPS = "9,11,13,15,16,17,18"

_RUN = "cd @{scratch} && PYTHONPATH=@{scratch}/src DSART_CONFIG_DIR=@{scratch}/configs @{py}"
_BENCH_CORR = "bench/timeres/b8_corr2x.py"
_BENCH_OWNER = "bench/timeres/b8_owner2x.py"
_FAKE = "bench/fake_corr_to_search.py"
_FAKE_V2 = _FAKE + "::prod_frame2|frame.?version"
_FAKE_C = _FAKE + "::send_?mmsg"
_SAMPLER = "@{repo}/tools/x2/cp_runner_nic_sample.sh"
_VM_SAMPLER = "@{repo}/tools/x2/cp_runner_vm_sample.sh"     # ledger E5 (NUMA hint faults, sys time)

_KILL_MARKED = (
    "me=$$; for p in $(pgrep -u \"$(id -u)\" .); do [ \"$p\" = \"$me\" ] && continue; "
    "tr '\\0' '\\n' < /proc/$p/environ 2>/dev/null | grep -qx 'DSART_X2_CP=@{tag}' && kill $p 2>/dev/null; "
    "done; sleep 3; for p in $(pgrep -u \"$(id -u)\" .); do [ \"$p\" = \"$me\" ] && continue; "
    "tr '\\0' '\\n' < /proc/$p/environ 2>/dev/null | grep -qx 'DSART_X2_CP=@{tag}' && kill -9 $p 2>/dev/null; "
    "done; rm -f /dev/shm/dsart-x2-*; true")
_NO_MARKED = (
    "for p in $(pgrep -u \"$(id -u)\" .); do tr '\\0' '\\n' < /proc/$p/environ 2>/dev/null "
    "| grep -qx 'DSART_X2_CP=@{tag}' && { echo \"x2 test process $p still running\"; exit 1; }; done; "
    "ls /dev/shm/dsart-x2-* 2>/dev/null && { echo 'x2 test shm left'; exit 1; }; true")
_GPU_FREE_OF_TESTS = (
    "for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do "
    "tr '\\0' '\\n' < /proc/$p/environ 2>/dev/null | grep -qx 'DSART_X2_CP=@{tag}' "
    "&& { echo \"x2 test process $p on a GPU\"; exit 1; }; done; true")
_GPU_IDLE = "test -z \"$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)\""
_DADA_CLEAN = (
    f"if ! pgrep -f '{PROD_PROC_RE}' > /dev/null; then for k in dada eada fada bada; do "
    "dada_db -d -k $k > /dev/null 2>&1 || true; done; fi; true")
_CHECKOUT = ("cd /home/ubuntu/proj/dsa110-rt && echo $(git rev-parse HEAD) "
             "$(git --no-optional-locks status --porcelain --untracked-files=no | wc -l)")

ALL_NODES = CORR_NODES + SEARCH_NODES


def _sampler_bg(nodes: str, n: int, period: int, outdir: str, script: str = _SAMPLER) -> str:
    """Local: start a read-only sampler (NIC by default) on every node in the background; PIDs in a
    file of ``outdir`` (one sampler kind per directory)."""
    return (f"mkdir -p {outdir} && : > {outdir}/.pids && for n in {nodes}; do "
            f"ssh -o BatchMode=yes $n.pro.pvt 'bash -s {n} {period}' < {script} > {outdir}/$n.txt & "
            f"echo $! >> {outdir}/.pids; done")


def _sampler_wait(outdir: str) -> str:
    return f"for p in $(cat {outdir}/.pids); do wait $p 2>/dev/null || true; done; true"


def _stage_cmds(node: str) -> list[Command]:
    return [
        Command("test -d @{scratch} || { mkdir -p /home/ubuntu/x2_scratch && "
                "nice -n 19 ionice -c3 cp -a /home/ubuntu/proj/dsa110-rt @{scratch}; }",
                node, 900, mark=False, group=node),
        Command("rsync -a --exclude='*.so' --exclude=__pycache__ --exclude='*.pdf' --exclude='*.png' "
                "@{repo}/src @{repo}/tests @{repo}/tools @{repo}/configs @{repo}/bench @{repo}/docs "
                f"@{{repo}}/setup.py {node}.pro.pvt:@{{scratch}}/", None, 600, mark=False, group=node),
        Command("cd @{scratch} && @{py} setup.py -q build_ext --inplace > /dev/null && "
                "PYTHONPATH=@{scratch}/src @{py} -c \"import dsart,sys; "
                "sys.exit(0 if dsart.__file__.startswith('@{scratch}/') else 3)\"",
                node, 900, mark=False, group=node),
    ]


def _cp1_pre() -> list[Step]:
    t = "pre"
    return [
        Step("pre.repo_state", t, "x2/main checkout clean; sha recorded", 1, [
            Command("mkdir -p @{state}/pre && cd @{repo} && git rev-parse HEAD | tee @{state}/pre/repo_sha.txt "
                    "&& test -z \"$(git status --porcelain --untracked-files=no)\"", None, 60, mark=False)],
            gate="clean checkout of the commit under test"),
        Step("pre.static_checks", t, "pyflakes + syntax of everything that will run (no bytecode written)", 2, [
            Command("cd @{repo} && @{pyflakes} src/dsart/x2 tools/x2/cp_runner.py tools/x2/cp_runner_plans.py "
                    "tools/ops/push_dsart_to_etcd.py tools/dashboard/dsa_monitor/opoint_switch.py && "
                    "@{py_h23} -c \"import ast,pathlib; [ast.parse(p.read_text()) for d in ('src','bench','tools') "
                    "for p in pathlib.Path(d).rglob('*.py')]\"", None, 300, mark=False)],
            gate="no pyflakes findings in x2 code; every .py parses"),
        Step("pre.compile_only", t, "O4: compile-only checks of the CUDA/Triton kernels on h23's CPU", 5, [
            Command(f"cd @{{repo}} && CUDA_VISIBLE_DEVICES= @{{py_h23}} {_BENCH_CORR} compile-check", None, 900,
                    mark=False),
            Command(f"cd @{{repo}} && CUDA_VISIBLE_DEVICES= @{{py_h23}} {_BENCH_OWNER} compile-check", None, 900,
                    mark=False)],
            gate="every kernel compiles; CPU twins / TRITON_INTERPRET pass (plan entry criteria)",
            requires=(_BENCH_CORR, _BENCH_OWNER), owner="WS-C / WS-S"),
        Step("pre.cpu_tests", t, "all x2 CPU tests on a search node (node_cpu_tests.sh, under prod_guard)", 25, [
            Command("cd @{repo} && tools/x2/node_cpu_tests.sh -n n13.pro.pvt -t @{tag}_pre -- "
                    "$(ls tests/x2/test_*.py)", None, 3600, mark=False)],
            gate="green; a production-guard trip aborts the run (exit 4)"),
        Step("pre.etcd_snapshot", t, "read-only snapshot of the live etcd configs, /cnf/opoint and injection K", 1, [
            Command("@{py_h23} @{repo}/tools/x2/cp_runner.py check snapshot --out @{state}/pre/etcd_snapshot.json",
                    None, 120, mark=False)],
            gate="/cnf/opoint is 1x (or unset); restore compares against this"),
        Step("pre.node_checkouts", t, "read-only: production checkout sha + dirty count on all 20 nodes", 2, [
            Command(f"for n in @{{all_nodes}}; do echo $n $(ssh -o BatchMode=yes $n.pro.pvt '{_CHECKOUT}'); "
                    "done | tee @{state}/pre/node_checkouts.txt", None, 300, mark=False)],
            gate="recorded; restore requires it unchanged"),
        Step("pre.search_health_1x", t, "read-only: 1x search health baseline (lag seeks, cube rate, scatter)", 3, [
            Command("@{py_h23} @{repo}/tools/x2/cp_runner.py check search-health --seconds 120 "
                    "--out @{state}/pre/search_health_1x.json", None, 300, mark=False)],
            gate="0 lag-recovery seeks, cube rate >= 4.9/s, scatter mean <= 90 ms on all 8 halves",
            optional=True),
        Step("pre.vm_baseline_1x", t, "read-only: 1x NUMA hint faults + system time on the search nodes "
             "(ledger E5 baseline, 2 min)", 3, [
            Command(_sampler_bg("@{search_nodes}", 25, 5, "@{state}/pre/vm_1x", _VM_SAMPLER) + "; "
                    + _sampler_wait("@{state}/pre/vm_1x") + "; @{py_h23} @{repo}/tools/x2/cp_runner.py check "
                    "vm-summary --dir @{state}/pre/vm_1x --nodes @{search_nodes_csv} "
                    "--out @{state}/pre/e5_numa_1x.json", None, 300, mark=False)],
            gate="recorded (today ~3,100 hint faults/s per node with numa_balancing=1)", optional=True),
        Step("pre.nic_baseline_1x", t, "read-only: 1x NIC counters on all 20 nodes while observing (5 min)", 6, [
            Command(_sampler_bg("@{all_nodes}", 61, 5, "@{state}/pre/nic_1x") + "; "
                    + _sampler_wait("@{state}/pre/nic_1x"), None, 600, mark=False)],
            gate="per-node loss / pps / CRC baseline for the relative gates (same-window 1x rates)"),
    ]


def _cp1_window() -> list[Step]:
    t = "window"
    stage = [c for n in ALL_NODES for c in _stage_cmds(n)]
    return [
        Step("window.stop_observing", t, "the user stops observing; the runner verifies READY", 5, [
            Command("@{py_h23} @{repo}/tools/x2/cp_runner.py check fleet-stopped", None, 60, mark=False)],
            manual="web UI Control tab: Stop (confirm=stop). Wait until the banner reads READY "
                   "(orchestrators up, nothing running). Do not stop the orchestrators themselves.",
            gate="etcd READY, no search orchestrator running"),
        Step("window.gpus_idle", t, "no process on any GPU of the 20 nodes", 1, [
            Command(_GPU_IDLE, n, 60, mark=False) for n in ALL_NODES],
            gate="nvidia-smi lists no compute app", parallel=True),
        Step("window.stage_scratch", t, "scratch copy of each node checkout + this commit + C build", 15, stage,
             touches=("host",), parallel=True,
             gate="dsart imports from @{scratch} on all 20 nodes; production checkout untouched"),
    ]


def _cp1_corr() -> list[Step]:
    t, h = "corr", "@{corr_node}"
    out = "@{scratch}/cp_results/corr"
    return [
        Step("corr.env_check", t, "GPU env on the corr node (torch, triton, cupy from the scratch copy)", 5, [
            Command(f"mkdir -p {out} && {_RUN} -c \"import torch, dsart; "
                    "assert dsart.__file__.startswith('@{scratch}/'); print(torch.__version__, "
                    "torch.cuda.device_count())\" && nvidia-smi -L", h, 300)],
            touches=("gpu",), gate="2 GPUs visible; dsart from scratch"),
        Step("corr.prod_profile_golden", t, "production 1x stage profile + golden capture (>= 64 blocks)", 30, [
            Command(f"{_RUN} {_BENCH_CORR} profile-1x --out {out}", h, 1500),
            Command(f"{_RUN} {_BENCH_CORR} golden-capture --blocks 64 --out {out}/golden", h, 1500)],
            touches=("gpu", "shm"), requires=(_BENCH_CORR,), owner="WS-C",
            gate="stage timings recorded; golden blocks incl. RFI, static-sky warm-up, array-burst and "
                 "injection blocks"),
        Step("corr.kernels_2x", t, "2x stage-1 / GEMM variants: both maps, n_sub 2 and 4", 25, [
            Command(f"{_RUN} {_BENCH_CORR} kernels --maps gridded,baselines --n-sub 2,4 --check-ref "
                    f"--out {out}", h, 2400)],
            touches=("gpu",), requires=(_BENCH_CORR,), owner="WS-C",
            gate="exact vs the CPU reference (fp32 bit-exact, GEMM <= 1e-5 rel then 1 fp16 ULP); peak memory; "
                 "GEMM + stage-1 <= 55 ms else escalation E2"),
        Step("corr.compat_1x", t, "1x-compat corr taps vs the golden capture", 20, [
            Command(f"{_RUN} {_BENCH_CORR} compat-1x --golden {out}/golden --out {out}", h, 1800)],
            touches=("gpu",), requires=(_BENCH_CORR,), owner="WS-C",
            gate="torch.equal per stage, byte-equal datagrams over >= 64 blocks"),
        Step("corr.live_2x", t, "corr 2x LIVE on one node (capture, corr_slow, voltage_retention; TX to a drain)", 30, [
            Command(f"mkdir -p {out}/nic && (bash @{{scratch}}/tools/x2/cp_runner_nic_sample.sh 290 5 "
                    f"> {out}/nic/@{{corr_node}}.txt &) && {_RUN} {_BENCH_CORR} live --opoint @{{live_format}} "
                    f"--minutes 20 --tx drain --reader-identity corr_fast --out {out}", h, 2700)],
            touches=("gpu", "shm", "nic", "services"), confirm=True, requires=(_BENCH_CORR,), owner="WS-C",
            gate="block time mean <= 85% of 134.218 ms, p99 <= 100%, zero fada/eada overruns; "
                 "SNAP NIC discards <= the 1x rate (corr.nic_gate)"),
        Step("corr.collect", t, "pull the corr results to h23", 2, [
            Command(f"mkdir -p @{{state}}/corr && rsync -a {h}.pro.pvt:{out}/ @{{state}}/corr/", None, 600,
                    mark=False)]),
        Step("corr.nic_gate", t, "relative SNAP NIC-discard gate (2x live vs 1x baseline)", 1, [
            Command("@{py_h23} @{repo}/tools/x2/cp_runner.py check nic-gate --baseline-dir @{state}/pre/nic_1x "
                    "--test-dir @{state}/corr/nic --nodes @{corr_node} --out @{state}/corr/nic_gate.json",
                    None, 60, mark=False)],
            requires=(_BENCH_CORR,), owner="WS-C", gate="discard + out-of-buffer fraction <= 1x + 1e-5"),
    ]


def _cp1_search() -> list[Step]:
    t, h = "search", "@{search_node}"
    out = "@{scratch}/cp_results/search"
    return [
        Step("search.env_check", t, "GPU env on the search node", 5, [
            Command(f"mkdir -p {out} && {_RUN} -c \"import torch, dsart; "
                    "assert dsart.__file__.startswith('@{scratch}/'); print(torch.__version__, "
                    "torch.cuda.device_count())\" && nvidia-smi -L", h, 300)],
            touches=("gpu",), gate="2 GPUs visible; dsart from scratch"),
        Step("search.kernels", t, "FFT variants, K3b boxcar + B-splines (b3x), K1 per format", 40, [
            Command(f"{_RUN} {_BENCH_OWNER} kernels --fft cufft,warp --k3b box,b3x --k1-formats all "
                    f"--check-1x-csr --out {out}", h, 3600)],
            touches=("gpu",), requires=(_BENCH_OWNER,), owner="WS-S",
            gate="K1 1x CSR bit-exact vs combine_sparse_v4; FFT |d| <= 2e-3 sigma vs float64; K3b compat "
                 "|dS/N| <= 0.02; us/image per stage (E1 if the chain > 4.8 us/image)"),
        Step("search.compat_1x", t, "1x-compat search taps", 20, [
            Command(f"{_RUN} {_BENCH_OWNER} compat-1x --out {out}", h, 1800)],
            touches=("gpu",), requires=(_BENCH_OWNER,), owner="WS-S",
            gate="1x-compat equal to production on the recorded taps"),
        Step("search.engine_synthetic", t, "owner engine on both GPUs from a synthetic ring", 15, [
            Command(f"(bash @{{scratch}}/tools/x2/cp_runner_vm_sample.sh 130 5 > {out}/vm_@{{search_node}}.txt &) "
                    f"&& {_RUN} {_BENCH_OWNER} engine --gpus 0,1 --input synthetic-ring --opoint @{{live_format}} "
                    f"--minutes 10 --out {out}", h, 1500)],
            touches=("gpu", "shm"), requires=(_BENCH_OWNER,), owner="WS-S",
            gate="GPU duty <= 0.75 per half (aim 0.6), bounded lag, zero ring overruns, host <= 30 ms/cycle; "
                 "E5: NUMA hint faults/s, system time and scatter recorded (vm_<node>.txt + the bench JSON)"),
        Step("search.collect", t, "pull the search results to h23", 2, [
            Command(f"mkdir -p @{{state}}/search && rsync -a {h}.pro.pvt:{out}/ @{{state}}/search/", None, 600,
                    mark=False)]),
    ]


def _cp1_fleet() -> list[Step]:
    t = "fleet"
    ipf = "@{state}/fleet/99-dsart-ipfrag.conf"
    keys = f"$(grep -Ev '^[[:space:]]*(#|$)' {ipf} | cut -d= -f1 | tr -d ' ')"
    drv = "--tx-nodes @{corr_nodes_csv} --rx-nodes @{search_nodes_csv} --scratch @{scratch} --marker @{tag}"
    return [
        Step("fleet.n01_ipfrag", t, "install n02's 99-dsart-ipfrag.conf on n01 + sysctl --system "
             "(the ONLY planned host change)", 10, [
            Command("sysctl net.ipv4.ipfrag_high_thresh net.ipv4.ipfrag_low_thresh net.ipv4.ipfrag_time "
                    "net.ipv4.ipfrag_max_dist; ls -l /etc/sysctl.d/", "n01", 60, mark=False),
            Command(f"mkdir -p @{{state}}/fleet && ssh -o BatchMode=yes n02.pro.pvt "
                    f"cat /etc/sysctl.d/99-dsart-ipfrag.conf > {ipf} && test -s {ipf} && cat {ipf}",
                    None, 60, mark=False),
            Command(f"scp -q -o BatchMode=yes {ipf} n01.pro.pvt:/tmp/99-dsart-ipfrag.conf", None, 60, mark=False),
            Command("{ test ! -e /etc/sysctl.d/99-dsart-ipfrag.conf || "
                    "cmp -s /tmp/99-dsart-ipfrag.conf /etc/sysctl.d/99-dsart-ipfrag.conf; } && "
                    "sudo -n install -m 0644 -o root -g root /tmp/99-dsart-ipfrag.conf "
                    "/etc/sysctl.d/99-dsart-ipfrag.conf && sudo -n sysctl --system > /dev/null", "n01", 120,
                    mark=False),
            Command(f"keys={keys}; diff <(ssh -o BatchMode=yes n02.pro.pvt sysctl $keys) "
                    "<(ssh -o BatchMode=yes n01.pro.pvt sysctl $keys)", None, 60, mark=False)],
            touches=("host", "nic"), confirm=True,
            gate="every key of the file reads the same on n01 as on n02; the previous n01 values are in the "
                 "step log (revert: sudo sysctl -w <key>=<old> and remove the file)"),
        Step("fleet.nic_detect", t, "read-only: the active data interface of every node (n14: f1)", 1, [
            Command(f"for n in @{{all_nodes}}; do echo $n $(ssh -o BatchMode=yes $n.pro.pvt 'bash -s 1 0' "
                    f"< {_SAMPLER} | head -1); done | tee @{{state}}/fleet/ifaces.txt", None, 300, mark=False)],
            gate="interfaces recorded (samplers and benches read them from the node, never hard-code)"),
        Step("fleet.capacity_sweep", t, "corner-turn capacity with v2 single-fragment frames, 16 corr -> 4 search, "
             f"{SWEEP_RATES_GBPS} Gb/s per search node, NIC counters + pps", 35, [
            Command(_sampler_bg("@{all_nodes}", 400, 5, "@{state}/fleet/nic_sweep") + "; "
                    f"@{{py_h23}} @{{repo}}/{_FAKE} sweep --frame-version 2 {drv} --rates-gbps {SWEEP_RATES_GBPS} "
                    "--dwell-s 120 --record-pps --nic-counters --out @{state}/fleet/capacity.json; rc=$?; "
                    + _sampler_wait("@{state}/fleet/nic_sweep") + "; exit $rc", None, 2700)],
            touches=("nic", "shm"), guard_hosts=ALL_NODES, requires=(_FAKE_V2,), owner="WS-T",
            gate="per node: clean = loss <= max(1e-4, that node's 1x loss) and no NIC-discard increase for the "
                 "dwell; record Gb/s AND datagrams/s (gridded ns2 cint5 needs ~305k/s); "
                 "capacity.json has clean_gbps_per_node"),
        Step("fleet.format_choice", t, "S1: choose the corner-turn format from the measured clean capacity", 5, [
            Command("@{py_h23} @{repo}/tools/x2/cp_runner.py check format-table "
                    "--capacity @{state}/fleet/capacity.json", None, 60, mark=False)],
            choose_format=True,
            gate="highest 'total' whose need <= measured clean ingress per node and within the GPU-duty aim; "
                 "gridded on ties; the user signs off S1 (none fits -> E4)"),
        Step("fleet.plan_build", t, "build + self-verify the v3 DM plan of the chosen format", 2, [
            Command("@{py_h23} @{repo}/tools/build_dm_plan_tiers.py --opoint @{format} "
                    "--out @{state}/fleet/plan_@{format}.npz", None, 300, mark=False)],
            gate="plan_id recorded; tables re-verified against a rebuild"),
        Step("fleet.plan_distribute", t, "copy the plan into every node's scratch copy", 2, [
            Command(f"scp -q -o BatchMode=yes @{{state}}/fleet/plan_@{{format}}.npz {n}.pro.pvt:@{{scratch}}/cp_plan.npz",
                    None, 120, mark=False, group=n) for n in ALL_NODES],
            parallel=True),
        Step("fleet.search_full_load", t, "search real time through the real RX path: v2 synthetic UDP at the "
             "chosen load +10% into search_rx v4 + owner engines on all 8 halves", 45, [
            Command(_sampler_bg("@{search_nodes}", 400, 5, "@{state}/fleet/nic_full_load") + "; "
                    + _sampler_bg("@{search_nodes}", 400, 5, "@{state}/fleet/vm_full_load", _VM_SAMPLER) + "; "
                    f"@{{py_h23}} @{{repo}}/{_FAKE} fleet-load --format @{{format}} --load-factor 1.1 "
                    f"--plan @{{scratch}}/cp_plan.npz {drv} --engine '{_BENCH_OWNER} engine --input rx-v4' "
                    "--minutes 30 --out @{state}/fleet/full_load.json; rc=$?; "
                    + _sampler_wait("@{state}/fleet/nic_full_load") + "; "
                    + _sampler_wait("@{state}/fleet/vm_full_load") + "; exit $rc", None, 3600),
            Command("@{py_h23} @{repo}/tools/x2/cp_runner.py check vm-summary --dir @{state}/fleet/vm_full_load "
                    "--nodes @{search_nodes_csv} --out @{state}/fleet/e5_numa.json", None, 120, mark=False)],
            touches=("gpu", "shm", "nic", "services"), guard_hosts=ALL_NODES,
            requires=(_FAKE_V2, _BENCH_OWNER), owner="WS-T / WS-S",
            gate="per half GPU duty <= 0.75 (aim 0.6), bounded lag, zero ring overruns, fan-in >= 99.9% of "
                 "cubes per DM, loss <= max(1e-4, 1x), host <= 30 ms/cycle; E5 (measure only): NUMA hint "
                 "faults/s and system-time fraction per search node vs the 1x ~3,100 faults/s, plus scatter"),
        Step("fleet.c_sender_snap_live", t, "C sendmmsg sender at the chosen per-corr load with the SNAPs live", 15, [
            Command(_sampler_bg("@{corr_nodes}", 130, 5, "@{state}/fleet/nic_snap_live") + "; "
                    f"@{{py_h23}} @{{repo}}/{_FAKE} snap-live --sender c --format @{{format}} {drv} "
                    "--minutes 10 --out @{state}/fleet/snap_live.json; rc=$?; "
                    + _sampler_wait("@{state}/fleet/nic_snap_live") + "; exit $rc", None, 1200),
            Command("@{py_h23} @{repo}/tools/x2/cp_runner.py check nic-gate --baseline-dir @{state}/pre/nic_1x "
                    "--test-dir @{state}/fleet/nic_snap_live --nodes @{corr_nodes_csv} "
                    "--out @{state}/fleet/snap_nic_gate.json", None, 120, mark=False)],
            touches=("nic", "services"), guard_hosts=CORR_NODES + SEARCH_NODES, confirm=True,
            requires=(_FAKE_C,), owner="WS-T",
            gate="SNAP NIC discard rate with 2x TX <= the 1x rate per corr node (relative gate)"),
        Step("fleet.collect", t, "pull node-side results to h23", 3, [
            Command(f"mkdir -p @{{state}}/nodes/{n} && rsync -a {n}.pro.pvt:@{{scratch}}/cp_results/ "
                    f"@{{state}}/nodes/{n}/ 2>/dev/null; true", None, 300, mark=False, group=n)
            for n in ALL_NODES], parallel=True),
    ]


def _teardown(track: str, nodes: tuple[str, ...] | list[str], *, gpu: bool, dada: bool = False,
              bench: str = "") -> list[Step]:
    cmds: list[Command] = []
    for n in nodes:
        if bench:
            cmds.append(Command(f"test -f @{{scratch}}/{bench} && {_RUN} {bench} teardown; true", n, 300,
                                mark=False, group=n))
        cmds.append(Command(_KILL_MARKED, n, 120, mark=False, group=n))
        if dada:
            cmds.append(Command(_DADA_CLEAN, n, 120, mark=False, group=n))
        cmds.append(Command(_NO_MARKED + ("; " + _GPU_FREE_OF_TESTS if gpu else ""), n, 60, mark=False,
                            group=n))
    return [Step(f"{track}.teardown", track, f"teardown {track}: kill this run's processes, remove test shm"
                 + (", destroy leftover PSRDADA buffers if no production process runs" if dada else ""),
                 3, cmds, parallel=True, gate="no x2 test process / shm / GPU use left")]


def _cp1_restore() -> list[Step]:
    t = "restore"
    return [
        Step("restore.kill_marked", t, "every node: kill this run's processes, remove test shm", 3, [
            Command(_KILL_MARKED, n, 120, mark=False) for n in ALL_NODES], parallel=True),
        Step("restore.no_test_leftovers", t, "every node: no x2 test process, shm or GPU use left", 1, [
            Command(_NO_MARKED + "; " + _GPU_FREE_OF_TESTS, n, 60, mark=False) for n in ALL_NODES],
            parallel=True, gate="clean"),
        Step("restore.checkouts_unchanged", t, "production checkouts on all 20 nodes as before the window", 2, [
            Command(f"for n in @{{all_nodes}}; do echo $n $(ssh -o BatchMode=yes $n.pro.pvt '{_CHECKOUT}'); "
                    "done > @{state}/restore_node_checkouts.txt; diff @{state}/pre/node_checkouts.txt "
                    "@{state}/restore_node_checkouts.txt", None, 300, mark=False)],
            gate="identical sha and dirty count"),
        Step("restore.etcd_unchanged", t, "live etcd configs, /cnf/opoint and K as before the window", 1, [
            Command("@{py_h23} @{repo}/tools/x2/cp_runner.py check verify-snapshot "
                    "--snapshot @{state}/pre/etcd_snapshot.json", None, 120, mark=False)],
            gate="sha256 equal; op-point 1x"),
        Step("restore.fleet_ready", t, "stopped in 1x: every orchestrator up, nothing running", 1, [
            Command("@{py_h23} @{repo}/tools/x2/cp_runner.py check fleet-state --want ready --require-all",
                    None, 120, mark=False)],
            gate="READY with 16/16 corr and 4/4 search orchestrators"),
        Step("restore.h20_pusher", t, "read-only: the h20 Influx pusher (outside update_dsart and the "
             "dashboard service inventory) is active", 1, [
            Command("ssh -o BatchMode=yes lxd110h20.pro.pvt systemctl is-active dsart_rt_to_influx.service",
                    None, 60, mark=False)],
            gate="active", optional=True),
        Step("restore.user_start", t, "the user starts 1x from the web UI and arms", 15, [
            Command("@{py_h23} @{repo}/tools/x2/cp_runner.py check fleet-state --want observing --wait-s 1200",
                    None, 1260, mark=False)],
            manual="web UI Control tab: Start with op-point 1x (the default), wait for PREPARED, then Arm "
                   "(utc_start).", in_abort=False, gate="OBSERVING within 20 min"),
        Step("restore.search_health", t, "incident signature on all 8 halves for 10 min after a 2 min settle", 13, [
            Command("@{py_h23} @{repo}/tools/x2/cp_runner.py check search-health --seconds 600 --settle-s 120 "
                    "--out @{state}/restore/search_health.json", None, 900, mark=False)],
            in_abort=False,
            gate="0 lag-recovery seeks; no two consecutive cube rates < 4.9/s; scatter mean <= 90 ms "
                 "(1x baseline: @{state}/pre/search_health_1x.json)"),
        Step("restore.nic_after", t, "read-only: NIC counters after the restore vs the 1x baseline (5 min)", 6, [
            Command(_sampler_bg("@{all_nodes}", 61, 5, "@{state}/restore/nic") + "; "
                    + _sampler_wait("@{state}/restore/nic"), None, 600, mark=False),
            Command("@{py_h23} @{repo}/tools/x2/cp_runner.py check nic-gate --baseline-dir @{state}/pre/nic_1x "
                    "--test-dir @{state}/restore/nic --nodes @{corr_nodes_csv},@{search_nodes_csv} "
                    "--out @{state}/restore/nic_gate.json", None, 120, mark=False)],
            in_abort=False, optional=True,
            gate="no node worse than its own 1x baseline (n01 should improve after the ipfrag sysctl)"),
    ]


def build_cp1() -> Plan:
    steps = _cp1_pre() + _cp1_window() + _cp1_corr() + _cp1_search() + _cp1_fleet() + _cp1_restore()
    teardown = {
        "corr": _teardown("corr", ["@{corr_node}"], gpu=True, dada=True, bench=_BENCH_CORR),
        "search": _teardown("search", ["@{search_node}"], gpu=True, bench=_BENCH_OWNER),
        "fleet": _teardown("fleet", list(ALL_NODES), gpu=True),
    }
    return Plan("cp1", steps, groups=[["window"], ["corr", "search"], ["fleet"], ["restore"]],
                teardown=teardown)


def build_plan(name: str) -> Plan:
    if name == "cp1":
        return build_cp1()
    raise ValueError(f"no plan {name!r}")
