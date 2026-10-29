"""G2 production wiring of the 2x op-point (CPU, no GPU, no etcd, nothing executed).

The repo 2x set rendered for ``2x-gridded-ns4-cint4`` + its v3 plan exactly as
``push_dsart_to_etcd.py --opoint`` renders it, then expanded per routine by
``dsart_rt``'s own token substitution (CN / CHGROUP / CALSB / CUSTOMDEC / SPL_*,
hostargs) for sample hosts:

* no unresolved ``${``, ``X2TODO`` or unknown upper-case token is left, and the
  target module's own argparse accepts the argv (parse only: ``parse_known_args`` is
  intercepted, so no service runs);
* the pattern ids that search_rx registers, the owner engine checks and corr_fast
  ships agree for the same (cal blob, DEC), each from its own rendered argv;
* dsart_rt mechanics: no sentinel gates, spawn order / stagger, no v3 ring on the
  search side, corr buffers + reader counts and every other corr routine unchanged;
* cross-file consistency (C1 endpoint, cutout ports / hosts, upload destination,
  retention, the c1 block vs the engine's values);
* the capture-epoch MJD helpers (``dsart.x2.prod_inputs``).
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import math
import re
import shlex
from pathlib import Path

import numpy as np
import pytest
import yaml

from dsart.x2 import opoint_runtime as opr
from dsart.x2 import prod_inputs as pi
from dsart.x2.opoint import opoint_from_id
from dsart.x2.plan import SEARCH_NODES, build_tiered_plan_for_opoint

dsart_rt = pytest.importorskip("dsart.services.dsart_rt")       # needs dsautils (nodes have it)

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
FIX = REPO / "tests" / "fixtures" / "x2" / "dsa110_antpos.npz"
OPID = "2x-gridded-ns4-cint4"
PROD_CAL = "/home/ubuntu/proj/dsa110-shell/dsa110-xengine/utils/antennas.out"
DEC = "71.63"
CORR_HOSTS = {"lxd110h03": 3, "lxd110h06": 6, "lxd110h22": 22}
SEARCH_HOSTS = {"lxd110h01": 1, "lxd110h02": 2, "lxd110h09": 9, "lxd110h13": 13}
DATA_IP = {1: "10.41.0.205", 2: "10.41.0.222", 9: "10.41.0.253", 13: "10.41.0.238"}
ALLOWED_UPPER = {"INFO"}
SUBST_TOKENS = ("CUSTOMDEC", "CHGROUP", "CALSB", "SPL_INTEGRATION_S", "SPL_NFREQ_INT")


# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("g2w")
    assert "CN" not in str(d)                       # dsart_rt would substitute it
    op = opoint_from_id(OPID)
    plan = build_tiered_plan_for_opoint(op)
    plan_path = d / "plan_ns4.npz"
    plan.to_npz(plan_path)
    rendered = opr.render_repo_set(OPID, CONFIGS, plan=plan, plan_path=str(plan_path))
    base = {inst: yaml.safe_load((CONFIGS / fname).read_text())
            for inst, fname in opr.CONFIG_FILES.items() if inst in opr.INSTANCES}
    # a cal blob with the fixture's E/N (corr_setup_96 station order; no sibling yaml, so the
    # production core mask from corr_setup_96 applies, as on the nodes)
    z = np.load(FIX)
    from dsart.cal.bf_weights import BF_WEIGHTS_FILE_SIZE
    blob = np.zeros(BF_WEIGHTS_FILE_SIZE // 4, dtype=np.float32)
    blob[:96] = z["antpos_e"]
    blob[96:192] = z["antpos_n"]
    cal = d / "cal" / "antennas.out"
    cal.parent.mkdir()
    blob.tofile(cal)
    return dict(op=op, plan=plan, plan_path=plan_path, rendered=rendered, base=base, cal=cal,
                dir=d)


def _orch(cn: int, host: str):
    """A dsart_rt orchestrator shell for its argv mechanics only (no etcd)."""
    o = dsart_rt.RtOrchestrator.__new__(dsart_rt.RtOrchestrator)
    o.cn_id, o.fqdn = int(cn), host
    o._spl = {"enabled": False, "integration_s": dsart_rt._SPL_DEFAULT_INTEGRATION_S,
              "nfreq_int": dsart_rt._SPL_DEFAULT_NFREQ_INT}
    return o


def _routines(env, inst):
    return dsart_rt.PipelineConfig.from_dict(env["rendered"][inst]).routines


def _argv(env, inst, name, host, cn, dec=DEC):
    r = next(r for r in _routines(env, inst) if r.name == name)
    o = _orch(cn, host)
    return o._build_argv(r, dec), {k: o._substitute(v, dec) for k, v in r.env.items()}


def _module_argv(argv):
    if "-m" not in argv:
        return None, None
    i = argv.index("-m")
    return argv[i + 1], argv[i + 2:]


class _Parsed(BaseException):
    def __init__(self, ns, rest):
        super().__init__("parsed")
        self.ns, self.rest = ns, rest


@contextlib.contextmanager
def _parse_only():
    """Every ``ArgumentParser.parse_known_args`` (``parse_args`` calls it) parses, then
    stops the program before it does anything else."""
    real = argparse.ArgumentParser.parse_known_args

    def fake(self, args=None, namespace=None):
        ns, rest = real(self, args, namespace)
        raise _Parsed(ns, rest)

    argparse.ArgumentParser.parse_known_args = fake
    try:
        yield
    finally:
        argparse.ArgumentParser.parse_known_args = real


def _parse(module: str, margs):
    import importlib
    mod = importlib.import_module(module)
    with _parse_only():
        try:
            mod.main(list(margs))
        except _Parsed as p:
            assert p.rest == [], f"{module}: unknown arguments {p.rest}"
            return p.ns
        except SystemExit as e:
            raise AssertionError(f"{module} rejected {margs} (exit {e.code})") from None
    raise AssertionError(f"{module}.main never parsed its arguments")


def _bad_tokens(tokens):
    bad = []
    for t in tokens:
        if "${" in t or "X2TODO" in t or any(s in t for s in SUBST_TOKENS):
            bad.append(t)
            continue
        for w in re.findall(r"(?<![A-Za-z0-9_])[A-Z][A-Z0-9_]+(?![a-z0-9])", t):
            if w not in ALLOWED_UPPER:
                bad.append(t)
    return bad


# -------------------------------------------- dsart_rt expansion + argparse


@pytest.mark.parametrize("inst,hosts", [("pipeline_rt", CORR_HOSTS), ("search_rt", SEARCH_HOSTS)])
def test_every_routine_expands_cleanly_and_parses(env, inst, hosts):
    rep = opr.validate_set(env["rendered"], OPID)
    assert rep.ok, rep.summary()                    # what the push / Start control check
    parsed = 0
    for r in _routines(env, inst):
        for host, cn in hosts.items():
            argv, envs = _argv(env, inst, r.name, host, cn)
            assert not _bad_tokens(argv + list(envs.values())), (r.name, host,
                                                                 _bad_tokens(argv))
            mod, margs = _module_argv(argv)
            if mod is None or not mod.startswith("dsart."):
                continue                            # binaries, dsamfs wrapper, bench replays
            _parse(mod, margs)
            parsed += 1
    assert parsed >= (5 if inst == "pipeline_rt" else 3) * len(hosts)


def test_search_routines_parse_to_the_2x_services(env):
    plan = env["plan"]
    for host, cn in SEARCH_HOSTS.items():
        idx = SEARCH_NODES.index(f"n{cn:02d}")
        argv, envs = _argv(env, "search_rt", "search_rx", host, cn)
        ns = _parse(*_module_argv(argv))
        assert (ns.ring_version, ns.opoint, ns.node_index, ns.mon_cn) == (4, OPID, idx, cn)
        assert ns.bind_host == DATA_IP[cn] and ns.shm_name == f"/dsart-rxring-{cn}"
        assert (ns.port_base, ns.n_ports, ns.rx_threads) == (6625, 16, 4)
        assert ns.plan_id == f"0x{plan.plan_id:016x}" and str(ns.plan) == str(env["plan_path"])
        assert ns.antpos_path == PROD_CAL and ns.obs_dec_deg == float(DEC)
        assert ns.ring4_spec is None and ns.pattern_id_file is None and ns.retention_s >= 20
        for g in (0, 1):
            argv, envs = _argv(env, "search_rt", f"search_compute_{g}", host, cn)
            mod, margs = _module_argv(argv)
            assert mod == "dsart.services.search_compute"
            ns = _parse(mod, margs)                 # --opoint 2x-* -> dsart.x2.search.service
            assert hasattr(ns, "fft_mode") and not hasattr(ns, "n_corr")   # the 2x parser
            assert (ns.opoint, ns.search_node_id, ns.gpu_half) == (OPID, cn, g)
            assert ns.shm_name == f"/dsart-rxring-{cn}" and ns.ring_access == "pread"
            assert (ns.fft_mode, ns.bank, ns.k1, ns.k3b) == ("fused", "b3x", "staged", "reg")
            assert ns.antpos_path == PROD_CAL and ns.obs_dec_deg == float(DEC)
            assert ns.plan_path == str(env["plan_path"])
            assert ns.plan_id == f"0x{plan.plan_id:016x}"
            assert ns.t_int_search_us == env["op"].t_ext_us
            assert ns.chan_sum_factor == env["op"].chan_sum_factor
            assert ns.mjd_epoch == "capture" and ns.mjd_at_native_0 == 0.0
            assert ns.cutout_bind_host == DATA_IP[cn] and ns.device == "cuda:0"
            assert envs["CUDA_VISIBLE_DEVICES"] == str(g)


def test_corr_fast_routine_runs_the_2x_path(env, monkeypatch):
    """corr_fast's own CLI with run() stubbed: the kwargs production starts with."""
    import dsart.services.corr_fast_integration as cfi
    cap = {}
    monkeypatch.setattr(cfi, "run", lambda fada, out, dev, cfg, **kw: cap.update(cfg=cfg, kw=kw))
    for host, cn in CORR_HOSTS.items():
        argv, envs = _argv(env, "pipeline_rt", "corr_fast", host, cn)
        mod, margs = _module_argv(argv)
        margs = [str(env["cal"]) if a == PROD_CAL else a for a in margs]
        assert cfi.main(margs) == 0 and cap
        cfg, kw = cap["cfg"], cap["kw"]
        assert kw["opoint"] == OPID and kw["plan_id"] == f"0x{env['plan'].plan_id:016x}"
        assert kw["transport_tx_mode"] == "v2" and kw["x2_epoch"] == "capture"
        assert kw["x2_tx_burst_bytes"] == 16384 and kw["x2_block0"] == 0   # 0 = capture epoch
        assert kw["x2_tx_bind_host"].startswith("10.41.0.")
        assert kw["transport_tx_worker_hosts"].split(",") == [DATA_IP[int(n[1:])]
                                                              for n in SEARCH_NODES]
        assert kw["transport_tx_port"] == 6625
        assert (cfg.t_int_fast_native, cfg.chan_sum_factor, cfg.n_sub) == (8, 2, 4)
        assert cfg.chgroup == dsart_rt.RtOrchestrator._cn_to_chgroup(_orch(cn, host))
        assert envs["DSART_HOST_PIN_PERSIST"] == "1" and envs["CUDA_VISIBLE_DEVICES"] == "1"


# --------------------------------------------------- pattern-id consistency


def _three_value_maps(env, monkeypatch, cal_path, dec):
    """(corr_fast, search_rx, engine) value maps + search_rx's registered ids, each from
    its own rendered argv on n13 / a corr node, with the cal blob at ``cal_path``."""
    import dsart.services.corr_fast_integration as cfi
    import dsart.services.search_rx as srx
    from dsart.x2.search import service as xs

    swap = (lambda a: str(cal_path) if a == PROD_CAL else a)
    # corr_fast (n06): its CLI -> cfg, then the value map exactly as run() builds it
    cap = {}
    monkeypatch.setattr(cfi, "run", lambda fada, out, dev, cfg, **kw: cap.update(cfg=cfg, kw=kw))
    argv, _ = _argv(env, "pipeline_rt", "corr_fast", "lxd110h06", 6, dec=dec)
    assert cfi.main([swap(a) for a in _module_argv(argv)[1]]) == 0
    cfg = cap["cfg"]
    e, n, mask = cfi.load_antpos_from_cal_blob(cfg.cal_path)
    vm_corr = pi.value_map(opoint_from_id(cap["kw"]["opoint"]), e, n, mask,
                           dec_deg=math.degrees(cfg.obs_dec_rad))
    # search_rx (n13): its CLI -> the in-process ring spec + expected ids
    argv, _ = _argv(env, "search_rt", "search_rx", "lxd110h13", 13, dec=dec)
    ns = _parse(*_module_argv([swap(a) for a in argv]))
    spec, pids = srx._spec_from_args(ns)
    # owner engine (n13, half 1): its CLI -> its value map
    argv, _ = _argv(env, "search_rt", "search_compute_1", "lxd110h13", 13, dec=dec)
    ns_e = _parse(*_module_argv([swap(a) for a in argv]))
    vm_eng = xs._value_map_for(opoint_from_id(ns_e.opoint), ns_e.antpos_path, ns_e.obs_dec_deg)
    return vm_corr, spec, pids, vm_eng, ns_e


def _assert_same_ids(env, vm_corr, spec, pids, vm_eng, ns_e):
    from dsart.x2.search.geometry import owner_plan_from_tiered
    ids_corr = [int(x) for x in vm_corr.stream_pattern_ids]
    ids_eng = [int(x) for x in vm_eng.stream_pattern_ids]
    assert len(ids_corr) == env["op"].n_streams == 64
    assert ids_corr == ids_eng == [pids[s] for s in range(len(pids))]
    assert vm_corr.pattern_id == vm_eng.pattern_id
    assert len(set(ids_corr)) == 64                 # one id per stream
    # the ring the engine half will attach: its DMs, stream counts and value counts
    half = pi.node_index_of_cn(ns_e.search_node_id) * 2 + ns_e.gpu_half
    oplan = owner_plan_from_tiered(env["plan"], half=half)
    assert {s.d for s in oplan.dms} <= set(spec.dms)
    assert spec.n_stream == 64 and list(spec.stream_n_values) == [int(x) for x in vm_eng.n_values]
    assert list(spec.stream_n_heavy) == [int(x) for x in vm_eng.n_heavy]


def test_pattern_ids_agree_corr_search_rx_engine(env, monkeypatch):
    out = _three_value_maps(env, monkeypatch, env["cal"], DEC)
    _assert_same_ids(env, *out)
    # the fixture npz (antpos_u ignored) gives the same map as the cal blob of its E/N
    from dsart.x2 import prod_inputs
    vm_npz = prod_inputs.value_map_for(env["op"], FIX, DEC)
    assert vm_npz.pattern_id == out[0].pattern_id


@pytest.mark.skipif(not Path(PROD_CAL).is_file(), reason="production cal blob not on this host")
def test_pattern_ids_agree_on_the_production_cal_blob(env, monkeypatch, tmp_path):
    cal = tmp_path / "antennas.out"                 # a copy: never hand the live file around
    cal.write_bytes(Path(PROD_CAL).read_bytes())
    _assert_same_ids(env, *_three_value_maps(env, monkeypatch, cal, DEC))


def test_dec_bin_edges_quantise_identically():
    """corr_fast carries DEC in radians; every process passes the verb's DEC through the
    same round trip, so a DEC on a 0.25 deg bin edge lands in one bin everywhere."""
    from dsart.grid.sparsity_pattern import quantise_dec_deg
    edges = [k * 0.125 for k in range(-720, 721) if k % 2]
    for x in edges + [71.63, 71.625, -10.375, 37.2341]:
        tok = str(x)                                # dsart_rt substitutes str(val)
        corr = math.degrees(math.radians(float(tok)))
        assert pi.canonical_dec_deg(tok) == corr
        assert quantise_dec_deg(pi.canonical_dec_deg(tok)) == quantise_dec_deg(corr)


# ---------------------------------------------------------- dsart_rt mechanics


def test_dsart_rt_mechanics(env):
    for inst in opr.INSTANCES:
        assert all(not r.gate_on_paths for r in _routines(env, inst))   # nothing waits on sentinels
    names = [r.name for r in _routines(env, "search_rt")]
    assert names == ["search_rx", "search_compute_0", "search_compute_1"]
    delay = {r.name: r.spawn_delay_s for r in _routines(env, "search_rt")}
    assert delay == {"search_rx": 4.0, "search_compute_0": 8.0, "search_compute_1": 0.0}
    assert env["rendered"]["search_rt"]["buffers"] == []
    for r in _routines(env, "search_rt"):           # no v3 ring anywhere on the search side
        toks = shlex.split(r.args)
        for v3 in ("--n-corr", "--n-filled", "--t-buf-samples", "--bytes-per-cell",
                   "--n-coarse-dm", "--ring4-spec", "--config-yaml", "--enable-cube-dump"):
            assert v3 not in toks, (r.name, v3)
        assert toks[toks.index("--opoint") + 1] == OPID
    rx = next(r for r in _routines(env, "search_rt") if r.name == "search_rx")
    assert "--ring-version 4" in rx.args


def test_corr_buffers_readers_and_other_routines_unchanged(env):
    two, one = env["rendered"]["pipeline_rt"], env["base"]["pipeline_rt"]
    assert two["buffers"] == one["buffers"]         # sizes + reader counts (fada r=3, bada r=2)
    assert two["captures"] == one["captures"] == {"mode": "real"}
    r1 = {r["name"]: r for r in one["routines"]}
    r2 = {r["name"]: r for r in two["routines"]}
    assert list(r1) == list(r2)
    for name in r1:
        if name == "corr_fast":
            continue
        a, b = copy.deepcopy(r1[name]), copy.deepcopy(r2[name])
        if name == "voltage_retention":
            assert "--n-pre 15" in b["args"] and "--n-pre 14" in a["args"]
            b["args"] = b["args"].replace("--n-pre 15", "--n-pre 14")
        assert a == b, name
    # corr_fast keeps today's reader role and every non-x2 argument
    t1 = shlex.split(r1["corr_fast"]["args"])
    t2 = shlex.split(r2["corr_fast"]["args"])
    gone = {"--transport-tx-workers", "--transport-tx-host", "--transport-tx-coarse-dm-mask",
            "--stage2-mode"}
    for flag in [t for t in t1 if t.startswith("--")]:
        if flag in gone:
            continue
        assert flag in t2, flag
    assert r1["corr_fast"]["env"].items() <= r2["corr_fast"]["env"].items()


def test_voltage_window_contains_the_2x_sweep(env):
    """tests/test_voltage_dump_containment.py's invariants with the 2x n_pre / n_post and
    the 2x plan's fine-DM maximum (event label = arrival at the band bottom)."""
    from dsart.common.constants import BLOCK_DURATION_S, NU_BOT_PROC_GHZ, NU_TOP_PROC_GHZ, freq_GHz
    from dsart.services.coincidencer import search_to_snap_specnum
    vr = next(r for r in env["rendered"]["pipeline_rt"]["routines"]
              if r["name"] == "voltage_retention")
    n_pre = int(opr.routine_arg({"routines": [vr]}, "voltage_retention", "--n-pre"))
    n_post = int(opr.routine_arg({"routines": [vr]}, "voltage_retention", "--n-post"))
    dm_max = float(np.max(env["plan"].fine_dm))
    k = 4.148808e-3

    def sweep(dm, hi, lo):
        return k * dm * (lo ** -2 - hi ** -2)

    tau = sweep(dm_max, NU_TOP_PROC_GHZ, NU_BOT_PROC_GHZ)
    assert dm_max > 2990 and n_pre * BLOCK_DURATION_S >= tau + BLOCK_DURATION_S
    assert n_post * BLOCK_DURATION_S >= 0.2
    for T in (38560 * 2048, 38560 * 2048 + 2047):
        blk = T // 2048
        lo, hi = (blk - n_pre) * 2048, (blk + n_post + 1) * 2048
        for g in range(16):
            earliest = T - sweep(dm_max, freq_GHz(g, 0), NU_BOT_PROC_GHZ) / 65.536e-6
            assert lo <= earliest < hi, (g, earliest, lo, hi)
    assert search_to_snap_specnum(1000, env["op"].t_ext_us) == 8000   # C2 -> SNAP specnum


# ------------------------------------------------------- cross-file consistency


def test_search_args_match_the_yaml_blocks(env):
    sr = env["rendered"]["search_rt"]
    c1, coinc = sr["c1"], sr["coinc"]
    for g in (0, 1):
        name = f"search_compute_{g}"
        arg = lambda f: opr.routine_arg(sr, name, f)          # noqa: E731
        assert arg("--c1-host") == c1["c2_endpoint"]["host"]
        assert int(arg("--c1-port")) == c1["c2_endpoint"]["port"] == coinc["bind"]["port"]
        assert int(arg("--c1-queue-depth")) == c1["emit_queue_depth"]
        assert int(arg("--cutout-base-port")) == coinc["dump_broadcast"]["port_base"]
        assert int(arg("--cutout-status-port")) == coinc["cutout"]["status_port"]
        assert f"{arg('--upload-dest-host')}:{arg('--upload-dest-root')}" == \
            c1["uploader"]["remote_root"]
        ret = c1["retention"]
        assert float(arg("--cutout-retention-max-age-h")) == ret["max_age_h"]
        assert float(arg("--cutout-retention-max-total-gb")) == ret["max_total_gb"]
        assert float(arg("--cutout-retention-low-water-gb")) == ret["low_water_gb"]
        r = next(r for r in sr["routines"] if r["name"] == name)
        for host, cn in SEARCH_HOSTS.items():
            want = f"--cutout-bind-host {coinc['dump_broadcast']['hosts'][str(cn)]}"
            assert r["hostargs"][host] == want and r["hostargs"][f"n{cn:02d}"] == want
    rx = next(r for r in sr["routines"] if r["name"] == "search_rx")
    cf = next(r for r in env["rendered"]["pipeline_rt"]["routines"] if r["name"] == "corr_fast")
    assert opr.routine_arg(sr, "search_rx", "--port-base") == \
        opr.routine_arg(env["rendered"]["pipeline_rt"], "corr_fast", "--transport-tx-base-port")
    tx_hosts = opr.routine_arg(env["rendered"]["pipeline_rt"], "corr_fast",
                               "--transport-tx-worker-hosts").split(",")
    for i, node in enumerate(SEARCH_NODES):
        cn = int(node[1:])
        want = f"--bind-host {DATA_IP[cn]} --node-index {i}"
        assert rx["hostargs"][f"lxd110h{cn:02d}"] == want == rx["hostargs"][node]
        assert tx_hosts[i] == DATA_IP[cn] == coinc["dump_broadcast"]["hosts"][str(cn)]
    assert "--apply-cal " + PROD_CAL in cf["args"]


def test_c1_block_equals_the_engine_values(env):
    """The 2x halves take these from EngineConfig / C1CycleConfig, not the yaml: the
    c1 block (read by h23 services and operators) must say the same."""
    from dsart.x2.search.c1_cycle import C1CycleConfig
    from dsart.x2.search.engine import EngineConfig
    c1 = env["rendered"]["search_rt"]["c1"]
    e, c = EngineConfig(), C1CycleConfig()
    m = c1["merger"]
    assert c1["snr_min"] == e.threshold
    assert (m["lm_max_cells"], m["dm_max_trials"], m["t_frac"], m["sample_period_specnum"]) == \
        (e.c1_lm_max_cells, e.c1_dm_max_trials, e.c1_t_frac, e.c1_sample_period_specnum)
    assert c1["noise_color_strength"] == c.noise_color_strength
    assert c1["noise_color_snr_floor"] == c.noise_color_snr_floor
    assert c1["dm_width_floor_frac"] == c.dm_width_floor_frac
    assert c1["max_c1c2_width_samples"] == c.max_width_ext
    assert c1["max_c1c2_width_snr_escape"] == c.max_width_snr_escape
    assert c1["max_c1c2_width_escape_per_block"] == c.escape_per_block
    assert c1["max_c1c2_width_escape_burst"] == c.escape_burst
    assert c1["max_c1c2_width_escape_refill_s"] == c.escape_refill_s
    assert c1["max_candidates_per_block"] == c.max_candidates_per_cycle
    assert c1["snr_protected_slots"] == c.snr_protected_slots
    assert env["rendered"]["search_rt"]["detector"]["input_clip_sigma"] == e.clip_sigma


# ----------------------------------------------------------- search_rx guards


def test_search_rx_v4_argument_guards(env):
    import dsart.services.search_rx as srx
    argv, _ = _argv(env, "search_rt", "search_rx", "lxd110h13", 13)
    good = _module_argv(argv)[1]

    def ns(extra=(), drop=()):
        a = [t for t in good]
        for flag in drop:
            i = a.index(flag)
            del a[i:i + 2]
        return _parse("dsart.services.search_rx", a + list(extra))

    with pytest.raises(SystemExit, match="--node-index 2 is not search node cn 13"):
        srx._spec_from_args(ns(extra=["--node-index", "2"]))
    with pytest.raises(SystemExit, match="--antpos-path"):
        srx._spec_from_args(ns(drop=["--antpos-path"]))
    with pytest.raises(SystemExit, match="--pattern-id-file"):
        srx._spec_from_args(ns(extra=["--pattern-id-file", "/nonexistent.json"]))
    bad = ns(extra=["--plan-id", "0x0123456789abcdef", "--antpos-path", str(env["cal"])])
    with pytest.raises(ValueError, match="plan_id"):
        srx._spec_from_args(bad)
    assert srx.main(["--bind-host", "127.0.0.1", "--shm-name", "/dsart-x2-g2w-none",
                     "--opoint", OPID]) == 2          # a 2x op-point never runs the v3 ring


def test_2x_service_checks_time_units_before_anything_else(env):
    from dsart.x2.search import service as xs
    argv, _ = _argv(env, "search_rt", "search_compute_0", "lxd110h01", 1)
    margs = _module_argv(argv)[1]
    for flag, bad in (("--t-int-search-us", "1048.576"), ("--chan-sum-factor", "8")):
        a = list(margs)
        a[a.index(flag) + 1] = bad
        with pytest.raises(SystemExit) as e:
            xs.main(a)
        assert e.value.code == 2
    with pytest.raises(SystemExit):                  # a 1x leftover argument fails loudly
        xs.main(list(margs) + ["--n-corr", "64"])


# ------------------------------------------------------------ capture epoch


def test_capture_native0_matches_corr_fast():
    from dsart.services.corr_fast_integration import x2_native0_from_specnum
    rng = np.random.default_rng(3)
    for u in [0, 16, 17, 31, 32, 1 << 34, (1 << 35) - 1] + [int(x) for x in
                                                           rng.integers(0, 1 << 36, 200)]:
        if 2 * u < 4096:
            continue
        assert pi.capture_native0(u) == x2_native0_from_specnum(u) + 4096
    assert pi.capture_native0(1000016) == 2000032    # a multiple of 16: exactly 2u


class _Store:
    def __init__(self, doc=None, fail=0):
        self.doc, self.fail, self.n = doc, fail, 0

    def get_dict(self, key):
        assert key == pi.ARM_RECORD_KEY
        self.n += 1
        if self.n <= self.fail:
            raise ConnectionError("etcd down")
        return self.doc


def test_read_arm_record():
    now = 1.8e9
    now_mjd = 40587.0 + now / 86400.0
    doc = {"armed_mjd": now_mjd - 10 / 86400.0, "arm_seq": 1 << 34, "source": "pps_epoch",
           "wrap_k": 0}
    rec, why = pi.read_arm_record(_Store(doc), now=lambda: now)
    assert why == "" and rec.arm_seq == 1 << 34 and rec.source == "pps_epoch"
    assert rec.mjd_at_native_0 == doc["armed_mjd"] - (2 << 34) * 32.768e-6 / 86400.0
    rec, why = pi.read_arm_record(_Store(doc, fail=2), now=lambda: now, retry_s=0.0)
    assert rec is not None                           # transient etcd errors are retried
    rec, why = pi.read_arm_record(_Store(doc, fail=3), now=lambda: now, retry_s=0.0)
    assert rec is None and "etcd down" in why
    rec, why = pi.read_arm_record(_Store(dict(doc, arm_seq=None)), now=lambda: now)
    assert rec is None and "arm_seq" in why          # pre-2026-07-20 record: refuse
    rec, why = pi.read_arm_record(_Store(dict(doc, armed_mjd=now_mjd - 40)), now=lambda: now)
    assert rec is None and "implausible" in why
    rec, why = pi.read_arm_record(_Store(None), now=lambda: now)
    assert rec is None and "missing" in why


def test_mjd_labels_from_the_arm_record_end_to_end():
    """dsart_rt's arm record -> MjdClock -> a C1 batch's mjd_start: a cycle's MJD is the
    arm MJD plus the data time since the arm, for an arm off the 16-specnum grid too."""
    from dsart.x2.search.geometry import CYCLE_NATIVE
    from dsart.x2.search.service import MjdClock
    wall = 1.8e9
    for arm_seq in (1 << 34, (1 << 34) + 7):         # on / off the 16-specnum grid
        n_arm = pi.capture_native0(arm_seq)          # corr_fast's label of the first sample
        armed = 40587.0 + (wall - 6.0) / 86400.0      # armed 6 s ago
        rec = pi.ArmRecord(armed_mjd=armed, arm_seq=arm_seq, source="wall_latch")
        clock = MjdClock(read_arm=lambda rec=rec: (rec, ""), wall=lambda: wall)
        k = (n_arm + 100_000) // CYCLE_NATIVE        # a cycle ~3.3 s after the arm
        got = clock.mjd_of_native(k * CYCLE_NATIVE, latch_n=(k + 1) * CYCLE_NATIVE)
        want = armed + (k * CYCLE_NATIVE - n_arm) * 32.768e-6 / 86400.0
        assert abs(got - want) * 86400.0 < 1e-5 and clock.source == "arm:wall_latch"
