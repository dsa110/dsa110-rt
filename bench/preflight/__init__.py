"""bench/preflight/ — operator-driven preflight gates run on demand.

Each script in this directory is a *standalone* command-line harness
designed to be run by the operator BEFORE pushing search-side changes
to the fleet. They are NOT auto-CI gates; the goal is to give the
operator a single command that exercises the search pipeline in
isolation (no real SNAP packets, no etcd, no live fleet) and either
prints a clear PASS / FAIL summary against the production budget /
correctness contract, or surfaces the exact regression so the
operator can iterate on a fix without coordinating fleet restarts.

Current scripts (M7.7 follow-up, 2026-06-04):

  * ``bench/preflight/search_speed_gate.py``
        Drives ``bench.search_node_throughput`` at the EXACT production
        op-point (n_grid=256, n_fdm=34, t_det=192, M7.7 symmetric-shift
        padding on, fp16/cuda/gpu-imager, real prod DM plan,
        --pipeline-overlap, all detector knobs matching
        ``configs/dsart_search_rt.yaml``). Asserts median cube-cadence
        ≤ ``--budget-ms`` (default 134, the 7.45 cubes/s production
        cadence) and prints a stage-by-stage breakdown so the operator
        can attribute any regression to imager / Layer-1 / detector.

Future scripts (in flight; this docstring is the spec):

  * ``bench/preflight/build_corr_fixture.py``
        Drives the corr-side fast-pipeline on synthetic noise +
        a known injection and saves the resulting visibility blobs +
        truth metadata to disk. Runs once per geometry; the output
        is loaded by both ``analyze_corr_fixture.py`` and
        ``search_e2e_correctness.py``.

  * ``bench/preflight/analyze_corr_fixture.py``
        Loads a saved fixture and renders per-baseline magnitude vs
        time + dirty image at the injection-DM trial. Operator-
        inspectable; lets the user confirm the synthetic pulse is
        actually present at the expected (l, m, t) BEFORE pushing it
        through the detector.

  * ``bench/preflight/search_e2e_correctness.py``
        Replays a saved fixture through the search-compute detector
        with M7.7 on and asserts the candidate lands at the expected
        (l, m, fine_dm, t) cell with SNR in tolerance.

The split-along-pipeline-stage design is deliberate: the speed gate
exercises the pure GPU compute cost (no transport, no real shifts), the
correctness gate exercises the algorithmic correctness on
deterministic data, and the fixture analyser keeps the per-stage
intermediate plots cheap to regenerate so the operator can debug a
detection miss WITHOUT re-running the whole fleet.
"""
