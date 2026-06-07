"""Quick target_max sweep reusing an existing corr_shared (no corr re-run)."""
import sys
from bench.preflight._inject_search_driver import run_search_driver

CORR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/allgpu_test/dm1200_w1ms/corr_shared"
FLUENCE = float(sys.argv[2]) if len(sys.argv) > 2 else 12.335
NBLK = int(sys.argv[3]) if len(sys.argv) > 3 else 12

for tmax in (20, 8):
    for o in (3, 0):
        r = run_search_driver(
            owner_idx=o, dm_pc_cm3=1200, dm_target=1200, width_ms=1.0,
            fluence_jy_ms=FLUENCE, n_blocks=NBLK, n_burnin=6,
            out_dir=f"/tmp/tmtest/o{o}_t{tmax}", run_noise_only=False,
            reuse_corr=True, corr_work_dir=CORR, corr_save_all_owners=True,
            audit_fp32=False, zero_dm_filter=True, quant_target_max=tmax,
            verbose=False,
        )
        cs = sorted(r["candidates"], key=lambda c: -c["snr"])
        t = cs[0] if cs else None
        cube_max = float(r["inj_cube_max"])
        snr = float(t["snr"]) if t else 0.0
        lp = int(t["l_pix"]) if t else -1
        mp = int(t["m_pix"]) if t else -1
        dm = float(t["dm_pc_cc"]) if t else -1.0
        fdm = int(t["fine_dm_idx"]) if t else -1
        print(f"RESULT tmax={tmax} owner={o} cube_max={cube_max:.1f} "
              f"snr={snr:.1f} pix=({lp},{mp}) dm={dm:.0f} fdm={fdm}", flush=True)
