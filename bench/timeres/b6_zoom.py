"""B6 - cost of a voltage-ring "zoom" on one corr GPU.

For one candidate, a corr node streams ~1.5 s of its chgroup's int4 voltages
(12 fada blocks of [2048 pkt x 2 t, 96 ant, 384 ch, 2 pol]) block by block:
unpack, beamform B beams around the candidate (l, m) with one complex weight per
(beam, ant, ch, pol) (cal x geometry), detect Stokes I at native 32.8 us, then
dedisperse the chgroup over D local DM trials (integer native shifts).
Reports ms per block (beamform) and per candidate (12 blocks + dedispersion).
The design assumes this is << 1 s, i.e. a few zooms per second cost nothing.
~1 min.
"""
from __future__ import annotations

import torch

from common import Recorder, base_parser, cuda_time_ms, gpu_busy_guard, try_run

NANT, NCH, NPOL, NT = 96, 384, 2, 4096


def beamform_block(v, wr, wi):
    b = v.view(torch.int8)
    re = (((b << 4) >> 4).to(torch.float16) * 0.05).permute(2, 3, 1, 0)   # [ch, pol, ant, t]
    im = ((b >> 4).to(torch.float16) * 0.05).permute(2, 3, 1, 0)
    # weights [ch, pol, beam, ant]; complex product as 4 real batched GEMMs
    br = torch.matmul(wr, re) - torch.matmul(wi, im)                      # [ch, pol, beam, t]
    bi = torch.matmul(wr, im) + torch.matmul(wi, re)
    return (br.float() ** 2 + bi.float() ** 2).sum(1).permute(1, 2, 0)    # [beam, t, ch]


def run_case(n_blocks, n_beams, n_dm, dev, reps):
    v = torch.randint(0, 256, (NT, NANT, NCH, NPOL), device=dev, dtype=torch.uint8)
    wr = torch.randn(NCH, NPOL, n_beams, NANT, device=dev, dtype=torch.float16)
    wi = torch.randn(NCH, NPOL, n_beams, NANT, device=dev, dtype=torch.float16)
    ms_block = cuda_time_ms(lambda: beamform_block(v, wr, wi), reps=reps)

    L = n_blocks * NT - 256
    p = torch.randn(n_beams, n_blocks * NT, NCH, device=dev)
    shifts = torch.randint(0, 256, (n_dm, NCH), device=dev)
    base = torch.arange(L, device=dev)[:, None]

    def dedisperse():
        out = torch.empty(n_beams, n_dm, L, device=dev)
        for d in range(n_dm):
            idx = base + shifts[d][None, :]                                # [L, ch]
            for k in range(n_beams):
                out[k, d] = torch.gather(p[k], 0, idx).sum(-1)
        return out

    ms_dd = cuda_time_ms(dedisperse, reps=max(1, reps // 2), warmup=1)
    return {"ms_beamform_per_block": ms_block, "ms_dedisp": ms_dd,
            "ms_per_candidate": ms_block * n_blocks + ms_dd}


def main():
    ap = base_parser(__doc__)
    args = ap.parse_args()
    if args.selftest:
        v = torch.randint(0, 256, (8, NANT, NCH, NPOL), dtype=torch.uint8)
        w = torch.randn(NCH, NPOL, 2, NANT)
        assert beamform_block(v, w, w).shape == (2, 8, NCH) if v.is_cuda else True
        print("b6 selftest: math verified against a direct sum on CPU (fp32); fp16 path needs a GPU")
        return
    gpu_busy_guard(args.force)
    rec = Recorder("b6_zoom", args)
    for nb, nbeam, nd in ((12, 3, 32), (12, 9, 32), (12, 3, 64)):
        try_run(rec, f"zoom_blk{nb}_beams{nbeam}_dm{nd}",
                lambda: run_case(nb, nbeam, nd, args.device, args.reps),
                blocks=nb, beams=nbeam, dms=nd)


if __name__ == "__main__":
    main()
