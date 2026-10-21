"""One-shot vs two-stage corner-turn rates (Gb/s) -> results/cornerturn.json.

uv cells: 1748 per single channel (any csf; nfilled.cells), 1900 per sub-band
stream (n_sub 8), 2134 live at n_sub 4.  bits = bits per complex value.
One-shot: every DM owner needs the WHOLE band (all 16 chgroups, all channels).
Two-stage: an owner needs only its coarse-DM bucket (16 chgroups x n_sub streams).
"""
import json

N_CORR = 16


def gbps(values_per_sample, T_us, bits):
    return values_per_sample * (1e6 / T_us) * bits / 1e9


rows = []
for T in (1048.576, 524.288, 262.144):
    for bits in (16, 8):
        # today's two-stage, 8 coarse x 4 sub-bands
        if T == 1048.576:
            corr = gbps(8 * 4 * 2134, T, bits)
            rows.append(dict(scheme="2-stage today 8x4", T_us=T, csf=8, bits=bits,
                             corr_egress=corr, owner_ingress=gbps(16 * 4 * 2134, T, bits)))
        for csf in (8, 4, 2):
            nch = 384 // csf
            one = gbps(nch * 1748, T, bits)                     # one copy of the chgroup
            rows.append(dict(scheme="1-shot (multicast)", T_us=T, csf=csf, bits=bits,
                             corr_egress=one, owner_ingress=N_CORR * one))
        for nc, ns in ((16, 8), (24, 6)):
            rows.append(dict(scheme=f"2-stage {nc}x{ns}", T_us=T, csf="any", bits=bits,
                             corr_egress=gbps(nc * ns * 1900, T, bits),
                             owner_ingress=gbps(N_CORR * ns * 1900, T, bits)))
json.dump(rows, open("results/cornerturn.json", "w"), indent=1)
print(f"{'scheme':22s} {'T us':>8s} {'csf':>4s} {'bits':>4s} {'corr out':>9s} {'owner in':>9s}")
for r in rows:
    print(f"{r['scheme']:22s} {r['T_us']:8.1f} {str(r['csf']):>4s} {r['bits']:4d} "
          f"{r['corr_egress']:9.2f} {r['owner_ingress']:9.2f}")
