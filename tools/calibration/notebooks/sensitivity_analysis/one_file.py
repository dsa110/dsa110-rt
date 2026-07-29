import numpy as np
import glob
import os
from astropy.time import Time
import sens_funcs as sf

band = "sb05"  # RFI-free band
calfls = glob.glob(f'/operations/beamformer_weights/applied/*{band}*.dat')
calfile = max(calfls, key=os.path.getmtime)

fls = glob.glob(f'/operations/correlator/*{band}*.hdf5')
fl = max(fls, key=os.path.getmtime)
tstamps = [i.split('/')[-1].split('_')[0] for i in fls]
times = Time(tstamps, format='isot').sort()
time0 = times[-1]

print(calfile,fl,time0)

uvdist,ant1,ant2,pl_A,pl_B = sf.calc_noise(fl,calfile)
noise_st = sf.mean_noise(uvdist,pl_A,pl_B,5,50)  # characterize potential self-talk
noise_bf = sf.mean_noise(uvdist,pl_A,pl_B,5,400)  # measure SEFD for beamformer baselines
noise_ou = sf.mean_noise(uvdist,pl_A,pl_B,400,2000)  # measure outrigger SEFDs

print(time0, noise_st, noise_bf, noise_ou)

