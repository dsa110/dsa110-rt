import numpy as np
import glob
from astropy.time import Time
import os
import sens_funcs as sf

calfile = '/operations/beamformer_weights/applied/beamformer_weights_sb05_2022-07-14T21:36:54.dat'


# obtain all files
fls = glob.glob('/operations/correlator/*sb05*.hdf5')
fls.sort(key=os.path.getmtime)
tstamps = [i.split('/')[-1].split('_')[0] for i in fls]
times = Time(tstamps,format='isot')

# find time limit and get mjds and files
time_limit = Time('2022-07-11T15:00:00',format='isot')
wrs = np.where(times > time_limit)[0].astype('int')
fls_lt = [fls[i] for i in wrs]
times = times[wrs].mjd

# process data
noise_5_50 = np.zeros(len(times))
noise_5_100 = np.zeros(len(times))
noise_5_200 = np.zeros(len(times))
noise_5_400 = np.zeros(len(times))
noise_300_1000 = np.zeros(len(times))

for i,fl in enumerate(fls_lt):

    uvdist,ant1,ant2,pl_A,pl_B = sf.calc_noise(fl,calfile)
    noise_5_50[i] = sf.mean_noise(uvdist,pl_A,pl_B,5,50)
    noise_5_100[i] = sf.mean_noise(uvdist,pl_A,pl_B,5,100)
    noise_5_200[i] = sf.mean_noise(uvdist,pl_A,pl_B,5,200)
    noise_5_400[i] = sf.mean_noise(uvdist,pl_A,pl_B,5,400)
    noise_300_1000[i] = sf.mean_noise(uvdist,pl_A,pl_B,300,1000)    
    print('New: ',times[i],noise_5_50[i],noise_5_100[i],noise_5_200[i],noise_5_400[i],noise_300_1000[i])

np.savez('sensitivities.npz',times=times,noise_5_50=noise_5_50,noise_5_100=noise_5_100,noise_5_200=noise_5_200,noise_5_400=noise_5_400,noise_300_1000=noise_300_1000,fls=fls_lt)

