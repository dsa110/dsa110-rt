import glob
import os
import numpy as np
import yaml
import astropy.units as u
import h5py
import dsacalib.config as configuration
config = configuration.Configuration()
from pyuvdata import UVData

def read_UV(files):
    UV = UVData()
    UV.read(files, file_type='uvh5', run_check_acceptability=False)
    return(UV)

def calc_noise(fl,calfile,path_to_fl = '',path_to_calfile = ''):

    

    UV = read_UV([path_to_fl+fl])
    nt = UV.Ntimes
    vis = UV.data_array
    vis = vis.reshape(nt, -1, 1, vis.shape[-2], 2).squeeze()
    f = UV.freq_array[0]
    t = UV.time_array.reshape((nt,2080))[:,0]
    ant1 = UV.ant_1_array.reshape(-1, 2080)[0, :]
    ant2 = UV.ant_2_array.reshape(-1, 2080)[0, :]
    u_m = UV.uvw_array[:,0].reshape((nt,2080))
    v_m = UV.uvw_array[:,1].reshape((nt,2080))

    # read in calibrations
    with open(path_to_calfile+calfile, 'rb') as f:
        caldata = np.fromfile(f, '<f4')

    # get calibration array

    # Parameters that are mostly constant
    antennas = config.antennas
    easting = caldata[:64]
    gains = caldata[64:].reshape(64, 48, 2, 2) # 64 ant, 48 chan, 2 pol, real/imag
    gains1 = gains[..., 0]+1.0j*gains[..., 1]

    # form baseline calibrations
    my_gains1 = np.zeros((2080, 48, 2), dtype=np.complex)
    for i in range(2080):
        idx1 = np.where(antennas==ant1[i]+1)[0][0]
        idx2 = np.where(antennas==ant2[i]+1)[0][0]
        my_gains1[i, ...] = np.conjugate(gains1[idx1, ...])*gains1[idx2, ...]

    # calibrate
    calvis = vis.copy()
    for i in np.arange(vis.shape[0]):
        calvis[i] = vis[i]*my_gains1

    # do calculation
    uvdist = np.mean(np.sqrt(u_m**2.+v_m**2.),axis=0)
    # we want noise in the time domain in channel-subtracted data
    # then take the mean estimate in the frequency domain
    sub_B = np.real(calvis[:,:,1:,0]) - np.real(calvis[:,:,:-1,0])
    sub_A = np.real(calvis[:,:,1:,1]) - np.real(calvis[:,:,:-1,1])
    noise_B = np.std(sub_B,axis=0)
    noise_A = np.std(sub_A,axis=0)
    noise_B = np.mean(noise_B,axis=1)
    noise_A = np.mean(noise_A,axis=1)
    
    # finally, scale by sqrt(BT)
    scfac = np.sqrt(3.221225472*250.e6/1024) # for channel width and time
    pl_B = noise_B*scfac
    pl_A = noise_A*scfac

    return uvdist,ant1,ant2,pl_A,pl_B

def mean_noise(uvdist,pl_A,pl_B,uv1,uv2):

    wrs = np.where(np.logical_and(uvdist>uv1,uvdist<uv2))
    return 0.5*(np.mean(pl_A[wrs])+np.mean(pl_B[wrs]))

