import glob, os
import dsautils.calstatus as cs
from dsautils.dsa_store import DsaStore
from astropy.time import Time
import time
import datetime
import yaml
from dsacalib.weights import average_beamformer_solutions
import glob
import os
import numpy as np
from pkg_resources import resource_filename
import astropy.units as u
from dsautils import cnf
from dsacalib.plotting import plot_beamformer_weights
from dsacalib.routines import get_files_for_cal, calibrate_measurement_set
from dsacalib.weights import get_good_solution, write_beamformer_solutions
from dsacalib.ms_io import convert_calibrator_pass_to_ms, uvh5_to_ms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
myconf = cnf.Conf(use_etcd=True)
ETCD = DsaStore()

#while True:
for i in [1]:
    
    #list_of_files = glob.glob('/operations/beamformer_weights/generated/*+*.yaml')
    #latest_file = max(list_of_files, key=os.path.getctime)[60:88]
    latest_file = sys.argv[1]
    print('Operating on',latest_file)

    CORR_PARAMS = myconf.get('corr')
    CAL_PARAMS = myconf.get('cal')
    MFS_PARAMS = myconf.get('fringe')

    REFANTS = CAL_PARAMS['refant']
    if isinstance(REFANTS, (str, int)):
        REFANTS = [REFANTS]
    MSDIR = CAL_PARAMS['msdir']

    BEAMFORMER_DIR = CAL_PARAMS['beamformer_dir']
    ANTENNAS = np.array(list(CORR_PARAMS['antenna_order'].values()))
    POLS = CORR_PARAMS['pols_voltage']
    ANTENNAS_NOT_IN_BF = CAL_PARAMS['antennas_not_in_bf']
    CORR_LIST = list(CORR_PARAMS['ch0'].keys())
    CURRENT_BEAMFORMER_DIR = CAL_PARAMS['bfarchivedir']
    current_weights = ETCD.get_dict('/mon/cal/bfweights')['val']['weight_files']
    print(current_weights)
    
    _ = plot_beamformer_weights([latest_file],ANTENNAS,BEAMFORMER_DIR,show=False,current_weights=current_weights,current_weights_dir=BEAMFORMER_DIR,outname='/home/ubuntu/data/webPLOTS/'+latest_file)

    os.system(f"scp /home/ubuntu/data/webPLOTS/{latest_file}* lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS") 
    #os.system("rm /operations/webPLOTS/calibration/*.p*")
    #os.system("for fl in `ls -drt /home/ubuntu/data/webPLOTS/calibration/* | tail -n 10`; do cp $fl /operations/webPLOTS/calibration/allpngs; done")
    #os.system("for fl in `ls -drt /operations/webPLOTS/calibration/allpngs/* | grep -v averagedweights | tail -n 5`; do cp $fl /operations/webPLOTS/calibration; done")    
    #os.system("for fl in `ls -drt /operations/webPLOTS/calibration/allpngs/*averagedweights* | tail -n 5`; do cp $fl /operations/webPLOTS/calibration; done")    

    time.sleep(1)

