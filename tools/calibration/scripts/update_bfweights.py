import argparse
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
from dsacalib.plotting import summary_plot, plot_current_beamformer_solutions
from dsacalib.plotting import plot_beamformer_weights
from dsacalib.routines import get_files_for_cal, calibrate_measurement_set
from dsacalib.weights import get_good_solution, write_beamformer_solutions
from dsacalib.ms_io import convert_calibrator_pass_to_ms, uvh5_to_ms
myconf = cnf.Conf(use_etcd=True)
ETCD = DsaStore()
import sys

parser = argparse.ArgumentParser()
parser.add_argument("bfweights", help="<SRC>_<DATESTRING> descriptor of bf weights, e.g., 1459+716_2022-09-26T22:29:30",type=str)
args = parser.parse_args()

print("Using...",args.bfweights)

CORR_PARAMS = myconf.get('corr')
CAL_PARAMS = myconf.get('cal')
MFS_PARAMS = myconf.get('fringe')
REFANTS = CAL_PARAMS['refant']
if isinstance(REFANTS, (str, int)):
    REFANTS = [REFANTS]
MSDIR = CAL_PARAMS['msdir']
BEAMFORMER_DIR = CAL_PARAMS['beamformer_dir']
ANTENNAS = np.array(list(CORR_PARAMS['antenna_order'].values()))
print(ANTENNAS)
POLS = CORR_PARAMS['pols_voltage']
ANTENNAS_NOT_IN_BF = CAL_PARAMS['antennas_not_in_bf']
#CORR_LIST = list(CORR_PARAMS['ch0'].keys())
#CORR_LIST = [int(cl.strip('corr')) for cl in CORR_LIST]

with open(
        '{0}/beamformer_weights_{1}.yaml'.format(
            BEAMFORMER_DIR,
            args.bfweights
        )
) as f:
    latest_solns = yaml.load(f, Loader=yaml.FullLoader)

now = Time(datetime.datetime.utcnow())
now.precision = 0
averaged_files, avg_flags = average_beamformer_solutions([args.bfweights],now,BEAMFORMER_DIR,ANTENNAS,58849.0)

latest_solns['cal_solutions']['weight_files'] = averaged_files
latest_solns['cal_solutions']['source'] = [args.bfweights.split('_')[0]]
latest_solns['cal_solutions']['caltime'] = [float(Time(args.bfweights.split('_')[1]).mjd)]
for key, value in latest_solns['cal_solutions']['flagged_antennas'].items():
    if 'casa solutions flagged' in value:
        value = value.remove('casa solutions flagged')
idxant, idxpol = np.nonzero(avg_flags)
for i, ant in enumerate(idxant):
    key = '{0} {1}'.format(ANTENNAS[ant], POLS[idxpol[i]])
    if key not in latest_solns['cal_solutions']['flagged_antennas'].keys():
        latest_solns['cal_solutions']['flagged_antennas'][key] = []
    latest_solns['cal_solutions']['flagged_antennas'][key] += ['casa solutions flagged']
latest_solns['cal_solutions']['flagged_antennas'] = {key: value for key, value in latest_solns['cal_solutions']['flagged_antennas'].items() if len(value) > 0}

with open('{0}/beamformer_weights_{1}.yaml'.format(BEAMFORMER_DIR, now.isot),'w') as file:
    print('writing bf weights')
    _ = yaml.dump(latest_solns, file)


print('Starting bfweights copy...')
os.system('systemctl --user start bfweights_copy.service')
time.sleep(3.)

with open('{0}/beamformer_weights_{1}.yaml'.format(BEAMFORMER_DIR,now.isot)) as f:
    latest_solutions = yaml.load(f, Loader=yaml.FullLoader)
ETCD.put_dict('/mon/cal/bfweights',{'cmd': 'update_weights','val': latest_solns['cal_solutions']})

time.sleep(60.)
time.sleep(3.)
print('Finished copy')
os.system('systemctl --user stop bfweights_copy.service')


