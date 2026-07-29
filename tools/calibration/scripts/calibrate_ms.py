from dsacalib.ms_io import convert_calibrator_pass_to_ms
from dsacalib.utils import generate_calibrator_source
import numpy as np
import astropy.units as u
from dsacalib.weights import write_beamformer_solutions
from astropy.time import Time
from dsautils import cnf
myconf = cnf.Conf()
import dsacalib.config as configuration
config = configuration.Configuration()
import yaml
from dsacalib.routines import calibrate_measurement_set

def extract_applied_delays(current_solns):
    """uses delays from a bf weights yaml file

    Parameters
    ----------
    file : str
        The full path to the yaml file

    Returns
    -------
    ndarray
        The applied delays in ns.
    """
    
    with open(current_solns) as yamlfile:
        calibration_params = yaml.load(yamlfile,Loader=yaml.FullLoader)
        applied_delays = np.array(calibration_params['delays'])*2
    return applied_delays

# input all info below
calnames = ['0137+331','1331+305']
ras = [24.42208096*u.deg,202.78453326666667*u.deg]
decs = [33.15975916*u.deg,30.509155236111113*u.deg]
fluxes = [16.5,15.0]
cals = []
for i in np.arange(len(ras)):
    cals.append(generate_calibrator_source(calnames[i], ra=ras[i], dec=decs[i], flux=fluxes[i]))
filenames = ['/operations/beamformer_weights_2022-04-27T20:57:53.yaml',
             '/operations/beamformer_weights_2022-04-27T20:57:53.yaml']
applied_delays = []
for i in np.arange(len(ras)):
    applied_delays.append(extract_applied_delays(filenames[i]))
msnames = ['/operations/calibration/2022-04-14_J013741+330935','/operations/calibration/2022-05-24_J133108+303032']

CORR_PARAMS = myconf.get('corr')
ANTENNAS = np.array(list(CORR_PARAMS['antenna_order'].values()))
CAL_PARAMS = myconf.get('cal')
ANTENNAS_NOT_IN_BF = CAL_PARAMS['antennas_not_in_bf']

ttime = Time.now()
ttime.precision = 0

i = 1

status = calibrate_measurement_set(
    msnames[i],
    cals[i],
    refants=['103'],
    bad_antennas=['10'],
    bad_uvrange='2~27m',
)

# Write beamformer solutions for one source
_ = write_beamformer_solutions(
    msnames[i],
    calnames[i],
    ttime,
    ANTENNAS,
    applied_delays[i],
    config.beamformer_dir,
    config.pols,
    config.nchan,
    config.nchan_spw,
    config.bw_GHz,
    config.chan_ascending,
    config.f0_GHz,
    config.ch0,
    config.refmjd,
    flagged_antennas=ANTENNAS_NOT_IN_BF
)