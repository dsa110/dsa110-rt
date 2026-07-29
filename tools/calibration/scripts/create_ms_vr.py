from dsacalib.ms_io import convert_calibrator_pass_to_ms
from dsacalib.utils import generate_calibrator_source
import astropy.units as u
from astropy.coordinates import Angle
import dsacalib.config as configuration
config = configuration.Configuration()

msdir = '/operations/calibration/manual_cal/'
hdf5dir = config.hdf5dir
calpasses = [
    {
        'date': "2022-07-05",
        'name': "fld1",
        'files': ["2022-07-05T13:06:47"],
        'ra': None,
        'dec': None
    }, 
    {
        'date': "2022-07-05",
        'name': "fld2",
        'files': ["2022-07-05T16:09:10"],
        'ra': None,
        'dec': None
    }, 
    {
        'date': "2022-07-05",
        'name': "fld3",
        'files': ["2022-07-05T19:06:32"],
        'ra': None,
        'dec': None
    }
]

for i, calpass in enumerate(calpasses):
    print(f'working on {calpass}')

    cal = generate_calibrator_source(calpass['name'], ra=calpass['ra'], dec=calpass['dec'])
    convert_calibrator_pass_to_ms(
        cal,
        calpass['date'],
        calpass['files'],
        msdir=msdir,
        hdf5dir=hdf5dir,
        refmjd=config.refmjd
    )
