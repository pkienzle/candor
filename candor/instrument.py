# This program is public domain
# Author: Paul Kienzle
import os.path

import numpy as np
from numpy import degrees, radians, arctan, arcsin, tan, sqrt

from . import nice
from .nice import (Motor, Map, Virtual, Instrument, InOut, Counter, Detector,
                   RateMeter, Experiment, Trajectory, TrajectoryData)

devices = {
    'Q': {
        'description': 'Candor Q Device',
        'fields': {
            'angleIndex': {
                'label': 'Q angle index',
                'mode': 'state',
                'note': 'Index in detectorBank.rowAngularOffsets used to calculate reflected angle in Qx,Qz calculations',
                'type': 'int32',
                'units': '',
            },
            'beamIndex': {
                'label': 'Q beam index',
                'mode': 'state',
                'note': 'Index in beam.angularOffsets used to calculate incident angle in Qx,Qz calculations.  Ignored for SingleBeam and WhiteBeam modes.',
                'type': 'int32',
                'units': '',
            },
            'beamMode': {
                'label': 'Q beam mode',
                'mode': 'state',
                'note': 'Source beam configuration for the measurement: SingleBeam - a single monochromatic beam with wavelength mono.waveLength WhiteBeam - a single beam with the full 4-6√Ö wavelength range MultiBeam - 4 incident white beams',
                'options': ['SINGLE_BEAM', 'MULTI_BEAM', 'WHITE_BEAM'],
                'type': 'string',
            },
            'wavelength': {
                'error': 0.001,
                'label': 'Q wavelength',
                'mode': 'state',
                'note': '',
                'type': 'float32',
                'units': '√Ö',
            },
            'wavelengthIndex': {
                'label': 'Q wavelength index',
                'mode': 'state',
                'note': 'Index in detectorBank.waveLengths used as the wavelength in Qx,Qz calculations.  In SingleBeam mode, use mono.waveLength.',
                'type': 'int32',
                'units': '',
            },
            'x': {
                'error': 0.001,
                'label': 'Q x',
                'mode': 'state',
                'note': 'X component of wave-vector transfer',
                'type': 'float32',
                'units': '1/√Ö',
            },
            'z': {
                'error': 0.001,
                'label': 'Q z',
                'mode': 'state',
                'note': 'Z component of wave-vector transfer',
                'type': 'float32',
                'units': '1/√Ö',
            },
        },
        'type': 'virtual',
    },
    # 'beam': {
    #     'description': '',
    #     'fields': {
    #         'angularOffsets': {
    #             'error': 0.001,
    #             'label': 'beam angular offsets',
    #             'mode': 'state',
    #             'note': 'the angular offset of each of the 4 beams in multibeam mode',
    #             'type': 'float32[]',
    #             'units': '¬∞',
    #         },
    #     },
    #     'type': 'virtual',
    # },
    'detectorTable': {
        'description': '',
        'fields': {
            'angularSpreads': {
                'error': 0.001,
                'label': 'detector table angular spreads',
                'mode': 'state',
                'note': 'The angular spread each neutrons detector receives Is constant per row of detectors',
                'type': 'float32[]',
                'units': '¬∞',
            },
            'rowAngularOffsets': {
                'error': 0.001,
                'label': 'detector table row angular offsets',
                'mode': 'state',
                'note': 'Angular offset of each row of detectors from detectorTableMotor‚Äôs angle',
                'type': 'float32[]',
                'units': '¬∞',
            },
            'wavelengthSpreads': {
                'error': 0.001,
                'label': 'detector table wavelength spreads',
                'mode': 'state',
                'note': 'The wavelength variation of neutrons each detector receives',
                'type': 'float32[]',
                'units': '',
            },
            'wavelengths': {
                'error': 0.001,
                'label': 'detector table wavelengths',
                'mode': 'state',
                'note': 'The wavelength of neutrons each detector is setup to detect',
                'type': 'float32[]',
                'units': '√Ö',
            },
        },
        'type': 'virtual',
    },
    'fastShutter': {
        'description': 'Fast shutter device',
        'fields': {
            'openState': {
                'label': 'fast shutter',
                'mode': 'state',
                'note': 'Indicates whether the Fast shutter is open or closed. If closed, could be due to an overcount or mechanical problem.',
                'type': 'bool',
            },
        },
        'primary': 'openState',
        'type': 'bit',
    },
    'frontPolarization': {
        'description': 'The state of the polarizer component - can be UP, DOWN, or OUT.  UP or DOWN indicate the polarization of neutrons passing through the device, while OUT indicates the device is not in the beam.',
        'fields': {
            'direction': {
                'label': 'front polarization',
                'mode': 'state',
                'note': '',
                'options': ['UP', 'DOWN', 'UNPOLARIZED'],
                'type': 'string',
            },
            'inBeam': {
                'label': 'front polarization in beam',
                'mode': 'state',
                'note': 'For systems which have motor-controlled spin filters, this controls whether that filter is in the IN or OUT position, otherwise manually set to indicate whether the filter is in or out',
                'type': 'bool',
            },
            'type': {
                'label': 'front polarization type',
                'mode': 'configure',
                'note': 'Type of polarization device (e.g. MEZEI, RF, HE3)',
                'type': 'string',
            },
        },
        'primary': 'direction',
        'type': 'polarization',
    },
    'frontSubDirection': {
        'description': 'The polarization direction of neutrons that would be passed through if this polarization unit is in the beam',
        'fields': {
            'flip': {
                'label': 'front sub direction flip',
                'mode': 'configure',
                'note': 'The state of the spin flipper - true = on and false = off',
                'type': 'bool',
            },
            'spinFilter': {
                'label': 'front sub direction spin filter',
                'mode': 'configure',
                'note': 'The polarization direction of neutrons that would be passed through if this polarization unit is in the beam',
                'options': ['UP', 'DOWN'],
                'type': 'string',
            },
            'substate': {
                'label': 'front sub direction',
                'mode': 'configure',
                'note': '',
                'options': ['UP', 'DOWN'],
                'type': 'string',
            },
        },
        'primary': 'substate',
        'type': 'virtual',
    },
    'mono': {
        'description': '',
        'fields': {
            'wavelength': {
                'error': 0.001,
                'label': 'mono wavelength',
                'mode': 'state',
                'note': 'The incoming neutron wavelength when the monochromator is in the beam.',
                'type': 'float32',
                'units': '√Ö',
            },
            'wavelengthSpread': {
                'error': 0.001,
                'label': 'mono wavelength spread',
                'mode': 'state',
                'note': 'The incoming neutron wavelength spread when the monochromator is in the beam.',
                'type': 'float32',
                'units': '',
            },
        },
        'type': 'virtual',
    },
    # # Reactor power sensors
    # 'reactorPower': {
    #     'description': 'Device for reactor power',
    #     'fields': {
    #         'coldSourcePressure': 'NXlog',
    #         'coldSourceTemperature': 'NXlog',
    #         'reactorPowerPercent': 'NXlog',
    #         'reactorPowerThermal': 'NXlog',
    #         'reactorState': 'NXlog',
    #     },
    # },
    'rfFlipperPowerSupply': {
        'description': '',
        'fields': {
            'outputEnabled': {
                'label': 'rf flipper power supply',
                'mode': 'state',
                'note': 'Node representing the output enabled property of the RF Flipper power supply',
                'type': 'bool',
            },
        },
        'primary': 'outputEnabled',
        'type': 'power_supply',
    },
    'sample': {
        'description': 'Device holding information about the sample in the beam.',
        'fields': {
            'description': {
                'label': 'sample description',
                'mode': 'state',
                'note': 'A description of the sample.',
                'type': 'string',
            },
            'id': {
                'label': 'sample id',
                'mode': 'state',
                'note': 'The id of the sample.',
                'type': 'string',
            },
            'mass': {
                'error': 0.001,
                'label': 'sample mass',
                'mode': 'state',
                'note': 'The mass of the sample.',
                'type': 'float32',
                'units': 'g',
            },
            'name': {
                'label': 'sample',
                'mode': 'state',
                'note': 'The name of the sample.',
                'type': 'string',
            },
            'thickness': {
                'error': 0.001,
                'label': 'sample thickness',
                'mode': 'state',
                'note': 'The thickness of the sample.',
                'type': 'float32',
                'units': 'mm',
            },
        },
        'primary': 'name',
        'type': 'virtual',
    },
    'sampleIndex': {
        'description': 'Sample selection device.',
        'fields': {
            'index': {
                'label': 'sample index',
                'mode': 'state',
                'note': 'The sample index.  Changing the sample index will cause sample property nodes to change their values to correspond to the newly selected sample.  It may also cause the selected sample to be moved into position.',
                'type': 'int32',
                'units': '',
            },
        },
        'primary': 'index',
        'type': 'virtual',
    },
    ## No longer part of the design.
    # 'slit1a': {
    #     'description': 'Multiblade slit device model device.',
    #     'fields': {
    #         'openingWidth': {
    #             'error': 0.001,
    #             'label': 'slit1a',
    #             'mode': 'state',
    #             'note': 'Width of the opening between the two blades.',
    #             'type': 'float32',
    #             'units': 'mm',
    #         },
    #     },
    #     'primary': 'openingWidth',
    #     'type': 'motor',
    # },
    # 'slit1b': {
    #     'description': 'Multiblade slit device model device.',
    #     'fields': {
    #         'openingWidth': {
    #             'error': 0.001,
    #             'label': 'slit1b',
    #             'mode': 'state',
    #             'note': 'Width of the opening between the two blades.',
    #             'type': 'float32',
    #             'units': 'mm',
    #         },
    #     },
    #     'primary': 'openingWidth',
    #     'type': 'motor',
    # },
    # 'slit1c': {
    #     'description': 'Multiblade slit device model device.',
    #     'fields': {
    #         'openingWidth': {
    #             'error': 0.001,
    #             'label': 'slit1c',
    #             'mode': 'state',
    #             'note': 'Width of the opening between the two blades.',
    #             'type': 'float32',
    #             'units': 'mm',
    #         },
    #     },
    #     'primary': 'openingWidth',
    #     'type': 'motor',
    # },
    # 'slit1d': {
    #     'description': 'Multiblade slit device model device.',
    #     'fields': {
    #         'openingWidth': {
    #             'error': 0.001,
    #             'label': 'slit1d',
    #             'mode': 'state',
    #             'note': 'Width of the opening between the two blades.',
    #             'type': 'float32',
    #             'units': 'mm',
    #         },
    #     },
    #     'primary': 'openingWidth',
    #     'type': 'motor',
    # },
    'ttl': {
        'description': 'Hardware ttl viper device',
        'fields': {
            'backgroundPollPeriod': {
                'error': 0.001,
                'label': 'ttl background poll period',
                'mode': 'configure',
                'note': 'The default time period between successive polls of the background-polled hardware properties of this device.  Positive infinity means poll once then never again.  NaN means never poll.',
                'type': 'float32',
                'units': 's',
            },
            'out_0': {
                'label': 'ttl out 0',
                'mode': 'log',
                'note': 'Proxy for driver property "out_0".',
                'type': 'bool',
            },
            'out_1': {
                'label': 'ttl out 1',
                'mode': 'log',
                'note': 'Proxy for driver property "out_1".',
                'type': 'bool',
            },
            'out_10': {
                'label': 'ttl out 10',
                'mode': 'log',
                'note': 'Proxy for driver property "out_10".',
                'type': 'bool',
            },
            'out_11': {
                'label': 'ttl out 11',
                'mode': 'log',
                'note': 'Proxy for driver property "out_11".',
                'type': 'bool',
            },
            'out_12': {
                'label': 'ttl out 12',
                'mode': 'log',
                'note': 'Proxy for driver property "out_12".',
                'type': 'bool',
            },
            'out_13': {
                'label': 'ttl out 13',
                'mode': 'log',
                'note': 'Proxy for driver property "out_13".',
                'type': 'bool',
            },
            'out_14': {
                'label': 'ttl out 14',
                'mode': 'log',
                'note': 'Proxy for driver property "out_14".',
                'type': 'bool',
            },
            'out_15': {
                'label': 'ttl out 15',
                'mode': 'log',
                'note': 'Proxy for driver property "out_15".',
                'type': 'bool',
            },
            'out_16': {
                'label': 'ttl out 16',
                'mode': 'log',
                'note': 'Proxy for driver property "out_16".',
                'type': 'bool',
            },
            'out_17': {
                'label': 'ttl out 17',
                'mode': 'log',
                'note': 'Proxy for driver property "out_17".',
                'type': 'bool',
            },
            'out_18': {
                'label': 'ttl out 18',
                'mode': 'log',
                'note': 'Proxy for driver property "out_18".',
                'type': 'bool',
            },
            'out_19': {
                'label': 'ttl out 19',
                'mode': 'log',
                'note': 'Proxy for driver property "out_19".',
                'type': 'bool',
            },
            'out_2': {
                'label': 'ttl out 2',
                'mode': 'log',
                'note': 'Proxy for driver property "out_2".',
                'type': 'bool',
            },
            'out_20': {
                'label': 'ttl out 20',
                'mode': 'log',
                'note': 'Proxy for driver property "out_20".',
                'type': 'bool',
            },
            'out_21': {
                'label': 'ttl out 21',
                'mode': 'log',
                'note': 'Proxy for driver property "out_21".',
                'type': 'bool',
            },
            'out_22': {
                'label': 'ttl out 22',
                'mode': 'log',
                'note': 'Proxy for driver property "out_22".',
                'type': 'bool',
            },
            'out_23': {
                'label': 'ttl out 23',
                'mode': 'log',
                'note': 'Proxy for driver property "out_23".',
                'type': 'bool',
            },
            'out_3': {
                'label': 'ttl out 3',
                'mode': 'log',
                'note': 'Proxy for driver property "out_3".',
                'type': 'bool',
            },
            'out_4': {
                'label': 'ttl out 4',
                'mode': 'log',
                'note': 'Proxy for driver property "out_4".',
                'type': 'bool',
            },
            'out_5': {
                'label': 'ttl out 5',
                'mode': 'log',
                'note': 'Proxy for driver property "out_5".',
                'type': 'bool',
            },
            'out_6': {
                'label': 'ttl out 6',
                'mode': 'log',
                'note': 'Proxy for driver property "out_6".',
                'type': 'bool',
            },
            'out_7': {
                'label': 'ttl out 7',
                'mode': 'log',
                'note': 'Proxy for driver property "out_7".',
                'type': 'bool',
            },
            'out_8': {
                'label': 'ttl out 8',
                'mode': 'log',
                'note': 'Proxy for driver property "out_8".',
                'type': 'bool',
            },
            'out_9': {
                'label': 'ttl out 9',
                'mode': 'log',
                'note': 'Proxy for driver property "out_9".',
                'type': 'bool',
            },
        },
        'type': 'hardware',
    },
}

class Candor(Instrument): # dimensions in millimeters
    MONOCHROMATOR_Z = -5216.5
    SOURCE_APERTURE_Z = -4600. # TODO: missing this number
    SOURCE_APERTURE = 60.
    SOURCE_SLIT_Z = -4335.86
    PRE_SAMPLE_SLIT_Z = -356.0
    POST_SAMPLE_SLIT_Z = 356.0
    DETECTOR_Z = 3496.
    SOURCE_LOUVER_Z = -4403.026

    DETECTOR_MASK_HEIGHT = 30.
    DETECTOR_MASK_WIDTHS = [10., 8., 6., 4.]
    DETECTOR_MASK_N = 30  # Must be a multiple of 3
    #DETECTOR_MASK_N = 3
    DETECTOR_MASK_SEPARATION = 12.84
    DETECTOR_WIDTH = (DETECTOR_MASK_N+1)*DETECTOR_MASK_SEPARATION
    DETECTOR_LEAF = 54  # Max of 54
    #DETECTOR_LEAF = 6  # Max of 54

    SOURCE_LOUVER_N = 4
    SOURCE_LOUVER_SEPARATION = 15.5  # center-center distance for source multi-slits
    SOURCE_LOUVER_CENTERS = np.linspace(-1.5*SOURCE_LOUVER_SEPARATION,
                                        +1.5*SOURCE_LOUVER_SEPARATION,
                                        SOURCE_LOUVER_N)
    SOURCE_LOUVER_ANGLES = np.arctan2(SOURCE_LOUVER_CENTERS, -SOURCE_LOUVER_Z)
    SOURCE_LOUVER_MAX = 14.5  # maximum opening for source multi-slit

    areaDetector = Detector(description="The main area detector for Candor",
                            dimension=[DETECTOR_MASK_N//3*2, DETECTOR_LEAF], offset=0, strides=[DETECTOR_LEAF, 1])
    attenuator = Map(label="attenuator", types=("int32", "float32"), description="CANDOR available attenuators.")
    attenuatorMotor = Motor(label="attenuator motor", units="cm", description="CANDOR attenuator motor.")
    convergingGuide = Motor(description="Horizontal converging guide", label="guide width", units="mm")
    convergingGuideMap = InOut(label="converging guide")
    counter = Counter()
    detectorMaskMap = Map(label="detector mask", types=("string", "float32"), description="")
    detectorMaskMotor = Motor(description="Vertically translates a mask over all detectors allowing for varying beam widths.", label="detector mask motor", units="mm")
    detectorTableMotor = Motor(description="Scattering Angle", label="detector table motor", units="degree")
    experiment = Experiment("CANDOR")
    monoTrans = InOut(label="monochromator")
    monoTransMotor = Motor(label="monochromator translator", units="mm", description="Translate the monochromator into and out of the beam path")
    polarizerTrans = Motor(description="Translates the polarizer in and out of the beam", label="polarizer trans", units="mm")
    rateMeter = RateMeter()

    sampleAngleMotor = Motor(description="Sample rotation", label="sample angle", units="degree")
    sampleIndexToDescription = Map(label="sample index to description", types=("int32", "string"), description="")
    sampleIndexToID = Map(label="sample index to ID", types=("int32", "string"), description="")
    sampleIndexToMass = Map(label="sample index to mass", types=("int32", "float32"), description="")
    sampleIndexToName = Map(label="sample index to name", types=("int32", "string"), description="")
    sampleIndexToThickness = Map(label="sample index to thickness", types=("int32", "float32"), description="")
    sampleTiltX = Motor(description="sample lower tilt", label="sample tilt x", units="degree")
    sampleTiltY = Motor(description="sample upper tilt", label="sample tilt y", units="degree")
    sampleTransX = Motor(description="sample lower translation", label="sample offset x", units="mm")
    sampleTransY = Motor(description="sample upper translation", label="sample offset y", units="mm")
    singleSlitApertureMap = InOut(label="source slit")
    slitAperture1 = Motor(description="source slit", label="source slit", units="mm")
    slitAperture2 = Motor(description="presample slit", label="presample slit", units="mm")
    slitAperture3 = Motor(description="postsample slit", label="postsample slit", units="mm")
    #multiSlit1TransMap = InOut(label="multislit")
    #multiSlitTransMotor = Motor(description="multislit stage translation", label="multislit", units="mm")
    #multiBladeSlit1aMotor = Motor(description="beam 1 source slit", label="multislit1", units="degree")
    #multiBladeSlit1bMotor = Motor(description="beam 2 source slit", label="multislit2", units="degree")
    #multiBladeSlit1cMotor = Motor(description="beam 3 source slit", label="multislit3", units="degree")
    #multiBladeSlit1dMotor = Motor(description="beam 4 source slit", label="multislit4", units="degree")
    trajectory = Trajectory()
    trajectoryData = TrajectoryData()

    monoTheta = Motor(description="monochromator angle", label="monocromator theta", units="degree")

    #deflectorTrans = Motor(description=">6 \u00c5 deflector", label="deflector", units="mm")
    #deflectorMap = InOut(label="deflector")
    #MezeiPolarizerMap = InOut(label="Mezei polarizer")
    #He3PolarizerMap = InOut(label="He3 polarizer")
    #FrontRFFlipperMap = InOut(label="front flipper")
    #MezeiAnalyzerMap = InOut(label="Mezei analyzer")
    #He3AnalyzerMap = InOut(label="He3 analyzer")
    #RearRFFlipperMap = InOut(label="rear flipper")

    #singleSlitApertureTrans = Motor(description="source slit stage translation", label="slit translation", units="mm")
    #singleSlitAperture1 = Motor(description="source slit", label="source slit", units="mm")

    @property
    def wavelengths(self):
        """
        Return wavelengths for each detector as a 2D array.
        """
        num_bank = self.detectorTable.rowAngularOffsets.size
        return np.reshape(self.detectorTable.wavelengths, (-1, num_bank)).T

# Add the virtual devices
for k, v in devices.items():
    setattr(Candor, k, Virtual(**v))
Candor.set_device_ids()
del Candor.ttl # skip this for now since it isn't needed for reduction

def load_spectrum():
    """
    Return the incident spectrum
    """
    datadir = os.path.abspath(os.path.dirname(__file__))
    L, I_in = np.loadtxt(os.path.join(datadir, 'CANDOR-incident.dat')).T
    _, I_out = np.loadtxt(os.path.join(datadir, 'CANDOR-detected.dat')).T
    L, I_in, I_out = L[::-1], I_in[::-1], I_out[::-1]
    # Maybe truncate the detector banks
    assert len(L) >= Candor.DETECTOR_LEAF
    s = slice(0, Candor.DETECTOR_LEAF)
    L, I_in, I_out = L[s], I_in[s], I_out[s]
    return L, I_in, L, I_out/I_in


def comb(n, width, separation):
    """
    Return bin edges with *n* bins.

    *separation* is the distance between bin centers and *width* is the
    size of each bin.
    """
    # Form n centers from n-1 intervals between centers, with
    # center-to-center spacing set to separation.  This puts the
    # final center at (n-1)*separation away from the first center.
    # Divide by two to arrange these about zero, giving (-limit, limit).
    # Edges are +/- width/2 from the centers.
    limit = separation*(n-1)/2
    centers = np.linspace(-limit, limit, n)
    edges = np.vstack([centers-width/2, centers+width/2]).T.flatten()
    return edges

def detector_mask(width=10.):
    """
    Return slit edges for candor detector mask.
    """
    #width = DETECTOR_MASK_WIDTHS[mask]
    edges = comb(n=Candor.DETECTOR_MASK_N,
                 width=width,
                 separation=Candor.DETECTOR_MASK_SEPARATION)
    # Every 3rd channel is dead (used for cooling)
    edges = edges.reshape(-1, 3, 2)[:, :2, :].flatten()
    # Aim at the center of the first bank
    edges -= (edges[0]+edges[1])/2
    return edges

def wavelength_dispersion(L, mask=10., detector_distance=Candor.DETECTOR_Z):
    """
    Return estimated wavelength dispersion for the candor channels.

    *L* (Ang) is the wavelength measured for each channel.

    *mask* (mm) is the mask opening at the start of each detector channel.

    *detector_distance* (mm) is the distance from the center of the sample
    to the detector mask.

    Note: result is 0.01 below the value shown in Fig 9 of ref.

    Running some approx. numbers to decide if changing the mask will
    significantly change the resolution::

        dQ/Q @ 1 degree for detectors between 4 A and 6 A

        4 mm:  [0.85 0.73 0.60 0.48 0.36 0.23]
        10 mm: [0.96 0.85 0.73 0.61 0.50 0.39]

    This value is dominated by the wavelength spread.

    **References**

    Jeremy Cook, "Estimates of maximum CANDOR detector count rates on NG-1"
    Oct 27, 2015
    """
    #: (Ang) d-spacing for graphite (002)
    dspacing = 3.354

    #: (rad) FWHM mosaic spread for graphite
    eta = radians(30./60.)  # convert arcseconds to radians

    #: (rad) FWHM incident divergence
    a0 = mask/detector_distance

    #: (rad) FWHM outgoing divergence
    a1 = a0 + 2*eta

    #: (rad) bragg angle for wavelength selector
    theta = arcsin(L/2/dspacing)
    #pylab.plot(degrees(theta))

    #: (unitless) FWHM dL/L, as given in Eqn 6 of the reference
    dLoL = (sqrt((a0**2*a1**2 + (a0**2 + a1**2)*eta**2)/(a0**2+a1**2+4*eta**2))
            / tan(theta))
    return dLoL

def candor_setup():
    # Save the incident/detected spectrum as a Candor class attribute
    Candor.spectrum = load_spectrum()

    # Multibeam beam centers and angles
    # Note: if width is 0, then both edges are at the center
    beam_centers = comb(4, 0, Candor.SOURCE_LOUVER_SEPARATION)[::2]
    beam_angles = arctan(beam_centers/-Candor.SOURCE_LOUVER_Z)

    # Detector bank wavelengths and angles
    # Note: assuming flat detector bank; a curved bank will give very slightly
    # different answers.
    wavelengths = Candor.spectrum[0]
    bank_centers = detector_mask(width=0.)[::2]
    bank_angles = arctan(bank_centers/Candor.DETECTOR_Z)
    #print("bank centers", bank_centers)
    #print("bank angles", degrees(bank_angles))
    #print("beam angles", degrees(beam_angles))

    num_leaf = len(wavelengths)
    num_bank = len(bank_angles)
    angular_spreads = 2.865*np.ones(num_bank*num_leaf)  # from nice vm
    wavelength_spreads = wavelength_dispersion(wavelengths)
    wavelength_spreads = 0.01*np.ones(num_bank*num_leaf)  # from nice vm
    #L = 6. - 0.037*np.arange(54)  # from nice vm
    wavelength_array = np.tile(wavelengths, (num_bank, 1)).T.flatten()

    # Initialize candor with fixed info fields
    candor = Candor()
    candor.load_nexus()
    candor.move(
        beam_angularOffsets=degrees(beam_angles),
        detectorTable_angularSpreads=angular_spreads,
        detectorTable_rowAngularOffsets=degrees(bank_angles),
        detectorTable_wavelengthSpreads=wavelength_spreads,
        detectorTable_wavelengths=wavelength_array,
    )

    # Initialize default values
    candor.move(
        Q_angleIndex=0,
        Q_wavelengthIndex=0,
        Q_beamIndex=0,
        #multiSlit1TransMap="OUT",
        singleSlitApertureMap="IN",
        monoTrans="OUT",
        mono_wavelength=4.75,
        mono_wavelengthSpread=0.01,
    )

    return candor

def demo():
    import time
    T0 = time.mktime(time.strptime("2018-01-01 12:00:00", "%Y-%m-%d %H:%M:%S"))
    candor = candor_setup()
    stream = nice.StreamWriter(candor, timestamp=T0)
    # This prints the record to the screen.  To save to a .stream.bz2 file
    # use the same methods without '_record'.
    print(stream.config_record())
    print(stream.open_record())
    candor.sampleAngleMotor.softPosition = 5
    print(stream.state_record(2.))
    candor.move(counter_liveMonitor=121, counter_liveTime=200.0)
    print(stream.counts_record())
    print(stream.close_record())
    print(stream.end_record())

if __name__ == "__main__":
    print(np.degrees(Candor.SOURCE_LOUVER_ANGLES))
    #demo()
