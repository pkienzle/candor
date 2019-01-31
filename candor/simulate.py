"""
Program to explore the beam profile and angular distribution for a simple
reflectometer with two front slits.
"""
from __future__ import division, print_function

import sys
import os

import numpy as np
from numpy import (sin, cos, tan, arcsin, arccos, arctan, arctan2, radians, degrees,
                   sign, sqrt, exp, log, std)
from numpy import pi, inf
import pylab

from .instrument import Candor
#from .nice import Instrument, Motor

# dimensions in millimeters
MONOCHROMATOR_Z = -5216.5
SOURCE_APERTURE_Z = -4600. # TODO: missing this number
SOURCE_APERTURE = 60.
SOURCE_LOUVER_Z = -4403.026
SOURCE_LOUVER_N = 4
SOURCE_LOUVER_SEPARATION = 15.5  # center-center distance for source multi-slits
SOURCE_LOUVER_MAX = 14.5  # maximum opening for source multi-slit
SOURCE_SLIT_Z = -4335.86
PRE_SAMPLE_SLIT_Z = -356.0
POST_SAMPLE_SLIT_Z = 356.0
DETECTOR_MASK_HEIGHT = 30.
DETECTOR_MASK_WIDTHS = [10., 8., 6., 4.]
DETECTOR_MASK_N = 30
DETECTOR_MASK_SEPARATION = 12.84
DETECTOR_Z = 3496.
DETECTOR_WIDTH = (DETECTOR_MASK_N+1)*DETECTOR_MASK_SEPARATION

WAVELENGTH_MIN = 4.
WAVELENGTH_MAX = 6.
WAVELENGTH_N = 54

SOURCE_LOUVER_CENTERS = np.linspace(-1.5*SOURCE_LOUVER_SEPARATION,
                                    1.5*SOURCE_LOUVER_SEPARATION,
                                    SOURCE_LOUVER_N)
SOURCE_LOUVER_ANGLES = np.arctan2(SOURCE_LOUVER_CENTERS, -SOURCE_LOUVER_Z)


def detector_mask(width=10.):
    """
    Return slit edges for candor detector mask.
    """
    #width = DETECTOR_MASK_WIDTHS[mask]
    edges = comb(n=DETECTOR_MASK_N,
                 width=width,
                 separation=DETECTOR_MASK_SEPARATION)
    # Every 3rd channel is dead (used for cooling)
    edges = edges.reshape(-1, 3, 2)[:, :2, :].flatten()
    return edges

def choose_sample_slit(louver, sample_width, sample_angle):
    theta = radians(sample_angle)
    index = np.nonzero(louver)[0]
    k = index[-1]
    x0, y0 = SOURCE_LOUVER_Z, SOURCE_LOUVER_CENTERS[k] + louver[k]/2
    x1, y1 = sample_width/2*cos(theta), sample_width/2*sin(theta)
    top = (y1-y0)/(x1-x0)*(PRE_SAMPLE_SLIT_Z - x1) + y1
    k = index[0]
    x0, y0 = SOURCE_LOUVER_Z, SOURCE_LOUVER_CENTERS[k] - louver[k]/2
    x1, y1 = -sample_width/2*cos(theta), -sample_width/2*sin(theta)
    bottom = (y1-y0)/(x1-x0)*(PRE_SAMPLE_SLIT_Z - x1) + y1
    #print(top, bottom)
    #slit = 2*max(top, -bottom)
    slit = 2*max(abs(top), abs(bottom))
    return slit

def load_spectrum():
    """
    Return the incident spectrum
    :return:
    """
    datadir = os.path.abspath(os.path.dirname(__file__))
    L, I_in = np.loadtxt(os.path.join(datadir, 'CANDOR-incident.dat')).T
    _, I_out = np.loadtxt(os.path.join(datadir, 'CANDOR-detected.dat')).T
    return L, I_in, L, I_out/I_in

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

def rotate(xy, theta):
    """
    Rotate points x,y through angle theta.
    """
    sin_theta, cos_theta = sin(theta), cos(theta)
    R = np.array([[cos_theta, -sin_theta],[sin_theta, cos_theta]])
    return np.dot(R, xy)

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

# TODO: avoid candor specific properties in simulation engine
class Neutrons(object):
    #: angle of the x-axis relative to source-sample line (rad)
    beam_angle = 0.
    #: x,y position of each neutron, with 0,0 at the center of rotation (mm)
    xy = None # type: np.ndarray
    #: direction of travel relative to beam (radians)
    angle = None # type: np.ndarray
    #: neutron wavelength (Angstroms)
    wavelength = None # type: np.ndarray
    #: relative intensity of individual neutrons, which depends on spectral
    #: weight, the percentage transmitted/reflected, and the percentage
    #: detected relative to incident.
    weight = None # type: np.ndarray
    #: set of neutrons masked by slits
    active = None  # type: np.ndarray
    #: 54 x 3 array of lambda, incident intensity (n/cm^2), detected intensity
    spectrum = None # type: np.ndarray
    #: sample angle if there is a sample in the beam
    sample_angle = 0.
    #: source beam for each neutron
    source = None # type: np.ndarray

    @classmethod
    def set_spectrum(cls, spectrum):
        cls.spectrum = spectrum

    @property
    def x(self):
        return self.xy[1]
    @x.setter
    def x(self, v):
        self.xy[1] = v

    @property
    def z(self):
        return self.xy[0]
    @z.setter
    def z(self, v):
        self.xy[0] = v

    def __init__(self, n, divergence=5., spectrum=None, trace=False):
        # type: (int, float, np.ndarray, float, Union[float, np.ndarray], bool) -> None
        if spectrum is not None:
            self.spectrum = spectrum
        L, I = spectrum[0], spectrum[1]
        I_weighted = I/np.sum(I)

        self.xy = np.zeros((2,n),'d')
        self.angle = (np.random.rand(n)-0.5)*radians(divergence)
        self.wavelength = np.random.rand(n)*L.ptp() + L.min()
        self.weight = np.interp(self.wavelength, L, I_weighted)
        self.active = (self.x == self.x)

        self.history = []
        self.trace = trace
        self.elements = []

    def converging_source(self, z, width, sample_low, sample_high):
        n = self.xy.shape[1]
        self.z = z
        x = (np.random.rand(n)-0.5)*width
        y = ((np.random.rand(n)-0.5)*(sample_high-sample_low)
             + (sample_high + sample_low)/2)
        self.x = x
        self.angle = arctan2(y-x, -z)
        self.source = 0
        self.add_trace()
        self.add_element((z, -width), (z, -width/2))
        self.add_element((z, width/2), (z, width))

    def slit_source(self, z, width, target=inf):
        n = self.xy.shape[1]
        self.z = z
        self.x = (np.random.rand(n)-0.5)*width
        self.angle += arctan(self.x/(self.z - target))
        self.source = 0
        self.add_trace()
        self.add_element((z, -width), (z, -width/2))
        self.add_element((z, width/2), (z, width))

    def comb_source(self, z, widths, separation):
        n = self.xy.shape[1]
        num_slits = len(widths)
        limit = separation*(num_slits-1)/2
        centers = np.linspace(-limit, limit, num_slits)
        edges = np.vstack([centers-widths/2, centers+widths/2]).T.flatten()
        break_points = np.cumsum(widths) + edges[0]
        total_width = break_points[-1] - edges[0]
        spaces = np.diff(edges)[1::2]
        shift = np.hstack([0, np.cumsum(spaces)])
        x = edges[0] + np.random.rand(n)*total_width
        index = np.searchsorted(break_points[:-1], x)
        self.source = index
        self.z = z
        self.x = x + shift[index]
        # Aim the center of the divergence at the pre-sample slit position
        # rather than the sample position
        self.angle += arctan(self.x/(self.z - PRE_SAMPLE_SLIT_Z))
        #self.angle += arctan(self.x/self.z)
        #self.angle -= SOURCE_LOUVER_ANGLES[index]
        self.add_trace()
        self.add_element((z, edges[0]-widths[0]/2), (z, edges[0]))
        for x1, x2 in pairwise(edges[1:-1]):
            self.add_element((z, x1), (z, x2))
        self.add_element((z, edges[-1]), (z, edges[-1]+widths[-1]/2))

    def trim(self):
        self.xy = self.xy[:,self.active]
        self.angle = self.angle[self.active]
        self.wavelength = self.wavelength[self.active]
        self.weight = self.weight[self.active]
        self.active = self.active[self.active]
        if self.source is not None:
            self.source = self.source[self.active]
        self.add_trace()

    def add_element(self, zx1, zx2):
        self.elements.append((rotate(zx1, -self.beam_angle),
                              rotate(zx2, -self.beam_angle)))
    def add_trace(self):
        if self.trace:
            self.history.append((rotate(self.xy, -self.beam_angle),
                                 self.active&True))

    def clear_trace(self):
        self.history = []

    def plot_trace_old(self, split=False):
        from matplotlib.lines import Line2D
        colors = 'rgby'
        for k, (zx, active) in enumerate(self.history[:-1]):
            zx_next, active_next = self.history[k+1]
            if active.shape != active_next.shape: continue
            z = np.vstack((zx[0], zx_next[0]))
            x = np.vstack((zx[1], zx_next[1]))
            # TODO: source will be the wrong length after trim
            for k, c in enumerate(colors):
                index = active & (self.source == k)
                if index.any():
                    if split:
                        pylab.subplot(2,2,k+1)
                    pylab.plot(z[:,index], x[:,index], '-', color=c, linewidth=0.1)

        entries = []
        for k, c in enumerate(colors):
            angle = degrees(SOURCE_LOUVER_ANGLES[k]) + self.sample_angle
            label = '%.3f degrees'%angle
            if split:
                pylab.subplot(2,2,k+1)
                pylab.title(label)
            else:
                line = Line2D([],[],color=c,marker=None,
                              label=label, linestyle='-')
                entries.append(line)
        if not split:
            pylab.legend(handles=entries, loc='best')
        #pylab.axis('equal')

        # draw elements, cutting slits off at the bounds of the data
        if split:
            for k, c in enumerate(colors):
                #pylab.axis('equal')
                #pylab.grid(True)
                pylab.subplot(2,2,k+1)
                for (z1, x1), (z2, x2) in self.elements:
                    pylab.plot([z1, z2], [x1, x2], 'k')
        else:
            #pylab.axis('equal')
            #pylab.grid(True)
            for (z1, x1), (z2, x2) in self.elements:
                pylab.plot([z1, z2], [x1, x2], 'k')

    def angle_hist(self):
        from scipy.stats import gaussian_kde
        pylab.figure(2)
        active_angle = self.angle[self.active] if self.active.any() else self.angle
        angles = degrees(active_angle) + self.sample_angle
        if 0:
            n = len(angles)
            x = np.linspace(angles.min(), angles.max(), 400)
            mu, sig = angles.mean(), angles.std()
            pdf = gaussian_kde((angles-mu)/sig, bw_method=0.001*n**0.2)
            pylab.plot(x, pdf(x)*sig + mu)
        else:
            pylab.hist(angles, bins=50, normed=True)
        pylab.xlabel('angle (degrees)')
        pylab.ylabel('P(angle)')
        pylab.figure(1)

    def plot_trace(self, split=None):
        from matplotlib.collections import LineCollection
        import matplotlib.colors as mcolors
        import matplotlib.cm as mcolormap

        active_angle = self.angle[self.active] if self.active.any() else self.angle
        active_angle = degrees(active_angle) + self.sample_angle
        vmin, vmax = active_angle.min(), active_angle.max()
        vpad = 0.05*(vmax-vmin)
        cnorm = mcolors.Normalize(vmin=vmin-vpad, vmax=vmax+vpad)
        cmap = mcolormap.ScalarMappable(norm=cnorm, cmap=pylab.get_cmap('jet'))
        colors = cmap.to_rgba(degrees(self.angle) + self.sample_angle)
        #colors = cmap.to_rgba(active_angle)
        for k, (zx, active) in enumerate(self.history[:-1]):
            zx_next, active_next = self.history[k+1]
            if active.shape != active_next.shape:
                continue
            if active.any():
                segs = np.hstack((zx[:, active].T, zx_next[:, active].T))
                segs = segs.reshape(-1, 2, 2)
                lines = LineCollection(segs, linewidth=0.1,
                                       linestyle='solid', colors=colors[active])
                pylab.gca().add_collection(lines)
                #pylab.plot(z[:,index], x[:,index], '-', color=c, linewidth=0.1)

        # draw elements, cutting slits off at the bounds of the data
        #pylab.axis('equal')
        #pylab.grid(True)
        for (z1, x1), (z2, x2) in self.elements:
            pylab.plot([z1, z2], [x1, x2], 'k')
        cmap.set_array(active_angle)
        h = pylab.colorbar(cmap)
        h.set_label('angle (degrees)')

    def plot_points(self):
        x, y = rotate(self.xy, -self.beam_angle)
        pylab.plot(x, y, '.')

    def detector_angle(self, angle):
        """
        Set the detector angle
        """
        self.rotate_rad(-radians(angle))

    def radial_collimator(self, z, n, w1, w2, length):
        raise NotImplementedError()

    def rotate_rad(self, angle):
        """
        Rotate the coordinate system through *angle*.
        """
        self.beam_angle += angle
        self.xy = rotate(self.xy, angle)
        self.angle += angle

    def move(self, z, add_trace=True):
        """
        Move neutrons to position z.
        """
        # Project neutrons onto perpendicular plane at the sample position
        dz = z - self.z
        self.x = dz*tan(self.angle) + self.x
        self.z = z
        if add_trace:
            self.add_trace()

    def sample(self, angle, width=100., offset=0., bow=0., diffuse=0.,
               refl=lambda kz: np.ones(kz.shape)):
        """
        Reflect off the sample.

        *angle* (degrees) is the angle of the sample plane relative
        to the beam.

        *width* (mm) is the width of the sample.

        *offset* (mm) is the offset of the sample from the center of rotation.

        *bow* (mm) is the amount of bowing in the sample, which is
        the height of the sample surface relative at the center of the
        sample relative to the edges. *bow* is positive for samples that
        are convex and negative for concave.

        *diffuse* (unitless) proportion of samples scattered in 4 pi.
        Note that we are using 2 pi rather than 4 pi for now since our model
        does not include vertical slits.  For a more complete treatment, the
        majority of the diffuse scattering events should be treated as
        absorption events, with only those incident on the solid angle of
        the detector bank propagated.  The diffuse scattering is proportional
        to the number of neutrons incident on the sample, not the number
        reflected, so even small rates of diffuse scattering may lead to
        significant crosstalk between channels.

        *refl* is a function which takes angle and wavelength, returning
        proportion of reflected neutrons.  No absorption in this model.
        """
        if bow != 0.:
            raise NotImplementedError("sample bow not supported")
        theta = radians(angle)

        # Determine where the neutrons intersect the plane through the
        # center of sample rotation (i.e., move them to z = 0)
        x = self.x - self.z * tan(self.angle)

        # Intersection of sample plane with individual neutrons
        # Note: divide by zero where self.angle == theta
        if theta == 0:
            xp = np.zeros_like(self.x)
            zp = self.z - self.x / tan(self.angle)
        else:
            xp = tan(theta) * x / (tan(theta) - tan(self.angle))
            zp = xp / tan(theta)

        # Find the location on the sample plane of the intercepted neutron.
        # The sample goes from (-w/2 - offset, w/2 - offset).  Anything with
        # position in that range has hit the sample.
        s = 2*(xp>=0)-1
        p = s*sqrt(xp**2 + zp**2) - offset
        #print("sample position", p)
        hit = (abs(p) < width/2.)
        dhit = hit & (np.random.rand(*hit.shape) < diffuse)

        # Calculate reflectivity, removing any neutrons that aren't reflected
        # TODO: maybe interpolate instead
        kz = 4 * pi * sin(theta - self.angle[hit]) / self.wavelength[hit]
        r = refl(kz)
        reflected = (r > np.random.rand(*r.shape))
        #print(theta, self.angle, self.wavelength, hit, kz, r)
        #print("reflected", np.mean(kz), np.mean(r), np.sum(reflected)/np.sum(hit),
        #      np.mean(self.angle), np.max(self.angle), np.min(self.angle))
        hit[hit] = reflected

        # Update the position of the neutrons which hit the sample
        self.angle[hit] = 2*theta - self.angle[hit]
        self.angle[dhit] = 2 * pi * np.random.rand(*hit.shape)[dhit]
        self.x[hit] = xp[hit]
        self.z[hit] = zp[hit]
        # Move the remaining neutrons to position 0
        self.x[~hit] = x[~hit]
        self.z[~hit] = 0
        self.sample_angle = angle
        self.add_trace()

        # sample is shifted along the z axis and then rotated by theta
        z1, x1 = rotate((-width/2+offset, 0), theta)
        z2, x2 = rotate((width/2+offset, 0), theta)
        self.add_element((z1, x1), (z2, x2))

    def slit(self, z, width, offset=0.):
        """
        Send
        :param width:
        :return:
        """
        self.move(z, add_trace=False)
        self.active &= (self.x >= -width/2+offset)
        self.active &= (self.x <= +width/2+offset)
        self.add_trace()
        self.add_element((z, -width+offset), (z, -width/2+offset))
        self.add_element((z, +width/2+offset), (z, width+offset))

    def comb_filter(self, z, n, width, separation):
        """
        Filter the neutrons through an *n* element comb filter.

        *width* is the filter opening and *separation* is the distance between
        consecutive openings. *width* can be a vector of length *n* if
        each opening is controlled independently.  The spacing between
        the centers is fixed.
        """
        self.move(z, add_trace=False)
        self.slit_array(z, comb(n, width, separation))

    def slit_array(self, z, edges):
        # Searching the neutron x positions in the list of comb edges
        # gives odd indices if they go through the edges, and even indices
        # if they encounter the edges of the comb.
        index = np.searchsorted(edges, self.x)
        self.active &= (index%2 == 1)
        self.add_trace()
        self.add_element((z, 2*edges[0]-edges[1]), (z, edges[0]))
        for x1, x2 in pairwise(edges[1:-1]):
            self.add_element((z, x1), (z, x2))
        self.add_element((z, edges[-1]), (z, 2*edges[-1]-edges[-2]))

    def reflect(self, q, r, sample_angle):
        # type: (np.ndarray, np.ndarray, float) -> Neutron
        """
        Interpolate neutrons in packet into the R(q) curve assuming specular
        reflectivity.

        Returns a weight associated with each neutron, which is the predicted
        reflectivity.
        """
        qk = 4.*pi*sin(self.angle+radians(sample_angle))/self.wavelength
        rk = np.interp(qk, q, r)
        self.weight *= rk
        return self

def wavelength_dispersion(L, mask=10., detector_distance=DETECTOR_Z):
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
    Neutrons.set_spectrum(load_spectrum())

    # Multibeam beam centers and angles
    # Note: if width is 0, then both edges are at the center
    beam_centers = comb(4, 0, SOURCE_LOUVER_SEPARATION)[::2]
    beam_angles = arctan(beam_centers/-SOURCE_LOUVER_Z)

    # Detector bank wavelengths and angles
    wavelengths = Neutrons.spectrum[0]
    bank_centers = detector_mask(width=0.)[::2]
    # Note: assuming flat detector bank; a curved bank will give
    bank_angles = arctan(bank_centers/DETECTOR_Z)
    bank_angles += bank_angles[0]
    #print("bank centers", bank_centers)
    #print("bank angles", degrees(bank_angles))
    #print("beam angles", degrees(beam_angles))

    num_leaf = len(wavelengths)
    num_bank = len(bank_angles)
    angular_spreads = 2.865*np.ones((1, num_bank*num_leaf))  # from nice vm
    wavelength_spreads = wavelength_dispersion(wavelengths)
    wavelength_spreads = 0.01*np.ones((1, num_bank*num_leaf))  # from nice vm
    #L = 6. - 0.037*np.arange(54)  # from nice vm
    wavelength_array = np.tile(wavelengths, (num_bank, 1)).T.flatten()

    # Initialize candor with fixed info fields
    candor = Candor()
    candor.move(
        beam_angularOffsets=[degrees(beam_angles)],
        detectorTable_angularSpreads=[angular_spreads],
        detectorTable_rowAngularOffsets=[degrees(bank_angles)],
        detectorTable_wavelengthSpreads=[wavelength_spreads],
        detectorTable_wavelengths=[wavelength_array],
    )
    return candor

def source_divergence(source_slit_z, source_slit_w,
                      sample_slit_z, sample_slit_w,
                      detector_slit_z, detector_slit_w,
                      sample_width, sample_offset,
                      sample_angle, detector_angle,
                      spill=True,
                     ):
    def angle(p1, p2):
        return arctan2(p2[1]-p1[1], p2[0]-p1[0])
    theta = radians(sample_angle)
    two_theta = radians(detector_angle)
    # source edges
    source_lo = source_slit_z, -0.5*source_slit_w
    source_hi = source_slit_z, +0.5*source_slit_w
    # pre-sample slit edges
    pre_lo = angle(source_hi, (sample_slit_z, -0.5*sample_slit_w))
    pre_hi = angle(source_lo, (sample_slit_z, +0.5*sample_slit_w))
    # sample edges (after rotation and shift)
    r_lo = -0.5*sample_width + sample_offset
    r_hi = +0.5*sample_width + sample_offset
    sample_lo = angle(source_hi, (arccos(theta)*r_lo, arcsin(theta)*r_lo))
    sample_hi = angle(source_lo, (arccos(theta)*r_hi, arcsin(theta)*r_hi))
    # post-sample slit edges (after rotation)
    alpha = arctan2(0.5*detector_slit_w, detector_slit_z)
    beta_lo = two_theta - alpha
    beta_hi = two_theta + alpha
    r = sqrt(detector_slit_z**2 + 0.25*detector_slit_w**2)
    post_lo = angle(source_hi, (arccos(beta_lo)*r, arcsin(beta_lo)*r))
    post_hi = angle(source_lo, (arccos(beta_hi)*r, arcsin(beta_hi)*r))

    if spill:
        max_angle = max(pre_hi, min(sample_hi, post_hi))
        min_angle = min(pre_lo, max(sample_lo, post_lo))
    else:
        max_angle = max(pre_hi, sample_hi)
        min_angle = min(pre_lo, sample_lo)
    return degrees(max(abs(max_angle), abs(min_angle)))


def simulate(counts, trace=False,
        sample_width=10., sample_offset=0., sample_diffuse=0.,
        sample_slit_offset=0.,
        refl=lambda kz: 1.,
        ):
    candor = Candor()

    # Set derived motors as needed.
    if candor.Q.wavelength is None:
        if candor.monoTransMap.key == "IN":
            candor.Q.wavelength =  candor.mono.wavelength
        else:
            bank, leaf = candor.Q.angleIndex, candor.Q.wavelengthIndex
            candor.Q.wavelength = candor.wavelengths[bank, leaf]

    # Retrieve values from candor instrument definition
    beam_mode = candor.Q.beamMode
    target_bank = candor.Q.angleIndex
    target_leaf = candor.Q.wavelengthIndex
    source_slit = candor.slitAperture1.softPosition
    sample_slit = candor.slitAperture2.softPosition
    detector_slit = candor.slitAperture3.softPosition
    sample_angle = candor.sampleAngleMotor.softPosition
    detector_angle = candor.detectorTableMotor.softPosition
    has_converging_guide = candor.convergingGuideMap.key == "IN"
    has_multibeam = candor.multiSlit1TransMap.key == "IN"
    has_single = candor.singleSlitApertureMap.key == "IN"
    detector_mask_width = float(candor.detectorMaskMap.key)
    has_mono = candor.monoTransMap.key == "IN"
    wavelength = candor.mono.wavelength
    wavelength_spread = candor.mono.wavelengthSpread
    louver = np.array([
        candor.multiBladeSlit1aMotor.softPosition,
        candor.multiBladeSlit1bMotor.softPosition,
        candor.multiBladeSlit1cMotor.softPosition,
        candor.multiBladeSlit1dMotor.softPosition,
    ])


    # Proceed with simulation
    has_sample = (sample_width > 0.)
    beam_mode = "single" if has_single else "multiple" if has_multibeam else "spread"

    # Enough divergence to cover the presample slit from the source aperture
    # Make sure we get the direct beam as well.
    divergence = degrees(arctan(sample_slit/abs(SOURCE_SLIT_Z-PRE_SAMPLE_SLIT_Z)))
    #print("divergence", divergence)
    #divergence = 5

    d2 = source_divergence(
        SOURCE_SLIT_Z, source_slit,
        PRE_SAMPLE_SLIT_Z, sample_slit,
        POST_SAMPLE_SLIT_Z, detector_slit,
        sample_width, sample_offset,
        sample_angle, detector_angle,
        )
    #divergence = d2
    #delta_theta *= 10

    if has_mono:
        L, I = Neutrons.spectrum[0], Neutrons.spectrum[1]
        rate = np.interp(wavelength, L, I)
        x = np.linspace(-3, 3, 21)
        mono_L = x*wavelength_spread/sqrt(log(256)) + wavelength
        mono_I = rate*exp(-x**2/2)/sqrt(2*pi)
        spectrum = mono_L, mono_I, Neutrons.spectrum[2:]
    else:
        spectrum = Neutrons.spectrum

    #counts = 100
    n = Neutrons(n=counts, trace=trace, spectrum=spectrum, divergence=divergence)
    if has_multibeam:
        # Use source louvers
        n.comb_source(SOURCE_LOUVER_Z, louver, SOURCE_LOUVER_SEPARATION)
    elif has_converging_guide:
        # Convergent guides remove any effects of slit collimation, and so are
        # equivalent to sliding the source toward the sample
        # Ignore divergence and use sample footprint to define beam
        spill = 0.1
        sample_xs = sample_width*sin(radians(sample_angle))
        sample_low = sample_xs*(-0.5 - spill + sample_offset/sample_width)
        sample_high = sample_xs*(0.5 + spill + sample_offset/sample_width)
        n.converging_source(PRE_SAMPLE_SLIT_Z, sample_slit, sample_low, sample_high)
    else:
        # Use fixed width source with given divergence
        # Aim the center of the divergence at the target position
        target = PRE_SAMPLE_SLIT_Z
        #target = POST_SAMPLE_SLIT_Z
        #target = DETECTOR_Z
        #target = inf  # target at infinity
        #target = -10  # just before sample
        #target = 0  # at the sample
        #target = 10  # just after sample
        n.slit_source(SOURCE_SLIT_Z, source_slit, target=target)

    if False and has_multibeam:
        # Play with a pre-sample comb selector (doesn't exist)
        comb_z = -1000
        comb_separation = SOURCE_LOUVER_SEPARATION*comb_z/SOURCE_LOUVER_Z
        comb_max = comb_separation - 1.
        comb_min = 0.1
        comb_width = np.maximum(np.minimum(louver, comb_max), comb_min)
        n.comb_filter(z=comb_z, n=SOURCE_LOUVER_N, width=comb_width,
                      separation=comb_separation)

    n.slit(z=PRE_SAMPLE_SLIT_Z, width=sample_slit, offset=sample_slit_offset)
    #return n
    if has_sample:
        n.sample(angle=sample_angle, width=sample_width, offset=sample_offset,
                 diffuse=sample_diffuse, refl=refl)
    else:
        n.move(z=0.)
    #return n
    n.detector_angle(angle=detector_angle)
    #n.clear_trace()
    n.slit(z=POST_SAMPLE_SLIT_Z, width=detector_slit, offset=sample_slit_offset)
    if False:
        # Play with a post-sample comb selector (doesn't exist)
        n.comb_filter(z=-comb_z, n=SOURCE_LOUVER_N, width=comb_width,
                      separation=comb_separation)
    n.move(z=DETECTOR_Z)
    n.slit_array(z=DETECTOR_Z, edges=detector_mask(detector_mask_width))
    n.move(z=DETECTOR_Z+1000)
    #n.angle_hist()

    return n

def single_point_demo(theta=2.5, count=150, trace=True, plot=True, split=False):
    #count = 10
    sample_width, sample_offset = 100., 0.
    #sample_width, sample_offset, source_slit = 2., 0., 0.02
    min_sample_angle = theta
    #mask = "4"  # 4 mm detector; posiion 3
    mask = "10"  # 10 mm detector; position 0
    #sample_diffuse = 0.01
    sample_diffuse = 0.0

    target_bank = 12  # detector bank to use for angle in Q calculations
    target_leaf = 2  # detector leaf to use for wavelength in Q calculations
    target_beam = 1  # beam number (in multibeam mode) to use for Q calculations
    qx, qz = 0., 0.3

    layers = [
        # depth rho rhoM thetaM phiM
        [ 0, 0.0, 0.0, 270, 0],
        [200, 4.0, 1.0, 359.9, 0.0],
        [200, 2.0, 1.0, 270, 0.0],
        [ 0, 4.0, 0.0, 270, 0.0],
    ]
    depth, rho, rhoM, thetaM, phiM = zip(*layers)
    rho = np.array(rho)*1000 #*400
    #from refl1d import abeles
    #refl = lambda kz: abs(abeles.refl(kz, depth, rho))**2
    refl = lambda kz: 0.9*np.ones(kz.shape)
    #pylab.semilogy(refl(np.linspace(0, 0.2, 1000))); pylab.show(); sys.exit()

    #sample_angle = min_sample_angle - degrees(SOURCE_LOUVER_ANGLES[0])
    sample_angle = min_sample_angle
    detector_angle = 2*sample_angle

    sample_slit_offset = 0.
    #sample_slit = choose_sample_slit(louver, sample_width, sample_angle)
    sample_slit = sample_width*sin(radians(sample_angle))
    #sample_slit = DETECTOR_WIDTH/DETECTOR_Z * POST_SAMPLE_SLIT_Z
    #sample_slit *= 0.2
    #sample_slit /= 2

    source_slit = sample_slit
    #source_slit = 3.  # narrow beam
    #source_slit = 50.  # wide beam
    #source_slit = 150.  # super-wide beam

    # TODO: should use slitAperture1a, 1b, 1c, 1d instead
    louver = (radians(sample_angle) + SOURCE_LOUVER_ANGLES)*sample_width
    louver = abs(louver)

    detector_slit = sample_slit
    #detector_slit = 3*sample_slit
    #detector_slit = sample_slit + (POST_SAMPLE_SLIT_Z - PRE_SAMPLE_SLIT_Z)*tan(radians(divergence))

    #louver[1:3] = 0.
    #sample_slit = louver[0]*2
    #sample_slit_offset = SOURCE_LOUVER_CENTERS[0]*PRE_SAMPLE_SLIT_Z/SOURCE_LOUVER_Z  # type: float

    beam_mode = "single"
    #beam_mode = "converging"
    #beam_mode = "multiple"

    wavelength_mode = "white"
    #wavelength_mode = "mono"
    wavelength = 4.75
    wavelength_spread = wavelength*0.01

    candor = Candor()
    # TODO: should this
    candor.move(
        # info fields
        Q_beamMode="SINGLE_BEAM" if wavelength_mode == "mono" else "WHITE_BEAM",
        Q_angleIndex=target_bank,
        Q_wavelengthIndex=target_leaf,
        Q_wavelength=None,  # value will be set automatically based on mode
        Q_beamIndex=target_beam,
        Q_x=qx,
        Q_z=qz,
        # slits
        singleSlitApertureMap="IN" if beam_mode != "multiple" else "OUT",
        multiSlit1TransMap="IN" if beam_mode == "multiple" else "OUT",
        multiBladeSlit1aMotor=louver[0],
        multiBladeSlit1bMotor=louver[1],
        multiBladeSlit1cMotor=louver[2],
        multiBladeSlit1dMotor=louver[3],
        slitAperture1=source_slit,
        slitAperture2=sample_slit,
        slitAperture3=detector_slit,
        detectorMaskMap=mask,
        convergingGuideMap="IN" if beam_mode == "converging" else "OUT",
        # angles
        sampleAngleMotor=sample_angle,
        detectorTableMotor=detector_angle,
        # wavelength
        monoTransMap="IN" if wavelength_mode == "mono" else "OUT",
        mono_wavelength=wavelength,
        mono_wavelengthSpread=wavelength_spread,
        )

    n = simulate(
        counts=count,
        sample_width=sample_width,
        sample_offset=sample_offset,
        sample_diffuse=sample_diffuse,
        sample_slit_offset=sample_slit_offset,
        refl=refl,  # Should replace this with sample scatter
        trace=trace,
    )

    if plot:
        n.plot_trace(split=split)
        pylab.xlabel('z (mm)')
        pylab.ylabel('x (mm)')
        pylab.title('sample width=%g'%sample_width)
        #pylab.axis('equal')


def scan_demo(count=100):
    pylab.subplot(2, 2, 1)
    single_point_demo(0.5, count)
    pylab.subplot(2, 2, 2)
    single_point_demo(1.0, count)
    pylab.subplot(2, 2, 3)
    single_point_demo(1.7, count)
    pylab.subplot(2, 2, 4)
    single_point_demo(2.5, count)

def main():
    candor_setup()
    if len(sys.argv) < 2:
        print("incident angle in degrees required")
        sys.exit(1)
    theta = float(sys.argv[1])
    single_point_demo(theta=theta, count=1500)
    #scan_demo(150)
    pylab.show()
    sys.exit()

if __name__ == "__main__":
    main()
