"""
Program to explore the beam profile and angular distribution for a simple
reflectometer with two front slits.
"""
from __future__ import division, print_function

import sys
import os
from warnings import warn
import time

import numpy as np
from numpy import (sin, cos, tan, arcsin, arccos, arctan, arctan2, radians, degrees,
                   sign, sqrt, exp, log)
from numpy import pi, inf
import pylab

from . import mpl_controls
from . import qcalc
from . import nice
from .instrument import Candor, candor_setup, comb, detector_mask
#from .nice import Instrument, Motor

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

class Neutrons(object):
    #: spectrum as [incident wavelength, spectral weight, detector efficiency]
    spectrum = None  # type: Tuple[np.ndarray, np.ndarray, np.ndarray]
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
    #: sample angle if there is a sample in the beam
    sample_angle = 0.
    #: source beam for each neutron
    source = None # type: np.ndarray
    #: detector bank for each neutron, or -1 if no detector
    detector = None # type: np.ndarray
    #: number of detector banks
    num_bank = 0
    #: number of wavelengths per detector bank
    num_leaf = 0
    #: poisson-noise based counts, with spectrum, scattering,
    #: detector efficiency and background included
    counts = None # type: np.ndarray

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
        L, I = spectrum[0], spectrum[1]
        assert L[0] > L[1]  # highest to lowest wavelength assumed
        L_min, L_max = 1.5*L[-1]-0.5*L[-2], 1.5*L[0]-0.5*L[1]  # limits go half bin beyond
        #print("limits", L_min, L_max, L[0], L[-1])
        I_weighted = I/np.mean(I)

        self.spectrum = spectrum
        self.xy = np.zeros((2, n), 'd')
        self.angle = (np.random.rand(n)-0.5)*radians(divergence)
        self.wavelength = np.random.rand(n)*(L_max-L_min) + L_min
        self.weight = np.interp(self.wavelength, L, I_weighted)
        self.active = (self.x == self.x)

        self.history = []
        self.trace = trace
        self.elements = []

    def converging_source(self, z, width, sample_low, sample_high,
                          sample_z=0, portion=1.0):
        n = int(self.xy.shape[1]*portion)
        self.z = z
        x = (np.random.rand(n)-0.5)*width
        y = ((np.random.rand(n)-0.5)*(sample_high-sample_low)
             + (sample_high + sample_low)/2)
        self.x[:n] = x
        self.angle[:n] = arctan2(y-x, sample_z-z)
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

    def comb_source(self, z, widths, separation, focus=0.):
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
        self.angle += arctan(self.x/(self.z - focus))
        #self.angle -= SOURCE_LOUVER_ANGLES[index]
        self.add_trace()
        self.add_element((z, edges[0]-widths[0]/2), (z, edges[0]))
        for x1, x2 in pairwise(edges[1:-1]):
            self.add_element((z, x1), (z, x2))
        self.add_element((z, edges[-1]), (z, edges[-1]+widths[-1]/2))

    def trim(self):
        self.xy = self.xy[:, self.active]
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

    def slit(self, z, width, offset=0., center_angle=0.):
        """
        Send beam through a pair of slits.
        :param width:
        :return:
        """
        offset += z * tan(radians(center_angle))
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
        #self.move(z, add_trace=False)
        self._slit_array(z, comb(n, width, separation))

    def detector_bank(self, z, edges, center_angle=0.):
        edges = edges - z * tan(radians(center_angle))
        index = self._slit_array(z, edges)
        # Record the slit array which each neutron goes through
        index[self.active] = (index[self.active] - 1)//2
        index[~self.active] = -1
        self.detector = index
        self.num_bank = len(edges)//2

    def dtheta(self):
        nbank = self.num_bank
        bank = self.detector[self.active]
        angle = self.angle[self.active]
        n = np.bincount(bank, minlength=nbank)
        sum_x = np.bincount(bank, weights=angle, minlength=nbank)
        sum_xsq = np.bincount(bank, weights=angle**2, minlength=nbank)
        mean = sum_x/(n + (n==0))
        var = sum_xsq/(n + (n==0)) - mean**2
        self.theta_mean = mean
        self.theta_std = np.sqrt(var)

    @staticmethod
    def _rlookup(edges, values):
        trimmed_edges = edges[1:]
        index = np.searchsorted(trimmed_edges[::-1], values)
        ret = len(trimmed_edges) - index
        return ret

    def count(self, intensity=1., background=0.):
        nbank = self.num_bank
        nleaf = len(self.spectrum[2])
        incident_wavelength = self.wavelength[self.active]
        weight = self.weight[self.active]
        bank = self.detector[self.active]
        leaf = self._rlookup(self.spectrum[2], incident_wavelength)
        binned = np.bincount(bank*nleaf+leaf, weights=weight, minlength=nbank*nleaf)
        expected = binned.reshape(nbank, nleaf)*self.spectrum[3] + background
        self.counts = np.random.poisson(intensity*expected)
        self.num_leaf = nleaf

    def _slit_array(self, z, edges):
        self.move(z, add_trace=False)
        # Searching the neutron x positions in the list of comb edges
        # gives odd indices if they go through the edges, and even indices
        # if they encounter the edges of the comb.
        index = np.searchsorted(edges, self.x)
        self.active &= (index%2 == 1)
        self.add_trace()
        # Draw the filter
        self.add_element((z, 2*edges[0]-edges[1]), (z, edges[0]))
        for x1, x2 in pairwise(edges[1:-1]):
            self.add_element((z, x1), (z, x2))
        self.add_element((z, edges[-1]), (z, 2*edges[-1]-edges[-2]))
        return index

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
        pylab.xlabel('z (mm)')
        pylab.ylabel('x (mm)')

    def _plot_trace_old(self, split=False):
        raise NotImplemented("_plot_trace_old requires source louvers")
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
            angle = degrees(Candor.SOURCE_LOUVER_ANGLES[k]) + self.sample_angle
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

def choose_sample_slit(louver, sample_width, sample_angle):
    raise NotImplemented("choose_sample_slit requires source louvers")
    theta = radians(sample_angle)
    index = np.nonzero(louver)[0]
    k = index[-1]
    x0, y0 = Candor.SOURCE_LOUVER_Z, Candor.SOURCE_LOUVER_CENTERS[k] + louver[k]/2
    x1, y1 = sample_width/2*cos(theta), sample_width/2*sin(theta)
    top = (y1-y0)/(x1-x0)*(Candor.PRE_SAMPLE_SLIT_Z - x1) + y1
    k = index[0]
    x0, y0 = Candor.SOURCE_LOUVER_Z, Candor.SOURCE_LOUVER_CENTERS[k] - louver[k]/2
    x1, y1 = -sample_width/2*cos(theta), -sample_width/2*sin(theta)
    bottom = (y1-y0)/(x1-x0)*(Candor.PRE_SAMPLE_SLIT_Z - x1) + y1
    #print(top, bottom)
    #slit = 2*max(top, -bottom)
    slit = 2*max(abs(top), abs(bottom))
    return slit


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

def simulate(
        count, trace=False,
        sample_width=10., sample_offset=0., sample_diffuse=0.,
        sample_slit_offset=0., detector_slit_offset=0.,
        refl=lambda kz: 1., intensity=1., background=0.,
        ):
    candor = Candor()

    # Retrieve values from candor instrument definition
    beam_mode = candor.Q.beamMode

    source_slit = candor.slitAperture1.softPosition
    sample_slit = candor.slitAperture2.softPosition
    detector_slit = candor.slitAperture3.softPosition
    detector_mask_width = float(candor.detectorMaskMap.key)


    ## No longer defining beams for broad multi-beam mode; now continuous
    #has_multibeam = candor.multiSlit1TransMap.key == "IN"
    #louver = np.array([
    #    candor.multiBladeSlit1aMotor.softPosition,
    #    candor.multiBladeSlit1bMotor.softPosition,
    #    candor.multiBladeSlit1cMotor.softPosition,
    #    candor.multiBladeSlit1dMotor.softPosition,
    #])

    target_bank = candor.Q.angleIndex
    target_leaf = candor.Q.wavelengthIndex
    source_beam = candor.Q.beamIndex
    sample_angle = candor.sampleAngleMotor.softPosition
    detector_table_angle = candor.detectorTableMotor.softPosition

    has_converging_guide = candor.convergingGuideMap.key == "IN"
    #has_single = candor.singleSlitApertureMap.key == "IN"
    has_mono = candor.monoTrans.key == "IN"
    mono_wavelength = candor.mono.wavelength
    mono_wavelength_spread = candor.mono.wavelengthSpread

    #beam_offset = candor.beam.angularOffsets[source_beam] if has_multibeam else 0.
    bank_angle = candor.detectorTable.rowAngularOffsets[target_bank]
    target_wavelength = candor.wavelengths[target_bank, target_leaf]

    # Set derived motors as needed.
    if candor.Q.wavelength is None:
        # TODO: check this
        candor.Q.wavelength = (mono_wavelength if has_mono
                               else candor.wavelengths[target_bank, target_leaf])

    lambda_i = mono_wavelength if has_mono else target_wavelength
    lambda_f = target_wavelength
    if candor.Q.x is None or candor.Q.z is None:
        theta_i = sample_angle #+ beam_offset
        theta_f = detector_table_angle + bank_angle
        qx, qz = qcalc.angle_to_qxz(theta_i, theta_f, lambda_i, lambda_f)
        candor.Q.x, candor.Q.z = qx, qz
        #print("angles to qx, qz")
    if sample_angle is None or detector_table_angle is None:
        L = target_wavelength
        qx, qz = candor.Q.x, candor.Q.z
        theta_i, theta_f = qcalc.qxz_to_angle(qx, qz, lambda_i, lambda_f)
        sample_angle = theta_i #- beam_offset
        detector_table_angle = theta_f - bank_angle
        candor.move(
            sampleAngleMotor=sample_angle,
            detectorTableMotor=detector_table_angle)
        #print("qx, qz to angles", lambda_i, lambda_f, theta_i, theta_f)
    detector_angle = detector_table_angle + bank_angle

    # If just moving motors and not counting, then stop early
    if count == 0:
        return None

    # Proceed with simulation
    has_sample = (sample_width > 0.)
    #beam_mode = "single" if has_single else "multiple" if has_multibeam else "spread"

    # Enough divergence to cover the presample slit from the source aperture
    # Make sure we get the direct beam as well.
    divergence = degrees(arctan(sample_slit/abs(Candor.SOURCE_SLIT_Z-Candor.PRE_SAMPLE_SLIT_Z)))
    #print("divergence", divergence)
    #divergence = 5

    d2 = source_divergence(
        Candor.SOURCE_SLIT_Z, source_slit,
        Candor.PRE_SAMPLE_SLIT_Z, sample_slit,
        Candor.POST_SAMPLE_SLIT_Z, detector_slit,
        sample_width, sample_offset,
        sample_angle, detector_angle,
        )
    #divergence = d2
    #delta_theta *= 10

    if has_mono:
        L, I = candor.spectrum[0], candor.spectrum[1]
        rate = np.interp(mono_wavelength, L, I)
        x = np.linspace(-3, 3, 21)
        mono_L = x*mono_wavelength_spread/sqrt(log(256)) + mono_wavelength
        mono_I = rate*exp(-x**2/2)/sqrt(2*pi)
        spectrum = mono_L, mono_I, candor.spectrum[2:]
    else:
        spectrum = candor.spectrum

    #count = 100
    n = Neutrons(n=count, trace=trace, spectrum=spectrum, divergence=divergence)
    #if has_multibeam:
    #    # Use source louvers
    #    n.comb_source(Candor.SOURCE_LOUVER_Z, louver,
    #                  Candor.SOURCE_LOUVER_SEPARATION,
    #                  focus=Candor.PRE_SAMPLE_SLIT_Z,
    #                  )
    spill = 0.1 # Spill for converging beam
    if has_converging_guide:
        # Convergent guides remove any effects of slit collimation, and so are
        # equivalent to sliding the source toward the sample
        # Ignore divergence and use sample footprint to define beam
        source_z = Candor.PRE_SAMPLE_SLIT_Z
        source_width = sample_slit
        sample_xs = sample_width*sin(radians(sample_angle))
        sample_low = sample_xs*(-0.5 - spill + sample_offset/sample_width)
        sample_high = sample_xs*(0.5 + spill + sample_offset/sample_width)
        n.converging_source(source_z, source_width, sample_low, sample_high)
    else:
        # Use fixed width source with given divergence
        # Aim the center of the divergence at the target position
        source_z = Candor.SOURCE_SLIT_Z
        sorce_width = source_slit
        target = Candor.PRE_SAMPLE_SLIT_Z
        #target = POST_SAMPLE_SLIT_Z
        #target = DETECTOR_Z
        #target = inf  # target at infinity
        #target = -10  # just before sample
        #target = 0  # at the sample
        #target = 10  # just after sample
        n.slit_source(source_z, source_width, target=target)

    _ = """
    # WRONG: this projects onto the sample position, not post-sample slit
    # postion.  I'm leaving it here because it says how to do so, but it
    # should be removed.

    # Point 10% of the beam at the post-sample slits
    # Need to find the post-sample slit end points first, then draw line from
    # top of source to top of detector and see where it intercepts z = 0.
    r = Candor.POST_SAMPLE_SLIT_Z
    sina, cosa = sin(radians(detector_angle)), cos(radians(detector_angle))
    x, y = r*cosa, r*sina
    d = detector_slit/2 + detector_slit_offset
    x1, y1 = source_z, source_width/2
    x2, y2 = x - d*sina, y + d*cosa
    target_high = y1 + (0 - x1) * (y2 - y1) / (x2 - x1)
    #print("top", (x1, y1), (x, y), (x2, y2), (x2-x1, y2-y1, 0-x1), target_high)
    # Same for bottoms:
    d = -detector_slit/2 + detector_slit_offset
    x1, y1 = source_z, -source_width/2
    x2, y2 = x - d*sina, y + d*cosa
    target_low = y1 + (0 - x1) * (y2 - y1) / (x2 - x1)
    #print("bottom", (x, y), (x2, y2), target_low)
    n.converging_source(source_z, source_width, target_low, target_high, portion=0.1)
    """

    # Point 10% of the beam at the post-sample slits
    # Need to find the post-sample slit end points first, then draw line from
    # top of source to top of detector and see where it intercepts z = 0.
    # WRONG: assumes that the post-sample slit is parallel to pre-sample slits
    # It is not in general, but averaging the z positions at the ends of the
    # openings is good enough even for 50 mm slits at 20 degrees.
    r = Candor.POST_SAMPLE_SLIT_Z
    d_low = -detector_slit/2 + detector_slit_offset
    d_high = detector_slit/2 + detector_slit_offset
    sina, cosa = sin(radians(detector_angle)), cos(radians(detector_angle))
    target_z = r*cosa - (d_low+d_high)*sina/2
    target_high = r*sina + d_high*cosa
    target_low = r*sina + d_low*cosa

    #print("bottom", (x, y), (x2, y2), target_low)
    n.converging_source(source_z, source_width, target_low, target_high,
                        sample_z=target_z, portion=0.1)

    ## Play with a pre-sample comb selector (doesn't exist on candor)
    #if False and has_multibeam:
    #    comb_z = -1000
    #    comb_separation = Candor.SOURCE_LOUVER_SEPARATION*comb_z/Candor.SOURCE_LOUVER_Z
    #    comb_max = comb_separation - 1.
    #    comb_min = 0.1
    #    comb_width = np.maximum(np.minimum(louver, comb_max), comb_min)
    #    n.comb_filter(z=comb_z, n=Candor.SOURCE_LOUVER_N, width=comb_width,
    #                  separation=comb_separation)

    n.slit(z=Candor.PRE_SAMPLE_SLIT_Z, width=sample_slit, offset=sample_slit_offset)
    #return n
    if has_sample:
        n.sample(angle=sample_angle, width=sample_width, offset=sample_offset,
                 diffuse=sample_diffuse, refl=refl)
    else:
        n.move(z=0.)
    #return n

    # Rotate beam through detector arm
    n.detector_angle(angle=detector_angle)

    # Send beam through post-sample slit.
    # Assume the slit is centered on target detector bank. The other option
    # is that it is aimed at the center of the detector, which does not
    # correspond to any particular bank.  Might want to do this for
    # the converging beam mode.
    #n.clear_trace()
    n.slit(z=Candor.POST_SAMPLE_SLIT_Z, width=detector_slit,
           offset=detector_slit_offset)

    ## Play with a post-sample comb selector (doesn't exist on candor)
    #if False:
    #    n.comb_filter(z=-comb_z, n=Candor.SOURCE_LOUVER_N, width=comb_width,
    #                  separation=comb_separation)

    # Send the neutrons through the detector mask
    n.detector_bank(z=Candor.DETECTOR_Z, edges=detector_mask(detector_mask_width),
                    center_angle=bank_angle)
    # Poisson simulation of detectors given "count" MC samples
    n.count(intensity/count, background)
    n.dtheta()  # compute wavelength divergence for each bank
    n.move(z=Candor.DETECTOR_Z+1000)
    #n.angle_hist()

    return n

def fake_sample():
    from . import abeles
    layers = [
        # depth rho rhoM thetaM phiM
        [ 0, 0.0, 0.0, 270, 0],
        [200, 4.0, 1.0, 359.9, 0.0],
        [200, 4.5, 1.0, 270, 0.0],
        [ 0, 2.07, 0.0, 270, 0.0],
    ]
    depth, rho, rhoM, thetaM, phiM = zip(*layers)
    rho = np.array(rho) #*400
    refl = lambda kz: (abs(abeles.refl(kz.flatten(), depth, rho))**2).reshape(kz.shape)
    #refl = lambda kz: 0.9*np.ones(kz.shape)
    #pylab.semilogy(refl(np.linspace(0, 0.2, 1000))); pylab.show(); sys.exit()
    return refl

def single_point_demo(theta=2.5, count=150, intensity=1e6, background=0., trace=False):
    #count = 10
    sample_width, sample_offset = 100., 0.
    #sample_width, sample_offset, source_slit = 2., 0., 0.02
    min_sample_angle = theta
    #mask = "4"  # 4 mm detector; posiion 3
    mask = "10"  # 10 mm detector; position 0
    #sample_diffuse = 0.01
    sample_diffuse = 0.0

    target_bank = 1  # detector bank to use for angle in Q calculations
    target_leaf = 3  # detector leaf to use for wavelength in Q calculations
    target_beam = 1  # beam number (in multibeam mode) to use for Q calculations

    #sample_angle = min_sample_angle - degrees(SOURCE_LOUVER_ANGLES[0])
    sample_angle = min_sample_angle
    detector_angle = 2*sample_angle
    #sample_angle, detector_angle = None, None
    qx, qz = None, None
    #qx, qz = 0., 0.3

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

    ## TODO: should use slitAperture1a, 1b, 1c, 1d instead
    #louver = (radians(sample_angle) + Candor.SOURCE_LOUVER_ANGLES)*sample_width
    #louver = abs(louver)

    detector_slit = sample_slit
    #detector_slit = 3*sample_slit
    #detector_slit = sample_slit + (POST_SAMPLE_SLIT_Z - PRE_SAMPLE_SLIT_Z)*tan(radians(divergence))

    #louver[1:3] = 0.
    #sample_slit = louver[0]*2
    #sample_slit_offset = SOURCE_LOUVER_CENTERS[0]*PRE_SAMPLE_SLIT_Z/SOURCE_LOUVER_Z  # type: float

    #beam_mode = "single"
    beam_mode = "converging"
    #beam_mode = "multiple"  # DEPRECATED

    wavelength_mode = "white"
    #wavelength_mode = "mono"
    wavelength = 4.75
    wavelength_spread = wavelength*0.01

    #source_slit *= 10
    #sample_slit *= 3
    #detector_slit *= 0.02
    source_slit *= 0.8
    sample_slit *= 0.8
    #detector_slit *= 0.02
    candor = Candor()
    if detector_angle is not None:
        bank_angle = candor.detectorTable.rowAngularOffsets[target_bank]
        detector_angle -= bank_angle
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
        #singleSlitApertureMap="IN" if beam_mode != "multiple" else "OUT",
        #multiSlit1TransMap="IN" if beam_mode == "multiple" else "OUT",
        #multiBladeSlit1aMotor=louver[0],
        #multiBladeSlit1bMotor=louver[1],
        #multiBladeSlit1cMotor=louver[2],
        #multiBladeSlit1dMotor=louver[3],
        slitAperture1=source_slit,
        slitAperture2=sample_slit,
        slitAperture3=detector_slit,
        detectorMaskMap=mask,
        convergingGuideMap="IN" if beam_mode == "converging" else "OUT",
        # angles
        sampleAngleMotor=sample_angle,
        detectorTableMotor=detector_angle,
        # wavelength
        monoTrans="IN" if wavelength_mode == "mono" else "OUT",
        mono_wavelength=wavelength,
        mono_wavelengthSpread=wavelength_spread,
        )

    n = simulate(
        count=count,
        sample_width=sample_width,
        sample_offset=sample_offset,
        sample_diffuse=sample_diffuse,
        sample_slit_offset=sample_slit_offset,
        refl=fake_sample(),  # Should replace this with sample scatter
        intensity=intensity,
        background=background,
        trace=trace,
    )

    return n

def make_sliders():
    sample = {
        'sample_width': 10.,
        'sample_offset': 0.,
        'sample_diffuse': 0.0,
        'sample_slit_offset': 0.0,
        'detector_slit_offset': 0.0,
        'refl': fake_sample(),
        'trace': True,
    }
    count = 150
    candor = Candor()
    def update(axis, value):
        #print("moving", axis, value)
        if axis in sample:
            sample[axis] = value
        else:
            candor.move(**{axis: value})
        # If changing Q then force recalculation of sample/detector angle
        #if axis in ("Q_x", "Q_z"):
        if axis.startswith("Q_"):
            candor.move(sampleAngleMotor=None, detectorTableMotor=None)
        n = simulate(count=count, **sample)
        if axis.startswith("Q_"):
            sliders.reset()
        fig = pylab.figure(2)
        limits = mpl_controls.get_zoom(fig.gca())
        pylab.clf()
        n.plot_trace(split=False)
        mpl_controls.set_zoom(fig.gca(), limits)
        pylab.gcf().canvas.draw_idle()

    def query(axis):
        return sample[axis] if axis in sample else candor[axis]

    slider_set = (
        ('sample_width', (0, 100), 'sample width (mm)'),
        ('sample_offset', (0, 100), 'sample offset (mm)'),
        #('sample_diffuse', (0, 1), 'diffuse portion'),
        ('Q_x', (0, 0.1), r'$Q_x$ 1/Ang'),
        ('Q_z', (0, 1.0), r'$Q_z$ 1/Ang'),
        ('sampleAngleMotor', (0, 20), r'$\theta$ ($^\circ$)'),
        ('detectorTableMotor', (0, 20), r'$2\theta$ ($^\circ$)'),
        ('slitAperture1', (0, 100), 'slit 1 (mm)'),
        ('slitAperture2', (0, 100), 'slit 2 (mm)'),
        ('slitAperture3', (0, 100), 'slit 3 (mm)'),
        ('sample_slit_offset', (-100, 100), 'slit 2 offset (mm)'),
        ('detector_slit_offset', (-100, 100), 'slit 3 offset (mm)'),
    )
    n_sliders = len(sample)
    pylab.figure(1)
    sliders = mpl_controls.SliderSet(len(slider_set), update, query)
    for name, limits, label in slider_set:
        value = sample[name] if name in sample else candor[name]
        sliders.add(name, limits=limits, value=value, label=label)
    pylab.figure(2)
    return sliders

def scan_demo():
    #: number of monte carlo samples
    count = 100000
    #: number of seconds per point
    count_time = 60.0
    #: counts per second for theta at 1 degree
    rate = 10000

    T0 = time.mktime(time.strptime("2018-01-01 12:00:00", "%Y-%m-%d %H:%M:%S"))
    candor = Candor()
    stream = nice.StreamWriter(candor, timestamp=T0)

    # set initial motor positions then open trajectory
    n = single_point_demo(theta=0.5, count=0, intensity=0.)
    candor.move(
        trajectory_trajectoryID=1,
        trajectory_entryID='demo:unpolarized',
        trajectory_length=4,
        trajectory_scanLength=4,
        )
    stream.config()
    stream.open()

    # run the points
    for theta in (0.5, 1.0, 1.7, 2.5):
        intensity = theta*rate*count_time
        n = single_point_demo(theta, count=count, intensity=intensity)
        stream.state()
        candor.move(
            counter_liveMonitor=np.random.poisson(rate*count_time*0.1),
            counter_liveTime=count_time,
            areaDetector_counts=n.counts,
            )
        stream.counts()

    # close the trajectory
    stream.close()
    stream.end()

def main():
    candor_setup()
    if len(sys.argv) < 2:
        print("incident angle in degrees required, or 'scan' for scan demo")
        sys.exit(1)
    if sys.argv[1] == 'scan':
        scan_demo()
        return
    theta = float(sys.argv[1])
    pylab.figure(1)
    sliders = make_sliders()
    pylab.figure(2)
    count = 1500
    n = single_point_demo(theta=theta, count=count, trace=True)
    n.plot_trace(split=False)
    #pylab.title('sample width=%g'%sample_width)
    #pylab.axis('equal')
    sliders.reset()
    #scan_demo(150)
    pylab.show()
    sys.exit()

if __name__ == "__main__":
    main()
