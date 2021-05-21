import numpy as np
from numpy import (radians, degrees, tan, sin, cos, arcsin, arccos, arctan,
    sign, sqrt, log, exp)

from . import simulate
from . import instrument

def plot_dLoL():
    spectrum = simulate.load_spectrum()
    L = spectrum[0]
    crystal = np.arange(1, 55)

    import pylab
    pylab.plot(crystal, 100*simulate.wavelength_dispersion(L, 10), label="10 mm")
    pylab.plot(crystal, 100*simulate.wavelength_dispersion(L, 8), label="8 mm")
    pylab.plot(crystal, 100*simulate.wavelength_dispersion(L, 6), label="6 mm")
    pylab.plot(crystal, 100*simulate.wavelength_dispersion(L, 4), label="4 mm")
    pylab.legend()
    pylab.xlabel("crystal #")
    pylab.ylabel(r"$\Delta\lambda/\lambda$ (FWHM) %")
    pylab.title("CANDOR wavelength dispersion")
    pylab.grid()
    pylab.show()

def plot_channels(mask=10.):
    spectrum = simulate.load_spectrum()
    L = spectrum[0]
    dL = L*simulate.wavelength_dispersion(L, mask)/2.35
    x = np.linspace(4,6,10000)
    y = [exp(-0.5*(x - Li)**2/dLi**2) for Li, dLi in zip(L,dL)]

    import pylab
    #pylab.plot(x, np.vstack(y).T)
    pylab.plot(x, np.sum(np.vstack(y),axis=0))
    pylab.grid(True)
    pylab.show()

def resolution_simulator():
    # s1, s2 are slit openings
    # theta is sample angle
    # width is sample width
    # offset is sample offset from center (horizontal)
    # beam_height is the vertical size of the beam; this is used to compute
    # the beam spill from sample disks, the effects of which can be reduced
    # by setting the beam height significantly below the disk width.

    ## Variants
    d1, d2 = instrument.Candor.SOURCE_SLIT_Z, instrument.Candor.PRE_SAMPLE_SLIT_Z
    beam_height = 8
    width=10
    #width=0.6
    #offset = -2.
    offset = 0
    theta = 3.5
    #s1, s2 = .5, .5
    #s1, s2 = 5, 5
    s1, s2 = 0.15, 0.10 # different slits
    #s1, s2 = 0.05, 0.10  # different slits
    #s1, s2, theta = 0.5, 0.5, 3.5  # far out
    #s1, s2, width = 0.1, 0.01, 4 # extreme slits
    #s1, s2 = 0.1, 0.2 # extreme slits
    #s1, s2 = 0.01, 0.1 # extreme slits

    # number of samples
    n = 1000000

    # use radians internally
    theta = radians(theta)

    # Maximum angle for any neutron is found by looking at the triangle for a
    # neutron entering at the bottom of slit 1 and leaving at the top of slit 2.
    # Simple trig gives the maximum angle we need to consider, with spread going
    # from negative of that angle to positive of that angle.  Assume all starting
    # positions and all angles are equiprobable and generate neutrons at angle
    # phi starting at position x1.
    spread = 2*arctan(0.5*(s1+s2)/(d2-d1))
    x1 = np.random.uniform(-s1/2, s1/2, size=n)
    phi = np.random.uniform(-spread/2, spread/2, size=n)

    # Determine where the neutrons intersect slit 2; tag those that make it through
    # the slits
    x2 = (d2-d1)*tan(phi) + x1
    through = (x2 > -s2/2) & (x2 < s2/2)
    n_through = np.sum(through)

    # Determine where the neutrons intersect the plane through the center of sample
    # rotation
    xs = -d1*tan(phi) + x1

    # Intersection of sample with individual neutrons
    def intersection(theta, phi, xs):
        xp = tan(theta) * xs / (tan(theta) - tan(phi))
        zp = xp / tan(theta)
        return xp, zp
    xp, zp = intersection(theta, phi, xs)

    # Find the location on the sample plane of the intercepted neutron.  Note that
    # this may lie outside the bounds of the sample, so check that it lies within
    # the sample.  The sample goes from (-w/2 - offset, w/2 - offset)
    z = sign(xp)*sqrt(xp**2 + zp**2) - offset
    hit = through & (abs(z) < width/2.)
    n_hit = np.sum(hit)

    # If phi > theta then the neutron must have come in from the back.  If it
    # hits the sample as well, then we need to deal with refraction (if it hits
    # the side) or reflection (if it is low angle and not transmitted).  Let's
    # count the number that hit the back of the sample.
    n_backside = np.sum((phi > theta)[hit])

    # For disk-shaped samples, weight according to chord length.  That is, for a
    # particular x position on the sample, r^2 = x^2 + y^2, chord length is 2*y.
    # Full intensity will be at the sample width, 2*r.
    # Ratio w = (r^2 - x^2)/r^2 = 1 - (x/r)^2.
    w = 1 - (2*z/width)**2
    # For ribbon beams (limited y), the total beam intensity only drops when
    # the ribbon width is greater than the chord length.  Rescale the chord
    # lengths to be relative to the beam height rather than the sample diameter
    # and clip anything bigger than one.  If the beam height is bigger than
    # the sample width, this will lead to beam spill even at the center.
    # Note: Could do the same for square samples.
    if np.isfinite(beam_height):
        w *= width/beam_height
        w[w>1] = 1
    #w = 1 - (np.minimum(abs(2*z), beam_height)/min(width, beam_height))**2
    w[w<0] = 0.
    weight = sqrt(w)
    n_hit_disk = np.sum(hit*weight)

    # End points of the sample
    sz1,sx1 = (-width/2+offset)*cos(theta), (-width/2+offset)*sin(theta)
    sz2,sx2 = (+width/2+offset)*cos(theta), (+width/2+offset)*sin(theta)

    # Simplified intersection: sample projects onto sample position
    hit_proj = through & (xs > sx1) & (xs < sx2)
    n_hit_proj = np.sum(hit_proj)

    # Simplified disk intersection: weight by the projection of the disk
    z_proj = xs / sin(theta) - offset
    w = 1 - (2*z_proj/width)**2
    w[w<0] = 0.
    weight_proj = sqrt(w)
    n_hit_disk_proj = np.sum(hit_proj*weight_proj)

    # beam profile is a trapezoid, 0 where a neutron entering at -s1/2 just
    # misses s2/2, and 1 where a neutron entering at s1/2 just misses s2/2.
    h1 = abs(d1/(d1-d2))*(s1+s2)/2 - s1/2
    h2 = abs(-abs(d1/(d1-d2))*(s1-s2)/2 + s1/2)
    profile = [-h1, -h2, h2, h1], [0, 1, 1, 0]

    # Compute divergence from slits and from sample
    def fwhm2sigma(s):
        return s/sqrt(8*log(2))  # gaussian
        #return s*sqrt(2/9.)       # triangular
    dT_beam = fwhm2sigma(degrees(0.5*(s1+s2)/abs(d1-d2)))
    dT_s1_sample = fwhm2sigma(degrees(0.5*(s1+width*sin(theta))))
    dT_s2_sample = fwhm2sigma(degrees(0.5*(s2+width*sin(theta))))

    dT_sample = min([dT_beam, dT_s1_sample, dT_s2_sample])
    dT_est = degrees(np.std(phi[hit]))
    # use resample to estimate divergence from disk-shaped sample
    resample = np.random.choice(phi[hit],p=weight[hit]/np.sum(weight[hit]),size=1000000)
    dT_disk = degrees(np.std(resample))

    # Hits on z_proj should match hits directly on z
    hit_proj_z = through & (abs(z_proj) < width/2.)
    assert (hit_proj == hit_proj_z).all()

    # Bins to use for intensity vs. position x at the center of rotation
    # The scale factor for the estimated counts comes from setting the
    # area of the beam trapezoid to the total number of neutrons that pass
    # through slit 2.  One should also be able to estimate this by integrating
    # the intensity at each point xs in bin k from its contribution from points
    # at x1, decreased over the distance d1 by the angular spread, but I couldn't
    # work out the details properly.
    bins = np.linspace(min(xs[through]),max(xs[through]),51)
    scale = (bins[1]-bins[0])*n_through/(h1+h2)
    beam = np.interp(bins, profile[0], profile[1], left=0, right=0) * scale
    rect = np.interp(bins, [sx1, sx2], [1, 1], left=0, right=0) * beam
    bins_z = bins / sin(theta) - offset
    bins_w = 1 - (2*bins_z/width)**2
    bins_w[bins_w < 0] = 0.
    disk = sqrt(bins_w) * beam

    phi_bins = degrees(np.linspace(min(phi[through]),max(phi[through]),51))
    phi_max = degrees(arctan(0.5*(s1+s2)/abs(d1-d2)))
    phi_flat = degrees(arctan(0.5*abs((s1-s2)/(d1-d2))))
    phi_scale = n_through/(phi_max+phi_flat)*(phi_bins[1]-phi_bins[0])
    def trapezoidal_variance(a, b):
        """
        Variance of a symmetric trapezoidal distribution.

        The trapezoid slopes up in (-b, -a), is flat in (-a, a) and slopes
        down in (a, b).
        """
        tails = (b-a)/6*(3*a**2 + 2*a*b + b**2)
        flat = (2/3)*a**3
        return (tails + flat)/(a+b)
    def trapezoidal_divergence(s1, s2, d1, d2):
        phi_max = degrees(arctan(0.5*(s1+s2)/abs(d1-d2)))
        phi_flat = degrees(arctan(0.5*abs(s1-s2)/abs(d1-d2)))
        return sqrt(trapezoidal_variance(phi_flat, phi_max))
    dT_beam_trap = sqrt(trapezoidal_variance(phi_flat, phi_max))
    dT_beam_est = np.std(degrees(phi[through]))

    print("spread: %f deg" % degrees(spread))
    print("acceptance: %.2f%%" % (n_through*100./n))
    print("footprint: %.2f%%, disk: %.2f%%" % (100*n_hit/n_through, 100*n_hit_disk/n_through))
    print("projection: %.2f%%, disk: %.2f%%" % (100*n_hit_proj/n_through, 100*n_hit_disk_proj/n_through))
    print("backside: %.2f%%" % (n_backside*100./n_through))
    print("dT beam: traditional %f  trapezoidal %f  estimated %f" % (dT_beam, dT_beam_trap, dT_beam_est))
    print("dT sample: %f  est: %f  disk: %f" % (dT_sample, dT_est, dT_disk))


    from pylab import subplot, hist, plot, legend, grid, show, xlabel

    subplot(221)
    #hist(xs[through], bins=bins, label="beam"); xlabel("x (mm)")
    #plot(bins, beam, '-r', label="_")
    #hist(x2, bins=bins, label="x2", alpha=0.5)
    #hist(xs, bins=bins, label="xs", alpha=0.5)
    hist(degrees(phi[through]), bins=phi_bins, label="beam"); xlabel("phi (deg)")
    plot([-phi_max, -phi_flat, phi_flat, phi_max], [0, phi_scale, phi_scale, 0], '-r', label="_")
    grid(True); legend()
    subplot(223)
    #hist(xs[hit], bins=bins, label="sample"); xlabel("x (mm)")
    #plot(bins, rect, '-r', label="_")
    hist(degrees(phi[hit]), bins=phi_bins, label="sample"); xlabel("phi (deg)")
    grid(True); legend()
    subplot(224)
    #hist(xs[hit], bins=bins, weights=weight[hit], label="disk"); xlabel("x (mm)")
    #plot(bins, disk, '-r', label="_")
    hist(degrees(phi[hit]), bins=phi_bins, weights=weight[hit], label="disk"); xlabel("phi (deg)")
    grid(True); legend()
    subplot(222)
    if False:  # plot collision points
        plot(zp[hit][:1000], xp[hit][:1000], '.')
        plot(zp[through&~hit][:1000], xp[through&~hit][:1000], '.', color=(0.8,0.,0.))
        #plot(zp[~through][:1000], xp[~through][:1000], '.', color=(0.5,0.5,0.5))
        plot([sz1,sz2],[sx1,sx2], '-', color=(0.0, 0.8, 0.0))  # sample edges
    else:
        # set p to position on sample of the hit
        p = z + width/2
        pmin, pmax = p[through].min(), p[through].max()
        # find beam spill from disk
        spill_weight = 1 - weight
        spill_weight[~hit] = 1
        spill_weight[~through] = 0
        bins = np.linspace(pmin, pmax, 100)
        hist(p[hit], label="sample", bins=bins)
        hist(p[hit], label="disk", weights=weight[hit], bins=bins)
        #hist(p, label="dspill", weights=spill_weight, bins=bins)
        hist(p[through&~hit], label="spill", bins=bins)
        legend()
    show()

if __name__ == "__main__":
    resolution_simulator()
    #plot_dLoL()
    #plot_channels(10)
