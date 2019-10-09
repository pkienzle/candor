import numpy as np
from numpy import pi, sin, radians, log10

from . import simulate
from . import instrument

def main():
    candor = instrument.candor_setup()
    sample = simulate.fake_sample()
    wavelengths, I_in, _, I_portion = candor.spectrum
    I_out = I_in * I_portion
    angles = candor.detectorTable.rowAngularOffsets
    print("events per second in detector bank: %e" % (np.sum(I_out),))
    #print(wavelengths)
    #print(angles)
    print("wavelengths: %d, angles: %d" % (len(wavelengths), len(angles)))
    def count(offset, measurement_time):
        kz = 2*pi/wavelengths[:, None] * sin(radians(offset + angles[None, :]))
        R = sample(kz)
        events = measurement_time*R*I_out[:, None]
        #events[kz < 0.025] /= 1000
        #events[events/measurement_time > 1e5] = 1e5 # clip at 100,000/s
        print("time: %g, angle: %g, events: %e"
            % (measurement_time, offset, np.sum(events)))
        return kz, R, events
    #kz, R, events = count(0.5, 600)
    #kz, R, events = count(0.05, 1)
    batches = (0.1, 1), (0.25, 1), (0.5, 60)
    #batches = batches[-1:]  # just the longest one
    counts = [count(c, t) for c, t in batches]
    kz, R, events = [np.vstack(v) for v in zip(*counts)]
    print("total events: %e" % (np.sum(events)))
    #print("max events/s %e"%(np.sum(I_out)*len(wavelengths),))

    import matplotlib.pyplot as plt
    if 0:
        q = np.linspace(0, 0.2, 400)
        plt.semilogy(q, sample(q/2))
        plt.xlabel("Q (1/Ang)")
        plt.ylabel("R")
    elif 0:
        #plt.pcolormesh(edges(angles), edges(wavelengths), kz)
        #plt.pcolormesh(edges(angles), edges(wavelengths), R)
        #plt.pcolormesh(edges(angles), edges(wavelengths), log10(R))
        plt.pcolormesh(edges(angles), edges(wavelengths), log10(events))
        plt.ylabel("wavelength (A)")
        plt.xlabel("Angle (degrees)")
        plt.colorbar()
    else:
        if 1:
            for k, ev in zip(kz.T, events.T):
                plt.semilogy(2*k, ev, '.')
            plt.ylabel("expected events")
        else:
            for k, r in zip(kz.T, R.T):
                plt.semilogy(2*k, r, '.')
            plt.ylabel("R")
        plt.xlabel("Q (1/Ang)")
        angles, times = zip(*batches)
        plt.title("time: %s  angle: %s"%(times, angles))
    plt.show()

def edges(v):
    mid = 0.5*(v[:-1] + v[1:])
    return np.hstack((2*v[0] - mid[0], mid, 2*v[-1] - mid[-1]))
if __name__ == "__main__":
    main()
