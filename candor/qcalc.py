from warnings import warn

import numpy as np
from numpy import (pi, radians, degrees, sin, cos, arctan, arctan2, arcsin,
                   arccos, sqrt, sign)


def angle_to_qxz(sample_theta, detector_theta, sample_lambda=5., detector_lambda=5.):
    lambda_i, lambda_f = sample_lambda, detector_lambda
    theta_i, theta_f = 0. - sample_theta, detector_theta - sample_theta
    k_i, k_f = 2 * pi / lambda_i, 2 * pi / lambda_f
    k_ix = k_i * cos(radians(theta_i))
    k_fx = k_f * cos(radians(theta_f))
    qx = k_fx - k_ix
    k_iz = k_i * sin(radians(theta_i))
    k_fz = k_f * sin(radians(theta_f))
    qz = k_fz - k_iz
    return qx, qz

def clip_angle(theta):
    return np.remainder(theta + 180, 360) - 180

def same_sign(x, y):
    return sign(x) == sign(y) or abs(x) < 1e-10 or abs(y) < 1e-10

def _error(old, new):
    num = np.linalg.norm((np.asarray(old) - new))
    den = np.linalg.norm(old)
    #print("error", num, den)
    return num/den

def qxz_to_angle_pak(qx, qz, sample_lambda=5., detector_lambda=5.):
    r"""
    Algorithm
    ---------
    Using the following:

    $q = k_i - k_f$

    $[k_{ix}, k_{iz}]^T = \tfrac{2\pi}{\lambda_i} [\cos\theta_i, \sin\theta_i]^T$

    $[k_{fx}, k_{fz}]^T = \tfrac{2\pi}{\lambda_f} [\cos\theta_f, \sin\theta_f]^T$

    solve for $\theta_f$,  giving:

    $\cos\theta_f = \lambda_f q_x/2\pi + \tfrac{\lambda_f}{\lambda_i}\cos \theta_i$

    $\sin\theta_f = \lambda_f q_z/2\pi + \tfrac{\lambda_f}{\lambda_i}\sin \theta_i$

    With some trig substitutions we get:

    ..math::

        (\lambda_i q_x/2\pi + \cos\theta_i)^2 + (\lambda_i q_z/2\pi + \sin\theta_i)^2
            = \left(\tfrac{\lambda_i}{\lambda_f}\right)^2

    Letting $X = \lambda_i q_x/2\pi$,
    $Z = \lambda_i q_z/2\pi$,
    $C = \left(\tfrac{\lambda_i}{\lambda_f}\right)^2$,
    and solving for $\theta_i$ gives:

    .. math::

        \tan \theta_i/2 = \frac{
            2 Z \pm \sqrt{2(C+1)(X^2 + Z^2) - (X^2 + Z^2)^2 - (C-1)^2}
        }{
            2 X - (X^2 + Z^2) + (C - 1)
        }

    and

    .. math::

        \sin \theta_f = q_z \lambda_f + \tfrac{\lambda_f}{\lambda_i}\sin \theta_i
    """
    # Use zero angles for q near zero
    if abs(qx) < 1e-10 and abs(qz) < 1e-10:
        return 0., 0.
    lambda_i, lambda_f = sample_lambda, detector_lambda
    kx, kz = qx/(2*pi), qz/(2*pi)

    # Solving the following:
    #   (lambda_i k_z + sin theta_i)^2 + (lambda_i k_x + cos theta_i)^2 = (lambda_i/lambda_f)^2


    # Construct quadratic solution parts
    X, Z = kx*lambda_i, kz*lambda_i
    C = (lambda_i/lambda_f)**2
    discriminant = 2*(C+1)*(X**2+Z**2) - (X**2+Z**2)**2 - (C-1)**2
    if discriminant < 0:
        warn("unsolvable position (qx, qz) = (%g,%g); discriminant is %g"
             %(qx, qz, discriminant))
        discriminant = 0.
    scale = 2*X - (X**2 + Z**2) + (C - 1)

    # Plus root discriminant solution
    theta_ip = 2*arctan2(2*Z + sqrt(discriminant), scale)
    theta_fp = arcsin(kz*lambda_f + lambda_f/lambda_i*sin(theta_ip))
    # Note that theta_i is (usually) negative, so sample angle is -theta_i
    # and detector angle is theta_f - theta_i.
    sample_theta_p = clip_angle(degrees(-theta_ip))
    detector_theta_p = clip_angle(degrees(theta_fp - theta_ip))

    qxz_p = angle_to_qxz(sample_theta_p, detector_theta_p, sample_lambda, detector_lambda)
    error_p = _error((qx, qz), qxz_p)
    if -10 <= detector_theta_p <= 120 and error_p < 1e-10:
        return sample_theta_p, detector_theta_p

    # Minus root discriminant solution
    theta_im = 2*arctan2(2*Z - sqrt(discriminant), scale)
    theta_fm = arcsin(kz*lambda_f + lambda_f/lambda_i*sin(theta_im))
    # Note that theta_i is (usually) negative, so sample angle is -theta_i
    # and detector angle is theta_f - theta_i.
    sample_theta_m = clip_angle(degrees(-theta_im))
    detector_theta_m = clip_angle(degrees(theta_fm - theta_im))

    # Pick the better branch (one of them will be really bad)
    qxz_m = angle_to_qxz(sample_theta_m, detector_theta_m, sample_lambda, detector_lambda)
    error_m = _error((qx, qz), qxz_m)
    if -10 <= detector_theta_m <= 120 and error_m < 1e-10:
        return sample_theta_m, detector_theta_m

    #print("err in both", error_p, error_m)
    if error_m > 1e-10 and error_p > 1e-10:
        warn("relative error solving for (qx, qz) = (%g,%g) is %g"
             % (qx, qz, min(error_m, error_p)))
    if error_p < error_m:
        return sample_theta_p, detector_theta_p
    else:
        return sample_theta_m, detector_theta_m

def qxz_to_angle_bbm(q_x, q_z, sample_lambda=5., detector_lambda=5.):
    qsq = q_x**2 + q_z**2
    if qsq < 1e-20:
        return 0, 0
    lambda_i, lambda_f = sample_lambda, detector_lambda
    k_i, k_f = 2 * pi / lambda_i, 2 * pi / lambda_f
    cos_delta = (k_i**2 + k_f**2 - qsq) / (2 * k_i * k_f)
    if abs(cos_delta) > 1:
        #warn("unsolvable delta q_x=%g, q_z=%g, L_i=%g, L_f=%g, cos(delta)=%g"
        #     %(q_x, q_z, sample_lambda, detector_lambda, cos_delta))
        return 0, 0
    delta = arccos(cos_delta)
    if q_z < 0:
        delta = -delta
    sin_theta = (k_f * q_x * sin(delta) - k_f * q_z*cos_delta + k_i*q_z)/qsq
    if abs(sin_theta) > 1:
        # Note: never encountered in 1 million trials; the cos_delta check
        # seems to remove all the invalid cases.
        warn("unsolvable delta q_x=%g, q_z=%g, L_i=%g, L_f=%g, sin(theta)=%g"
             %(q_x, q_z, sample_lambda, detector_lambda, sin_theta))
        return 0, 0
    detector_theta = degrees(delta)
    sample_theta = degrees(arcsin(sin_theta))

    # Check answer and chose different arcsin branch if it fails.  Do phase
    # unwrapping if necessary to keep sample_theta in [-180, 180]
    qxz = angle_to_qxz(sample_theta, detector_theta, lambda_i, lambda_f)
    if _error((q_x, q_z), qxz) > 1e-8:
        sample_theta = 180 - sample_theta if sample_theta > 0 else -180 - sample_theta

    return sample_theta, detector_theta

def qxz_to_angle_nice(qx, qz, sample_lambda=5., detector_lambda=5.):
    # Use zero angles for q near zero
    if abs(qx) < 1e-10 and abs(qz) < 1e-10:
        return 0., 0.
    lambda_i, lambda_f = sample_lambda, detector_lambda
    k_i, k_f = 2 * pi / lambda_i, 2 * pi / lambda_f
    qsq = qx**2 + qz**2
    A = k_f**2 - k_i**2 - qsq
    zl = - qz * A / (2 * qsq)
    zrsq = zl**2 + qx**2 * k_i**2 / qsq - (A/2)**2 / qsq
    if zrsq < 0:
        return 0., 0.
        warn("unsolvable position (qx, qz) = (%g,%g); discriminant is %g"
             %(qx, qz, zrsq))
        zrsq = 0.
    zr = sqrt(zrsq)

    z_p = zl + zr
    x_p = -(A + 2*qz*z_p)/(2 * qx)
    sample_theta_p = arctan2(z_p, -x_p)
    detector_theta_p = arctan2(qz - z_p, qx - x_p) + sample_theta_p
    sample_theta_p = clip_angle(degrees(sample_theta_p))
    detector_theta_p = clip_angle(degrees(detector_theta_p))

    #if sign(sample_theta_p) == sign(qz):
    if -10 <= detector_theta_p <= 120:
        return sample_theta_p, detector_theta_p

    z_m = zl - zr
    x_m = -(A + 2*qz*z_m)/(2 * qx)
    sample_theta_m = arctan2(z_m, -x_m)
    detector_theta_m = arctan2(qz - z_m, qx - x_m) + sample_theta_m
    sample_theta_m = clip_angle(degrees(sample_theta_m))
    detector_theta_m = clip_angle(degrees(detector_theta_m))
    return sample_theta_m, detector_theta_m
qxz_to_angle = qxz_to_angle_nice

def _check_angle(angles, wavelengths, tol=1e-8):
    #wavelengths = 4.0, 5.0
    qxz = angle_to_qxz(angles[0], angles[1], wavelengths[0], wavelengths[1])
    result = qxz_to_angle_bbm(qxz[0], qxz[1], wavelengths[0], wavelengths[1])
    qxz_new = angle_to_qxz(result[0], result[1], wavelengths[0], wavelengths[1])

    ## Diff from original angle
    #error = np.linalg.norm(angles - result)/np.linalg.norm(angles)
    #if error > tol:
    #    print("round trip fails for (%g, %g) => (%g, %g) with error %g"
    #          % (angles[0], angles[1], result[0], result[1], error))

    # Check if this differs from the NICE equation results

    ## Diff between nice and new
    #error = np.linalg.norm(np.array(nice) - result)/np.linalg.norm(result)
    #if error > tol:
    #    print("angles from (%g, %g) => (%g, %g) vs nice(%g, %g)"
    #          % (angles[0], angles[1], result[0], result[1], nice[0], nice[1]))
    #    print("qxz from (%g, %g) back to q (%g, %g) vs nice(%g, %g)"
    #          % (qxz[0], qxz[1], qxz_new[0], qxz_new[1], qxz_nice[0], qxz_nice[1]))
    if _error(qxz, qxz_new) > tol:
        print("bbm bad", angles, "===> qx/qz", qxz, "===> angles", result, "===> qx/qz", qxz_new)
    elif _error(angles, result) > tol:
        #print("bbm new", angles, "===>", result)
        pass

    return  # Don't check nice

    nice = qxz_to_angle_nice(qxz[0], qxz[1], wavelengths[0], wavelengths[1])
    qxz_nice = angle_to_qxz(nice[0], nice[1], wavelengths[0], wavelengths[1])
    if _error(qxz, qxz_nice) > tol:
        print("nice bad", angles, "===> qx/qz", qxz, "===> angles", nice, "===> qx/qz", qxz_new)
    elif _error(angles, nice) > tol:
        #print("nice new", angles, "===>", nice)
        pass

def _check_qxz(qxz, wavelengths, tol=1e-8):
    result = qxz_to_angle_bbm(qxz[0], qxz[1], wavelengths[0], wavelengths[1])
    qxz_new = angle_to_qxz(result[0], result[1], wavelengths[0], wavelengths[1])
    if result != (0, 0) and _error(qxz, qxz_new) > tol:
        #return  # don't check nic if bbm bad
        print("bbm bad","qx/qz", qxz, "===> angles", result, "===> qx/qz", qxz_new)

    return

    nice = qxz_to_angle_nice(qxz[0], qxz[1], wavelengths[0], wavelengths[1])
    qxz_nice = angle_to_qxz(nice[0], nice[1], wavelengths[0], wavelengths[1])
    if _error(qxz, qxz_nice) > tol:
        print("nic bad","qx/qz", qxz, "===> angles", nice, "===> qx/qz", qxz_nice)

    #nice = qxz_to_angle_nice(qxz[0], qxz[1], wavelengths[0], wavelengths[1])
    #qxz_nice = angle_to_qxz(nice[0], nice[1], wavelengths[0], wavelengths[1])
    #if _error(qxz, qxz_nice) > tol:
    #    print("nic bad","qx/qz", qxz, "===> angles", nice, "===> qx/qz", qxz_nice)

def _check_many_angles(count=1, tol=1e-8):
    """
    angle => qxz => angle
    """
    #sample_angle = np.random.uniform(-90, 90, count)
    sample_angle = np.random.uniform(-180, 180, count)
    detector_angle = np.random.uniform(-10, 120, count)
    lambda_i = np.random.uniform(4.0, 6.0, count)
    lambda_f = np.random.uniform(4.0, 6.0, count)
    for i in range(count):
        if i and i%10000 == 0: print(i, "of", count)
        angles = sample_angle[i], detector_angle[i]
        wavelengths = lambda_i[i], lambda_f[i]
        wavelengths = 4., 5.
        _check_angle(angles, wavelengths, tol)


def _check_many_qxz(count=1, tol=1e-8):
    """
    qxz => angle => qxz
    """
    qx = np.random.uniform(-2.6, 1.5, count)
    qz = np.random.uniform(-2.6, 2.6, count)
    lambda_i = np.random.uniform(4.0, 6.0, count)
    lambda_f = np.random.uniform(4.0, 6.0, count)
    for i in range(count):
        if i and i%10000 == 0: print(i, "of", count)
        qxz = qx[i], qz[i]
        wavelengths = lambda_i[i], lambda_f[i]
        #wavelengths = 4., 5.
        _check_qxz(qxz, wavelengths, tol)



if __name__ == "__main__":
    import sys
    count = int(sys.argv[1]) if sys.argv[1:] else 1
    #_check_many_angles(count, tol=1e-7)
    _check_many_qxz(count)
    #_check_angle((83.37818939681772, 48.71538285854942), (4.,5.))
    #_check_angle((100.46183025394834, 71.39230850016918), (4.,5.))