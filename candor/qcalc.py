from warnings import warn

import numpy as np
from numpy import (pi, radians, degrees, sin, cos, arctan, arctan2, arcsin,
                   sqrt, sign)


def angle_to_qxz(sample_theta, detector_theta, sample_lambda=5., detector_lambda=5.):
    lambda_i, lambda_f = sample_lambda, detector_lambda
    theta_i, theta_f = 0. - sample_theta, detector_theta - sample_theta
    k_ix = 2 * pi / lambda_i * cos(radians(theta_i))
    k_iz = 2 * pi / lambda_i * sin(radians(theta_i))
    k_fx = 2 * pi / lambda_f * cos(radians(theta_f))
    k_fz = 2 * pi / lambda_f * sin(radians(theta_f))
    qx = k_fx - k_ix
    qz = k_fz - k_iz
    return qx, qz

def clip_angle(theta):
    return np.remainder(theta + 180, 360) - 180

def same_sign(x, y):
    return sign(x) == sign(y) or abs(x) < 1e-10 or abs(y) < 1e-10

def qxz_to_angle(qx, qz, sample_lambda=5., detector_lambda=5.):
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

    $(\lambda_i q_x/2\pi + \cos\theta_i)^2 + (\lambda_i q_z/2\pi + \sin\theta_i)^2 = (\tfrac{\lambda_i}{\lambda_f})^2$

    Letting $X = \lambda_i q_x/2\pi$, $Z = \lambda_i q_z/2\pi$, $C = (\tfrac{\lambda_i}{\lambda_f})^2$,
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
    discriminant = 2*(C+1)*(X**2+Z**2) - (X**2 + Z**2)**2 - (C-1)**2
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
    same_sign_p = same_sign(sample_theta_p, detector_theta_p)
    distance_p = sqrt(sample_theta_p**2 + detector_theta_p**2)
    result_p = sample_theta_p, detector_theta_p

    # Minus root discriminant solution
    theta_im = 2*arctan2(2*Z - sqrt(discriminant), scale)
    theta_fm = arcsin(kz*lambda_f + lambda_f/lambda_i*sin(theta_im))
    # Note that theta_i is (usually) negative, so sample angle is -theta_i
    # and detector angle is theta_f - theta_i.
    sample_theta_m = clip_angle(degrees(-theta_im))
    detector_theta_m = clip_angle(degrees(theta_fm - theta_im))
    same_sign_m = same_sign(sample_theta_m, detector_theta_m)
    distance_m = sqrt(sample_theta_m**2 + detector_theta_m**2)
    result_m = sample_theta_m, detector_theta_m

    # Prefer solution with with the same sign on the angles, or if both have
    # the same sign, choose the one with the smallest angles.  Treat values
    # near zero is treated as the sign of the other value.
    #print(result_m, result_p)
    if same_sign_m == same_sign_p:
        return result_p if distance_p <= distance_m else result_m
    elif same_sign_p:
        return result_p
    elif same_sign_m:
        return result_m
    else:
        raise RuntimeError("unreachable code")

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
    result_p = sample_theta_p, detector_theta_p
    same_sign_p = same_sign(sample_theta_p, detector_theta_p)
    distance_p = sqrt(sample_theta_p**2 + detector_theta_p**2)

    z_m = zl - zr
    x_m = -(A + 2*qz*z_m)/(2 * qx)
    sample_theta_m = arctan2(z_m, -x_m)
    detector_theta_m = arctan2(qz - z_m, qx - x_m) + sample_theta_m
    sample_theta_m = clip_angle(degrees(sample_theta_m))
    result_m = sample_theta_m, detector_theta_m
    detector_theta_m = clip_angle(degrees(detector_theta_m))
    same_sign_m = same_sign(sample_theta_m, detector_theta_m)
    distance_m = sqrt(sample_theta_m**2 + detector_theta_m**2)

    # Prefer solution with with the same sign on the angles, or if both have
    # the same sign, choose the one with the smallest angles.  Treat values
    # near zero is treated as the sign of the other value.
    #print(result_m, result_p)
    if same_sign_m == same_sign_p:
        return result_p if distance_p <= distance_m else result_m
    elif same_sign_p:
        return result_p
    elif same_sign_m:
        return result_m
    else:
        raise RuntimeError("unreachable code")
#qxz_to_angle = qxz_to_angle_nice

def _check_qxz_to_angle(tol=1e-8):
    wavelengths = 4.0, 5.0
    # random angles in [-45, +45], with incident matching reflected
    angles = np.random.rand(2)*45*np.random.choice([-1, 1])
    #angles[0] = 0
    #angles[1] = 0
    qxz = angle_to_qxz(angles[0], angles[1], wavelengths[0], wavelengths[1])
    result = qxz_to_angle(qxz[0], qxz[1], wavelengths[0], wavelengths[1])
    error = np.linalg.norm(angles - result)/np.linalg.norm(angles)
    if error > tol:
         print("round trip fails for (%g, %g) => (%g, %g) with error %g"
               % (angles[0], angles[1], result[0], result[1], error))
    #assert error < tol

if __name__ == "__main__":
    _check_qxz_to_angle()