import logging
import time

import numpy as num

"""Ellispoidal Cavity Model (ECM), triaxial elipsoidal deformation source.

Code is after Medhi Nikkhoo Matlab scripts found on
(http://www.volcanodeformation.com)

2017/05 - Marius Isken

Functions defined in this file serve as a backend for
kite.source.ellipsoidal_source

After

    Nikkhoo, M., Walter, T. R., Lundgren, P. R., Prats-Iraola, P. (2017):
    Compound dislocation models (CDMs) for volcano deformation analyses.
    Geophys Journal International, 208 (2): 877-894. doi:10.1093/gji/ggw427

    website:
    http://www.volcanodeformation.com
"""

logger = logging.getLogger("ECM")

d2r = num.pi / 180.0
r2d = 180.0 / num.pi
sqrt = num.sqrt
pi = num.pi


def cosd(deg):
    return num.cos(deg * d2r)


def sind(deg):
    return num.sin(deg * d2r)


def strike_dip(rot_mat, idx):
    Vstrike = num.array([-rot_mat[1, idx], rot_mat[0, idx], 0.0])
    Vstrike = Vstrike / num.linalg.norm(Vstrike)
    strike = num.arctan2(Vstrike[0], Vstrike[1]) * r2d
    if num.isnan(strike):
        strike = 0.0
    dip = num.arccos(rot_mat[2, idx]) * r2d
    return strike, dip


def rotation_matrix(rotx, roty, rotz):
    Rx = num.matrix(
        [[1.0, 0.0, 0.0], [0.0, cosd(rotx), sind(rotx)], [0.0, -sind(rotx), cosd(rotx)]]
    )
    Ry = num.matrix(
        [[cosd(roty), 0.0, -sind(roty)], [0.0, 1.0, 0.0], [sind(roty), 0.0, cosd(roty)]]
    )
    Rz = num.matrix(
        [[cosd(rotz), sind(rotz), 0.0], [-sind(rotz), cosd(rotz), 0.0], [0.0, 0.0, 1.0]]
    )

    return Rz * Ry * Rx


def pointCDM(coords, x0, y0, z0, rotx, roty, rotz, dVx, dVy, dVz, nu):
    """Point Compound Dislocation Model for surface displacements

    :param coords: Coordinates upon displacement is calculated
    :type coords: :class:`numpy.ndarray` of shape Nx2
    :param x0: Horizontal x-coordinate of the source, same unit as `coords`
    :type x0: float
    :param y0: Horizontal y-coordinate of the source, same unit as `coords`
    :type y0: float
    :param z0: Depth of the source, same unit as `coords`
    :type z0: float
    :param rotx: Clockwise rotation of ellipsoid around x-axis, [deg]
    :type rotx: float
    :param roty: Clockwise rotation of ellipsoid around y-axis, [deg]
    :type roty: float
    :param rotz: Clockwise rotation of ellipsoid around z-axis, [deg]
    :type rotz: float
    :param dVx: Volume change in axis-x, same unit as `coords`
    :type dVx: float
    :param dVy: Volume change in axis-y, same unit as `coords`
    :type dVy: float
    :param dVz: Volume change in axis-z, same unit as `coords`
    :type dVz: float
    :param nu: Poisson's ratio
    :type nu: float
    :returns: Volume change in axis-x
    :rtype: tuple of :class:`numpy.ndarray`
    """
    ncoords = coords.shape[0]
    rot_mat = rotation_matrix(rotx, roty, rotz)

    Ue = num.zeros(ncoords)
    Un = num.zeros(ncoords)
    Uv = num.zeros(ncoords)

    coords_shifted = coords.copy()
    coords_shifted[:, 0] -= x0
    coords_shifted[:, 1] -= y0

    component_names = ["dVx", "dVy", "dVz"]

    for icomp, comp in enumerate([dVx, dVy, dVz]):
        if num.all(comp):
            t0 = time.time()
            strike, dip = strike_dip(rot_mat, icomp)
            comp_ue, comp_un, comp_uv = PointDisplacementSurface(
                coords_shifted, z0, strike, dip, comp, nu
            )
            Ue += comp_ue
            Un += comp_un
            Uv += comp_uv
            logger.debug(
                "Calculated component %s [%.6f s]"
                % (component_names[icomp], time.time() - t0)
            )

    return Ue, Un, Uv


def ECM(coords, x0, y0, z0, rotx, roty, rotz, ax, ay, az, P, mu, lamda):
    """Calculate 2D surface displacement of a triaxial elipsoidal source.

    After:

        Nikkhoo, M., Walter, T. R., Lundgren, P. R., Prats-Iraola, P. (2017):
        Compound dislocation models (CDMs) for volcano deformation analyses.
        Geophys Journal International, 208 (2): 877-894. doi:10.1093/gji/ggw427

        website:
        http://www.volcanodeformation.com

    :param coords: Coordinates upon displacement is calculated
    :type coords: :class:`numpy.ndarray` of shape Nx2
    :param x0: Horizontal x-coordinate of the source, same unit as `coords`
    :type x0: float
    :param y0: Horizontal y-coordinate of the source, same unit as `coords`
    :type y0: float
    :param z0: Depth of the source, same unit as `coords`
    :type z0: float
    :param rotx: Clockwise rotation of ellipsoid around x-axis, [deg]
    :type rotx: float
    :param roty: Clockwise rotation of ellipsoid around y-axis, [deg]
    :type roty: float
    :param rotz: Clockwise rotation of ellipsoid around z-axis, [deg]
    :type rotz: float
    :param ax: Length of x semi-axis of ellisoid before rotation,
        same unit as `coords`
    :type ax: float
    :param ay: Length of y semi-axis of ellisoid before rotation,
        same unit as `coords`
    :type ay: float
    :param az: Length of z semi-axis of ellisoid before rotation,
        same unit as `coords`
    :type az: float
    :param P: Pressure on the cavity walls, same unit as the Lame constants
    :type P: float
    :param mu: Lame constant
    :type mu: float
    :param lamda: Lame constant
    :type lamda: float
    """
    ncoords = coords.shape[0]

    nu = lamda / (lamda + mu) / 2  # Poison's ratio
    K = lamda + 2 * mu / 3  # Bulk Modulus

    r0 = 1e-12  # Instability threshold for shape tensor
    ax = ax if ax > r0 else r0
    ay = ay if ay > r0 else r0
    az = az if az > r0 else r0

    a_arr = num.array([ax, ay, az])
    ia_sort = num.argsort(a_arr)[::-1]
    shape_tensor = shapeTensor(*a_arr[ia_sort], nu=nu)
    # Transform strain
    eT = -num.linalg.inv(shape_tensor) * P * num.ones((3, 1)) / 3.0 / K
    sT = (2 * mu * eT) + lamda * eT.sum()
    V = 4.0 / 3 * pi * ax * ay * az

    stress_tensor = sT[ia_sort]
    moment_tensor = V * stress_tensor

    dV = (eT.sum() - P / K) * V
    dV = dV + P * V / K  # Potency
    dVx, dVy, dVz = (
        1.0 / 2.0 / mu * (moment_tensor - lamda / 3.0 / K * moment_tensor.sum())
    )

    #####

    rot_mat = rotation_matrix(rotx, roty, rotz)

    Ue = num.zeros(ncoords)
    Un = num.zeros(ncoords)
    Uv = num.zeros(ncoords)

    coords_shifted = coords.copy()
    coords_shifted[:, 0] -= x0
    coords_shifted[:, 1] -= y0

    component_names = ["dVx", "dVy", "dVz"]
    for icomp, comp in enumerate([dVx, dVy, dVz]):
        if num.all(comp):
            t0 = time.time()
            strike, dip = strike_dip(rot_mat, icomp)
            comp_ue, comp_un, comp_uv = PointDisplacementSurface(
                coords_shifted, z0, strike, dip, float(comp), nu
            )
            Ue += comp_ue
            Un += comp_un
            Uv += comp_uv
            logger.debug(
                "Calculated component %s [%.6f s]"
                % (component_names[icomp], time.time() - t0)
            )

    return Ue, Un, Uv, dV, dV


def shapeTensor(a1, a2, a3, nu):
    """Calculates the Eshelby (1957) shape tensor components."""

    if a1 == 0.0 and a2 == 0.0 and a3 == 0:
        return num.zeros((3, 3)).view(num.matrix)

    # General case: triaxial ellipsoid
    if a1 > a2 and a2 > a3 and a3 > 0:
        logger.debug("General case: triaxial ellipsoid")
        sin_theta = sqrt(1 - a3**2 / a1**2)
        k = sqrt((a1**2 - a2**2) / (a1**2 - a3**2))

        # Calculate Legendre's incomplete elliptic integrals of the first and
        # second kind using Carlson (1995) method (see Numerical computation of
        # real or complex elliptic integrals. Carlson, B.C. Numerical
        # Algorithms (1995) 10: 13. doi:10.1007/BF02198293)
        tol = 1e-16
        c = 1 / sin_theta**2
        F = RF(c - 1, c - k**2, c, tol)
        E = F - k**2 / 3 * RD(c - 1, c - k**2, c, tol)

        I1 = (
            (4 * pi * a1 * a2)
            * a3
            / (a1**2 - a2**2)
            / sqrt(a1**2 - a3**2)
            * (F - E)
        )
        I3 = (
            (4 * pi * a1 * a2)
            * a3
            / (a2**2 - a3**2)
            / sqrt(a1**2 - a3**2)
            * (a2 * sqrt(a1**2 - a3**2) / a1 / a3 - E)
        )
        I2 = 4 * pi - I1 - I3

        I12 = (I2 - I1) / (a1**2 - a2**2)
        I13 = (I3 - I1) / (a1**2 - a3**2)
        I11 = (4 * pi / a1**2 - I12 - I13) / 3

        I23 = (I3 - I2) / (a2**2 - a3**2)
        I21 = I12
        I22 = (4 * pi / a2**2 - I23 - I21) / 3

        I31 = I13
        I32 = I23
        I33 = (4 * pi / a3**2 - I31 - I32) / 3

    # Special case-1: Oblate ellipsoid
    elif a1 == a2 and a2 > a3 and a3 > 0:
        logger.debug("Special case 1: oblate ellipsoid")
        I1 = (
            (2.0 * pi * a1 * a2)
            * a3
            / (a1**2 - a3**2) ** 1.5
            * (num.arccos(a3 / a1) - a3 / a1 * sqrt(1.0 - a3**2 / a1**2))
        )
        I2 = I1
        I3 = 4 * pi - 2 * I1

        I13 = (I3 - I1) / (a1**2 - a3**2)
        I11 = pi / a1**2 - I13 / 4
        I12 = I11

        I23 = I13
        I22 = pi / a2**2 - I23 / 4
        I21 = I12

        I31 = I13
        I32 = I23
        I33 = (4 * pi / a3**2 - 2 * I31) / 3

    # Special case-2: Prolate ellipsoid
    elif a1 > a2 and a2 == a3 and a3 > 0:
        logger.debug("Special case: prolate ellipsoid")
        I2 = (
            (2 * pi * a1 * a2)
            * a3
            / (a1**2 - a3**2) ** 1.5
            * (a1 / a3 * sqrt(a1**2 / a3**2 - 1) - num.arccosh(a1 / a3))
        )
        I3 = I2
        I1 = 4 * pi - 2 * I2

        I12 = (I2 - I1) / (a1**2 - a2**2)
        I13 = I12
        I11 = (4 * pi / a1**2 - 2 * I12) / 3

        I21 = I12
        I22 = pi / a2**2 - I21 / 4
        I23 = I22

        I32 = I23
        I31 = I13
        I33 = (4 * pi / a3**2 - I31 - I32) / 3

    # Special case-3: Sphere
    if a1 == a2 and a2 == a3:
        logger.debug("Special case: sphere")
        S1111 = (7.0 - 5 * nu) / 15.0 / (1.0 - nu)
        S1122 = (5 * nu - 1.0) / 15.0 / (1.0 - nu)
        S1133 = (5 * nu - 1.0) / 15.0 / (1.0 - nu)
        S2211 = (5 * nu - 1.0) / 15.0 / (1.0 - nu)
        S2222 = (7.0 - 5 * nu) / 15.0 / (1.0 - nu)
        S2233 = (5 * nu - 1.0) / 15.0 / (1.0 - nu)
        S3311 = (5 * nu - 1.0) / 15.0 / (1.0 - nu)
        S3322 = (5 * nu - 1.0) / 15.0 / (1.0 - nu)
        S3333 = (7.0 - 5 * nu) / 15.0 / (1.0 - nu)
    # General triaxial, oblate and prolate ellipsoids
    else:
        logger.debug("General case: triaxial, oblate and prolate ellipsoid")
        S1111 = (3.0 / 8.0 / pi) / (1.0 - nu) * (a1**2 * I11) + (
            1.0 - 2 * nu
        ) / 8.0 / pi / (1.0 - nu) * I1
        S1122 = (1.0 / 8.0 / pi) / (1.0 - nu) * (a2**2 * I12) - (
            1.0 - 2 * nu
        ) / 8.0 / pi / (1.0 - nu) * I1
        S1133 = (1.0 / 8.0 / pi) / (1.0 - nu) * (a3**2 * I13) - (
            1.0 - 2 * nu
        ) / 8.0 / pi / (1.0 - nu) * I1
        S2211 = (1.0 / 8.0 / pi) / (1.0 - nu) * (a1**2 * I21) - (
            1.0 - 2 * nu
        ) / 8.0 / pi / (1.0 - nu) * I2
        S2222 = (3.0 / 8.0 / pi) / (1.0 - nu) * (a2**2 * I22) + (
            1.0 - 2 * nu
        ) / 8.0 / pi / (1.0 - nu) * I2
        S2233 = (1.0 / 8.0 / pi) / (1.0 - nu) * (a3**2 * I23) - (
            1.0 - 2 * nu
        ) / 8.0 / pi / (1.0 - nu) * I2
        S3311 = (1.0 / 8.0 / pi) / (1.0 - nu) * (a1**2 * I31) - (
            1.0 - 2 * nu
        ) / 8.0 / pi / (1.0 - nu) * I3
        S3322 = (1.0 / 8.0 / pi) / (1.0 - nu) * (a2**2 * I32) - (
            1.0 - 2 * nu
        ) / 8.0 / pi / (1.0 - nu) * I3
        S3333 = (3.0 / 8.0 / pi) / (1.0 - nu) * (a3**2 * I33) + (
            1.0 - 2 * nu
        ) / 8.0 / pi / (1.0 - nu) * I3

    return num.matrix(
        [
            [S1111 - 1, S1122, S1133],
            [S2211, S2222 - 1, S2233],
            [S3311, S3322, S3333 - 1],
        ]
    )


def RF(x, y, z, r):
    """Calculates the RF term, Carlson (1995) method for elliptic integrals"""
    if x < 0 or y < 0 or z < 0:
        raise ArithmeticError("x, y and z values must be positive!")
    elif num.count_nonzero([x, y, z]) < 2:
        raise ArithmeticError("At most one of the x, y and z values can be zero!")

    xm = x
    ym = y
    zm = z
    A0 = (x + y + z) / 3
    Q = max([abs(A0 - x), abs(A0 - y), abs(A0 - z)]) / (3.0 * r) ** (1.0 / 6)
    n = 0
    Am = A0
    while abs(Am) <= Q / (4.0**n):
        lambdam = sqrt(xm * ym) + sqrt(xm * zm) + sqrt(ym * zm)
        Am = (Am + lambdam) / 4.0
        xm = (xm + lambdam) / 4.0
        ym = (ym + lambdam) / 4.0
        zm = (zm + lambdam) / 4.0
        n += 1
    X = (A0 - x) / 4**n / Am
    Y = (A0 - y) / 4**n / Am
    Z = -X - Y
    E2 = X * Y - Z**2
    E3 = X * Y * Z
    rf = (1.0 - E2 / 10 + E3 / 14 + E2**2.0 / 24 - 3.0 * E2 * E3 / 44.0) / sqrt(Am)
    return rf


def RD(x, y, z, r):
    """Calculates the RF term, Carlson (1995) method for elliptic integrals"""
    if z == 0:
        raise ArithmeticError("z value must be nonzero!")
    elif x == 0 and y == 0:
        raise ArithmeticError("At most one of the x and y values can be zero!")

    xm = x
    ym = y
    zm = z
    A0 = (x + y + 3 * z) / 5
    Q = max([abs(A0 - x), abs(A0 - y), abs(A0 - z)]) / (r / 4) ** (1.0 / 6)
    n = 0
    Am = A0
    S = 0
    while abs(Am) <= Q / (4**n):
        lambdam = sqrt(xm * ym) + sqrt(xm * zm) + sqrt(ym * zm)
        S = S + (1.0 / 4**n) / sqrt(zm) / (zm + lambdam)
        Am = (Am + lambdam) / 4
        xm = (xm + lambdam) / 4
        ym = (ym + lambdam) / 4
        zm = (zm + lambdam) / 4
        n += 1

    X = (A0 - x) / 4.0**n / Am
    Y = (A0 - y) / 4.0**n / Am
    Z = -(X + Y) / 3.0
    E2 = X * Y - 6 * Z**2
    E3 = (3 * X * Y - 8 * Z**2) * Z
    E4 = 3 * (X * Y - Z**2) * Z**2
    E5 = X * Y * Z**3
    rd = (
        1.0
        - 3 * E2 / 14
        + E3 / 6
        + 9 * E2**2 / 88
        - 3 * E4 / 22
        - 9 * E2 * E3 / 52
        + 3 * E5 / 26
    ) / 4**n / Am**1.5 + 3.0 * S
    return rd


def PointDisplacementSurface(coords_shifted, z0, strike, dip, dV, nu):
    """calculates surface displacements associated with a tensile
    point dislocation (PDF) in an elastic half-space (Okada, 1985).
    """
    ncoords = coords_shifted.shape[0]

    beta = strike - 90.0
    rot_mat = num.matrix([[cosd(beta), -sind(beta)], [sind(beta), cosd(beta)]])
    r_beta = rot_mat * coords_shifted.conj().T
    x = r_beta[0, :].view(num.ndarray).ravel()
    y = r_beta[1, :].view(num.ndarray).ravel()

    r = (x**2 + y**2 + z0**2) ** 0.5
    d = z0
    q = y * sind(dip) - d * cosd(dip)

    r3 = r**3
    rpd = r + d
    rpd2 = rpd**2
    a = (3 * r + d) / r3 / rpd**3

    I1 = (1.0 - 2 * nu) * y * (1.0 / r / rpd2 - x**2 * a)
    I3 = (1.0 - 2 * nu) * x / r3 - ((1.0 - 2 * nu) * x * (1.0 / r / rpd2 - y**2 * a))
    I5 = (1.0 - 2 * nu) * (1.0 / r / rpd - x**2 * (2 * r + d) / r3 / rpd2)

    # Note: For a PDF M0 = dV*mu!
    u = num.empty((ncoords, 3))

    u[:, 0] = x
    u[:, 1] = y
    u[:, 2] = d

    u *= (3.0 * q**2 / r**5)[:, num.newaxis]
    u[:, 0] -= I3 * sind(dip) ** 2
    u[:, 1] -= I1 * sind(dip) ** 2
    u[:, 2] -= I5 * sind(dip) ** 2
    u *= dV / 2 / pi

    r_beta = rot_mat.conj().T * u[:, :2].conj().T
    return (
        r_beta[0, :].view(num.ndarray).ravel(),  # ue
        r_beta[1, :].view(num.ndarray).ravel(),  # un
        u[:, 2],
    )  # uv


if __name__ == "__main__":
    nrows = 500
    ncols = 500

    x0 = 250.0
    y0 = 250.0
    depth = 30.0

    rotx = 0.0
    roty = 0.0
    rotz = 0.0

    ax = 1.0
    ay = 1.0
    az = 0.25

    # ax = 1.
    # ay = 1.
    # az = 1.

    P = 1e6
    mu = 0.33e11
    lamda = 0.33e11

    X, Y = num.meshgrid(num.arange(nrows), num.arange(ncols))

    coords = num.empty((nrows * ncols, 2))
    coords[:, 0] = X.ravel()
    coords[:, 1] = Y.ravel()

    ue, un, uv = ECM(coords, x0, y0, depth, rotx, roty, rotz, ax, ay, az, P, mu, lamda)
