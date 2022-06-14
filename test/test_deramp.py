import numpy as num

from kite import Scene


def test_deramp():
    c = num.arange(20, dtype=num.float)
    E, N = num.meshgrid(c, c)

    displ = (-3 + 5.4 * E) + (10 + 2.5 * N)
    sc = Scene(displacement=displ, llLat=0, llLon=0.0, dLat=0.3, dLon=0.3)

    sc.displacement_deramp(demean=True, inplace=True)

    coeffs = sc.get_ramp_coefficients()
    num.testing.assert_almost_equal(coeffs, num.zeros_like(coeffs))
