import numpy as np

from kite import Scene


def test_deramp():
    c = np.arange(20, dtype=float)
    E, N = np.meshgrid(c, c)

    displ = (-3 + 5.4 * E) + (10 + 2.5 * N)
    sc = Scene(displacement=displ, llLat=0, llLon=0.0, dLat=0.3, dLon=0.3)

    sc.deramp.get_ramp_coefficients(sc.displacement)
