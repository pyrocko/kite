import matplotlib.pyplot as plt
import numpy as num

from kite import SandboxScene
from kite.sources import OkadaSource, PyrockoRectangularSource

km = 1e3

sandbox = SandboxScene()
# Set the LOS incident angles, remember :class:`kite.Scene`
sandbox.phi.fill(num.rad2deg(100.0))
sandbox.theta.fill(num.rad2deg(23.0))

okada = OkadaSource(
    northing=40 * km,
    easting=40 * km,
    depth=4 * km,
    length=8 * km,
    width=4 * km,
    strike=63.0,
    dip=33.0,
    slip=3.0,
    opening=1,
)

pyrocko_rectangular = PyrockoRectangularSource(
    northing=40 * km,
    easting=40 * km,
    depth=4 * km,
    length=8 * km,
    width=4 * km,
    strike=63.0,
    dip=33.0,
    slip=3.0,
    store_dir="gfstore_halfspace",
)

sandbox.addSource(okada)

sandbox.processSources()

# Plot the resulting surface displacements
fig, axis = plt.subplots(nrows=2, ncols=2)
axis[0][0].imshow(sandbox.north)
axis[0][1].imshow(sandbox.east)
axis[1][0].imshow(sandbox.down)
axis[1][1].imshow(sandbox.displacement)  # Displacement in LOS
fig.show()
