import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as num

from kite import SandboxScene
from kite.sources import OkadaSource

km = 1e3

sandbox = SandboxScene()

nframes = 20
depths = num.linspace(4 * km, 4 * km, nframes)
strikes = num.linspace(0, 180, nframes)

okada = OkadaSource(
    northing=40 * km,
    easting=40 * km,
    depth=4 * km,
    length=8 * km,
    width=4 * km,
    strike=63.0,
    rake=0,
    dip=0.0,
    slip=3.0,
    opening=0,
)

sandbox.addSource(okada)

sandbox.processSources()

fig, axis = plt.subplots(nrows=2, ncols=2)


# Plot the resulting surface displacements


def imargs(data):
    max_value = max(num.abs(data.max()), num.abs(data.min()))
    return {
        "X": data,
        "cmap": "bwr",
        "vmin": -max_value,
        "vmax": max_value,
        "animated": True,
    }


components = [sandbox.north, sandbox.east, sandbox.down, sandbox.displacement]
titles = ["North", "East", "Down", "LOS"]
images = []


for ax, comp, title in zip(fig.axes, components, titles):
    im = ax.imshow(**imargs(comp))
    images.append(im)
    ax.set_title(title)


def update_figure(iframe, *args):
    print("Updating figure! (frame %03d)" % iframe)

    okada.depth = depths[iframe]
    okada.strike = strikes[iframe]
    sandbox.processSources()

    for im, comp in zip(images, components):
        args = imargs(comp)
        im.set_data(comp)
    return images


ani = animation.FuncAnimation(
    fig, update_figure, interval=50, frames=nframes, blit=True
)

fig.show()
