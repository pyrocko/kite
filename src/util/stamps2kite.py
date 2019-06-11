import logging
import glob
from os import path as op

import numpy as num
from scipy import stats
try:
    import h5py
except ImportError as e:
    raise e('Please install h5py library')

import matplotlib.pyplot as plt
from matplotlib import colors

from kite.scene import Scene, SceneConfig

log = logging.getLogger()


def read_mat_data(fn_locations='ps2.mat', fn_psvel='ps_plot_v-d.mat'):
    loc_mat = h5py.File(fn_locations, 'r')
    vel_mat = h5py.File(fn_psvel, 'r')

    coords = num.asarray(loc_mat['lonlat'])
    ps_velocities = num.asarray(vel_mat['ph_disp']).ravel()

    lon = coords[0, :]
    lat = coords[1, :]

    # plot_scatter_vel(lat, lon, psvel)
    return lat, lon, ps_velocities


def bin_data(lats, lons, vels, bins=(50, 50), statistic='mean', calcstd=False):
    binned_vels, x_edg, y_edg, binnumber = stats.binned_statistic_2d(
        lats, lons, vels,
        statistic=statistic, bins=bins)

    if calcstd:
        std, _, _, _ = stats.binned_statistic_2d(
            lats, lons, vels,
            statistic='std', bins=[x_edg, y_edg])

        return binned_vels, std

    return binned_vels, x_edg, y_edg


def stamps_to_kite_scene(
        dirname='.', fn_ps=None, fn_ps_plot=None,
        bins_east=800, bins_north=800):
    '''Convert StaMPS velocity data to a Kite Scene

    Loads a StaMPS project, bins the PS velocities from ``ps_plot(..., -1)`` to
    a regular grid and converts the surface displacement data to a
    :class:`~kite.Scene`.

    If explicit filenames `fn_ps` or `fn_ps_plot` are not passed an
    auto-discovery of :file:`ps?.mat` and :file:`ps_plot*.mat` in
    directory :param:`dirname` is approached.

    :param dirname: Folder containing data from the StaMPS project.
        Defaults to ``None``.
    :type dirname: str
    :param fn_ps: :file:`ps?.mat` file with lat/lon information about the PS.
        Defaults to ``None``.
    :type fn_ps: str
    :param fn_ps_plot: Processed output from StaMPS ``ps_plot`` function, e.g.
        :file:`ps_plot*.mat`. Defaults to ``None``.
    :type fn_ps_plot: str

    :param bins_east: Number of pixels/bins in East dimension. Default 500.
    :type bins_east: int
    :param bins_north: Number of pixels/bins in North dimension. Default 500.
    :type bins_north: int

    :returns: Kite Scene from the StaMPS data
    :rtype: :class:`kite.Scene`
    '''

    if fn_ps_plot is None:
        fn_locations = glob.glob(op.join(dirname, 'ps?.mat'))
        if len(fn_locations) != 1:
            raise NameError('Cannot find a ps?.mat file in %s', dirname)
        fn_locations = fn_locations[0]

    if fn_ps is None:
        fn_psvel = glob.glob(op.join(dirname, 'ps_plot*.mat'))
        if len(fn_psvel) != 1:
            raise NameError('Cannot find a ps_plot*.mat file in %s', dirname)
        fn_psvel = fn_psvel[0]

    lats, lons, vels = read_mat_data(fn_locations, fn_psvel)

    binned_vels, e_edg, n_edg = bin_data(
        lats, lons, vels, bins=(bins_east, bins_north))

    config = SceneConfig()
    config.frame.llLat = lats.min()
    config.frame.llLon = lons.min()
    config.frame.dE = e_edg[1] - e_edg[0]
    config.frame.dN = n_edg[1] - n_edg[0]
    config.frame.spacing = 'degree'

    scene = Scene(
        theta=num.zeros_like(binned_vels),
        phi=num.zeros_like(binned_vels),
        displacement=binned_vels,
        config=config)

    return scene


def plot_scatter_vel(lat, lon, vel):
    vmax = abs(max(vel.min(), vel.max()))
    norm = colors.Normalize(vmin=-vmax, vmax=vmax)

    plt.scatter(lat, lon, s=5, c=vel, cmap='RdYlBu', norm=norm)
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    lats, lons, vels = read_mat_data()
