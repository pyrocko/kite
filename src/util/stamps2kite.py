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

log = logging.getLogger('stamp2kite')

d2r = num.pi/180.
r2d = 180./num.pi


class ADict(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def read_mat_data(fn_locations='ps2.mat', fn_psvel='ps_plot_v-d.mat',
                  fn_look_angle='la2.mat', fn_parms='parms.mat'):
    log.debug('Reading in StaMPS project...')
    data = ADict()

    loc_mat = h5py.File(fn_locations, 'r')
    vel_mat = h5py.File(fn_psvel, 'r')

    look_angle_mat = h5py.File(fn_look_angle, 'r')
    params_mat = h5py.File(fn_parms, 'r')

    data.ll_coords = num.asarray(loc_mat['ll0'])

    geo_coords = num.asarray(loc_mat['lonlat'])
    data.lats = geo_coords[0, :]
    data.lons = geo_coords[1, :]

    data.ps_velocities = num.asarray(vel_mat['ph_disp']).ravel()

    data.look_angles = num.asarray(look_angle_mat['la'])
    data.heading = float(num.asarray(params_mat['heading']))

    return data


def bin_ps_data(data, bins=(800, 800)):
    log.debug('binning/gridding PS velocity data')
    bin_look_angles, edg_lat, edg_lon, binnumber = stats.binned_statistic_2d(
        data.lats, data.lons, data.look_angles,
        statistic='mean', bins=bins)

    log.debug('binning/gridding look angle data')
    bin_vels, edg_lat, edg_lon, binnumber = stats.binned_statistic_2d(
        data.lats, data.lons, data.ps_velocities,
        statistic='mean', bins=bins)

    data.bin_ps_velocities = bin_vels.reshape(*bins)
    data.bin_look_angles = bin_look_angles.reshape(*bins)
    data.bin_edg_lat = edg_lat
    data.bin_edg_lon = edg_lon

    return data


def _get_file(dirname, fname):
    fns = glob.glob(op.join(dirname, fname))
    if len(fns) == 0:
        raise OSError('Cannot find a %s file in %s', fname, dirname)
    if len(fns) > 1:
        raise OSError('Found multiple files for %s: %s', fname, ', '.join(fns))
    fn = fns[0]

    log.debug('Got file %s', fn)
    return fn


def stamps2kite(
        dirname='.',
        fn_ps=None, fn_ps_plot=None, fn_parms=None, fn_la2=None,
        bins_east=800, bins_north=800):
    '''Convert StaMPS velocity data to a Kite Scene

    Loads a StaMPS project, bins the PS velocities from ``ps_plot(..., -1)`` to
    a regular grid and converts the surface displacement data to a
    :class:`~kite.Scene`.

    If explicit filenames `fn_ps` or `fn_ps_plot` are not passed an
    auto-discovery of :file:`ps?.mat` and :file:`ps_plot*.mat` in
    directory :param:`dirname` is approached.

    .. note ::

        Running the stamps2kite script on your StaMPS project does
        the trick in most cases.

    :param dirname: Folder containing data from the StaMPS project.
        Defaults to ``None``.
    :type dirname: str
    :param fn_ps: :file:`ps2.mat` file with lat/lon information about the PS.
        Defaults to ``None``.
    :type fn_ps: str
    :param fn_ps_plot: Processed output from StaMPS ``ps_plot`` function, e.g.
        :file:`ps_plot*.mat`. Defaults to ``None``.
    :type fn_ps_plot: str
    :param fn_parms: :file:`parms.mat` from StaMPS, holding essential meta-
        information.
    :type fn_parms: str
    :param fn_ln2: The :file:`la2.mat` file from StaMPS containing the look
        angle data.

    :param bins_east: Number of pixels/bins in East dimension. Default 500.
    :type bins_east: int
    :param bins_north: Number of pixels/bins in North dimension. Default 500.
    :type bins_north: int

    :returns: Kite Scene from the StaMPS data
    :rtype: :class:`kite.Scene`
    '''

    if fn_ps_plot is None:
        fn_locations = _get_file(dirname, 'ps?.mat')

    if fn_ps is None:
        fn_psvel = _get_file(dirname, 'ps_plot*.mat')

    if fn_parms is None:
        fn_parms = _get_file(dirname, 'parms.mat')

    if fn_la2 is None:
        fn_la2 = _get_file(dirname, 'la2.mat')

    log.info('Found a StaMPS project at %s', op.abspath(dirname))

    data = read_mat_data(fn_locations, fn_psvel, fn_la2, fn_parms)

    data = bin_ps_data(data, bins=(bins_east, bins_north))

    log.debug('Processing of LOS angles')
    data.bin_theta = (num.pi/2) - data.bin_look_angles

    phi_angle = -data.heading * d2r
    data.bin_phi = num.full_like(data.bin_theta, phi_angle)
    data.bin_phi[num.isnan(data.bin_theta)] = num.nan

    log.debug('Setting up the Kite Scene')
    config = SceneConfig()
    config.frame.llLat = data.bin_edg_lat.min()
    config.frame.llLon = data.bin_edg_lon.min()
    config.frame.dE = data.bin_edg_lon[1] - data.bin_edg_lon[0]
    config.frame.dN = data.bin_edg_lat[1] - data.bin_edg_lat[0]
    config.frame.spacing = 'degree'

    scene = Scene(
        theta=data.bin_theta.T,
        phi=data.bin_phi.T,
        displacement=data.bin_ps_velocities.T,
        config=config)

    return scene


def plot_scatter_vel(lat, lon, vel):
    vmax = abs(max(vel.min(), vel.max()))
    norm = colors.Normalize(vmin=-vmax, vmax=vmax)

    plt.scatter(lat, lon, s=5, c=vel, cmap='RdYlBu', norm=norm)
    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='''Convert a StaMPS project to a Kite scene.

Running the stamps2kite script on your StaMPS project does
the trick in most cases.''')
    parser.add_argument(
        'folder', nargs='?', type=str,
        default='.',
        help='StaMPS project folder. Default is current directory.')
    parser.add_argument(
        '--out', default=None, type=str,
        help='Filename to save the Kite scene to.')
    parser.add_argument(
        '--force', default=False, action='store_true',
        help='Force overwrite of an existing scene.')
    parser.add_argument(
        '-v', action='count',
        default=0,
        help='Verbosity, add mutliple to increase verbosity.')

    args = parser.parse_args()

    log_level = logging.INFO - args.v * 10
    logging.basicConfig(level=log_level if log_level > 0 else 0)

    fn_save = args.out
    if args.out:
        for fn in (fn_save, fn_save + '.yml', fn_save + '.npz'):
            if op.exists(fn) and not args.force:
                raise UserWarning('File %s exists! Use --force to overwrite.' %
                                  fn_save)

    scene = stamps2kite(dirname=args.folder)

    if fn_save:
        fn_save.rstrip('.yml')
        fn_save.rstrip('.npz')

        log.info('Saving StaMPS scene to file %s[.yml/.npz]...', fn_save)
        scene.save(args.out)

    else:
        scene.spool()


if __name__ == '__main__':
    main()
