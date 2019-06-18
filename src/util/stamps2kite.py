import logging
import glob
from os import path as op
from datetime import datetime, timedelta

import numpy as num
from scipy import stats, interpolate

try:
    import h5py
except ImportError as e:
    raise e('Please install h5py library')

import matplotlib.pyplot as plt
from matplotlib import colors

from kite.scene import Scene, SceneConfig

log = logging.getLogger('stamps2kite')

d2r = num.pi/180.
r2d = 180./num.pi


class ADict(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def _get_file(dirname, fname):
    fns = glob.glob(op.join(dirname, fname))
    if len(fns) == 0:
        raise ImportError('Cannot find %s file in %s' % (fname, dirname))
    if len(fns) > 1:
        raise ImportError('Found multiple files for %s: %s' %
                          (fname, ', '.join(fns)))
    fn = fns[0]

    log.debug('Got file %s', fn)
    return fn


def read_mat_data(dirname, **kwargs):
    # TODO: Add old matlab import
    data = ADict()
    log.debug('Reading in StaMPS project...')

    fn_ps2 = kwargs.get(
        'fn_ps2', _get_file(dirname, 'ps2.mat'))
    fn_mv2 = kwargs.get(
        'fn_mv2', _get_file(dirname, 'mv2.mat'))
    fn_ps_plot = kwargs.get(
        'fn_ps_plot', _get_file(dirname, 'ps_plot*.mat'))
    fn_parms = kwargs.get(
        'fn_parms', _get_file(dirname, 'parms.mat'))
    fn_width = kwargs.get(
        'fn_width', _get_file(dirname, 'width.txt'))
    fn_len = kwargs.get(
        'fn_len', _get_file(dirname, 'len.txt'))
    fn_look_angle = kwargs.get(
        'fn_look_angle', _get_file(dirname, 'look_angle.1.in'))

    ps2_mat = h5py.File(fn_ps2, 'r')
    mv2_mat = h5py.File(fn_mv2, 'r')
    ps_plot_mat = h5py.File(fn_ps_plot, 'r')

    params_mat = h5py.File(fn_parms, 'r')

    data.ll_coords = num.asarray(ps2_mat['ll0'])
    data.radar_coords = num.asarray(ps2_mat['ij'])
    data.ps_mean_v = num.asarray(ps_plot_mat['ph_disp']).ravel()
    data.ps_mean_std = num.asarray(mv2_mat['mean_v_std']).ravel()

    geo_coords = num.asarray(ps2_mat['lonlat'])
    data.lats = geo_coords[0, :]
    data.lons = geo_coords[1, :]

    days = num.asarray(ps2_mat['day'])
    data.tmin = timedelta(days=days.min() - 366) + datetime(1, 1, 1)
    data.tmax = timedelta(days=days.max() - 366) + datetime(1, 1, 1)

    with open(fn_len) as rl, open(fn_width) as rw:
        data.px_length = int(rl.readline())
        data.px_width = int(rw.readline())

    data.look_angles = num.loadtxt(fn_look_angle)[::2]
    data.heading = float(num.asarray(params_mat['heading']))

    return data


def bin_ps_data(data, bins=(800, 800)):
    log.debug('Binning StaMPS velocity data...')
    bin_vels, edg_lat, edg_lon, _ = stats.binned_statistic_2d(
        data.lats, data.lons, data.ps_mean_v,
        statistic='mean', bins=bins)

    log.debug('Binning radar coordinates...')
    bin_radar_x, _, _, _ = stats.binned_statistic_2d(
        data.lats, data.lons, data.radar_coords[1],
        statistic='mean', bins=bins)
    bin_radar_y, _, _, _ = stats.binned_statistic_2d(
        data.lats, data.lons, data.radar_coords[2],
        statistic='mean', bins=bins)

    log.debug('Binning mean velocity variance...')
    bin_mean_std, _, _, _ = stats.binned_statistic_2d(
        data.lats, data.lons, data.ps_mean_std,
        statistic='mean', bins=bins)

    data.bin_ps_mean_v = bin_vels
    data.bin_mean_var = bin_mean_std**2  # We want variance

    data.bin_radar_x = bin_radar_x
    data.bin_radar_y = bin_radar_y

    data.bin_edg_lat = edg_lat
    data.bin_edg_lon = edg_lon

    return data


def interpolate_look_angles(data):
    log.info('Interpolating look angles from radar coordinates...')
    width_coords = num.linspace(0, data.px_width, 50)
    len_coords = num.linspace(0, data.px_length, 50)
    coords = num.asarray(num.meshgrid(width_coords, len_coords))\
        .reshape(2, 2500)

    radar_coords = num.vstack(
        [data.bin_radar_y.ravel(), data.bin_radar_x.ravel()])

    interp = interpolate.LinearNDInterpolator(coords.T, data.look_angles*d2r)
    data.bin_look_angles = interp(radar_coords.T).reshape(
        *data.bin_ps_mean_v.shape)
    return interp


def stamps2kite(dirname='.', bins=(800, 800), convert_m=True, import_var=False,
                **kwargs):
    '''Convert StaMPS velocity data to a Kite Scene

    Loads the mean PS velocities (from e.g. ``ps_plot(..., -1)``) from a
    StaMPS project, and grids the data into mean velocity bins. The LOS
    velocities will be converted to a Kite Scene (:class:`~kite.Scene`).

    If explicit filenames `fn_ps` or `fn_ps_plot` are not passed an
    auto-discovery of :file:`ps?.mat` and :file:`ps_plot*.mat` in
    directory :param:`dirname` is approached.

    .. note ::

        Running the stamps2kite script on your StaMPS project does
        the trick in most cases.

    :param dirname: Folder containing data from the StaMPS project.
        Defaults to ``.``.
    :type dirname: str
    :param fn_ps2: :file:`ps2.mat` file with lat/lon information about the PS
        scene. Defaults to ``ps2.mat``.
    :type fn_ps2: str
    :param fn_mv2: :file:`mv2.mat` file with mean velocity standard deviations
        of every PS point. Defaults to ``mv2.mat``.
    :type fn_mv2: str
    :param fn_ps_plot: Processed output from StaMPS ``ps_plot`` function, e.g.
        :file:`ps_plot*.mat`. Defaults to ``ps_plot*.mat``.
    :type fn_ps_plot: str
    :param fn_parms: :file:`parms.mat` from StaMPS, holding essential meta-
        information. Defaults to ``parms.mat``.
    :type fn_parms: str
    :param fn_look_angle: The :file:`look_angle.1.in` holds the a 50x50 grid
        with look angle information. Defaults to ``look_angle.1.in``.
    :type fn_look_angle: str
    :param fn_width: The :file:`width.txt` containing number of radar columns.
        Defaults to ``width.txt``.
    :type fn_width: str
    :param fn_len: The :file:`len.txt` containing number of rows in the
        interferogram. Defaults to ``len.txt``.
    :type fn_len: str

    :param bins: Number of pixels/bins in East and North dimension.
        Default (800, 800).
    :type bins: tuple

    :returns: Kite Scene from the StaMPS data
    :rtype: :class:`kite.Scene`
    '''
    data = read_mat_data(dirname, **kwargs)
    log.info('Found a StaMPS project at %s', op.abspath(dirname))

    if convert_m:
        data.ps_mean_v /= 1e3
        data.ps_mean_std /= 1e3

    bin_ps_data(data, bins=bins)
    interpolate_look_angles(data)

    log.debug('Processing of LOS angles')
    data.bin_theta = data.bin_look_angles

    phi_angle = -data.heading * d2r + num.pi
    if phi_angle > num.pi:
        phi_angle -= 2*num.pi
    data.bin_phi = num.full_like(data.bin_theta, phi_angle)
    data.bin_phi[num.isnan(data.bin_theta)] = num.nan

    log.debug('Setting up the Kite Scene')
    config = SceneConfig()
    config.frame.llLat = data.bin_edg_lat.min()
    config.frame.llLon = data.bin_edg_lon.min()
    config.frame.dE = data.bin_edg_lon[1] - data.bin_edg_lon[0]
    config.frame.dN = data.bin_edg_lat[1] - data.bin_edg_lat[0]
    config.frame.spacing = 'degree'

    scene_name = op.basename(op.abspath(dirname))
    config.meta.scene_title = '%s (StamPS import)' % scene_name
    config.meta.scene_id = scene_name
    config.meta.time_master = data.tmin.timestamp()
    config.meta.time_slave = data.tmax.timestamp()

    scene = Scene(
        theta=data.bin_theta.T,
        phi=data.bin_phi.T,
        displacement=data.bin_ps_mean_v.T,
        config=config)

    if import_var:
        scene.displacement_px_var = data.bin_mean_var.T

    return scene


def plot_scatter_vel(lat, lon, vel):
    vmax = abs(max(vel.min(), vel.max()))
    norm = colors.Normalize(vmin=-vmax, vmax=vmax)

    plt.scatter(lat, lon, s=5, c=vel, cmap='RdYlBu', norm=norm)
    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='''Convert StaMPS data into a Kite scene.

Loads the PS velocities from a StaMPS project (i.e. from ps_plot(..., -1);),
and grids the data into mean velocity bins. The LOS velocities will be
converted into a Kite Scene.

The data has to be fully processed through StaMPS and may stem from the master
project or the processed small baseline pairs. Required files are:

 - ps2.mat          Meta information and geographical coordinates.
 - parms.mat        Meta information about the scene (heading, etc.).
 - ps_plot*.mat     Processed and corrected LOS velocities.
 - mv2.mat          Mean velocity's standard deviation.

 - look_angle.1.in  Look angles for the scene.
 - width.txt        Width dimensions of the interferogram and
 - len.txt          length.
 ''',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        'folder', nargs='?', type=str,
        default='.',
        help='StaMPS project folder. Default is current directory.')
    parser.add_argument(
        '--resolution', '-r', nargs=2, metavar=('pxE', 'pxN'),
        dest='resolution', type=int, default=(800, 800),
        help='resolution of the output grid in East and North (pixels). '
             'Default is 800 by 800 px.')
    parser.add_argument(
        '--save', '-s', default=None, type=str, dest='save',
        help='filename to save the Kite scene to. If not defined, the scene'
             ' will be opened in spool GUI.')
    parser.add_argument(
        '--force', '-f', default=False, action='store_true', dest='force',
        help='force overwrite of an existing scene.')
    parser.add_argument(
        '-v', action='count',
        default=0,
        help='verbosity, add mutliple to increase verbosity.')
    parser.add_argument(
        '--keep-mm', action='store_true',
        default=False,
        help='keep mm/a and do not convert to m/a.')
    parser.add_argument(
        '--import-var', action='store_true', dest='import_var',
        default=False,
        help='import the variance from mv2.mat, which is added to Kite\'s'
             ' scene covariance matrix.')

    args = parser.parse_args()

    log_level = logging.INFO - args.v * 10
    logging.basicConfig(level=log_level if log_level > 0 else 0)

    fn_save = args.save
    if args.save:
        for fn in (fn_save, fn_save + '.yml', fn_save + '.npz'):
            if op.exists(fn) and not args.force:
                raise UserWarning(
                    'File %s exists! Use --force to overwrite.' % fn_save)

    scene = stamps2kite(dirname=args.folder, bins=args.resolution,
                        convert_m=not args.keep_mm,
                        import_var=args.import_var)

    if fn_save:
        fn_save.rstrip('.yml')
        fn_save.rstrip('.npz')

        log.info('Saving StaMPS scene to file %s[.yml/.npz]...', fn_save)
        scene.save(args.save)

    else:
        scene.spool()


if __name__ == '__main__':
    main()
