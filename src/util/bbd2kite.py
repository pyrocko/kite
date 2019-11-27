import shapefile
import logging
import numpy as num
import os.path as op

from scipy import stats

from kite.scene import Scene, SceneConfig


log = logging.getLogger('shp2kite')

d2r = num.pi/180.
r2d = 180./num.pi


class DataStruct(dict):

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def read_shapefile(filename):
    log.info('Loading data from %s', filename)
    shp = shapefile.Reader(filename)

    npoints = shp.numRecords

    data = DataStruct()
    data.ll_coords = shp.bbox[:1]

    data.ps_mean_v = num.zeros(npoints)
    data.ps_mean_var = num.zeros(npoints)
    los_n = num.zeros(npoints)
    los_e = num.zeros(npoints)
    los_u = num.zeros(npoints)

    coords = num.zeros((npoints, 2))

    for isr, sr in enumerate(shp.iterShapeRecords()):
        shape = sr.shape
        record = sr.record
        assert shape.shapeType == 1

        los_n[isr] = record.Los_North
        los_e[isr] = record.Los_East
        los_u[isr] = -record.Los_Up

        data.ps_mean_v[isr] = record.Mean_Velo
        data.ps_mean_v[isr] = record.Var_Mean_V

        coords[isr] = shape.points[0]

    data.phi = num.arctan2(los_n, los_e)
    data.theta = num.arcsin(los_u)

    data.lons = coords[:, 0]
    data.lats = coords[:, 1]

    return data


def bin_ps_data(data, bins=(800, 800)):
    log.debug('Binning mean velocity data...')
    bin_vels, edg_lat, edg_lon, _ = stats.binned_statistic_2d(
        data.lats, data.lons, data.ps_mean_v,
        statistic='mean', bins=bins)

    log.debug('Binning LOS angles...')
    bin_phi, _, _, _ = stats.binned_statistic_2d(
        data.lats, data.lons, data.phi,
        statistic='mean', bins=bins)
    bin_theta, _, _, _ = stats.binned_statistic_2d(
        data.lats, data.lons, data.theta,
        statistic='mean', bins=bins)

    log.debug('Binning mean velocity variance...')
    bin_mean_var, _, _, _ = stats.binned_statistic_2d(
        data.lats, data.lons, data.ps_mean_var,
        statistic='mean', bins=bins)

    data.bin_mean_var = bin_mean_var
    data.bin_ps_mean_v = bin_vels

    data.bin_phi = bin_phi
    data.bin_theta = bin_theta

    data.bin_edg_lat = edg_lat
    data.bin_edg_lon = edg_lon

    return data


def bbd2kite(filename, bins=(800, 800), import_var=False):
    data = read_shapefile(filename)
    bin_ps_data(data, bins=bins)

    log.debug('Setting up the Kite Scene')
    config = SceneConfig()
    config.frame.llLat = data.bin_edg_lat.min()
    config.frame.llLon = data.bin_edg_lon.min()
    config.frame.dE = data.bin_edg_lon[1] - data.bin_edg_lon[0]
    config.frame.dN = data.bin_edg_lat[1] - data.bin_edg_lat[0]
    config.frame.spacing = 'degree'

    scene_name = op.basename(op.abspath(filename))
    config.meta.scene_title = '%s (BodenBewegunsDienst import)' % scene_name
    config.meta.scene_id = scene_name
    # config.meta.time_master = data.tmin.timestamp()
    # config.meta.time_slave = data.tmax.timestamp()

    scene = Scene(
        theta=data.bin_theta,
        phi=data.bin_phi,
        displacement=data.bin_ps_mean_v,
        config=config)

    if import_var:
        scene.displacement_px_var = data.bin_mean_var

    return scene


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='''Convert BodenBewegunsDienst PS displacements into a Kite scene.

Loads the PS velocities delivered by BGR BodenBewegunsDienst
(https://bodenbewegungsdienst.bgr.de) and grids the data in mean velocity bins.
The mean LOS velocities will be converted into a Kite Scene.

The data is delivered in ESRI shapefile format. Python module pyshp is required.
''',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        'file', type=str,
        default='.',
        help='BodenBewegunsDienst shape file project folder.')
    parser.add_argument(
        '--resolution', '-r', nargs=2, metavar=('pxN', 'pxE'),
        dest='resolution', type=int, default=(800, 800),
        help='resolution of the output grid in North and East (pixels). '
             'Default is 800 by 800 px.')
    parser.add_argument(
        '--save', '-s', default=None, type=str, dest='save',
        help='filename to save the Kite scene to. If not given, the scene'
             ' will be opened in spool GUI.')
    parser.add_argument(
        '--force', '-f', default=False, action='store_true', dest='force',
        help='force overwrite of an existing scene.')
    parser.add_argument(
        '--import-var', action='store_true', dest='import_var',
        default=False,
        help='import the variance, which is added to Kite\'s'
             ' scene covariance matrix.')
    parser.add_argument(
        '-v', action='count',
        default=0,
        help='verbosity, add mutliple to increase verbosity.')

    args = parser.parse_args()

    log_level = logging.INFO - args.v * 10
    logging.basicConfig(level=log_level if log_level > 0 else 0)

    fn_save = args.save
    if args.save:
        for fn in (fn_save, fn_save + '.yml', fn_save + '.npz'):
            if op.exists(fn) and not args.force:
                raise UserWarning(
                    'File %s exists! Use --force to overwrite.' % fn_save)

    scene = bbd2kite(filename=args.file, bins=args.resolution,
                     import_var=args.import_var)

    if fn_save is not None:
        fn_save.rstrip('.yml')
        fn_save.rstrip('.npz')

        log.info('Saving BGR BodenBewegunsDienst scene to file %s[.yml/.npz]...',   # noqa
                 fn_save)
        scene.save(args.save)

    elif scene:
        scene.spool()


if __name__ == '__main__':
    main()
