import logging
import os.path as op
import re

import numpy as num
import pyrocko.orthodrome as od
import shapefile
import utm
from scipy import stats

from kite.scene import Scene, SceneConfig

log = logging.getLogger("bbd2kite")

d2r = num.pi / 180.0
r2d = 180.0 / num.pi


class DataStruct(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def read_projection(filename):
    prj_filename = op.splitext(filename)[0] + ".prj"
    if not op.exists(prj_filename):
        log.warning("Could not find %s, defaulting to UTM Zone 32N")
        return 32, "N"

    with open(prj_filename, "r") as f:
        matches = re.findall(r'[\'"](\w*)[\'"]', f.read())

    if not matches:
        log.warning("Could not find projection in %s", prj_filename)
        return 32, "N"

    projection = matches[0]
    if "UTM" not in projection:
        raise AttributeError("Projection is not UTM: %s" % projection)

    zone, letter = int(projection[-3:-1]), projection[-1]
    return zone, letter


def read_shapefile(filename):
    log.info("Loading data from %s", filename)
    shp = shapefile.Reader(filename)

    npoints = shp.numRecords
    field_name_map = {fld[0].lower(): fld[0] for fld in shp.fields}

    data = DataStruct()
    data.bbox = shp.bbox

    data.ps_mean_v = num.zeros(npoints)
    data.ps_mean_var = num.zeros(npoints)
    los_n = num.zeros(npoints)
    los_e = num.zeros(npoints)
    los_u = num.zeros(npoints)

    coords = num.zeros((npoints, 2))

    for isr, sr in enumerate(shp.iterShapeRecords()):
        shape = sr.shape
        record = sr.record
        # assert shape.shapeType == 11

        los_n[isr] = getattr(record, field_name_map["los_north"])
        los_e[isr] = getattr(record, field_name_map["los_east"])
        los_u[isr] = -getattr(record, field_name_map["los_up"])

        data.ps_mean_v[isr] = getattr(record, field_name_map["mean_velo"])
        data.ps_mean_var[isr] = getattr(record, field_name_map["var_mean_v"])

        coords[isr] = shape.points[0]

    data.phi = num.arctan2(los_n, los_e)
    data.theta = num.arcsin(los_u)

    data.easts = coords[:, 0]
    data.norths = coords[:, 1]

    return data


def bin_ps_data(data, bins=(800, 800)):
    log.debug("Binning mean velocity data...")
    bin_vels, edg_E, edg_N, _ = stats.binned_statistic_2d(
        data.easts, data.norths, data.ps_mean_v, statistic="mean", bins=bins
    )

    log.debug("Binning LOS angles...")
    bin_phi, _, _, _ = stats.binned_statistic_2d(
        data.easts, data.norths, data.phi, statistic="mean", bins=bins
    )
    bin_theta, _, _, _ = stats.binned_statistic_2d(
        data.easts, data.norths, data.theta, statistic="mean", bins=bins
    )

    log.debug("Binning mean velocity variance...")
    bin_mean_var, _, _, _ = stats.binned_statistic_2d(
        data.easts, data.norths, data.ps_mean_var, statistic="mean", bins=bins
    )

    data.bin_mean_var = bin_mean_var
    data.bin_ps_mean_v = bin_vels

    data.bin_phi = bin_phi
    data.bin_theta = bin_theta

    data.bin_edg_N = edg_N
    data.bin_edg_E = edg_E

    return data


def bbd2kite(filename, px_size=(500, 500), import_var=False, convert_m=True):
    """Convert BGR BodenBewegungsdienst PS velocity data to a Kite Scene

    Loads the mean PS velocities (from e.g. ``ps_plot(..., -1)``) from a
    BGR BodenBewegungsdienst, and grids the data into mean velocity bins.
    The LOS velocities will be converted to a Kite Scene
    (:class:`~kite.Scene`).

    :param filename: Name of the BGR BBD as ESRI shapefile.
    :type filename: str
    :param px_size: Size of pixels in North and East in meters.
        Default (500, 500).
    :type px_size: tuple
    :param convert_m: Convert displacement to meters, default True.
    :type convert_m: bool
    :param import_var: Import the mean velocity variance, this information
        is used by the Kite scene to define the covariance.
    :param import_var: bool
    """
    data = read_shapefile(filename)

    if convert_m:
        data.ps_mean_v /= 1e3
        data.ps_mean_var /= 1e3

    # lengthN = od.distance_accurate50m(
    #     data.bbox[1], data.bbox[0],
    #     data.bbox[3], data.bbox[0])
    # lengthE = od.distance_accurate50m(
    #     data.bbox[1], data.bbox[0],
    #     data.bbox[1], data.bbox[2])

    lengthE = data.bbox[2] - data.bbox[0]
    lengthN = data.bbox[3] - data.bbox[1]

    bins = (round(lengthE / px_size[0]), round(lengthN / px_size[1]))

    bin_ps_data(data, bins=bins)

    log.debug("Setting up the Kite Scene")
    config = SceneConfig()
    zone, letter = read_projection(filename)

    llLat, llLon = utm.to_latlon(data.bbox[0], data.bbox[1], zone, letter)
    config.frame.llLat = llLat
    config.frame.llLon = llLon

    config.frame.dE = data.bin_edg_E[1] - data.bin_edg_E[0]
    config.frame.dN = data.bin_edg_N[1] - data.bin_edg_N[0]
    config.frame.spacing = "meter"

    scene_name = op.basename(op.abspath(filename))
    config.meta.scene_title = "%s (BodenbewegungsDienst import)" % scene_name
    config.meta.scene_id = scene_name
    # config.meta.time_master = data.tmin.timestamp()
    # config.meta.time_slave = data.tmax.timestamp()

    scene = Scene(
        theta=data.bin_theta,
        phi=data.bin_phi,
        displacement=data.bin_ps_mean_v,
        config=config,
    )

    if import_var:
        scene.displacement_px_var = data.bin_mean_var

    return scene


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="""Convert BodenbewegungsDienst PS displacements into a Kite scene.

Loads the PS velocities delivered by BGR BodenbewegungsDienst
(https://bodenbewegungsdienst.bgr.de) and grids the data in mean velocity bins.
The mean LOS velocities will be converted into a Kite Scene.

The data is delivered in ESRI shapefile format.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "file", type=str, default=".", help="BodenbewegungsDienst shape file."
    )
    parser.add_argument(
        "--resolution",
        "-r",
        nargs=2,
        metavar=("mE", "mN"),
        dest="resolution",
        type=int,
        default=(500, 500),
        help="pixel size of the output grid in east and north (meter)."
        " Default is 500 m by 500 m.",
    )
    parser.add_argument(
        "--save",
        "-s",
        default=None,
        type=str,
        dest="save",
        help="filename to save the Kite scene to. If not given, the scene"
        " will be opened in spool GUI.",
    )
    parser.add_argument(
        "--force",
        "-f",
        default=False,
        action="store_true",
        dest="force",
        help="force overwrite of an existing scene.",
    )
    parser.add_argument(
        "--keep-mm",
        action="store_true",
        default=False,
        help="keep mm/a and do not convert to m/a.",
    )
    parser.add_argument(
        "--import-var",
        action="store_true",
        dest="import_var",
        default=False,
        help="import the variance, which is added to Kite's"
        " scene covariance matrix.",
    )
    parser.add_argument(
        "-v",
        action="count",
        default=0,
        help="verbosity, add multiple to increase verbosity.",
    )

    args = parser.parse_args()

    log_level = logging.INFO - args.v * 10
    logging.basicConfig(level=log_level if log_level > 0 else 0)

    fn_save = args.save
    if args.save:
        for fn in (fn_save, fn_save + ".yml", fn_save + ".npz"):
            if op.exists(fn) and not args.force:
                raise UserWarning("File %s exists! Use --force to overwrite." % fn_save)

    scene = bbd2kite(
        filename=args.file,
        px_size=args.resolution,
        import_var=args.import_var,
        convert_m=not args.keep_mm,
    )

    if fn_save is not None:
        fn_save.rstrip(".yml")
        fn_save.rstrip(".npz")

        log.info(
            "Saving BGR BodenBewegunsdienst scene to file %s[.yml/.npz]...",  # noqa
            fn_save,
        )
        scene.save(args.save)

    elif scene:
        scene.spool()


if __name__ == "__main__":
    main()
