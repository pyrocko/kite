import glob
import logging
from datetime import datetime, timedelta
from os import path as op

import numpy as np
import pyrocko.orthodrome as od
from scipy import interpolate, io, stats

try:
    import h5py
except ImportError as e:
    raise e("Please install h5py library")

import matplotlib.pyplot as plt
from matplotlib import colors

from kite.scene import Scene, SceneConfig

log = logging.getLogger("stamps2kite")

d2r = np.pi / 180.0
r2d = 180.0 / np.pi


class DataStruct(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def _get_file(dirname, fname):
    fns = glob.glob(op.join(dirname, fname))
    if len(fns) == 0:
        raise ImportError("Cannot find %s file in %s" % (fname, dirname))
    if len(fns) > 1:
        raise ImportError("Found multiple files for %s: %s" % (fname, ", ".join(fns)))
    fn = fns[0]

    log.debug("Found file %s", fn)
    return fn


def _read_mat(filename):
    try:
        mat = h5py.File(filename, "r")
    except OSError:
        log.debug("using old scipy import for %s", filename)
        mat = io.loadmat(filename)
    return mat


def read_mat_data(dirname, import_mv2=False, **kwargs):
    # TODO: Add old matlab import
    log.debug("Reading in StaMPS project...")

    fn_ps2 = kwargs.get("fn_ps2", _get_file(dirname, "ps2.mat"))
    fn_ps_plot = kwargs.get("fn_ps_plot", _get_file(dirname, "ps_plot*.mat"))
    fn_parms = kwargs.get("fn_parms", _get_file(dirname, "parms.mat"))
    fn_width = kwargs.get("fn_width", _get_file(dirname, "width.txt"))
    fn_len = kwargs.get("fn_len", _get_file(dirname, "len.txt"))
    fn_look_angle = kwargs.get("fn_look_angle", _get_file(dirname, "look_angle.1.in"))

    ps2_mat = _read_mat(fn_ps2)
    ps_plot_mat = _read_mat(fn_ps_plot)
    params_mat = _read_mat(fn_parms)

    data = DataStruct()
    data.ll_coords = np.asarray(ps2_mat["ll0"])
    data.radar_coords = np.asarray(ps2_mat["ij"])
    data.ps_mean_v = np.asarray(ps_plot_mat["ph_disp"]).ravel()

    geo_coords = np.asarray(ps2_mat["lonlat"])
    data.lons = geo_coords[0, :]
    data.lats = geo_coords[1, :]

    days = np.asarray(ps2_mat["day"])
    data.tmin = timedelta(days=days.min() - 366.25) + datetime(1, 1, 1)
    data.tmax = timedelta(days=days.max() - 366.25) + datetime(1, 1, 1)

    if import_mv2:
        fn_mv2 = kwargs.get("fn_mv2", _get_file(dirname, "mv2.mat"))
        mv2_mat = h5py.File(fn_mv2, "r")
        data.ps_mean_std = np.asarray(mv2_mat["mean_v_std"]).ravel()

    with open(fn_len) as rl, open(fn_width) as rw:
        data.px_length = int(rl.readline())
        data.px_width = int(rw.readline())

    data.look_angles = np.loadtxt(fn_look_angle)[::2]

    heading = float(np.asarray(params_mat["heading"]))
    if np.isnan(heading):
        raise ValueError("Heading information in parms.mat is missing!")

    data.heading = heading

    return data


def bin_ps_data(data, bins=(800, 800)):
    log.debug("Binning StaMPS velocity data...")
    bin_vels, edg_lat, edg_lon, _ = stats.binned_statistic_2d(
        data.lats, data.lons, data.ps_mean_v, statistic="mean", bins=bins
    )

    log.debug("Binning radar coordinates...")
    bin_radar_i, _, _, _ = stats.binned_statistic_2d(
        data.lats, data.lons, data.radar_coords[1], statistic="mean", bins=bins
    )
    bin_radar_j, _, _, _ = stats.binned_statistic_2d(
        data.lats, data.lons, data.radar_coords[2], statistic="mean", bins=bins
    )

    if "ps_mean_std" in data.keys():
        log.debug("Binning mean velocity variance...")
        bin_mean_std, _, _, _ = stats.binned_statistic_2d(
            data.lats, data.lons, data.ps_mean_std, statistic="mean", bins=bins
        )
        data.bin_mean_var = bin_mean_std**2  # We want variance

    data.bin_ps_mean_v = bin_vels

    data.bin_radar_i = bin_radar_i
    data.bin_radar_j = bin_radar_j

    data.bin_edg_lat = edg_lat
    data.bin_edg_lon = edg_lon

    return data


def interpolate_look_angles(data):
    log.info("Interpolating look angles from radar coordinates...")
    log.debug(
        "Radar coordinates extent width %d; length %d", data.px_width, data.px_length
    )
    log.debug(
        "Radar coordinates data: length %d - %d; width %d - %d",
        data.radar_coords[1].min(),
        data.radar_coords[1].max(),
        data.radar_coords[0].min(),
        data.radar_coords[0].max(),
    )
    log.debug(
        "Binned radar coordinate ranges: length %d - %d; width %d - %d",
        np.nanmin(data.bin_radar_i),
        np.nanmax(data.bin_radar_i),
        np.nanmin(data.bin_radar_j),
        np.nanmax(data.bin_radar_j),
    )

    width_coords = np.linspace(0, data.px_width, 50)
    len_coords = np.linspace(0, data.px_length, 50)
    coords = np.asarray(np.meshgrid(width_coords, len_coords)).reshape(2, 2500)

    radar_coords = np.vstack(
        [
            data.bin_radar_j.ravel() - data.radar_coords[0].min(),
            data.bin_radar_i.ravel() - data.radar_coords[1].min(),
        ]
    )

    interp = interpolate.LinearNDInterpolator(coords.T, data.look_angles)
    data.bin_look_angles = interp(radar_coords.T).reshape(*data.bin_ps_mean_v.shape)
    return interp


def stamps2kite(
    dirname=".", px_size=(800, 800), convert_m=True, import_var=False, **kwargs
):
    """Convert StaMPS velocity data to a Kite Scene

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
        of every PS point, created by `ps_plot(...)`. Defaults to ``mv2.mat``.
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
    :param px_size: Size of pixels in North and East in meters.
        Default (200, 200).
    :type px_size: tuple
    :param convert_m: Convert displacement to meters, default True.
    :type convert_m: bool
    :param import_var: Import the mean velocity variance, this information
        is used by the Kite scene to define the covariance.
    :param import_var: bool

    :returns: Kite Scene from the StaMPS data
    :rtype: :class:`kite.Scene`
    """
    data = read_mat_data(dirname, import_mv2=import_var, **kwargs)
    log.info("Found a StaMPS project at %s", op.abspath(dirname))

    bbox = (data.lons.min(), data.lats.min(), data.lons.max(), data.lats.max())

    lengthN = od.distance_accurate50m(bbox[1], bbox[0], bbox[3], bbox[0])
    lengthE = od.distance_accurate50m(bbox[1], bbox[0], bbox[1], bbox[2])
    bins = (round(lengthE / px_size[0]), round(lengthN / px_size[1]))

    if convert_m:
        data.ps_mean_v /= 1e3

    if convert_m and import_var:
        data.ps_mean_std /= 1e3

    bin_ps_data(data, bins=bins)
    interpolate_look_angles(data)

    log.debug("Processing of LOS angles")
    data.bin_theta = data.bin_look_angles * d2r

    phi_angle = -data.heading * d2r + np.pi
    if phi_angle > np.pi:
        phi_angle -= 2 * np.pi
    data.bin_phi = np.full_like(data.bin_theta, phi_angle)
    data.bin_phi[np.isnan(data.bin_theta)] = np.nan

    log.debug("Setting up the Kite Scene")
    config = SceneConfig()
    config.frame.llLat = data.bin_edg_lat.min()
    config.frame.llLon = data.bin_edg_lon.min()
    config.frame.dE = data.bin_edg_lon[1] - data.bin_edg_lon[0]
    config.frame.dN = data.bin_edg_lat[1] - data.bin_edg_lat[0]
    config.frame.spacing = "degree"

    scene_name = op.basename(op.abspath(dirname))
    config.meta.scene_title = "%s (StamPS import)" % scene_name
    config.meta.scene_id = scene_name
    config.meta.time_master = data.tmin.timestamp()
    config.meta.time_slave = data.tmax.timestamp()

    scene = Scene(
        theta=data.bin_theta,
        phi=data.bin_phi,
        displacement=data.bin_ps_mean_v,
        config=config,
    )

    if import_var:
        scene.displacement_px_var = data.bin_mean_var

    return scene


def plot_scatter_vel(lat, lon, vel):
    vmax = abs(max(vel.min(), vel.max()))
    norm = colors.Normalize(vmin=-vmax, vmax=vmax)

    plt.scatter(lat, lon, s=5, c=vel, cmap="RdYlBu", norm=norm)
    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="""Convert StaMPS data into a Kite scene.

Loads the PS velocities from a StaMPS project (i.e. from ps_plot(..., -1);),
and grids the data into mean velocity bins. The mean LOS velocities will be
converted into a Kite Scene.

The data has to be fully processed through StaMPS and may stem from the master
project or the processed small baseline pairs. Required files are:

 - ps2.mat          Meta information and geographical coordinates.
 - parms.mat        Meta information about the scene (heading, etc.).
 - ps_plot*.mat     Processed mean LOS velocities from ps_plot(..., -1).
 - mv2.mat          Mean velocity's standard deviation from ps_plot(..., -1).

 - look_angle.1.in  Look angles for the scene.
 - width.txt        Width dimensions of the interferogram and
 - len.txt          length.
 """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("folder", type=str, default=".", help="StaMPS project folder.")
    parser.add_argument(
        "--resolution",
        "-r",
        nargs=2,
        metavar=("mN", "mE"),
        dest="resolution",
        type=int,
        default=(500, 500),
        help="pixel size of the output grid in North and East (meter)."
        "Default is 500 m by 500 m.",
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
        help="import the variance from mv2.mat, which is added to Kite's"
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

    scene = stamps2kite(
        dirname=args.folder,
        px_size=args.resolution,
        convert_m=not args.keep_mm,
        import_var=args.import_var,
    )

    if fn_save:
        fn_save.rstrip(".yml")
        fn_save.rstrip(".npz")

        log.info("Saving StaMPS scene to file %s[.yml/.npz]...", fn_save)
        scene.save(args.save)

    else:
        scene.spool()


if __name__ == "__main__":
    main()
