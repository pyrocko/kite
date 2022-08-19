#!/bin/python
import argparse as ap
import logging
import sys

from kite import Scene, TestScene
from kite.spool import spool


def main(args=None):
    """
    Spool app deployed through setuptools
    """
    if args is None:
        args = sys.argv[1:]

    epilog = """Spool is part of the kite InSAR framework.

Author: Marius Paul Isken (marius.isken@gfz-potsdam.de)
Documentation: https://pyrocko.org"""
    desc = "InSAR deformation inspector, quadtree and covariance"

    parser = ap.ArgumentParser(
        prog="spool",
        epilog=epilog,
        description=desc,
        parents=[],
        formatter_class=ap.RawTextHelpFormatter,
        prefix_chars="-",
        fromfile_prefix_chars=None,
        argument_default=ap.SUPPRESS,
        conflict_handler="resolve",
        add_help=True,
    )
    parser.add_argument(
        "file",
        type=str,
        help="Load native kite container (.npz & .yml)",
        default=None,
        nargs="?",
    )
    parser.add_argument(
        "--load",
        metavar="file",
        type=str,
        default=None,
        help="""Import file or directory
Supported formats are:
 - Matlab   *.mat
 - GAMMA    * binary and *.par file
 - GMTSAR   *.grd binary and *.los.* file
 - ISCE     *.unw.geo with *.unw.geo.xml and; *.rdr.geo for LOS data
 - ROI_PAC  * binary and *.rsc file
 - SARSCAPE *los_ll.grd and *.los.enu file
 - SNAP     *.rsc and *.Abstracted_Metadata.txt file (GAMMA Export)
 - ARIA     Extracted layers: unwrappedPhase, lookAngle, incidenceAngle,
             connectedComponents
 - LiCSAR   *.unw.tif and LOS data, see client.download_licsar

 For more information see the online documentation at https://pyrocko.org""",
    )
    parser.add_argument(
        "--synthetic",
        type=str,
        default=None,
        choices=["fractal", "sine", "gauss"],
        help="""Synthetic Tests
Available Synthetic Displacement:
 * fractal (Atmospheric model; after Hanssen, 2001)
 * sine
 * gauss
""",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="Verbosity, add multiple to increase verbosity.",
    )

    ns = parser.parse_args(args)

    log_level = logging.WARNING - ns.verbose * 10
    log_level = log_level if log_level > logging.DEBUG else logging.DEBUG
    logging.basicConfig(level=log_level if log_level > 0 else 0)

    if ns.load is None and ns.synthetic is None and ns.file is None:
        parser.print_help()
        sys.exit(0)

    sc = None
    if ns.synthetic is not None:
        if ns.synthetic == "fractal":
            sc = TestScene.createFractal()
        elif ns.synthetic == "sine":
            sc = TestScene.createSine()
        elif ns.synthetic == "gauss":
            sc = TestScene.createFractal()
        else:
            parser.print_help()
            sys.exit(0)

    elif ns.file is not None:
        sc = Scene.load(ns.file)

    if sc:
        spool(scene=sc)
    elif ns.load is not None:
        spool(import_file=ns.load)


if __name__ == "__main__":
    main()
