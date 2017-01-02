#!/bin/python
import sys
import argparse as ap
from kite.spool import Spool


def main(args=None):
    '''
    Spool app deployed through setuptools
    '''
    epilog = 'Spool is part of the kite InSAR framework.'
    epilog += '\nMore at http://pyrocko.org'
    epilog += '\n\n - Marius Isken (marius.isken@gfz-potsdam.de)'
    desc = 'InSAR deformation inspector, quadtree and covariance'
    parser = ap.ArgumentParser(
        prog='spool',
        epilog=epilog,
        description=desc,
        version='0.0.1',
        parents=[],
        formatter_class=ap.RawTextHelpFormatter,
        prefix_chars='-',
        fromfile_prefix_chars=None,
        argument_default=None,
        conflict_handler='resolve',
        add_help=True)
    parser.add_argument('--file', type=str,
                        help='''Import file or directory
Supported formats are:
 * Matlab (*.mat)
 * GAMMA  (*.* and *.par in same directory)
 * GMTSAR (*.grd and binary *.los.* file)
 * ISCE   (*.unw.geo, *.unw.geo.xml and *.rdr.geo for LOS data)''',
                        default=None)
    parser.add_argument('--log-lvl', type=str, help='Debug level (CRITICAL,'
                        'ERROR, WARNING, INFO, DEBUG)',
                        default='INFO')

    parser.add_argument('--syn', type=str, help='''Synthetic Tests
Available Synthetic Displacement are:
 * fractal (Atmospheric model, after Hanssen, 2001)
 * sine
 * gauss
''',
                        default=None)

    ns = parser.parse_args(args)
    if ns.file is None and ns.syn is None:
        parser.print_help()
        sys.exit(0)

    sc = None
    if ns.syn is not None:
        from kite import SceneTest
        if ns.syn == 'fractal':
            sc = SceneTest.createFractal()
        elif ns.syn == 'sine':
            sc = SceneTest.createSine()
        elif ns.syn == 'gauss':
            sc = SceneTest.createFractal()
        else:
            parser.print_help()
            sys.exit(0)
    # sc = Scene()
    # sc.setLogLevel(ns.log_lvl)
    # try:
    #     sc.import_data(ns.file[0])
    # except ImportError:
    #     sc.load(ns.file[0])
    # sc.spool()
    if sc:
        spool = Spool(scene=sc)
    else:
        spool = Spool(filename=ns.file)
    spool.spool_win.buildViews()
