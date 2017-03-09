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
        argument_default=ap.SUPPRESS,
        conflict_handler='resolve',
        add_help=True)
    parser.add_argument('file', type=str,
                        help='Load Kite native container (*.npz/*.yml)',
                        default=None, nargs='?')
    parser.add_argument('--load', metavar='file', type=str,
                        default=None,
                        help='''Import file or directory
Supported formats are:
 * Matlab (*.mat)
 * GAMMA  (*.* and *.par in same directory)
 * GMTSAR (*.grd and binary *.los.* file)
 * ISCE   (*.unw.geo, *.unw.geo.xml and *.rdr.geo for LOS data)''')
    parser.add_argument('--synthetic', type=str, default=None,
                        choices=['fractal', 'sine', 'gauss'],
                        help='''Synthetic Tests
Available Synthetic Displacement:
 * fractal (Atmospheric model, after Hanssen, 2001)
 * sine
 * gauss
''')

    ns = parser.parse_args(args)
    if ns.load is None and ns.synthetic is None and ns.file is None:
        parser.print_help()
        sys.exit(0)

    sc = None
    if ns.synthetic is not None:
        from kite import SceneTest
        if ns.synthetic == 'fractal':
            sc = SceneTest.createFractal()
        elif ns.synthetic == 'sine':
            sc = SceneTest.createSine()
        elif ns.synthetic == 'gauss':
            sc = SceneTest.createFractal()
        else:
            parser.print_help()
            sys.exit(0)

    if sc:
        Spool(scene=sc)
    elif ns.load is not None:
        Spool(import_data=ns.load)
    elif ns.file is not None:
        Spool(load_file=ns.file)
    sys.exit(0)
