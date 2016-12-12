#!/bin/python
import sys
import argparse as ap
from kite import Scene


def main(args=None):
    '''
    Spool app deployed through setuptools
    '''
    usage = 'Spool is part of the kite InSAR framework'
    desc = 'Quickly inspect deformation data, quadtree and covariance'
    parser = ap.ArgumentParser(
        prog='spool',
        usage=usage,
        description=desc,
        epilog='http://gitext.gfz-potsdam.de - misken@geophysik.uni-kiel.de',
        version='0.0.1',
        parents=[],
        formatter_class=ap.RawTextHelpFormatter,
        prefix_chars='-',
        fromfile_prefix_chars=None,
        argument_default=None,
        conflict_handler='resolve',
        add_help=True)
    parser.add_argument('file', nargs=1, type=str,
                        help='''Import filename or directory
Supported formats are:
 * Matlab (*.mat)
 * GAMMA  (*.* and *.par in same directory)
 * GMTSAR (*.grd and binary *.los.* file)
 * ISCE   (*.unw.geo, *.unw.geo.xml and *.rdr.geo for LOS data''',
                        default=None)
    parser.add_argument('--log-lvl', type=str, help='Debug level (CRITICAL,'
                        'ERROR, WARNING, INFO, DEBUG)',
                        default='INFO')
    ns = parser.parse_args(args)
    if ns.file[0] is None:
        parser.print_help()
        sys.exit(0)
    sc = Scene()
    sc._log_stream.setLevel(ns.log_lvl)
    sc.import_data(ns.file[0])
    sc.spool()
