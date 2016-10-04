#!/bin/python
import sys
import argparse as ap
from kite import Scene


def main(args=None):
    '''
    Spool app deployed through setuptools
    '''
    usage = 'Spool is part of the Kite InSAR framework'
    desc = 'Quickly inspect deformation data and manipulate its quadtree'
    parser = ap.ArgumentParser(
        prog='spool',
        usage=usage,
        description=desc,
        epilog='http://gitext.gfz-potsdam.de - misken@geophysik.uni-kiel.de',
        version='0.0.1',
        parents=[],
        # formatter_class=HelpFormatter,
        prefix_chars='-',
        fromfile_prefix_chars=None,
        argument_default=None,
        conflict_handler='resolve',
        add_help=True)
    parser.add_argument('file', nargs=1, type=str,
                        help='Import Matlab/Gamma filename', default=None)
    ns = parser.parse_args(args)

    print 'Importing file %s' % ns.file[0]
    sc = Scene.import_file(ns.file[0])
    sc.spool()
