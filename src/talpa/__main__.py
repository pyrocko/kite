#!/usr/bin/python2
import sys  # noqa
import argparse as ap
from kite.talpa import Talpa


def main(args=None):
    '''
    Talpa app deployed through setuptools
    '''
    epilog = '''Talpa is part of the kite InSAR framework.
More at http://pyrocko.org, http://github.com/pyrocko

BriDGes DFG Project, University of Kiel

 Marius Isken (marius.isken@gfz-potsdam.de)
 Henriette Sudhaus'''
    desc = 'Crust deformation modeling'

    parser = ap.ArgumentParser(
        prog='talpa',
        epilog=epilog,
        description=desc,
        version='0.1',
        parents=[],
        formatter_class=ap.RawTextHelpFormatter,
        prefix_chars='-',
        fromfile_prefix_chars=None,
        argument_default=ap.SUPPRESS,
        conflict_handler='resolve',
        add_help=True)

    ns = parser.parse_args(args)
    ns
    Talpa()
