#!/usr/bin/python2
import argparse as ap
import logging
import sys

from kite.talpa.talpa import talpa


def main(args=None):
    """
    Talpa app deployed through setuptools
    """
    if args is None:
        args = sys.argv[1:]

    epilog = """Talpa is part of the kite InSAR framework.
More at http://pyrocko.org, http://github.com/pyrocko

BriDGes DFG Project, University of Kiel

 Marius Isken (marius.isken@gfz-potsdam.de)
 Henriette Sudhaus"""
    desc = "Crust deformation modeling"

    parser = ap.ArgumentParser(
        prog="talpa",
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
        help="Load SandboxScene from file (.yml)",
        default=None,
        nargs="?",
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

    logging.basicConfig()
    stream_handler = logging.root.handlers[0]
    stream_handler.setLevel(level=log_level if log_level > 0 else 0)

    talpa(filename=ns.file)


if __name__ == "__main__":
    main()
