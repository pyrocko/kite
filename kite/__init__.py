#!/usr/bin/env python3
import pkg_resources

from .scene import Scene, TestScene, read  # noqa
from .quadtree import Quadtree  # noqa
from .covariance import Covariance  # noqa
from .sandbox_scene import SandboxScene, TestSandboxScene  # noqa

__version__ = pkg_resources.get_distribution("kite").version
