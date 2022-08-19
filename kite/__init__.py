#!/usr/bin/env python3
import pkg_resources

from .covariance import Covariance  # noqa
from .quadtree import Quadtree  # noqa
from .sandbox_scene import SandboxScene, TestSandboxScene  # noqa
from .scene import Scene, TestScene, read  # noqa

__version__ = pkg_resources.get_distribution("kite").version
