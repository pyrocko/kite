from kite import SandboxScene
from kite.sources import EllipsoidSource

km = 1e3

sandbox = SandboxScene()

ellipsoid_source = EllipsoidSource(
    northing=40 * km,
    easting=40 * km,
    depth=4 * km,
    length_x=100,
    length_y=200,
    length_z=350,
    roation_x=42.0,
)
sandbox.addSource(ellipsoid_source)

sandbox.save("/tmp/sandbox_scene.yml")
playground = SandboxScene.load("/tmp/sandbox_scene.yml")
