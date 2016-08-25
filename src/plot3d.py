from vtk import vtkPolyData, vtkDelaunay3D
from vtk.util import numpy_support
import numpy as num


class vtkSceneDisplacement(vtkPolyData):
    pass


class vtkSceneDEM(vtkPolyData):
    pass


class _vtkSceneDelaunay3D(vtkDelaunay3D):
    pass


class vtkSceneDisplacementDelaunay(_vtkSceneDelaunay3D):
    pass


class vtkSceneDEMDelaunay(_vtkSceneDelaunay3D):
    pass


class Plot3D(object):
    def __init__(self, scene):
        self.scene = scene

    def setScene(self, scene):
        self.scene = scene
