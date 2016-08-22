from vtk import vtkPolyData, vtkDelaunay2D
from vtk.util import numpy_support
import numpy as num


class Plot3D(object):

    class vtkSceneDisplacement(vtkPolyData):
        pass

    class vtkSceneDEM(vtkPolyData):
        pass

    class _vtkSceneDelaunay2D(vtkDelaunay2D):
        pass

    class vtkSceneDisplacementDelaunay(_vtkSceneDelaunay2D):
        pass

    class vtkSceneDEMDelaunay(_vtkSceneDelaunay2D):
        pass

    def __init__(self, scene):
        self.scene = scene

    def setScene(self, scene):
        self.scene = scene
