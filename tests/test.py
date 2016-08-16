from kite.scene import Scene
from kite.quadtree import Quadtree
from kite.plot2d import Plot2DQuadTree
import scipy.io as io
import os

_file = 'data/20110214_20110401_ml4_sm.unw.geo_ig_dsc_ionnocorr.mat'
mat = io.loadmat(os.path.join(os.path.dirname(__file__), _file))

sc = Scene()
sc.meta.title = 'Matlab Input - Myanmar 2011-02-14'

sc.los.phi = mat['phi_dsc_defo']
sc.los.theta = mat['theta_dsc_defo']
sc.los.displacement = mat['ig_dc']
sc.x = mat['xx_ig']
sc.y = mat['yy_ig']


#sc.los.plot()
a = Plot2DQuadTree(sc.quadtree)
a.plotInteractive()
#qt = Quadtree(sc)
#qp = Plot2DQuadTree(qt, cmap='RdBu')
#qp.plotInteractive()
