from kite.scene import Scene
import scipy.io as io
import os

_file = 'data/20110214_20110401_ml4_sm.unw.geo_ig_dsc_ionnocorr.mat'
mat = io.loadmat(os.path.join(os.path.dirname(__file__), _file))

sc = Scene()
sc.meta.title = 'Matlab Input - Myanmar 2011-02-14'

sc.phi = mat['phi_dsc_defo']
sc.theta = mat['theta_dsc_defo']
sc.displacement = mat['ig_dc']
sc.utm_x = mat['xx_ig']
sc.utm_y = mat['yy_ig']

sc.quadtree.plot.interactive()
# qt = Quadtree(sc)
# qp = Plot2DQuadTree(qt, cmap='RdBu')
# qp.plotInteractive()
