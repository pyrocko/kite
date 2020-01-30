import matplotlib.pyplot as plt
from kite.gacos import GACOSCorrection


def test_gacos():
    corr = GACOSCorrection()
    corr.load('/home/marius/Development/testing/kite/GACOS/20180826.ztd')

    grd = corr.grids[0]

    d = grd.get_corrections(grd.llLat, grd.llLon, -grd.dLat, grd.dLon,
                            grd.rows, grd.cols)

    d = grd.get_corrections(grd.llLat, grd.llLon, -grd.dLat*2, grd.dLon*2,
                            grd.rows//2, grd.cols//2)

    d = grd.get_corrections(grd.llLat+.01, grd.llLon+.01, -grd.dLat/1.5, grd.dLon/1.5,
                            int(grd.rows*1.), int(grd.cols*1.))

    plt.imshow(grd.data)
    plt.show()
    plt.imshow(d)
    plt.show()


if __name__ == '__main__':
    test_gacos()
