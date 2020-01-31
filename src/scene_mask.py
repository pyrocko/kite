import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as num

def click_mask_matplotlib(sc):
    """ Open a matplotlib window to click a closed polygon to mask

    :param sc: scene to mask
    :type sc: kite.Scene
    """

    # Open a figure
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.imshow(sc.displacement, origin='lower')

    #  Click polygon to mask
    lst = plt.ginput(-1)

    #  Make the mask
    p = Path(lst)
    pts = [(c, r) for r in range(sc.frame.rows) for c in range(sc.frame.cols)]
    mask = p.contains_points(pts).reshape(sc.frame.rows, sc.frame.cols)

    return mask 
