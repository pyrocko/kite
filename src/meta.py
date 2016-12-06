#!/bin/python
import numpy as num
import scipy as sp


def derampMatrix(displ):
    """ Deramp through fitting a bilinear plane
    Data is also de-meaned
    """
    if displ.ndim != 2:
        raise TypeError('Displacement has to be 2-dim array')
    mx = num.nanmedian(displ, axis=0)
    my = num.nanmedian(displ, axis=1)
    mx[~num.isnan(mx)]
    my[~num.isnan(my)]

    ix = num.arange(mx.size)[~num.isnan(mx)]
    iy = num.arange(my.size)[~num.isnan(my)]
    dx, cx, _, _, _ = sp.stats.linregress(ix, mx)
    dy, cy, _, _, _ = sp.stats.linregress(iy, my)

    rx = (ix * dx)
    ry = (iy * dy)
    data = displ - (rx[num.newaxis, :] + ry[:, num.newaxis])
    data -= num.nanmean(data)
    return data


def derampMatrix2(displ):
    """ Deramp through fitting a bilinear plane
    using scipy.stats.linregress
    """
    if displ.ndim != 2:
        raise TypeError('Displacement has to be 2-dim array')

    c_grid = num.mgrid[0:displ.shape[0], 0:displ.shape[1]]

    # separate and flatten coordinate grid into x and y vectors for each !point
    ix = c_grid[0].flatten()
    iy = c_grid[1].flatten()
    displ_f = displ.flatten()

    # reduce vectors taking out all NaN's
    displ_nonan = displ_f[num.isfinite(displ_f)]
    ix = ix[num.isfinite(displ_f)]
    iy = iy[num.isfinite(displ_f)]

    dx, cx, _, _, _ = sp.stats.linregress(ix, displ_nonan)
    dy, cy, _, _, _ = sp.stats.linregress(iy, displ_nonan)

    rx = (ix * dx + cx)
    ry = (iy * dy + cy)

    ramp_x = num.multiply(displ_f, 0.)
    ramp_y = num.multiply(displ_f, 0.)
    # insert ramp values in full vectors
    num.place(ramp_x, num.isfinite(displ_f), num.array(rx).flatten())
    num.place(ramp_y, num.isfinite(displ_f), num.array(ry).flatten())
    ramp_x = ramp_x.reshape(*displ.shape)
    ramp_y = ramp_y.reshape(*displ.shape)

    return displ - ramp_x - ramp_y


def derampGMatrix(displ):
    """ Deramp through lsq a bilinear plane
    Data is also de-meaned
    """
    if displ.ndim != 2:
        raise TypeError('Displacement has to be 2-dim array')

    # form a relative coordinate grid
    c_grid = num.mgrid[0:displ.shape[0], 0:displ.shape[1]]

    # separate and flatten coordinate grid into x and y vectors for each !point
    ix = c_grid[0].flat
    iy = c_grid[1].flat
    displ_f = displ.flat

    # reduce vectors taking out all NaN's
    displ_nonan = displ_f[num.isfinite(displ_f)]
    ix = ix[num.isfinite(displ_f)]
    iy = iy[num.isfinite(displ_f)]

    # form kernel/design derampMatrix (c, x, y)
    GT = num.matrix([num.ones(len(ix)), ix, iy])
    G = GT.T

    # generalized kernel matrix (quadtratic)
    GTG = GT * G
    # generalized inverse
    GTGinv = GTG.I

    # lsq estimates of ramp parameter
    ramp_paras = displ_nonan * (GTGinv * GT).T

    # ramp values
    ramp_nonan = ramp_paras * GT
    ramp_f = num.multiply(displ_f, 0.)

    # insert ramp values in full vectors
    num.place(ramp_f, num.isfinite(displ_f), num.array(ramp_nonan).flatten())
    ramp_f = ramp_f.reshape(displ.shape[0], displ.shape[1])

    return displ - ramp_f


def trimMatrix(displ):
    """Trim displacement matrix from all NaN rows and columns
    """
    if displ.ndim != 2:
        raise TypeError('Displacement has to be 2-dim array')
    r1 = r2 = False
    c1 = c2 = False
    for r in xrange(displ.shape[0]):
        if not num.all(num.isnan(displ[r, :])):
            if not r1:
                r1 = r
            else:
                r2 = r
    for c in xrange(displ.shape[1]):
        if not num.all(num.isnan(displ[:, c])):
            if not c1:
                c1 = c
            else:
                c2 = c
    return displ[r1:r2, c1:c2]


def property_cached(func):
    var_name = '_cached_' + func.__name__
    func_doc = "**Property:** "
    if func.__doc__ is not None:
        func_doc += func.__doc__
    else:
        func_doc += "Undocumented"

    def cache_return(instance, *args, **kwargs):
        cache_return.__doc__ = func.__doc__
        if instance.__dict__.get(var_name, None) is None:
            instance.__dict__[var_name] = func(instance)
        return instance.__dict__[var_name]

    def cache_return_setter(instance, value):
        instance.__dict__[var_name] = value

    return property(fget=cache_return,
                    fset=cache_return_setter,
                    doc=func_doc)


class Subject(object):
    """
    Subject - Obsever model realization
    """
    def __init__(self):
        self._listeners = list()

    def subscribe(self, listener):
        """
        Subscribe a listening callback to this subject
        """
        if len(self._listeners) > 0:
            self._listeners.insert(0, listener)
        else:
            self._listeners.append(listener)

    def unsubscribe(self, listener):
        """
        Unsubscribe a listening callback from this subject
        """
        try:
            self._listeners.remove(listener)
        except:
            raise AttributeError('%s was not subscribed to ')

    def notify(self, msg=''):
        for l in self._listeners:
            if 'msg' in l.__code__.co_varnames:
                l(msg)
            else:
                l()


__all__ = '''
Subject
'''.split()
