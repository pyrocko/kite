#!/bin/python
import numpy as num
import scipy as sp

C = 299792458  # m/s


def squareMatrix(mat):
    if mat.shape[0] == mat.shape[1]:
        return mat
    min_a = num.argmin(mat.shape)
    max_a = num.argmax(mat.shape)

    width = mat.shape[max_a] - mat.shape[min_a]

    if min_a == 0:
        padding = ((width, 0), (0, 0))
    elif min_a == 1:
        padding = ((0, 0), (0, width))
    return num.pad(mat, pad_width=padding, mode="constant", constant_values=0.0)


def derampMatrix(displ):
    """Deramp through fitting a bilinear plane
    Data is also de-meaned
    """
    if displ.ndim != 2:
        raise TypeError("Displacement has to be 2-dim array")
    mx = num.nanmedian(displ, axis=0)
    my = num.nanmedian(displ, axis=1)

    ix = num.arange(mx.size)
    iy = num.arange(my.size)
    dx, cx, _, _, _ = sp.stats.linregress(ix[~num.isnan(mx)], mx[~num.isnan(mx)])
    dy, cy, _, _, _ = sp.stats.linregress(iy[~num.isnan(my)], my[~num.isnan(my)])

    rx = ix * dx
    ry = iy * dy
    data = displ - (rx[num.newaxis, :] + ry[:, num.newaxis])
    data -= num.nanmean(data)
    return data


def derampGMatrix(displ):
    """Deramp through lsq a bilinear plane
    Data is also de-meaned
    """
    if displ.ndim != 2:
        raise TypeError("Displacement has to be 2-dim array")

    # form a relative coordinate grid
    c_grid = num.mgrid[0 : displ.shape[0], 0 : displ.shape[1]]

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
    ramp_f = num.multiply(displ_f, 0.0)

    # insert ramp values in full vectors
    num.place(ramp_f, num.isfinite(displ_f), num.array(ramp_nonan).flatten())
    ramp_f = ramp_f.reshape(displ.shape[0], displ.shape[1])

    return displ - ramp_f


def trimMatrix(displ, data=None):
    """Trim displacement matrix from all NaN rows and columns"""
    if displ.ndim != 2:
        raise ValueError("Displacement has to be 2-dim array")

    if num.all(num.isnan(displ)):
        raise ValueError("Displacement is all NaN.")

    r1 = r2 = False
    c1 = c2 = False
    for r in range(displ.shape[0]):
        if not num.all(num.isnan(displ[r, :])):
            if r1 is False:
                r1 = r
            r2 = r
    for c in range(displ.shape[1]):
        if not num.all(num.isnan(displ[:, c])):
            if c1 is False:
                c1 = c
            c2 = c

    if data is not None:
        return data[r1 : (r2 + 1), c1 : (c2 + 1)]

    return displ[r1 : (r2 + 1), c1 : (c2 + 1)]


def greatCircleDistance(alat, alon, blat, blon):
    R1 = 6371009.0
    d2r = num.deg2rad
    sin = num.sin
    cos = num.cos
    a = (
        sin(d2r(alat - blat) / 2) ** 2
        + cos(d2r(alat)) * cos(d2r(blat)) * sin(d2r(alon - blon) / 2) ** 2
    )
    c = 2.0 * num.arctan2(num.sqrt(a), num.sqrt(1.0 - a))
    return R1 * c


def property_cached(func):
    var_name = "_cached_" + func.__name__
    func_doc = ":getter: *(Cached)*"
    if func.__doc__ is not None:
        func_doc += func.__doc__
    else:
        func_doc += " Undocumented"

    def cache_return(instance, *args, **kwargs):
        cache_return.__doc__ = func.__doc__
        if instance.__dict__.get(var_name, None) is None:
            instance.__dict__[var_name] = func(instance)
        return instance.__dict__[var_name]

    def cache_return_setter(instance, value):
        instance.__dict__[var_name] = value

    return property(fget=cache_return, fset=cache_return_setter, doc=func_doc)


def calcPrecision(data):
    # number of floating points:
    mn = num.nanmin(data)
    mx = num.nanmax(data)
    if not num.isfinite(mx) or num.isfinite(mn):
        return 3, 6
    precision = int(round(num.log10((100.0 / (mx - mn)))))
    if precision < 0:
        precision = 0
    # length of the number in the label:
    length = max(len(str(int(mn))), len(str(int(mx)))) + precision
    return precision, length


def formatScalar(v, ndigits=7):
    if num.isinf(v):
        return "inf"
    elif num.isnan(v):
        return "nan"

    if v % 1 == 0.0:
        return "{value:d}".format(value=v)

    if abs(v) < (10.0 ** -(ndigits - 2)):
        return "{value:e}".format(value=v)

    p = num.ceil(num.log10(num.abs(v)))
    if p <= 0.0:
        f = {"d": 1, "f": ndigits - 1}
    else:
        p = int(p)
        f = {"d": p, "f": ndigits - p}

    return "{value:{d}.{f}f}".format(value=v, **f)


class Subject(object):
    """
    Subject - Obsever model realization
    """

    def __init__(self):
        self._listeners = list()
        self._mute = False

    def __call__(self, *args, **kwargs):
        return self.notify(*args, **kwargs)

    def mute(self):
        self._mute = True

    def unmute(self):
        self._mute = False

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
        except Exception:
            raise AttributeError("%s was not subscribed!", listener.__name__)

    def unsubscribeAll(self):
        for l in self._listeners:
            self.unsubscribe(l)

    def notify(self, *args, **kwargs):
        if self._mute:
            return
        for l in self._listeners:
            if callable(l):
                self._call(l, *args, **kwargs)

    @staticmethod
    def _call(func, *args, **kwargs):
        try:
            for k in kwargs.keys():
                if k not in func.__code__.co_varnames:
                    k.pop(k)
        except AttributeError:
            kwargs = {}
        func(*args, **kwargs)


class ADict(dict):
    def __getattribute__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


__all__ = """
Subject
""".split()
