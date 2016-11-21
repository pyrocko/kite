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
    cx = num.nanmean(mx)
    cy = num.nanmean(my)
    mx -= cx
    my -= cy
    mx[num.isnan(mx)] = 0.
    my[num.isnan(my)] = 0.

    ix = num.arange(mx.size)
    iy = num.arange(my.size)
    dx, _, _, _, _ = sp.stats.linregress(ix, mx)
    dy, _, _, _, _ = sp.stats.linregress(iy, my)

    rx = (ix * dx + cx)
    ry = (iy * dy + cy)

    return displ - rx[num.newaxis, :] - ry[:, num.newaxis]


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

    def _notify(self, msg=''):
        for l in self._listeners:
            if 'msg' in l.__code__.co_varnames:
                l(msg)
            else:
                l()


__all__ = '''
Subject
'''.split()
