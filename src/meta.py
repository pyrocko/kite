#!/bin/python


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
            l()

__all__ = '''
Subject
'''.split()
