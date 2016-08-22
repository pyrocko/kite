#!/bin/python


def property_cached(func):
    var_name = '_cached_' + func.__name__

    @property
    def cache_return(instance, *args, **kwargs):
        if instance.__dict__.get(var_name, None) is None:
            instance.__dict__[var_name] = func(instance)
        return instance.__dict__[var_name]

    @cache_return.setter
    def cache_return(instance, value):
        instance.__dict__[var_name] = value

    return cache_return


class Subject(object):
    '''Subject - Obsever model realization '''
    def __init__(self):
        self._listeners = list()

    def subscribe(self, listener):
        self._listeners.append(listener)

    def unsubscribe(self, listener):
        self._listeners.remove(listener)

    def _notify(self, msg=''):
        for l in self._listeners:
            l()

__all__ = '''
Subject
'''.split()
