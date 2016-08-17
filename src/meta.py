#!/bin/python


class Subject(object):
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
