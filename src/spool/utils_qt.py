#!/usr/bin/python2
from __future__ import (print_function, division, unicode_literals,
                        absolute_import)
from PySide import QtGui

import os

from PySide.QtCore import QMetaObject
from PySide.QtUiTools import QUiLoader


# -*- coding: utf-8 -*-
# Copyright (c) 2011 Sebastian Wiesner <lunaryorn@gmail.com>
# Modifications by Charl Botha <cpbotha@vxlabs.com>
# * customWidgets support (registerCustomWidget() causes segfault in
#   pyside 1.1.2 on Ubuntu 12.04 x86_64)
# * workingDirectory support in loadUi

# found this here:
# https://github.com/lunaryorn/snippets/blob/master/qt4/designer/pyside_dynamic.py

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
    How to load a user interface dynamically with PySide.
    .. moduleauthor::  Sebastian Wiesner  <lunaryorn@gmail.com>
"""

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

_viridis_data = [[68, 1, 84],
                 [69, 6, 90],
                 [70, 12, 95],
                 [71, 18, 101],
                 [71, 24, 106],
                 [72, 29, 111],
                 [72, 34, 115],
                 [71, 39, 119],
                 [71, 44, 123],
                 [70, 49, 126],
                 [69, 54, 129],
                 [67, 59, 131],
                 [66, 64, 133],
                 [64, 68, 135],
                 [62, 73, 137],
                 [60, 77, 138],
                 [58, 82, 139],
                 [56, 86, 139],
                 [54, 90, 140],
                 [52, 94, 141],
                 [50, 98, 141],
                 [49, 102, 141],
                 [47, 106, 141],
                 [45, 110, 142],
                 [44, 114, 142],
                 [42, 118, 142],
                 [40, 122, 142],
                 [39, 125, 142],
                 [38, 129, 142],
                 [36, 133, 141],
                 [35, 137, 141],
                 [33, 140, 141],
                 [32, 144, 140],
                 [31, 148, 139],
                 [30, 152, 138],
                 [30, 155, 137],
                 [30, 159, 136],
                 [31, 163, 134],
                 [33, 167, 132],
                 [36, 170, 130],
                 [40, 174, 127],
                 [44, 177, 125],
                 [50, 181, 122],
                 [56, 185, 118],
                 [62, 188, 115],
                 [69, 191, 111],
                 [77, 194, 107],
                 [85, 198, 102],
                 [94, 201, 97],
                 [103, 204, 92],
                 [112, 206, 86],
                 [121, 209, 81],
                 [131, 211, 75],
                 [141, 214, 68],
                 [151, 216, 62],
                 [162, 218, 55],
                 [173, 220, 48],
                 [183, 221, 41],
                 [194, 223, 34],
                 [205, 224, 29],
                 [215, 226, 25],
                 [225, 227, 24],
                 [236, 228, 26],
                 [246, 230, 31]]
_viridis_data.reverse()


class UiLoader(QUiLoader):
    """
    Subclass :class:`~PySide.QtUiTools.QUiLoader` to create the user interface
    in a base instance.
    Unlike :class:`~PySide.QtUiTools.QUiLoader` itself this class does not
    create a new instance of the top-level widget, but creates the user
    interface in an existing instance of the top-level class.
    This mimics the behaviour of :func:`PyQt4.uic.loadUi`.
    """

    def __init__(self, baseinstance, customWidgets=None):
        """
        Create a loader for the given ``baseinstance``.
        The user interface is created in ``baseinstance``, which must be an
        instance of the top-level class in the user interface to load, or a
        subclass thereof.
        ``customWidgets`` is a dictionary mapping from class name to class
        object for widgets that you've promoted in the Qt Designer interface.
        Usually, this should be done by calling registerCustomWidget on the
        QUiLoader, but with PySide 1.1.2 on Ubuntu 12.04 x86_64 this causes a
        segfault.
        ``parent`` is the parent object of this loader.
        """

        QUiLoader.__init__(self, baseinstance)
        self.baseinstance = baseinstance
        self.customWidgets = customWidgets

    def createWidget(self, class_name, parent=None, name=''):
        """
        Function that is called for each widget defined in ui file,
        overridden here to populate baseinstance instead.
        """

        if parent is None and self.baseinstance:
            # supposed to create the top-level widget, return the base instance
            # instead
            return self.baseinstance

        else:
            if class_name in self.availableWidgets():
                # create a new widget for child widgets
                widget = QUiLoader.createWidget(self, class_name, parent, name)

            else:
                # if not in the list of availableWidgets, must be a custom
                # widget this will raise KeyError if the user has not supplied
                # the relevant class_name in the dictionary, or TypeError, if
                # customWidgets is None
                try:
                    widget = self.customWidgets[class_name](parent)

                except (TypeError, KeyError):
                    raise Exception('No custom widget ' + class_name +
                                    'found in customWidgets param ' +
                                    'of UiLoader __init__.')

            if self.baseinstance:
                # set an attribute for the new child widget on the base
                # instance, just like PyQt4.uic.loadUi does.
                setattr(self.baseinstance, name, widget)

                # this outputs the various widget names, e.g.
                # sampleGraphicsView, dockWidget, samplesTableView etc.

            return widget


def loadUi(uifile, baseinstance=None, customWidgets=None,
           workingDirectory=None):
    """
    Dynamically load a user interface from the given ``uifile``.
    ``uifile`` is a string containing a file name of the UI file to load.
    If ``baseinstance`` is ``None``, the a new instance of the top-level widget
    will be created.  Otherwise, the user interface is created within the given
    ``baseinstance``.  In this case ``baseinstance`` must be an instance of the
    top-level widget class in the UI file to load, or a subclass thereof.  In
    other words, if you've created a ``QMainWindow`` interface in the designer,
    ``baseinstance`` must be a ``QMainWindow`` or a subclass thereof, too.  You
    cannot load a ``QMainWindow`` UI file with a plain
    :class:`~PySide.QtGui.QWidget` as ``baseinstance``.
    ``customWidgets`` is a dictionary mapping from class name to class object
    for widgets that you've promoted in the Qt Designer interface. Usually,
    this should be done by calling registerCustomWidget on the QUiLoader, but
    with PySide 1.1.2 on Ubuntu 12.04 x86_64 this causes a segfault.
    :method:`~PySide.QtCore.QMetaObject.connectSlotsByName()` is called on the
    created user interface, so you can implemented your slots according to its
    conventions in your widget class.
    Return ``baseinstance``, if ``baseinstance`` is not ``None``.  Otherwise
    return the newly created instance of the user interface.
    """

    loader = UiLoader(baseinstance, customWidgets)

    if workingDirectory is not None:
        loader.setWorkingDirectory(workingDirectory)

    widget = loader.load(uifile)
    QMetaObject.connectSlotsByName(widget)
    return widget


class QDoubleSlider(QtGui.QSlider):
    ''' DoublePrecision slider for Qt
    '''
    def __init__(self, *args, **kwargs):
        QtGui.QSlider.__init__(self, *args, **kwargs)

        super(QDoubleSlider, self).setMinimum(0)
        self._max_int = 10000
        super(QDoubleSlider, self).setMaximum(self._max_int)

        self._min_value = 0.0
        self._max_value = 100.0

    @property
    def _value_range(self):
        return self._max_value - self._min_value

    def value(self):
        return float(super(QDoubleSlider, self).value()) \
            / self._max_int * self._value_range

    def setValue(self, value):
        super(QDoubleSlider, self).setValue(int(value /
                                            self._value_range * self._max_int))

    def setMinimum(self, value):
        self.setRange(value, self._max_value)

    def setMaximum(self, value):
        self.setRange(self._min_value, value)

    def setRange(self, minimum, maximum):
        old_value = self.value()
        self._min_value = minimum
        self._max_value = maximum
        self.setValue(old_value)

    def proportion(self):
        return (self.value() - self._min_value) / self._value_range


class QSceneLogger(QtGui.QWidget):
    pass
