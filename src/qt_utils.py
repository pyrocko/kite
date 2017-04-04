#!/usr/bin/python2
from __future__ import (print_function, division, unicode_literals,
                        absolute_import)

import os
import logging
import numpy as num

from os import path as op
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.parametertree.parameterTypes import WidgetParameterItem
from PySide.QtCore import QMetaObject
from PySide.QtUiTools import QUiLoader


SCRIPT_DIRECTORY = op.dirname(op.abspath(__file__))

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


def validateFilename(filename):
    filedir = op.dirname(filename)
    if filename == '' or filedir == '':
        return False
    if op.isdir(filename) or not os.access(filedir, os.W_OK):
        QtGui.QMessageBox.critical(None, 'Path Error',
                                   'Could not access file <b>%s</b>'
                                   % filename)
        return False
    return True


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


class SliderWidget(QtGui.QWidget):
    '''
    shows a horizontal/vertical slider with a label showing its value
    '''
    sigValueChanged = QtCore.Signal(object)  # value

    def __init__(self, horizontal=True, parent=None):
        '''
        horizontal -> True/False
        '''
        QtGui.QWidget.__init__(self, parent)
        self.mn, self.mx = None, None
        self.precission = 0
        self.step = 100
        self.valueLen = 2
        self.suffix = None

        self.label = QtGui.QLabel()
        self.label.setFont(QtGui.QFont('Courier'))
        self.slider = QtGui.QSlider(QtCore.Qt.Orientation(
                        1 if horizontal else 0), self)  # 1...horizontal
        self.slider.setTickPosition(
            QtGui.QSlider.TicksAbove if horizontal
            else QtGui.QSlider.TicksLeft)
        # self.slider.setRange (0, 100)
        self.slider.sliderMoved.connect(self._updateLabel)
        self._updateLabel(self.slider.value())

        layout = QtGui.QHBoxLayout() if horizontal else QtGui.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.slider)
        layout.addWidget(self.label)

    def value(self):
        return self._value

    def setValue(self, val):
        if val is None:
            val = self.mn
        if self.mn is not None:
            val = (val-self.mn) / (self.mx-self.mn)
            val *= 99.0
            val = int(round(val))
        self.slider.setValue(val)
        self._updateLabel(val)

    def setRange(self, mn, mx):
        '''
        mn, mx -> arbitrary values that are not equal
        '''
        if mn == mx:
            raise ValueError('limits must be different values')
        self.mn = float(min(mn, mx))
        self.mx = float(max(mn, mx))
        self._calcPrecission()
        self._updateLabel(self.slider.value())

    def _calcPrecission(self):
        # number of floating points:
        self.precission = int(round(num.log10(
                (self.step / (self.mx-self.mn)))))
        if self.precission < 0:
            self.precission = 0
        # length of the number in the label:
        self.valueLen = max(len(str(int(self.mn))), len(str(int(self.mx))))\
            + self.precission

    def setOpts(self, bounds=None):
        if bounds is not None:
            self.setRange(*bounds)

    def setSuffix(self, suffix=None):
        self.suffix = suffix

    def _updateLabel(self, val):
        if self.mn is not None:
            val /= 99.0  # val->0...1
            val = val * (self.mx-self.mn) + self.mn

        self._value = round(val, self.precission)
        self.sigValueChanged.emit(self._value)

        text = format(self._value, '%s.%sf'
                      % (self.valueLen, self.precission))
        if self.suffix is not None:
            text += ' %s' % self.suffix
        self.label.setText(text)


class SliderWidgetParameterItem(WidgetParameterItem):
    ''' Enabling Slider widget for Parameter
    '''
    def makeWidget(self):
        opts = self.param.opts
        w = SliderWidget()
        w.sigChanged = w.sigValueChanged
        w.sigChanging = w.sigValueChanged
        l = opts.get('limits')
        if l:
            w.setRange(*l)
        v = opts.get('value')
        if l:
            w.setValue(v)
        w.setSuffix(opts.get('suffix', None))
        self.hideWidget = False
        return w


class SceneLogModel(QtCore.QAbstractTableModel, logging.Handler):

    def __init__(self, model, *args, **kwargs):
        QtCore.QAbstractTableModel.__init__(self, *args, **kwargs)
        logging.Handler.__init__(self)

        self.log_records = []
        self.app = None
        self.model = model
        self.model.sigLogRecord.connect(self.newRecord)

    def data(self, idx, role):
        rec = self.log_records[idx.row()]

        if role == QtCore.Qt.DisplayRole:
            if idx.column() == 0:
                return int(rec.levelno)
            elif idx.column() == 1:
                return '%s:%s' % (rec.levelname, rec.name)
            elif idx.column() == 2:
                return rec.getMessage()

        elif role == QtCore.Qt.ItemDataRole:
            return rec

        elif role == QtCore.Qt.ToolTipRole:
            if idx.column() == 0:
                return rec.levelname
            elif idx.column() == 1:
                return '%s.%s' % (rec.module, rec.funcName)
            elif idx.column() == 2:
                return 'Line %d' % rec.lineno

    def rowCount(self, idx):
        return len(self.log_records)

    def columnCount(self, idx):
        return 3

    @QtCore.Slot(object)
    def newRecord(self, record):
        self.beginInsertRows(QtCore.QModelIndex(), 0, 0)
        self.log_records.append(record)
        self.endInsertRows()


class SceneLog(QtGui.QDialog):

    levels = {
        50: 'Critical',
        40: 'Error',
        30: 'Warning',
        20: 'Info',
        10: 'Debug',
    }

    class LogEntryDelegate(QtGui.QStyledItemDelegate):

        def paint(self, painter, option, idx):
            pass

    class LogFilter(QtGui.QSortFilterProxyModel):
        def __init__(self, *args, **kwargs):
            QtGui.QSortFilterProxyModel.__init__(self, *args, **kwargs)
            self.level = 0

        def setLevel(self, level):
            self.level = level
            self.setFilterRegExp('%s' % self.level)

    def __init__(self, app=None):
        QtGui.QDialog.__init__(self, app)

        logging_ui = op.join(
            op.dirname(op.realpath(__file__)),
            'spool', 'res', 'logging.ui')
        loadUi(logging_ui, baseinstance=self)

        self.closeButton.setIcon(self.style().standardPixmap(
                                 QtGui.QStyle.SP_DialogCloseButton))

        self.table_filter = self.LogFilter()
        self.table_filter.setFilterKeyColumn(0)
        self.table_filter.setDynamicSortFilter(True)
        self.table_filter.setSourceModel(app.model.log)

        self.table_filter.rowsInserted.connect(self.popupOnWarning)

        self.tableView.setModel(self.table_filter)

        self.tableView.setColumnWidth(0, 30)
        self.tableView.setColumnWidth(1, 200)

        self.filterBox.addItems(
            [l for l in self.levels.values()] + ['All'])
        self.filterBox.setCurrentIndex(0)

        def changeFilter():
            for lvl, lvl_name in self.levels.iteritems():
                if lvl_name == self.filterBox.currentText():
                    self.table_filter.setLevel(lvl)
                    return

            self.table_filter.setLevel(0)

        self.filterBox.currentIndexChanged.connect(changeFilter)

    def popupOnWarning(self, idx, first, last):
        record = self.table_filter.sourceModel().log_records[-1]
        if record.levelno >= 30 and self.autoBox.isChecked():
            self.show()
