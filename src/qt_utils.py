import os
import logging
from os import path as op

from PyQt5 import QtGui, QtCore, uic
from pyqtgraph.parametertree.parameterTypes import WidgetParameterItem
import pyqtgraph.parametertree.parameterTypes as pTypes


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
    if not filename:
        return False
    filename = op.abspath(filename)
    filedir = op.dirname(filename)
    if filename == '' or filedir == '':
        return False
    if op.isdir(filename) or not os.access(filedir, os.W_OK):
        QtGui.QMessageBox.critical(None, 'Path Error',
                                   'Could not access file <b>%s</b>'
                                   % filename)
        return False
    return True


def loadUi(uifile, baseinstance=None):
    return uic.loadUi(uifile, baseinstance)


class SliderWidget(QtGui.QWidget):
    """
    shows a horizontal/vertical slider with a label showing its value
    """
    sigValueChanged = QtCore.Signal(object)  # value

    def __init__(self, horizontal=True, parent=None, decimals=3, step=.005,
                 slider_exponent=1):
        QtGui.QWidget.__init__(self, parent)
        self.vmin = None
        self.vmax = None
        self.slider_exponent = slider_exponent
        self.decimals = decimals
        self.step = 100
        self.valueLen = 2
        self.suffix = None
        self._value = None

        self.spin = QtGui.QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.valueChanged.connect(self._spin_updated)
        self.spin.setFrame(False)

        self.slider = QtGui.QSlider(
            QtCore.Qt.Orientation(1 if horizontal else 0), self)  # 1 = hor.
        self.slider.setTickPosition(
            QtGui.QSlider.TicksAbove if horizontal
            else QtGui.QSlider.TicksLeft)
        self.slider.setRange(0, 99)
        self.slider.sliderMoved.connect(self._slider_updated)

        layout = QtGui.QHBoxLayout() if horizontal else QtGui.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.slider)
        layout.addWidget(self.spin)

    def value(self):
        return self._value

    def setValue(self, val):
        self.setSliderValue(val)
        self.spin.setValue(val)
        self._value = val

    def setSliderValue(self, val):
        if val is None:
            val = self.vmin

        elif self.vmin is not None and self.vmax is not None:
            if val <= self.vmin:
                val = self.vmin
            val = (val-self.vmin) / (self.vmax-self.vmin)

            val **= 1./self.slider_exponent
            val *= 99
            val = int(round(val))

        self.slider.setValue(val)

    def setRange(self, vmin, vmax):
        """
        vmin, vmax -> arbitrary values that are not equal
        """
        if vmin == vmax:
            raise ValueError('limits must be different values')
        self.vmin = float(min(vmin, vmax))
        self.vmax = float(max(vmin, vmax))

        self.spin.setRange(self.vmin, self.vmax)

    def setOpts(self, bounds=None):
        if bounds is not None:
            self.setRange(*bounds)

    def setSuffix(self, suffix=None):
        self.suffix = suffix
        self.spin.setSuffix(suffix)

    @QtCore.pyqtSlot(int)
    def _slider_updated(self, val):
        val /= 99  # val -> 0...1
        val **= self.slider_exponent

        if self.vmin is not None and self.vmax is not None:
            val = val * (self.vmax-self.vmin) + self.vmin

        val = round(val, self.decimals)

        if self._value != val:
            self._value = val
            self.sigValueChanged.emit(val)

        self.spin.setValue(val)

    @QtCore.pyqtSlot(float)
    def _spin_updated(self, val):
        self.setSliderValue(self._value)

        if self._value != val:
            self._value = val
            self.sigValueChanged.emit(val)


class SliderWidgetParameterItem(WidgetParameterItem):
    """ Enabling Slider widget for Parameter
    """
    def makeWidget(self):
        opts = self.param.opts

        step = opts.get('step', .1)
        decimals = opts.get('decimals', 2)
        slider_exponent = opts.get('slider_exponent', 1)

        w = SliderWidget(
            decimals=decimals, step=step,
            slider_exponent=slider_exponent)

        limits = opts.get('limits', None)
        if limits is not None:
            w.setRange(*limits)

        value = opts.get('value', None)
        if value is not None:
            w.setValue(value)

        suffix = opts.get('suffix', None)
        w.setSuffix(suffix)

        self.hideWidget = False
        w.sigChanged = w.sigValueChanged
        w.sigChanging = w.sigValueChanged

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

    @property
    def nlogs(self):
        return len(self.log_records)

    def rowCount(self, idx):
        return self.nlogs

    def columnCount(self, idx):
        return 3

    @QtCore.pyqtSlot(object)
    def newRecord(self, record):
        self.beginInsertRows(QtCore.QModelIndex(), self.nlogs, self.nlogs)
        self.log_records.append(record)
        self.endInsertRows()


class SceneLog(QtGui.QDialog):

    levels = {
        50: 'Critical',
        40: 'Error',
        30: 'Warning',
        20: 'Info',
        10: 'Debug',
        0: 'All'
    }

    class LogEntryDelegate(QtGui.QStyledItemDelegate):

        levels = {
            50: QtGui.QStyle.SP_MessageBoxCritical,
            40: QtGui.QStyle.SP_MessageBoxCritical,
            30: QtGui.QStyle.SP_MessageBoxWarning,
            20: QtGui.QStyle.SP_MessageBoxInformation,
            10: QtGui.QStyle.SP_FileIcon,
        }

        def paint(self, painter, option, idx):
            # paint icon instead of log_lvl
            if idx.column() == 0:
                getIcon = QtGui.QApplication.style().standardIcon
                icon = getIcon(self.levels[idx.data()])
                icon.paint(painter, option.rect)
            else:
                QtGui.QStyledItemDelegate.paint(self, painter, option, idx)

    class LogFilter(QtCore.QSortFilterProxyModel):
        def __init__(self, *args, **kwargs):
            QtCore.QSortFilterProxyModel.__init__(self, *args, **kwargs)
            self.setLevel(30)

        def setLevel(self, level):
            self.level = level
            self.setFilterRegExp(str(level))
            self.invalidate()

    def __init__(self, app, model):
        QtGui.QDialog.__init__(self, app)
        logging_ui = op.join(
            op.dirname(
                op.realpath(__file__)), 'spool', 'res', 'logging.ui')
        loadUi(logging_ui, baseinstance=self)

        self.move(
            self.parent().window().mapToGlobal(
                self.parent().window().rect().center()) -
            self.mapToGlobal(self.rect().center()))

        self.closeButton.setIcon(
            self.style().standardIcon(QtGui.QStyle.SP_DialogCloseButton))

        self.table_filter = self.LogFilter()
        self.table_filter.setFilterKeyColumn(0)
        self.table_filter.setDynamicSortFilter(True)
        self.table_filter.setSourceModel(model.log)

        model.log.rowsInserted.connect(self.newLogRecord)

        self.tableView.setModel(self.table_filter)
        self.tableView.setItemDelegate(self.LogEntryDelegate())

        self.tableView.setColumnWidth(0, 30)
        self.tableView.setColumnWidth(1, 200)

        self.filterBox.addItems(
            [lvl_name for lvl_name in self.levels.values()])

        def changeFilter():
            for lvl, lvl_name in self.levels.items():
                if lvl_name == self.filterBox.currentText():
                    self.table_filter.setLevel(lvl)
                    break

            self.tableView.update()

        self.filterBox.currentIndexChanged.connect(changeFilter)
        self.filterBox.setCurrentText('Warning')

    @QtCore.pyqtSlot(QtCore.QModelIndex, int, int)
    def newLogRecord(self, idx, first, last):
        self.tableView.scrollToBottom()

        record = self.table_filter.sourceModel().log_records[idx.row()]
        if record.levelno >= 30 and self.autoBox.isChecked():
            self.show()
