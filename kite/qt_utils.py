import logging
import os
from os import path as op

import pyqtgraph.parametertree.parameterTypes as pTypes
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from pyqtgraph.parametertree.parameterTypes import WidgetParameterItem

SCRIPT_DIRECTORY = op.dirname(op.abspath(__file__))

_viridis_data = [
    [68, 1, 84],
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
    [246, 230, 31],
]
_viridis_data.reverse()


def validateFilename(filename):
    if not filename:
        return False
    filename = op.abspath(filename)
    filedir = op.dirname(filename)
    if filename == "" or filedir == "":
        return False
    if op.isdir(filename) or not os.access(filedir, os.W_OK):
        QtWidgets.QMessageBox.critical(
            None, "Path Error", f"Could not access file <b>{filename}</b>"
        )
        return False
    return True


def loadUi(uifile, baseinstance=None):
    return uic.loadUi(uifile, baseinstance)


class SliderWidget(QtWidgets.QWidget):
    """
    shows a horizontal/vertical slider with a label showing its value
    """

    sigValueChanged = QtCore.pyqtSignal(object)  # value

    def __init__(
        self, horizontal=True, parent=None, decimals=3, step=0.005, slider_exponent=1
    ):
        QtWidgets.QWidget.__init__(self, parent)
        self.vmin = None
        self.vmax = None
        self.slider_exponent = slider_exponent
        self.decimals = decimals
        self.step = 100
        self.valueLen = 2
        self.suffix = None
        self._value = None

        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setFrame(False)
        self.spin.valueChanged.connect(self._spin_updated)

        self.slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation(1 if horizontal else 0), self
        )  # 1 = hor.
        self.slider.setTickPosition(
            QtWidgets.QSlider.TicksAbove if horizontal else QtWidgets.QSlider.TicksLeft
        )
        self.slider.setRange(0, 99)
        self.slider.sliderMoved.connect(self._slider_updated)

        layout = QtWidgets.QHBoxLayout() if horizontal else QtWidgets.QVBoxLayout()
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
            val = (val - self.vmin) / (self.vmax - self.vmin)

            val **= 1.0 / self.slider_exponent
            val *= 99
            val = int(round(val))

        self.slider.setValue(val)

    def setRange(self, vmin, vmax):
        """
        vmin, vmax -> arbitrary values that are not equal
        """
        if vmin == vmax:
            raise ValueError("limits must be different values")
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
            val = val * (self.vmax - self.vmin) + self.vmin

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
    """Enabling Slider widget for Parameter"""

    def makeWidget(self):
        opts = self.param.opts

        step = opts.get("step", 0.1)
        decimals = opts.get("decimals", 2)
        slider_exponent = opts.get("slider_exponent", 1)

        w = SliderWidget(decimals=decimals, step=step, slider_exponent=slider_exponent)

        limits = opts.get("limits", None)
        if limits is not None:
            w.setRange(*limits)

        value = opts.get("value", None)
        if value is not None:
            w.setValue(value)

        suffix = opts.get("suffix", None)
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
                return "%s:%s" % (rec.levelname, rec.name)
            elif idx.column() == 2:
                return rec.getMessage()

        elif role == QtCore.Qt.ItemDataRole:
            return rec

        elif role == QtCore.Qt.ToolTipRole:
            if idx.column() == 0:
                return rec.levelname
            elif idx.column() == 1:
                return "%s.%s" % (rec.module, rec.funcName)
            elif idx.column() == 2:
                return "Line %d" % rec.lineno

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


class SceneLog(QtWidgets.QDialog):

    levels = {
        50: "Critical",
        40: "Error",
        30: "Warning",
        20: "Info",
        10: "Debug",
        0: "All",
    }

    class LogEntryDelegate(QtWidgets.QStyledItemDelegate):

        levels = {
            50: QtWidgets.QStyle.SP_MessageBoxCritical,
            40: QtWidgets.QStyle.SP_MessageBoxCritical,
            30: QtWidgets.QStyle.SP_MessageBoxWarning,
            20: QtWidgets.QStyle.SP_MessageBoxInformation,
            10: QtWidgets.QStyle.SP_FileIcon,
        }

        def paint(self, painter, option, idx):
            # paint icon instead of log_lvl
            if idx.column() == 0:
                getIcon = QtWidgets.QApplication.style().standardIcon
                icon = getIcon(self.levels[idx.data()])
                icon.paint(painter, option.rect)
            else:
                QtWidgets.QStyledItemDelegate.paint(self, painter, option, idx)

    class LogFilter(QtCore.QSortFilterProxyModel):
        def __init__(self, *args, **kwargs):
            QtCore.QSortFilterProxyModel.__init__(self, *args, **kwargs)
            self.setLevel(30)

        def setLevel(self, level):
            self.level = level
            self.setFilterRegExp(str(level))
            self.invalidate()

    def __init__(self, app, model):
        QtWidgets.QDialog.__init__(self, app)
        logging_ui = op.join(
            op.dirname(op.realpath(__file__)), "spool", "res", "logging.ui"
        )
        loadUi(logging_ui, baseinstance=self)

        self.move(
            self.parent().window().mapToGlobal(self.parent().window().rect().center())
            - self.mapToGlobal(self.rect().center())
        )

        self.closeButton.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogCloseButton)
        )

        self.table_filter = self.LogFilter()
        self.table_filter.setFilterKeyColumn(0)
        self.table_filter.setDynamicSortFilter(True)
        self.table_filter.setSourceModel(model.log)

        model.log.rowsInserted.connect(self.newLogRecord)

        self.tableView.setModel(self.table_filter)
        self.tableView.setItemDelegate(self.LogEntryDelegate())

        self.tableView.setColumnWidth(0, 30)
        self.tableView.setColumnWidth(1, 200)

        self.filterBox.addItems([lvl_name for lvl_name in self.levels.values()])

        def changeFilter():
            for lvl, lvl_name in self.levels.items():
                if lvl_name == self.filterBox.currentText():
                    self.table_filter.setLevel(lvl)
                    break

            self.tableView.update()

        self.filterBox.currentIndexChanged.connect(changeFilter)
        self.filterBox.setCurrentText("Warning")

    @QtCore.pyqtSlot(QtCore.QModelIndex, int, int)
    def newLogRecord(self, idx, first, last):
        self.tableView.scrollToBottom()

        record = self.table_filter.sourceModel().log_records[idx.row()]
        if record.levelno >= 30 and self.autoBox.isChecked():
            self.show()


DEFAULT_CSS = """
QRangeSlider * {
    border: 0px;
    padding: 0px;
}
QRangeSlider #Head {
    background: #222;
}
QRangeSlider #Span {
    background: #393;
}
QRangeSlider #Span:active {
    background: #282;
}
QRangeSlider #Tail {
    background: #222;
}
QRangeSlider > QSplitter::handle {
    background: #393;
}
QRangeSlider > QSplitter::handle:vertical {
    height: 4px;
}
QRangeSlider > QSplitter::handle:pressed {
    background: #ca5;
}

"""


def scale(val, src, dst):
    """
    Scale the given value from the scale of src to the scale of dst.
    """
    return int(((val - src[0]) / float(src[1] - src[0])) * (dst[1] - dst[0]) + dst[0])


class Ui_Form(object):
    """default range slider form"""

    def setupUi(self, Form):
        Form.setObjectName("QRangeSlider")
        Form.resize(300, 30)
        Form.setStyleSheet(DEFAULT_CSS)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        # self.gridLayout.setMargin(0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")

        self._splitter = QtWidgets.QSplitter(Form)
        self._splitter.setMinimumSize(QtCore.QSize(0, 0))
        self._splitter.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self._splitter.setOrientation(QtCore.Qt.Horizontal)
        self._splitter.setObjectName("splitter")

        self._head = QtWidgets.QGroupBox(self._splitter)
        self._head.setTitle("")
        self._head.setObjectName("Head")
        self._handle = QtWidgets.QGroupBox(self._splitter)
        self._handle.setTitle("")
        self._handle.setObjectName("Span")

        self._tail = QtWidgets.QGroupBox(self._splitter)
        self._tail.setTitle("")
        self._tail.setObjectName("Tail")
        self.gridLayout.addWidget(self._splitter, 0, 0, 1, 1)

        # self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        encoding = QtWidgets.QApplication.UnicodeUTF8
        Form.setWindowTitle(
            QtWidgets.QApplication.translate(
                "QRangeSlider", "QRangeSlider", None, encoding
            )
        )


class Element(QtWidgets.QGroupBox):
    def __init__(self, parent, main):
        super().__init__(parent)
        self.main = main

    def setStyleSheet(self, style):
        """redirect style to parent groupbox"""
        self.parent().setStyleSheet(style)

    def textColor(self):
        """text paint color"""
        return getattr(self, "__textColor", QtGui.QColor(125, 125, 125))

    def format(self, value):
        if self.main._formatter is None:
            return str(value)
        return self.main._formatter(value)

    def setTextColor(self, color):
        """set the text paint color"""
        if type(color) == tuple and len(color) == 3:
            color = QtGui.QColor(color[0], color[1], color[2])
        elif type(color) == int:
            color = QtGui.QColor(color, color, color)
        setattr(self, "__textColor", color)

    def paintEvent(self, event):
        """overrides paint event to handle text"""
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.main.drawValues():
            self.drawText(event, qp)
        qp.end()


class Head(Element):
    """area before the handle"""

    def __init__(self, parent, main):
        super().__init__(parent, main)

    def drawText(self, event, qp):
        qp.setPen(self.textColor())
        qp.setFont(QtGui.QFont("Arial", 10))
        qp.drawText(event.rect(), QtCore.Qt.AlignLeft, self.format(self.main.min()))


class Tail(Element):
    """area after the handle"""

    def __init__(self, parent, main):
        super().__init__(parent, main)

    def drawText(self, event, qp):
        qp.setPen(self.textColor())
        qp.setFont(QtGui.QFont("Arial", 10))
        qp.drawText(event.rect(), QtCore.Qt.AlignRight, self.format(self.main.max()))


class Handle(Element):
    """handle area"""

    def __init__(self, parent, main):
        super().__init__(parent, main)

    def drawText(self, event, qp):
        qp.setPen(self.textColor())
        qp.setFont(QtGui.QFont("Arial", 10))
        qp.drawText(event.rect(), QtCore.Qt.AlignLeft, self.format(self.main.start()))
        qp.drawText(event.rect(), QtCore.Qt.AlignRight, self.format(self.main.end()))

    def mouseMoveEvent(self, event):
        event.accept()
        mx = event.globalX()
        _mx = getattr(self, "__mx", None)

        if not _mx:
            setattr(self, "__mx", mx)
            dx = 0
        else:
            dx = mx - _mx

        setattr(self, "__mx", mx)

        if dx == 0:
            event.ignore()
            return
        elif dx > 0:
            dx = 1
        elif dx < 0:
            dx = -1

        s = self.main.start() + dx
        e = self.main.end() + dx
        if s >= self.main.min() and e <= self.main.max():
            self.main.setRange(s, e)


class QRangeSlider(QtWidgets.QWidget, Ui_Form):
    """
    The QRangeSlider class implements a horizontal range slider widget.

    Inherits QWidget.

    Methods

        * __init__ (self, QWidget parent = None)
        * bool drawValues (self)
        * int end (self)
        * (int, int) getRange (self)
        * int max (self)
        * int min (self)
        * int start (self)
        * setBackgroundStyle (self, QString styleSheet)
        * setDrawValues (self, bool draw)
        * setEnd (self, int end)
        * setStart (self, int start)
        * setRange (self, int start, int end)
        * setSpanStyle (self, QString styleSheet)

    Signals

        * endValueChanged (int)
        * maxValueChanged (int)
        * minValueChanged (int)
        * startValueChanged (int)

    Customizing QRangeSlider

    You can style the range slider as below:
    ::
        QRangeSlider * {
            border: 0px;
            padding: 0px;
        }
        QRangeSlider #Head {
            background: #222;
        }
        QRangeSlider #Span {
            background: #393;
        }
        QRangeSlider #Span:active {
            background: #282;
        }
        QRangeSlider #Tail {
            background: #222;
        }

    Styling the range slider handles follows QSplitter options:
    ::
        QRangeSlider > QSplitter::handle {
            background: #393;
        }
        QRangeSlider > QSplitter::handle:vertical {
            height: 4px;
        }
        QRangeSlider > QSplitter::handle:pressed {
            background: #ca5;
        }
    """

    endValueChanged = QtCore.pyqtSignal(int)
    maxValueChanged = QtCore.pyqtSignal(int)
    minValueChanged = QtCore.pyqtSignal(int)
    startValueChanged = QtCore.pyqtSignal(int)

    # define splitter indices
    _SPLIT_START = 1
    _SPLIT_END = 2

    # signals
    minValueChanged = QtCore.pyqtSignal(int)
    maxValueChanged = QtCore.pyqtSignal(int)
    startValueChanged = QtCore.pyqtSignal(int)
    endValueChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        """Create a new QRangeSlider instance.

        :param parent: QWidget parent
        :return: New QRangeSlider instance.

        """
        super().__init__(parent)
        self.setupUi(self)
        self.setMouseTracking(False)

        # self._splitter.setChildrenCollapsible(False)
        self._splitter.splitterMoved.connect(self._handleMoveSplitter)

        # head layout
        self._head_layout = QtWidgets.QHBoxLayout()
        self._head_layout.setSpacing(0)
        # self._head_layout.setMargin(0)
        self._head.setLayout(self._head_layout)
        self.head = Head(self._head, main=self)
        self._head_layout.addWidget(self.head)

        # handle layout
        self._handle_layout = QtWidgets.QHBoxLayout()
        self._handle_layout.setSpacing(0)
        # self._handle_layout.setMargin(0)
        self._handle.setLayout(self._handle_layout)
        self.handle = Handle(self._handle, main=self)
        self.handle.setTextColor((150, 255, 150))
        self._handle_layout.addWidget(self.handle)

        # tail layout
        self._tail_layout = QtWidgets.QHBoxLayout()
        self._tail_layout.setSpacing(0)
        # self._tail_layout.setMargin(0)
        self._tail.setLayout(self._tail_layout)
        self.tail = Tail(self._tail, main=self)
        self._tail_layout.addWidget(self.tail)

        self._formatter = None

        # defaults
        self.setMin(0)
        self.setMax(99)
        self.setStart(0)
        self.setEnd(99)
        self.setDrawValues(True)

    def min(self):
        """:return: minimum value"""
        return getattr(self, "__min", None)

    def max(self):
        """:return: maximum value"""
        return getattr(self, "__max", None)

    def setMin(self, value):
        """sets minimum value"""
        assert type(value) is int
        setattr(self, "__min", value)
        self.minValueChanged.emit(value)

    def setMax(self, value):
        """sets maximum value"""
        assert type(value) is int
        setattr(self, "__max", value)
        self.maxValueChanged.emit(value)

    def start(self):
        """:return: range slider start value"""
        return getattr(self, "__start", None)

    def end(self):
        """:return: range slider end value"""
        return getattr(self, "__end", None)

    def _setStart(self, value):
        """stores the start value only"""
        setattr(self, "__start", value)
        self.startValueChanged.emit(value)

    def setFormatter(self, func):
        self._formatter = func

    def setStart(self, value):
        """sets the range slider start value"""
        assert type(value) is int
        v = self._valueToPos(value)
        self._splitter.splitterMoved.disconnect()
        self._splitter.moveSplitter(v, self._SPLIT_START)
        self._splitter.splitterMoved.connect(self._handleMoveSplitter)
        self._setStart(value)

    def _setEnd(self, value):
        """stores the end value only"""
        setattr(self, "__end", value)
        self.endValueChanged.emit(value)

    def setEnd(self, value):
        """set the range slider end value"""
        assert type(value) is int
        v = self._valueToPos(value)
        self._splitter.splitterMoved.disconnect()
        self._splitter.moveSplitter(v, self._SPLIT_END)
        self._splitter.splitterMoved.connect(self._handleMoveSplitter)
        self._setEnd(value)

    def drawValues(self):
        """:return: True if slider values will be drawn"""
        return getattr(self, "__drawValues", None)

    def setDrawValues(self, draw):
        """sets draw values boolean to draw slider values"""
        assert type(draw) is bool
        setattr(self, "__drawValues", draw)

    def getRange(self):
        """:return: the start and end values as a tuple"""
        return (self.start(), self.end())

    def setRange(self, start, end):
        """set the start and end values"""
        self.setStart(start)
        self.setEnd(end)

    def keyPressEvent(self, event):
        """overrides key press event to move range left and right"""
        key = event.key()
        if key == QtCore.Qt.Key_Left:
            s = self.start() - 1
            e = self.end() - 1
        elif key == QtCore.Qt.Key_Right:
            s = self.start() + 1
            e = self.end() + 1
        else:
            event.ignore()
            return
        event.accept()
        if s >= self.min() and e <= self.max():
            self.setRange(s, e)

    def setBackgroundStyle(self, style):
        """sets background style"""
        self._tail.setStyleSheet(style)
        self._head.setStyleSheet(style)

    def setSpanStyle(self, style):
        """sets range span handle style"""
        self._handle.setStyleSheet(style)

    def _valueToPos(self, value):
        """converts slider value to local pixel x coord"""
        return scale(value, (self.min(), self.max()), (0, self.width()))

    def _posToValue(self, xpos):
        """converts local pixel x coord to slider value"""
        return scale(xpos, (0, self.width()), (self.min(), self.max()))

    def _handleMoveSplitter(self, xpos, index):
        """private method for handling moving splitter handles"""
        hw = self._splitter.handleWidth()

        def _lockWidth(widget):
            width = widget.size().width()
            widget.setMinimumWidth(width)
            widget.setMaximumWidth(width)

        def _unlockWidth(widget):
            widget.setMinimumWidth(0)
            widget.setMaximumWidth(16777215)

        v = self._posToValue(xpos)

        if index == self._SPLIT_START:
            _lockWidth(self._tail)
            if v >= self.end():
                return

            offset = -20
            w = xpos + offset
            self._setStart(v)

        elif index == self._SPLIT_END:
            _lockWidth(self._head)
            if v <= self.start():
                return

            offset = -40
            w = self.width() - xpos + offset
            self._setEnd(v)

        _unlockWidth(self._tail)
        _unlockWidth(self._head)
        _unlockWidth(self._handle)
