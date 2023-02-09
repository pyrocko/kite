import logging

from PyQt5 import QtCore

# Importing available models
from kite.qt_utils import SceneLogModel

from . import sources
from .util import SourceEditorDialog, SourceROI

available_delegates = {}
for module in sources.__sources__:
    available_delegates[module.__represents__] = module


class CursorTracker(QtCore.QObject):
    sigCursorMoved = QtCore.pyqtSignal(object)
    sigMouseMoved = QtCore.pyqtSignal(object)


class SandboxModel(QtCore.QObject):
    sigModelUpdated = QtCore.pyqtSignal()
    sigModelChanged = QtCore.pyqtSignal()
    sigLogRecord = QtCore.pyqtSignal(object)

    sigProcessingFinished = QtCore.pyqtSignal()
    sigProcessingStarted = QtCore.pyqtSignal(str)

    def __init__(self, sandbox_model=None, *args, **kwargs):
        QtCore.QObject.__init__(self)

        self.model = sandbox_model
        self.log = SceneLogModel(self)
        self.sources = SourceModel(self)
        self.cursor_tracker = CursorTracker(self)

        self._log_handler = logging.Handler()
        self._log_handler.setLevel(logging.DEBUG)
        self._log_handler.emit = self.sigLogRecord.emit

        logging.root.setLevel(logging.DEBUG)
        logging.root.addHandler(self._log_handler)

        self.worker_thread = QtCore.QThread(self)
        self.moveToThread(self.worker_thread)
        self.worker_thread.start()

        if self.model:
            self.setModel(self.model)

    def setModel(self, model):
        self.disconnectSlots()

        self.model = model
        self.frame = model.frame

        self.connectSlots()
        self.sigModelChanged.emit()
        self.sigModelUpdated.emit()

    def connectSlots(self):
        self.model._log.addHandler(self._log_handler)
        self.model.evModelUpdated.subscribe(self.sigModelUpdated.emit)

    def disconnectSlots(self):
        if self.model is None:
            return
        try:
            self.model._log.removeHandler(self._log_handler)
            self.model.evModelUpdated.unsubscribe(self.sigModelUpdated.emit)
        except AttributeError:
            pass
        finally:
            self.model = None

    def addSource(self, source):
        self.model.addSource(source)

    def removeSource(self, source):
        self.model.removeSource(source)

    @QtCore.pyqtSlot()
    def optimizeSource(self):
        self.sigProcessingStarted.emit("Optimizing source, stay tuned!")
        self.model.reference.optimizeSource()
        self.sigProcessingFinished.emit()

    @classmethod
    def randomOkada(cls, nsources=1):
        from ..sandbox_scene import TestSandboxScene

        model = TestSandboxScene.randomOkada(nsources)
        sandbox = cls(model)
        return sandbox

    @classmethod
    def simpleOkada(cls, **kwargs):
        from ..sandbox_scene import TestSandboxScene

        model = TestSandboxScene.simpleOkada(**kwargs)
        sandbox = cls(model)
        return sandbox

    @classmethod
    def empty(cls, parent=None):
        from ..sandbox_scene import SandboxScene

        sandbox = cls(parent=parent)
        sandbox.setModel(SandboxScene())
        return sandbox


class SourceModel(QtCore.QAbstractTableModel):
    selectionModelChanged = QtCore.pyqtSignal()

    def __init__(self, sandbox, *args, **kwargs):
        QtCore.QAbstractTableModel.__init__(self, *args, **kwargs)

        self.sandbox = sandbox
        self.selection_model = None
        self._createSources()

        self.sandbox.sigModelUpdated.connect(self.modelUpdated)
        self.sandbox.sigModelChanged.connect(self.modelChanged)

    def _createSources(self):
        self._sources = []
        for isrc, src in enumerate(self.model_sources):
            source_model = available_delegates[src.__class__.__name__]
            idx = self.createIndex(isrc, 0)
            src = source_model(self, src, idx, parent=self)

            self._sources.append(src)

    @property
    def model_sources(self):
        if self.sandbox.model is None:
            return []
        else:
            return self.sandbox.model.sources

    def rowCount(self, idx):
        return len(self.model_sources)

    def columnCount(self, idx):
        return 1

    def flags(self, idx):
        return (
            QtCore.Qt.ItemIsSelectable
            | QtCore.Qt.ItemIsEditable
            | QtCore.Qt.ItemIsEnabled
        )

    def setSelectionModel(self, selection_model):
        self.selection_model = selection_model
        self.selectionModelChanged.emit()

    def data(self, idx, role):
        src = self._sources[idx.row()]
        if role == QtCore.Qt.DisplayRole:
            return src.formatListItem()
        elif role == SourceROI:
            return src.getROIItem()
        elif role == SourceEditorDialog:
            return src.getEditingDialog()

    def itemData(self, idx):
        src = self._sources[idx.row()]
        return src.getSourceParameters()

    def setItemData(self, idx, parameters):
        src = self._sources[idx.row()]
        src.setSourceParameters(parameters)
        self.dataChanged.emit(idx, idx)
        return True

    def setData(self, idx, value, role):
        print("Set %s with role %s to value %s" % (idx, value, role))

    def removeSource(self, idx):
        src = self._sources[idx.row()]
        self.sandbox.removeSource(src.source)

    @QtCore.pyqtSlot()
    def modelUpdated(self, force=False):
        if len(self._sources) != len(self.model_sources) or force:
            self.beginResetModel()
            self._createSources()
            self.endResetModel()

    @QtCore.pyqtSlot()
    def modelChanged(self):
        self.modelUpdated(force=True)
