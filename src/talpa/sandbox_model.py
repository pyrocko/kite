from PySide import QtCore

# Importing available models
from .sources import OkadaSourceDelegate
from .common import SourceROI, SourceEditorDialog
from ..qt_utils import SceneLogModel

import logging


available_delegates = {}
for module in [OkadaSourceDelegate]:
    available_delegates[module.__represents__] = module


class CursorTracker(QtCore.QObject):
    sigCursorMoved = QtCore.Signal(object)
    sigMouseMoved = QtCore.Signal(object)


class SandboxModel(QtCore.QObject):

    sigModelUpdated = QtCore.Signal()
    sigModelChanged = QtCore.Signal()
    sigLogRecord = QtCore.Signal(object)

    def __init__(self, scene_model, *args, **kwargs):
        QtCore.QObject.__init__(self, *args, **kwargs)
        self.model = None
        self.log = SceneLogModel(self)
        self.sources = SourceModel(self)

        self._log_handler = logging.Handler()
        self._log_handler.emit = self.sigLogRecord.emit

        self.cursor_tracker = CursorTracker()
        self.setModel(scene_model)

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
        self.model._log.removeHandler(self._log_handler)
        self.model.evModelUpdated.unsubscribe(self.sigModelUpdated.emit)
        self.model = None

    def addSource(self, source):
        self.model.addSource(source)

    def removeSource(self, source):
        self.model.removeSource(source)

    @classmethod
    def randomOkada(cls, nsources=1):
        from ..model_scene import TestModelScene
        model = TestModelScene.randomOkada(nsources)
        sandbox = cls(model)
        return sandbox

    @classmethod
    def simpleOkada(cls, **kwargs):
        from ..model_scene import TestModelScene
        model = TestModelScene.simpleOkada(**kwargs)
        sandbox = cls(model)
        return sandbox

    @classmethod
    def empty(cls):
        from ..model_scene import ModelScene
        model = ModelScene()
        sandbox = cls(model)
        return sandbox


class SourceModel(QtCore.QAbstractTableModel):

    selectionModelChanged = QtCore.Signal()

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
            src = source_model(self, src, idx)

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
        return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable |\
            QtCore.Qt.ItemIsEnabled

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
        print idx

    def removeSource(self, idx):
        src = self._sources[idx.row()]
        self.sandbox.removeSource(src.source)

    @QtCore.Slot()
    def modelUpdated(self, force=False):
        if len(self._sources) != len(self.model_sources) or force:
            self.beginResetModel()
            self._createSources()
            self.endResetModel()

    @QtCore.Slot()
    def modelChanged(self):
        self.modelUpdated(force=True)
