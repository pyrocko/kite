from PySide import QtCore, QtGui

# Importing available models
from .sources import OkadaSourceModel
from .common import PyQtGraphROI


available_models = {}
for module in [OkadaSourceModel]:
    available_models[module.__represents__] = module


class CursorTracker(QtCore.QObject):
    sigCursorMoved = QtCore.Signal(object)
    sigMouseMoved = QtCore.Signal(object)


class SandboxModel(QtCore.QObject):

    sigModelChanged = QtCore.Signal()

    def __init__(self, model, *args, **kwargs):
        QtCore.QObject.__init__(self, *args, **kwargs)
        self.model = None

        self.cursor_tracker = CursorTracker()
        self.setModel(model)

    def setModel(self, model):
        self.disconnectModel()

        self.model = model
        self.frame = model.frame
        self.sources = SourceModel(self)

        self.connectModel()
        self.sigModelChanged.emit()

    def connectModel(self):
        self.model.evModelChanged.subscribe(self.sigModelChanged.emit)

    def disconnectModel(self):
        if self.model is None:
            return
        self.model.evModelChanged.unsubscribe(self.sigModelChanged.emit)

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


class SourceModel(QtCore.QAbstractTableModel):

    def __init__(self, sandbox, *args, **kwargs):
        QtCore.QAbstractTableModel.__init__(self, *args, **kwargs)

        self.sandbox = sandbox
        self.selection_model = None
        self._createSources()
        self.sandbox.sigModelChanged.connect(self.modelUpdated)

    def _createSources(self):
        self._sources = []
        for isrc, src in enumerate(self.sandbox.model.sources):
            source_model = available_models[src.__class__.__name__]
            idx = self.createIndex(isrc, 0)
            src = source_model(self, src, idx)

            self._sources.append(src)

    def rowCount(self, idx):
        return len(self.sandbox.model.sources)

    def setSelectionModel(self, selection_model):
        self.selection_model = selection_model
        for src in self._sources:
            src.ROISelected.connect(self.highlightSourceListItem)

    @QtCore.Slot(object)
    def highlightSourceROI(self, idx):
        self._sources[idx.row()].sigHighlightROI.emit()

    @QtCore.Slot(object)
    def highlightSourceListItem(self, idx):
        print idx
        self.selection_model.select(idx, QtGui.QItemSelectionModel.ToggleCurrent)

    def data(self, idx, role):
        src = self._sources[idx.row()]
        if role == QtCore.Qt.DisplayRole:
            return src.formatListItem()
        elif role == PyQtGraphROI:
            return src.getROIItem()

    def columnCount(self, idx):
        return 1

    @QtCore.Slot()
    def modelUpdated(self):
        self.beginResetModel()
        # self._createSources()
        self.endResetModel()
