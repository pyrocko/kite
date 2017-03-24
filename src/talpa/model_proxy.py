from PySide import QtCore

# Importing available models
from .sources import OkadaSourceDelegate
from .common import SourceROI, SourceEditorDialog


available_delegates = {}
for module in [OkadaSourceDelegate]:
    available_delegates[module.__represents__] = module


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

    selectionModelChanged = QtCore.Signal()

    def __init__(self, sandbox, *args, **kwargs):
        QtCore.QAbstractTableModel.__init__(self, *args, **kwargs)

        self.sandbox = sandbox
        self.selection_model = None
        self._createSources()
        self.sandbox.sigModelChanged.connect(self.modelUpdated)

    def _createSources(self):
        self._sources = []
        for isrc, src in enumerate(self.sandbox.model.sources):
            source_model = available_delegates[src.__class__.__name__]
            idx = self.createIndex(isrc, 0)
            src = source_model(self, src, idx)

            self._sources.append(src)

    def rowCount(self, idx):
        return len(self.sandbox.model.sources)

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

    @QtCore.Slot()
    def modelUpdated(self):
        return
