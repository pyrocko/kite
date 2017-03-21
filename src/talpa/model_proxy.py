import QtCore


class ModelProxy(QtCore.QObject):

    sigModelChanged = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        QtCore.QObject.__init__(self, *args, **kwargs)
        self.model = None

        model = self.kwargs.pop('model', None)
        if model is not None:
            self.setModel(model)

    def setModel(self, model):
        self.disconnectModel()
        self.model = model
        self.connectModel()
        self.sigModelChanged.emit()

    def connectModel(self):
        pass

    def disconnectModel(self):
        if self.model is None:
            return
