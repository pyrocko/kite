from PyQt5 import QtCore
from pyqtgraph import SignalProxy
import logging
from datetime import datetime

from kite import Scene
from kite.qt_utils import SceneLogModel


class SceneModel(QtCore.QObject):
    ''' Proxy for :class:`kite.Scene` so we can change the scene '''
    sigSceneModelChanged = QtCore.pyqtSignal(object)

    sigSceneChanged = QtCore.pyqtSignal()
    sigConfigChanged = QtCore.pyqtSignal()

    sigFrameChanged = QtCore.pyqtSignal()
    sigQuadtreeChanged = QtCore.pyqtSignal()
    _sigQuadtreeChanged = QtCore.pyqtSignal()
    sigQuadtreeConfigChanged = QtCore.pyqtSignal()
    sigCovarianceChanged = QtCore.pyqtSignal()
    sigCovarianceConfigChanged = QtCore.pyqtSignal()

    sigProcessingStarted = QtCore.pyqtSignal(str)
    sigProcessingFinished = QtCore.pyqtSignal()
    sigCalculateWeightMatrixFinished = QtCore.pyqtSignal(object)

    sigLogRecord = QtCore.pyqtSignal(object)

    def __init__(self):
        QtCore.QObject.__init__(self)

        self.scene = None
        self.frame = None
        self.quadtree = None
        self.covariance = None
        self.log = SceneLogModel(self)

        self._ = SignalProxy(
            self._sigQuadtreeChanged,
            rateLimit=10,
            delay=0,
            slot=lambda: self.sigQuadtreeChanged.emit())

        self._log_handler = logging.Handler()
        self._log_handler.setLevel(logging.DEBUG)
        self._log_handler.emit = self.sigLogRecord.emit

        logging.root.addHandler(self._log_handler)

        self.qtproxy = QSceneQuadtreeProxy(self)

        self.worker_thread = QtCore.QThread()
        self.moveToThread(self.worker_thread)
        self.worker_thread.start()

    def setScene(self, scene):
        self.disconnectSlots()

        self.scene = scene
        self.frame = scene.frame
        self.quadtree = scene.quadtree
        self.covariance = scene.covariance

        self.connectSlots()
        self.sigSceneModelChanged.emit(object)

    def getScene(self):
        return self.scene

    def disconnectSlots(self):
        if self.scene is None:
            return

        self.scene.evChanged.unsubscribe(
            self.sigSceneChanged.emit)
        self.scene.evConfigChanged.unsubscribe(
            self.sigConfigChanged.emit)

        self.scene.frame.evChanged.unsubscribe(
            self.sigFrameChanged.emit)

        self.quadtree.evChanged.unsubscribe(
            self._sigQuadtreeChanged.emit)
        self.quadtree.evConfigChanged.unsubscribe(
            self.sigQuadtreeConfigChanged.emit)

        self.covariance.evChanged.unsubscribe(
            self.sigCovarianceChanged.emit)
        self.covariance.evConfigChanged.unsubscribe(
            self.sigCovarianceConfigChanged.emit)

    def connectSlots(self):
        self.scene.evChanged.subscribe(
            self.sigSceneChanged.emit)
        self.scene.evConfigChanged.subscribe(
            self.sigCovarianceConfigChanged.emit)

        self.scene.frame.evChanged.subscribe(
            self.sigFrameChanged.emit)

        self.quadtree.evChanged.subscribe(
            self._sigQuadtreeChanged.emit)
        self.quadtree.evConfigChanged.subscribe(
            self.sigQuadtreeConfigChanged.emit)

        self.covariance.evChanged.subscribe(
            self.sigCovarianceChanged.emit)
        self.covariance.evConfigChanged.subscribe(
            self.sigCovarianceConfigChanged.emit)

    @QtCore.pyqtSlot(str)
    def exportWeightMatrix(self, filename):
        self.sigProcessingStarted.emit(
            'Calculating <span style="font-family: monospace">'
            'Covariance.weight_matrix</span>, this can take a few minutes...')
        self.scene.covariance.export_weight_matrix(filename)
        self.sigProcessingFinished.emit()

    @QtCore.pyqtSlot()
    def calculateWeightMatrix(self):
        t0 = datetime.now()
        self.sigProcessingStarted.emit(
            'Calculating <span style="font-family: monospace">'
            'Covariance.weight_matrix</span>,'
            ' this can take a few minutes...')
        self.scene.covariance.weight_matrix
        self.sigProcessingFinished.emit()
        self.sigCalculateWeightMatrixFinished.emit(datetime.now()-t0)

    @QtCore.pyqtSlot(str)
    def importFile(self, filename):
        self.sigProcessingStarted.emit('Importing scene...')
        self.setScene(Scene.import_data(filename))
        self.sigProcessingFinished.emit()

    @QtCore.pyqtSlot(str)
    def loadFile(self, filename):
        self.sigProcessingStarted.emit('Loading scene...')
        self.setScene(Scene.load(filename))
        self.sigProcessingFinished.emit()

    @QtCore.pyqtSlot(str)
    def loadConfig(self, filename):
        self.scene.load_config(filename)


class QSceneQuadtreeProxy(QtCore.QObject):

    def __init__(self, scene_proxy):
        QtCore.QObject.__init__(self, scene_proxy)
        self.scene_proxy = scene_proxy

    @QtCore.pyqtSlot(float)
    def setEpsilon(self, value):
        self.scene_proxy.quadtree.epsilon = value

    @QtCore.pyqtSlot(float)
    def setNanFraction(self, value):
        self.scene_proxy.quadtree.nan_allowed = value

    @QtCore.pyqtSlot(float)
    def setTileMaximum(self, value):
        self.scene_proxy.quadtree.tile_size_max = value

    @QtCore.pyqtSlot(float)
    def setTileMinimum(self, value):
        self.scene_proxy.quadtree.tile_size_min = value
