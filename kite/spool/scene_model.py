import logging
from datetime import datetime

from PyQt5 import QtCore
from pyqtgraph import SignalProxy

try:
    from pyrocko.dataset.util import set_download_callback as pyrocko_download_callback
except ImportError:
    pyrocko_download_callback = None

from kite import Scene
from kite.qt_utils import SceneLogModel


class SceneModel(QtCore.QObject):
    """Proxy for :class:`kite.Scene` so we can change the scene"""

    sigSceneModelChanged = QtCore.pyqtSignal(object)

    sigSceneChanged = QtCore.pyqtSignal()
    sigConfigChanged = QtCore.pyqtSignal()

    sigFrameChanged = QtCore.pyqtSignal()
    sigQuadtreeChanged = QtCore.pyqtSignal()
    _sigQuadtreeChanged = QtCore.pyqtSignal()
    sigQuadtreeConfigChanged = QtCore.pyqtSignal()
    sigCovarianceChanged = QtCore.pyqtSignal()
    sigCovarianceConfigChanged = QtCore.pyqtSignal()

    sigProgressStarted = QtCore.pyqtSignal(object)
    sigProgressFinished = QtCore.pyqtSignal()
    sigCalculateWeightMatrixFinished = QtCore.pyqtSignal(object)

    sigHasElevation = QtCore.pyqtSignal()
    sigLogRecord = QtCore.pyqtSignal(object)

    def __init__(self, spool):
        QtCore.QObject.__init__(self)

        self.spool = spool
        self.scene = None
        self.frame = None
        self.quadtree = None
        self.covariance = None
        self.aps = None

        self.log = SceneLogModel(self)

        self._ = SignalProxy(
            self._sigQuadtreeChanged,
            rateLimit=10,
            delay=0,
            slot=lambda: self.sigQuadtreeChanged.emit(),
        )

        self._log_handler = logging.Handler()
        self._log_handler.setLevel(logging.DEBUG)
        self._log_handler.emit = self.sigLogRecord.emit

        logging.root.addHandler(self._log_handler)

        self._download_status = None
        if pyrocko_download_callback:
            pyrocko_download_callback(self.download_progress)

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
        self.aps = scene.aps

        self.connectSlots()
        self.sigSceneModelChanged.emit(object)

    def getScene(self):
        return self.scene

    def disconnectSlots(self):
        if self.scene is None:
            return

        self.scene.evChanged.unsubscribe(self.sigSceneChanged.emit)
        self.scene.evConfigChanged.unsubscribe(self.sigConfigChanged.emit)

        self.scene.frame.evChanged.unsubscribe(self.sigFrameChanged.emit)

        self.quadtree.evChanged.unsubscribe(self._sigQuadtreeChanged.emit)
        self.quadtree.evConfigChanged.unsubscribe(self.sigQuadtreeConfigChanged.emit)

        self.covariance.evChanged.unsubscribe(self.sigCovarianceChanged.emit)
        self.covariance.evConfigChanged.unsubscribe(
            self.sigCovarianceConfigChanged.emit
        )

        self.aps.evChanged.unsubscribe(self.sigAPSChanged.emit)

    def connectSlots(self):
        self.scene.evChanged.subscribe(self.sigSceneChanged.emit)
        self.scene.evConfigChanged.subscribe(self.sigCovarianceConfigChanged.emit)

        self.scene.frame.evChanged.subscribe(self.sigFrameChanged.emit)

        self.quadtree.evChanged.subscribe(self._sigQuadtreeChanged.emit)
        self.quadtree.evConfigChanged.subscribe(self.sigQuadtreeConfigChanged.emit)

        self.covariance.evChanged.subscribe(self.sigCovarianceChanged.emit)
        self.covariance.evConfigChanged.subscribe(self.sigCovarianceConfigChanged.emit)

    @QtCore.pyqtSlot(str)
    def exportWeightMatrix(self, filename):
        t0 = datetime.now()
        quadtree = self.quadtree
        covariance = self.covariance

        def progress_func():
            return covariance.finished_combinations

        ncombinations = quadtree.nleaves * (quadtree.nleaves + 1) / 2

        self.sigProgressStarted.emit(
            (
                'Calculating <span style="font-family: monospace">'
                "Covariance.weight_matrix</span>, this can take a few minutes...",
                ncombinations,
                progress_func,
            )
        )

        self.scene.covariance.export_weight_matrix(filename)
        self.sigProgressFinished.emit()
        self.sigCalculateWeightMatrixFinished.emit(datetime.now() - t0)

    @QtCore.pyqtSlot()
    def calculateWeightMatrix(self):
        t0 = datetime.now()
        quadtree = self.quadtree
        covariance = self.covariance

        def progress_func():
            return covariance.finished_combinations

        ncombinations = quadtree.nleaves * (quadtree.nleaves + 1) / 2

        self.sigProgressStarted.emit(
            (
                'Calculating <span style="font-family: monospace">'
                "Covariance.weight_matrix</span>,"
                " this can take a few minutes...",
                ncombinations,
                progress_func,
            )
        )

        self.scene.covariance.weight_matrix
        self.sigProgressFinished.emit()
        self.sigCalculateWeightMatrixFinished.emit(datetime.now() - t0)

    @QtCore.pyqtSlot(str)
    def importFile(self, filename):
        self.sigProgressStarted.emit(("Importing scene...",))
        self.setScene(Scene.import_data(filename))
        self.sigProgressFinished.emit()

    @QtCore.pyqtSlot(str)
    def loadFile(self, filename):
        self.sigProgressStarted.emit(("Loading scene...",))
        self.setScene(Scene.load(filename))
        self.sigProgressFinished.emit()

    @QtCore.pyqtSlot(str)
    def loadConfig(self, filename):
        self.scene.load_config(filename)

    def download_progress(self, context_str, status):
        progress = self.spool.progress

        progress.setWindowTitle("Downloading...")
        progress.setLabelText(context_str)
        progress.setMaximum(status.get("ntotal_bytes_all_files", 0))
        progress.setValue(status.get("nread_bytes_all_files", 0))
        QtCore.QCoreApplication.processEvents()

        if progress.isHidden():
            progress.show()
            QtCore.QCoreApplication.processEvents()

        if status["finished"]:
            progress.reset()


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
