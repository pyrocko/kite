import logging
import os
from os import path

from PyQt5 import QtCore, QtGui, QtWidgets
from pyrocko.guts import Bool, Float, Int, Object, String, Tuple, load

from kite.qt_utils import loadUi

from .util import get_resource

config_file = path.expanduser("~/.config/kite/talpa_config.yml")
logger = logging.getLogger("TalpaConfig")

config_instance = None


class TalpaConfig(Object):
    show_cursor = Bool.T(default=True)

    default_gf_dir = String.T(default="")  # noqa

    nvectors = Int.T(
        default=200, help="Number of horizontal displacement vectors to show"
    )

    vector_color = Tuple.T(
        default=(0, 0, 0, 100),
        help="Color of the displacement arrows, RGBA (0, 0, 0, 100)",
    )

    vector_relative_length = Int.T(default=100, help="Relative length of the arrow.")

    vector_pen_thickness = Float.T(default=1.0, help="Thickness of the arrows.")

    view_north = Bool.T(default=True, help="Show the north view of displacement.")

    view_east = Bool.T(default=True, help="Show the east view of displacement.")

    view_down = Bool.T(default=True, help="Show the down view of displacement.")

    view_los = Bool.T(default=True, help="Show the los view of displacement.")

    def __init__(self, *args, **kwargs):
        class QConfig(QtCore.QObject):
            updated = QtCore.pyqtSignal()

        Object.__init__(self, *args, **kwargs)
        self.qconfig = QConfig()

    def saveConfig(self):
        self.regularize()
        self.dump(filename=config_file)
        self.qconfig.updated.emit()


def createDefaultConfig():
    logger.info("Creating new config...")
    import os

    try:
        os.makedirs(path.dirname(config_file))
    except OSError as e:
        if e.errno == 17:
            pass
        else:
            raise e

    config = TalpaConfig()
    config.dump(filename=config_file)


def getConfig():
    global config_instance

    if not config_instance:
        if not path.isfile(config_file):
            createDefaultConfig()
        try:
            logger.info("Loading config from %s..." % config_file)
            config_instance = load(filename=config_file)
        except KeyError:
            createDefaultConfig()
            config_instance = TalpaConfig()

    return config_instance


class ConfigDialog(QtWidgets.QDialog):
    attributes = [
        "show_cursor",
        "default_gf_dir",
        "nvectors",
        "vector_color",
        "vector_relative_length",
        "vector_pen_thickness",
        "view_east",
        "view_north",
        "view_down",
        "view_los",
    ]

    def __init__(self, *args, **kwargs):
        QtWidgets.QDialog.__init__(self, *args, **kwargs)

        self.completer = QtWidgets.QCompleter()
        self.completer_model = QtGui.QFileSystemModel(self.completer)
        self.completer.setModel(self.completer_model)
        self.completer.setMaxVisibleItems(8)

        loadUi(get_resource("dialog_config.ui"), self)

        self.ok_button.released.connect(self.setAttributes)
        self.ok_button.released.connect(self.close)

        self.apply_button.released.connect(self.setAttributes)

        self.vector_color_picker = QtWidgets.QColorDialog(self)
        self.vector_color_picker.setCurrentColor(
            QtGui.QColor(*getConfig().vector_color)
        )
        self.vector_color_picker.setOption(self.vector_color_picker.ShowAlphaChannel)
        self.vector_color_picker.colorSelected.connect(self.updateVectorColor)
        self.vector_color_picker.setModal(True)
        self.vector_color.clicked.connect(self.vector_color_picker.show)

        self.vector_color.setValue = self.setButtonColor
        self.vector_color.value = self.getButtonColor

        self.chooseStoreDirButton.released.connect(self.chooseStoreDir)
        self.completer_model.setRootPath("")
        self.completer.setParent(self.default_gf_dir)
        self.default_gf_dir.setCompleter(self.completer)

        self.getAttributes()

    @QtCore.pyqtSlot()
    def chooseStoreDir(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open Pyrocko GF Store", os.getcwd()
        )
        if folder != "":
            self.default_gf_dir.setText(folder)

        self.setAttributes()

    def getAttributes(self):
        for attr in self.attributes:
            qw = self.__getattribute__(attr)
            value = getConfig().__getattribute__(attr)
            if isinstance(value, bool):
                qw.setChecked(value)
            elif isinstance(value, str):
                qw.setText(value)
            else:
                qw.setValue(value)

    @QtCore.pyqtSlot()
    def setAttributes(self):
        for attr in self.attributes:
            qw = self.__getattribute__(attr)
            if isinstance(qw, QtWidgets.QCheckBox):
                value = qw.isChecked()
            elif isinstance(qw, QtWidgets.QLineEdit):
                value = str(qw.text())
            else:
                value = qw.value()

            getConfig().__setattr__(attr, value)

        getConfig().saveConfig()

    def setButtonColor(self, rgba):
        self.vector_color.setStyleSheet(
            "background-color: rgb(%d, %d, %d, %d);" "border: none;" % rgba
        )

    def getButtonColor(self):
        return getConfig().vector_color

    @QtCore.pyqtSlot(QtGui.QColor)
    def updateVectorColor(self, qcolor):
        getConfig().vector_color = (
            qcolor.red(),
            qcolor.green(),
            qcolor.blue(),
            qcolor.alpha(),
        )
        self.setButtonColor(getConfig().vector_color)
