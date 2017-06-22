from PySide import QtGui, QtCore
from common import get_resource
from kite.qt_utils import loadUi
from pyrocko.guts import Object, Bool, String, Int, Tuple, load
from os import path

import os
import logging


config_file = path.expanduser('~/.config/kite/talpa_config.yml')
logger = logging.getLogger('TalpaConfig')


class TalpaConfig(Object):
    show_cursor = Bool.T(
        default=True)

    default_gf_dir = String.T(
        default='')  # noqa

    nvectors = Int.T(
        default=200,
        help='Number of horizontal displacement vectors to show')

    vector_color = Tuple.T(
        default=(0, 0, 0, 100),
        help='Color of displacement arrow, RGBA (0, 0 ,0 , 100)')

    class QConfig(QtCore.QObject):
        updated = QtCore.Signal()

    qconfig = QConfig()

    def saveConfig(self):
        self.regularize()
        self.dump(filename=config_file)
        self.qconfig.updated.emit()


def createDefaultConfig():
    logger.info('Creating new config...')
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
    if not path.isfile(config_file):
        createDefaultConfig()
    try:
        logger.info('Loading config from %s...' % config_file)
        config = load(filename=config_file)
    except KeyError:
        createDefaultConfig()
        config = TalpaConfig()
    return config


config = getConfig()


class ConfigDialog(QtGui.QDialog):

    attributes = ['show_cursor', 'default_gf_dir', 'nvectors', 'vector_color']

    completer = QtGui.QCompleter()
    completer_model = QtGui.QFileSystemModel(completer)
    completer.setModel(completer_model)
    completer.setMaxVisibleItems(8)

    def __init__(self, *args, **kwargs):
        QtGui.QDialog.__init__(self, *args, **kwargs)
        loadUi(get_resource('dialog_config.ui'), self)

        self.ok_button.released.connect(
            self.setAttributes)
        self.ok_button.released.connect(
            self.close)

        self.apply_button.released.connect(
            self.setAttributes)

        self.vector_color_picker = QtGui.QColorDialog(self)
        self.vector_color_picker.\
            setCurrentColor(QtGui.QColor(*config.vector_color))
        self.vector_color_picker.\
            setOption(self.vector_color_picker.ShowAlphaChannel)
        self.vector_color_picker.colorSelected.connect(
            self.updateVectorColor)
        self.vector_color_picker.setModal(True)
        self.vector_color.clicked.connect(
            self.vector_color_picker.show)

        self.vector_color.setValue = self.setButtonColor
        self.vector_color.value = self.getButtonColor

        self.chooseStoreDirButton.released.connect(
            self.chooseStoreDir)
        self.completer_model.setRootPath('')
        self.completer.setParent(self.default_gf_dir)
        self.default_gf_dir.setCompleter(self.completer)

        self.getAttributes()

    @QtCore.Slot()
    def chooseStoreDir(self):
        folder = QtGui.QFileDialog.getExistingDirectory(
            self, 'Open Pyrocko GF Store', os.getcwd())
        if folder != '':
            self.default_gf_dir.setText(folder)

        self.setAttributes()

    def getAttributes(self):
        for attr in self.attributes:
            qw = self.__getattribute__(attr)
            value = config.__getattribute__(attr)
            if isinstance(value, bool):
                qw.setChecked(value)
            elif isinstance(value, str):
                qw.setText(value)
            else:
                qw.setValue(value)

    def setAttributes(self):
        for attr in self.attributes:
            qw = self.__getattribute__(attr)
            if isinstance(qw, QtGui.QCheckBox):
                value = qw.isChecked()
            elif isinstance(qw, QtGui.QLineEdit):
                value = str(qw.text())
            else:
                value = qw.value()

            config.__setattr__(attr, value)

        config.saveConfig()

    def setButtonColor(self, rgba):
        self.vector_color.setStyleSheet(
            'background-color: rgb(%d, %d, %d, %d);'
            'border: none;' % rgba)

    def getButtonColor(self):
        return config.vector_color

    @QtCore.Slot()
    def updateVectorColor(self, qcolor):
        config.vector_color = (qcolor.red(), qcolor.green(), qcolor.blue(),
                               qcolor.alpha())
        self.setButtonColor(config.vector_color)
