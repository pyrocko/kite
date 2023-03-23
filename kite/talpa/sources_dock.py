from PyQt5 import QtCore, QtGui, QtWidgets

from .sources import __sources__
from .util import SourceEditorDialog

km = 1e3


class SourcesListDock(QtWidgets.QDockWidget):
    def __init__(self, sandbox, *args, **kwargs):
        QtWidgets.QDockWidget.__init__(self, "Sources", *args, **kwargs)
        self.sandbox = sandbox

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(3, 3, 3, 3)
        sources = SourcesList(sandbox, parent=self)
        sources_add_menu = SourcesAddButton(sandbox)

        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetMovable
        )

        self.widget = QtWidgets.QWidget()
        self.widget.setLayout(layout)
        self.widget.layout().addWidget(sources)
        self.widget.layout().addWidget(sources_add_menu)

        self.setWidget(self.widget)


class SourcesAddButton(QtWidgets.QToolButton):
    class SourcesAddMenu(QtWidgets.QMenu):
        def __init__(self, sandbox, *args, **kwargs):
            QtWidgets.QMenu.__init__(self)
            self.setTitle("Add Source")
            self.sandbox = sandbox

            backends = set(src.display_backend for src in __sources__)

            for ibe, be in enumerate(backends):
                self.addSection(be)

                for src in [s for s in __sources__ if s.display_backend == be]:
                    self.addSourceDelegate(src)

                if ibe is not len(backends) - 1:
                    self.addSeparator()

        def addSourceDelegate(self, src):
            def addSource():
                source = src.getRepresentedSource(self.sandbox)
                source.lat = self.sandbox.frame.llLat
                source.lon = self.sandbox.frame.llLon
                source.easting = self.sandbox.frame.Emeter[-1] / 2
                source.northing = self.sandbox.frame.Nmeter[-1] / 2
                source.depth = 4 * km

                if source:
                    self.sandbox.addSource(source)

            action = QtWidgets.QAction(src.display_name, self)
            action.setToolTip(
                '<span style="font-family: monospace;">'
                "%s</span>" % src.__represents__
            )

            action.triggered.connect(addSource)
            self.addAction(action)
            return action

        def addSection(self, text):
            action = QtWidgets.QAction(text, self)
            # action.setSeparator(True)
            font = action.font()
            font.setPointSize(9)
            font.setItalic(True)
            action.setFont(font)
            action.setEnabled(False)
            self.addAction(action)
            return action

    def __init__(self, sandbox, parent=None):
        QtWidgets.QToolButton.__init__(self, parent)

        menu = self.SourcesAddMenu(sandbox, self, "Availables sources")

        self.setText("Add Source")
        self.setMenu(menu)

        self.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView)
        )
        self.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)


class SourcesList(QtWidgets.QListView):
    class SourceItemDelegate(QtWidgets.QStyledItemDelegate):
        def paint(self, painter, option, index):
            options = QtWidgets.QStyleOptionViewItem(option)
            self.initStyleOption(options, index)

            style = (
                QtWidgets.QApplication.style()
                if options.widget is None
                else options.widget.style()
            )

            doc = QtGui.QTextDocument()
            doc.setHtml(options.text)

            options.text = ""
            style.drawControl(QtWidgets.QStyle.CE_ItemViewItem, options, painter)

            ctx = QtGui.QAbstractTextDocumentLayout.PaintContext()

            textRect = style.subElementRect(
                QtWidgets.QStyle.SE_ItemViewItemText, options, options.widget
            )
            painter.save()
            painter.translate(textRect.topLeft())
            painter.setClipRect(textRect.translated(-textRect.topLeft()))
            doc.documentLayout().draw(painter, ctx)

            painter.restore()

        def sizeHint(self, option, index):
            options = QtWidgets.QStyleOptionViewItem(option)
            self.initStyleOption(options, index)

            doc = QtGui.QTextDocument()
            doc.setHtml(options.text)
            doc.setTextWidth(options.rect.width())

            return QtCore.QSize(int(doc.idealWidth()), int(doc.size().height()))

    class SourceContextMenu(QtWidgets.QMenu):
        def __init__(self, parent, idx, *args, **kwargs):
            QtWidgets.QMenu.__init__(self, parent, *args, **kwargs)
            self.parent = parent
            self.sandbox = parent.sandbox
            self.idx = idx

            def removeSource():
                self.sandbox.sources.removeSource(self.idx)

            editAction = self.addAction(
                "Edit", lambda: self.parent.editSource(self.idx)
            )

            self.addMenu(SourcesAddButton.SourcesAddMenu(self.sandbox, self))
            self.addSeparator()

            removeAction = self.addAction(
                self.style().standardIcon(QtWidgets.QStyle.SP_DialogCloseButton),
                "Remove",
                removeSource,
            )

            if self.sandbox.sources.rowCount(QtCore.QModelIndex()) == 0:
                editAction.setEnabled(False)
                removeAction.setEnabled(False)

    def __init__(self, sandbox, *args, **kwargs):
        QtWidgets.QListView.__init__(self, *args, **kwargs)
        self.sandbox = sandbox
        self.setModel(sandbox.sources)
        self.setItemDelegate(self.SourceItemDelegate())
        self.setAlternatingRowColors(True)
        sandbox.sources.setSelectionModel(self.selectionModel())

    def edit(self, idx, trigger, event):
        if (
            trigger == QtWidgets.QAbstractItemView.DoubleClicked
            or trigger == QtWidgets.QAbstractItemView.SelectedClicked
        ):
            self.editSource(idx)
        return False

    @QtCore.pyqtSlot(QtCore.QObject)
    def editSource(self, idx):
        editing_dialog = idx.data(SourceEditorDialog)
        editing_dialog.show()
        editing_dialog.raise_()

    @QtCore.pyqtSlot(object)
    def contextMenuEvent(self, event):
        idx = self.indexAt(event.pos())
        menu = self.SourceContextMenu(self, idx)
        menu.popup(event.globalPos())
