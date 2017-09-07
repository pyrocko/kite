from PySide import QtGui, QtCore
from functools import partial

from .sources import __sources__
from .util import SourceEditorDialog


class SourcesListDock(QtGui.QDockWidget):

    def __init__(self, sandbox, *args, **kwargs):
        QtGui.QDockWidget.__init__(self, 'Sources', *args, **kwargs)
        self.sandbox = sandbox

        layout = QtGui.QVBoxLayout()
        layout.setContentsMargins(3, 3, 3, 3)
        sources = SourcesList(sandbox)
        sources_add_menu = SourcesAddButton(sandbox)

        self.setFeatures(QtGui.QDockWidget.DockWidgetFloatable |
                         QtGui.QDockWidget.DockWidgetMovable)

        self.widget = QtGui.QWidget()
        self.widget.setLayout(layout)
        self.widget.layout().addWidget(sources)
        self.widget.layout().addWidget(sources_add_menu)

        self.setWidget(self.widget)


class SourcesAddButton(QtGui.QToolButton):

    class SourcesAddMenu(QtGui.QMenu):

        def __init__(self, sandbox, *args, **kwargs):
            QtGui.QMenu.__init__(self, *args, **kwargs)
            self.setTitle('Add Source')
            self.sandbox = sandbox

            backends = set(src.display_backend for src in __sources__)

            for ibe, be in enumerate(backends):
                self.addSection(be)

                for src in [s for s in __sources__
                            if s.display_backend == be]:
                    self.addSourceDelegate(src)

                if ibe is not len(backends) - 1:
                    self.addSeparator()

        def addSourceDelegate(self, src):

            def addSource(source):
                if source:
                    self.sandbox.addSource(source)

            action = QtGui.QAction(src.display_name, self)
            action.setToolTip('<span style="font-family: monospace;">'
                              '%s</span>' % src.__represents__)
            action.triggered.connect(
                partial(addSource, src.getRepresentedSource(self.sandbox)))
            self.addAction(action)
            return action

        def addSection(self, text):
            action = QtGui.QAction(text, self)
            # action.setSeparator(True)
            font = action.font()
            font.setPointSize(9)
            font.setItalic(True)
            action.setFont(font)
            action.setEnabled(False)
            self.addAction(action)
            return action

    def __init__(self, sandbox, parent=None):
        QtGui.QToolButton.__init__(self, parent)

        menu = self.SourcesAddMenu(sandbox, self, 'Availables sources')

        self.setText('Add Source')
        self.setMenu(menu)

        self.setIcon(self.style().standardPixmap(
                     QtGui.QStyle.SP_FileDialogDetailedView))
        self.setPopupMode(QtGui.QToolButton.InstantPopup)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)


class SourcesList(QtGui.QListView):

    class SourceItemDelegate(QtGui.QStyledItemDelegate):

        def paint(self, painter, option, index):
            options = QtGui.QStyleOptionViewItemV4(option)
            self.initStyleOption(options, index)

            style = QtGui.QApplication.style() if options.widget is None\
                else options.widget.style()

            doc = QtGui.QTextDocument()
            doc.setHtml(options.text)

            options.text = ""
            style.drawControl(QtGui.QStyle.CE_ItemViewItem, options, painter)

            ctx = QtGui.QAbstractTextDocumentLayout.PaintContext()

            textRect = style.subElementRect(
                QtGui.QStyle.SE_ItemViewItemText, options, options.widget)
            painter.save()
            painter.translate(textRect.topLeft())
            painter.setClipRect(textRect.translated(-textRect.topLeft()))
            doc.documentLayout().draw(painter, ctx)

            painter.restore()

        def sizeHint(self, option, index):
            options = QtGui.QStyleOptionViewItemV4(option)
            self.initStyleOption(options, index)

            doc = QtGui.QTextDocument()
            doc.setHtml(options.text)
            doc.setTextWidth(options.rect.width())

            return QtCore.QSize(doc.idealWidth(), doc.size().height())

    class SourceContextMenu(QtGui.QMenu):

        def __init__(self, sandbox, idx, *args, **kwargs):
            QtGui.QMenu.__init__(self, *args, **kwargs)
            self.sandbox = sandbox
            self.idx = idx

            def removeSource():
                self.sandbox.sources.removeSource(self.idx)

            def editSource():
                editing_dialog = self.sandbox.sources.data(
                    self.idx, SourceEditorDialog)
                editing_dialog.show()

            editAction = self.addAction(
                'Edit', editSource)

            self.addMenu(
                SourcesAddButton.SourcesAddMenu(self.sandbox, self))
            self.addSeparator()

            removeAction = self.addAction(
                self.style().standardPixmap(
                    QtGui.QStyle.SP_DialogCloseButton),
                'Remove', removeSource)

            if self.sandbox.sources.rowCount(QtCore.QModelIndex()) == 0:
                editAction.setEnabled(False)
                removeAction.setEnabled(False)

    def __init__(self, sandbox, *args, **kwargs):
        QtGui.QListView.__init__(self, *args, **kwargs)
        self.sandbox = sandbox
        self.setModel(sandbox.sources)
        self.setItemDelegate(self.SourceItemDelegate())
        self.setAlternatingRowColors(True)
        sandbox.sources.setSelectionModel(self.selectionModel())

    def edit(self, idx, trigger, event):
        if trigger == QtGui.QAbstractItemView.EditTrigger.DoubleClicked or\
          trigger == QtGui.QAbstractItemView.EditTrigger.SelectedClicked:
            editing_dialog = idx.data(SourceEditorDialog)
            editing_dialog.show()
        return False

    @QtCore.Slot()
    def contextMenuEvent(self, event):
        idx = self.indexAt(event.pos())
        menu = self.SourceContextMenu(self.sandbox, idx, self)
        menu.popup(event.globalPos())
