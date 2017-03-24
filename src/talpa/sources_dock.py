from PySide import QtGui, QtCore
from .common import SourceEditorDialog


class SourcesList(QtGui.QListView):

    def __init__(self, sandbox, *args, **kwargs):
        QtGui.QListView.__init__(self, *args, **kwargs)
        self.sandbox = sandbox
        self.setModel(sandbox.sources)
        self.setItemDelegate(SourceItemDelegate())
        self.setAlternatingRowColors(True)
        sandbox.sources.setSelectionModel(self.selectionModel())

        self.setEditTriggers(
            QtGui.QAbstractItemView.EditTrigger.DoubleClicked |
            QtGui.QAbstractItemView.EditTrigger.SelectedClicked)

    def edit(self, idx, trigger, event):
        if trigger == QtGui.QAbstractItemView.EditTrigger.DoubleClicked or\
          trigger == QtGui.QAbstractItemView.EditTrigger.SelectedClicked:
            editing_dialog = idx.data(SourceEditorDialog)
            editing_dialog.show()
        return False


class SourcesListDock(QtGui.QDockWidget):

    def __init__(self, sandbox, *args, **kwargs):
        QtGui.QDockWidget.__init__(self, 'Sources', *args, **kwargs)
        self.sandbox = sandbox

        self.list = SourcesList(sandbox)
        self.setWidget(self.list)


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
