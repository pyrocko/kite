from PySide import QtGui, QtCore


class SourcesList(QtGui.QListView):

    def __init__(self, sandbox, *args, **kwargs):
        QtGui.QListView.__init__(self, *args, **kwargs)
        self.sandbox = sandbox
        self.setModel(sandbox.sources)
        self.setItemDelegate(SourceItemDelegate())

        sandbox.sources.setSelectionModel(self.selectionModel())


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

        # Highlighting text if item is selected
        # if (optionV4.state & QStyle::State_Selected)
            # ctx.palette.setColor(QPalette::Text, optionV4.palette.color(QPalette::Active, QPalette::HighlightedText));

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
