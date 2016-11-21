#!/usr/bin/python2
from PySide import QtGui
from PySide import QtCore

import numpy as num

import pyqtgraph as pg
from pyqtgraph import dockarea

__all__ = ['QKiteDock', 'QKitePlot',
           'QKiteToolComponents', 'QKiteToolColormap']


class QKiteDock(dockarea.DockArea):
    def __init__(self, container):
        dockarea.DockArea.__init__(self)

        main_widget = self.main_widget(container)
        main_dock = dockarea.Dock(self.title,
                                  autoOrientation=False,
                                  widget=main_widget)

        histogram_dock = dockarea.Dock('Colormap',
                                       autoOrientation=False,
                                       widget=QKiteToolColormap(main_widget))
        histogram_dock.setStretch(.3, None)

        for i, (name, tool) in enumerate(self.tools.iteritems()):
            tool_dock = dockarea.Dock(name, widget=tool(main_widget),
                                      size=(2, 3),
                                      autoOrientation=False)

            self.addDock(tool_dock, position='bottom')

        self.addDock(main_dock, position='left')
        self.addDock(histogram_dock, position='left')


class QKitePlot(pg.PlotWidget):
    def __init__(self, container):
        pg.PlotWidget.__init__(self)

        self.container = container

        self.image = pg.ImageItem(None)

        self.setAspectLocked(True)
        self.plotItem.getAxis('left').setZValue(100)
        self.plotItem.getAxis('bottom').setZValue(100)
        self.setLabels(bottom={'East', 'm'},
                       left={'North', 'm'},)

        self.hint_text = pg.LabelItem(text='East %d m | North %d m | %s %.2f'
                                      % (0, 0, self.component.title(), 0),
                                      justify='right', size='8pt',
                                      parent=self.plotItem)
        self.hint_text.anchor(itemPos=(1., 0.), parentPos=(1, 0))
        font = self.hint_text.font()
        font.setBold(True)
        self.addItem(self.image)
        self.update()

        self.transformToUTM()
        self._move_sig = pg.SignalProxy(self.image.scene().sigMouseMoved,
                                        rateLimit=30, slot=self.mouseMoved)
        # self.addIsocurve()
        # self.scalebar()

    def transformToUTM(self):
        padding = 100
        ll_x, ll_y, ur_x, ur_y, dx, dy = \
            (self.container.frame.llE, self.container.frame.llN,
             self.container.frame.urE, self.container.frame.urN,
             self.container.frame.dE, self.container.frame.dN)

        self.image.translate(ll_x, ll_y)
        self.image.scale(dx, dy)
        self.setLimits(xMin=ll_x-dx*padding,
                       xMax=ur_x+dx*padding,
                       yMin=ll_y-dy*padding,
                       yMax=ur_y+dy*padding)

    def scalebar(self):
        ''' Not working '''
        self.scale_bar = pg.ScaleBar(10, width=5, suffix='m')
        self.scale_bar.setParentItem(self.plotItem)
        self.scale_bar.anchor((1, 1), (1, 1), offset=(-20, -20))

    @property
    def component(self):
        return self._component

    @component.setter
    def component(self, component):
        if component not in self.components_available.keys():
            raise AttributeError('Invalid component %s' % component)
        self._component = component
        self.update()

    @property
    def data(self):
        _data = self.components_available[self.component][1](self.container)
        return _data  # num.nan_to_num(_data)

    def update(self):
        self.image.updateImage(self.data.T, autoDownsample=True)
        self.mouseMoved()
        # self.addIsocurves()

    def addIsocurve(self, level=0.):
        iso = pg.IsocurveItem(level=level, pen='g')
        iso.setZValue(1000)
        iso.setData(pg.gaussianFilter(self.data, (5, 5)))
        iso.setParentItem(self.image)

        self.iso = iso

    def mouseMoved(self, event=None):
        if event is None:
            self.hint_text.setText('East %d m | North %d m | %s %.2f'
                                   % (0, 0, self.component.title(), 0))
            return
        pos = event[0]
        if self.image.sceneBoundingRect().contains(pos):
            map_pos = self.plotItem.vb.mapSceneToView(pos)
            if map_pos.isNull():
                return
            img_pos = self.image.mapFromScene(event).data
            z = self.image.image[int(img_pos().x()),
                                 int(img_pos().y())]
            if self.component == 'displacement':
                z *= 1e2
            text = '<span style="font-family: monospace; color: #fff">' \
                'East {east:08.2f} m | North {north:08.2f} m | '\
                '{measure} {z:05.1f}</span>'.format(
                    north=map_pos.x(), east=map_pos.y(),
                    measure=self.component.title(),
                    z=z)
            self.hint_text.setText(text)
            return


class QKiteToolComponents(QtGui.QWidget):
    def __init__(self, plot=None):
        QtGui.QWidget.__init__(self)
        self.plot = plot

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.getComponentsGroup())
        self.layout.addWidget(self.getInfoPanel())
        self.layout.addStretch(3)

    def getComponentsGroup(self):
        from functools import partial

        layout = QtGui.QVBoxLayout()
        self._btn_grp = QtGui.QButtonGroup()
        self._btn_grp.setExclusive(True)

        def changeComponent(component):
            self.plot.component = component

        components = self.plot.components_available.keys()
        components.sort()
        for comp in components:
            btn = QtGui.QPushButton(self)

            layout.addWidget(btn)
            self._btn_grp.addButton(btn)

            btn.setText(self.plot.components_available[comp][0])
            btn.setToolTip('Scene.%s' % comp)
            btn.setCheckable(True)
            btn.setChecked(comp == self.plot.component)

            btn.clicked_str = QtCore.Signal(str)
            btn.clicked.connect(partial(changeComponent, comp))

        layout.addStretch(3)

        group = QtGui.QGroupBox('Scene Components')
        group.setLayout(layout)

        return group

    def getInfoPanel(self):
        layout = QtGui.QVBoxLayout()
        info_text = QtGui.QLabel()

        def updateInfoText():
            table_content = [
                ('Component', '<b>%s</b>' %
                 self.plot.components_available[self.plot.component][0]),
                ('Max value', '%0.4f' % num.nanmax(self.plot.data)),
                ('Min value', '%0.4f' % num.nanmin(self.plot.data)),
                ('Mean value', '%0.4f' % num.nanmean(self.plot.data)),
                ('Resolution px', '%d x %d' % (self.plot.data.shape[0],
                                               self.plot.data.shape[1])),
                ('dx', '%.2f m' % self.plot.container.frame.dE),
                ('dy', '%.2f m' % self.plot.container.frame.dN),
                ]
            rstr = '<table>'
            for (metric, value) in table_content:
                rstr += '<tr><td style="padding-right: 15px">%s:</td>' \
                        '<td>%s</td></tr>' % (metric, value)
            rstr += '</table>'

            info_text.setText(rstr)

        updateInfoText()
        self.plot.image.sigImageChanged.connect(updateInfoText)

        layout.addWidget(info_text)
        group = QtGui.QGroupBox('Scene Information')
        group.setLayout(layout)

        return group


class QKiteToolColormap(pg.HistogramLUTWidget):
    def __init__(self, plot):
        pg.HistogramLUTWidget.__init__(self, image=plot.image)

        self._plot = plot

        _zero_marker = pg.InfiniteLine(pos=0, angle=0, pen='w', movable=False)
        _zero_marker.setValue(0.)
        _zero_marker.setZValue(1000)
        self.vb.addItem(_zero_marker)

        self.axis.setLabel('Displacement / m')
        # self.plot.rotate(-90)
        # self.layout.rotate(90)
        # self.gradient.setOrientation('bottom')
        self.setSymColormap()
        self._plot.image.sigImageChanged.connect(self.setSymColormap)
        # self.isoCurveControl()

    def setSymColormap(self):
        default_cmap = {'ticks':
                        [[0., (0, 0, 0, 255)],
                         [1e-3, (106, 0, 31, 255)],
                         [.5, (255, 255, 255, 255)],
                         [1., (8, 54, 104, 255)]],
                        'mode': 'rgb'}
        default_cmap = {'ticks':
                        [[0., (0, 0, 0, 255)],
                         [1e-3, (51, 53, 120)],
                         [.5, (255, 255, 255, 255)],
                         [1., (172, 56, 56)]],
                        'mode': 'rgb'}
        lvl_min = num.nanmin(self._plot.data)
        lvl_max = num.nanmax(self._plot.data)
        abs_range = max(abs(lvl_min), abs(lvl_max))

        self.gradient.restoreState(default_cmap)
        self.setLevels(-abs_range, abs_range)

    def isoCurveControl(self):
        iso_ctrl = pg.InfiniteLine(pos=0, angle=0, pen='g', movable=True)
        iso_ctrl.setValue(0.)
        iso_ctrl.setZValue(1000)

        def isolineChange():
            self._plot.iso.setLevel(iso_ctrl.value())

        iso_ctrl.sigDragged.connect(isolineChange)
        self.vb.addItem(iso_ctrl)
