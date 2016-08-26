from PySide import QtGui


class QDoubleSlider(QtGui.QSlider):
    ''' DoublePrecision slider for Qt
    '''
    def __init__(self, *args, **kwargs):
        QtGui.QSlider.__init__(self, *args, **kwargs)

        super(QDoubleSlider, self).setMinimum(0)
        self._max_int = 10000
        super(QDoubleSlider, self).setMaximum(self._max_int)

        self._min_value = 0.0
        self._max_value = 100.0

    @property
    def _value_range(self):
        return self._max_value - self._min_value

    def value(self):
        return float(super(QDoubleSlider, self).value()) \
            / self._max_int * self._value_range

    def setValue(self, value):
        super(QDoubleSlider, self).setValue(int(value /
                                            self._value_range * self._max_int))

    def setMinimum(self, value):
        self.setRange(value, self._max_value)

    def setMaximum(self, value):
        self.setRange(self._min_value, value)

    def setRange(self, minimum, maximum):
        old_value = self.value()
        self._min_value = minimum
        self._max_value = maximum
        self.setValue(old_value)

    def proportion(self):
        return (self.value() - self._min_value) / self._value_range


class QSceneLogger(QtGui.QWidget):
    pass
