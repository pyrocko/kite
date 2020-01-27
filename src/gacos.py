import logging
import re
import os.path as op
import numpy as num
from datetime import datetime, timedelta

from pyrocko.guts import Object, String, Bool, List


def load_params(filename):
    rc = re.compile(r'([\w]*)\s*([\w.+-]*)')
    params = {}

    with open(filename, mode='r') as par:
        for line in par:
            parsed = rc.match(line)
            if parsed is None:
                continue

            groups = parsed.groups()
            params[groups[0]] = groups[1].strip()

    return params


class GACOSGrid(object):

    def __init__(self, filename, time, data,
                 ulLat, ulLon, dLat, dLon):
        self.filename = filename
        self.time = time

        self.data = data
        self.rows, self.cols = data.shape

        self.dLat = dLat
        self.dLon = dLon

        self.ulLat = ulLat
        self.ulLon = ulLon

        self.llLat = self.ulLat + self.rows * self.dLat
        self.llLon = ulLon

        self.urLon = ulLon + self.cols * self.dLon

    def get_corrections(self, llLat, llLon, dLat, dLon, rows, cols):
        ulLat = llLat + dLat * rows
        ulLon = llLon
        urLon = llLon + dLon * cols

        assert llLat >= self.llLat
        assert llLon >= self.llLon

        assert urLon <= self.urLon
        assert ulLat <= self.ulLat

        row_offset = (self.ulLat - ulLat) // -self.dLat
        col_offset = (ulLon - self.llLon) // self.dLon

        idx_rows = row_offset + (num.arange(rows) * dLat // -self.dLat)
        idx_cols = col_offset + (num.arange(cols) * dLon // self.dLon)

        idx_rows = num.repeat(idx_rows, cols)
        idx_cols = num.tile(idx_cols, rows)

        idx = num.c_[idx_rows.astype(num.intp), idx_cols.astype(num.intp)]
        return self.data[idx[:, 0], idx[:, 1]].reshape(rows, cols)

    @classmethod
    def load(cls, filename):
        rsc_file = filename + '.rsc'
        if not op.exists(filename) and not op.exists(rsc_file):
            raise FileNotFoundError('Could not find %s or .rsc file %s'
                                    % (filename, rsc_file))

        params = load_params(rsc_file)

        time = datetime.strptime(op.basename(filename)[:8], '%Y%m%d')
        hour = timedelta(hours=float(params['TIME_OF_DAY'].rstrip('UTC')))
        time += hour

        rows = int(params['FILE_LENGTH'])
        cols = int(params['WIDTH'])

        ulLat = float(params['Y_FIRST'])
        ulLon = float(params['X_FIRST'])

        dLat = float(params['Y_STEP'])
        dLon = float(params['X_STEP'])

        data = num.memmap(
            filename, dtype=num.float32, mode='r', shape=(rows, cols))

        return cls(filename, time, data, ulLat, ulLon, dLat, dLon)


class GACOSConfig(Object):
    grd_filename = List.T(String.T, default=[])
    applied = Bool.T(default=False)


class GACOSCorrection(object):

    def __init__(self, scene=None, config=None):
        self.config = config or GACOSConfig()

        if scene:
            self._log = scene._log.getChild('GACOSCorrection')
        else:
            self._log = logging.getLogger('GACOSCorrection')

        self.grids = []

        for filename in self.config.grd_filename:
            self.load(filename)

    def is_applied(self):
        return self.config.applied

    def load(self, filename):
        if len(self.grids) > 2:
            raise AttributeError('We already loaded two GACOS grids!')

        filename = op.abspath(filename)
        if not op.exists(filename):
            raise OSError('cannot find GACOS grid %s' % filename)

        self._log.info('Loading %s', filename)
        grid = GACOSGrid.load(filename)
        self.grids.append(grid)
        self.config.append(filename)

    def get_correction(self):
        if self.grids != 2:
            raise AttributeError('We need to load two GACOS grids to apply!')

        frame = self.scene.frame
        grids = sorted(self.grids, key=lambda grd: grd.time)

        extent = {
            'llLat': frame.llLat,
            'llLon': frame.llLon,
            'dLat': frame.dLat,
            'dLon': frame.dLon,
            'cols': frame.cols,
            'rows': frame.rows
        }

        corr_date1 = grids[0].get_corrections(**extent)
        corr_date2 = grids[1].get_corrections(**extent)

        return corr_date2 - corr_date1

    def apply_model(self):
        if self.is_applied():
            self._log.warning('GACOS correction already applied!')
            return

        self._log.info('Applying GACOS model to displacement')
        correction = self.get_correction()
        self.scene.displacement -= correction
        self.config.applied = True

        self.scene.evChanged()

    def remove_model(self):
        if not self.is_applied():
            self._log.warning('GACOS correction is not applied!')
            return

        self._log.info('Removing GACOS model from displacement')
        correction = self.get_correction()
        self.scene.displacement += correction
        self.config.applied = False

        self.scene.evChanged()
