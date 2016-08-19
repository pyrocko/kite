import importlib


class SceneIO(object):
    """ Prototype class for SARIO objects """

    def __init__(self):
        self.data_items = {
            'phi': None,
            'theta': None,
            'displacement': None,
            'utm_x': None,
            'utm_y': None
        }

    def read(self, filename, **kwargs):
        """Read function of the file format

        :param filename: file to read
        :type filename: string
        :param **kwargs: Keyword arguments
        :type **kwargs: {dict}
        """
        raise NotImplementedError('read not implemented')

    def write(self, filename, **kwargs):
        """Write method for IO

        :param filename: file to write to
        :type filename: string
        :param **kwargs: Keyword arguments
        :type **kwargs: {dict}
        """
        raise NotImplementedError('write not implemented')

    def validate(self, filename, **kwargs):
        """Validate file format

        :param filename: file to validate
        :type filename: string
        :returns: Validation
        :rtype: {bool}
        """
        pass
        raise NotImplementedError('validate not implemented')


class Matlab(SceneIO):
    """Reads Matlab .mat files into :py:class:`kite.scene.Scene`

    Variable naming convenctions in .mat variables
    ============ ==================
    Property     .mat name contains
    ============ ==================
    Displacement `ig_`
    Phi          `phi`
    Theta        `theta`
    UTM_X        `xx`
    UTM_Y        `yy`
    ============ ==================
    """
    def __init__(self):
        self.io = importlib.import_module('scipy.io')
        SceneIO.__init__(self)

    def validate(self, filename):
        try:
            self.io.loadmat(filename)
            return True
        except:
            return False

    def read(self, filename):
        mat = self.io.loadmat(filename)

        for mat_k, v in mat.iteritems():
            for io_k in self.data_items.iterkeys():
                if io_k in mat_k:
                    self.data_items[io_k] = mat[mat_k]
                elif 'ig_' in mat_k:
                    self.data_items['displacement'] = mat[mat_k]
                elif 'xx' in mat_k:
                    self.data_items['utm_x'] = mat[mat_k]
                elif 'yy' in mat_k:
                    self.data_items['utm_y'] = mat[mat_k]

        return self.data_items


class Gamma(SceneIO):
    """Reads Gamma binary files

    [description]
    """
    def _getParameterFile(self, filename):
        import os
        import glob

        path = os.path.dirname(os.path.realpath(filename))
        try:
            return glob.glob('%s/*.gc_par' % path)[0]
        except ValueError:
            raise ImportError('Could not find Gamma parameter file (.gc_par)')

    def _parseParameterFile(self, parameter_file):
        import re

        parameters = {}
        rc = re.compile(r'^(\w*):\s*([a-zA-Z0-9+-.*]*\s[a-zA-Z0-9_]*).*')

        with open(parameter_file, mode='r') as par:
            for line in par:
                parsed = rc.match(line)
                if parsed is None:
                    continue

                groups = parsed.groups()
                try:
                    parameters[groups[0]] = float(groups[1])
                except ValueError:
                    parameters[groups[0]] = groups[1].strip()
        return parameters or None

    def validate(self, filename, parameter_file=None):
        if parameter_file is None:
            parameter_file = self._getParameterFile(filename)
        par = self._parseParameterFile(parameter_file)

        if par is None:
            return False
            raise ValueError('Parameter file %s is empty' % parameter_file)

        return True

    def read(self, filename, parameter_file=None):
        import numpy as num
        import utm

        if parameter_file is None:
            parameter_file = self._getParameterFile(filename)
        par = self._parseParameterFile(parameter_file)

        try:
            par['width'] = int(par['width'])
            par['nlines'] = int(par['nlines'])
        except:
            raise ValueError('Error parsing width and nlines from %s' %
                             parameter_file)

        self.data_items['displacement'] = num.fromfile(filename, dtype='>f4')

        # Resize array last line is not scanned completely
        _fill = num.empty(par['width'] -
                          self.data_items['displacement'].size % par['width'])
        _fill.fill(num.nan)
        self.data_items['displacement'] = num.append(
                          self.data_items['displacement'], _fill)

        # Cast vector
        self.data_items['displacement'] = \
            self.data_items['displacement'].reshape(par['nlines'],
                                                    par['width'])

        # LatLon UTM Conversion
        utm_x, utm_y, utm_zone, _ = utm.from_latlon(par['corner_lat'],
                                                    par['corner_lon'])
        last_x, last_y, _, _ = utm.from_latlon(
                            par['corner_lat'] + par['post_lat']*par['width'],
                            par['corner_lon'] + par['post_lon']*par['nlines'])
        self.data_items['utm_x'] = num.linspace(utm_x, last_x, par['width'])
        self.data_items['utm_y'] = num.linspace(utm_y, last_y, par['nlines'])

        # Theta and Phi
        self.data_items['theta'] = 0.
        self.data_items['phi'] = 0.
        # raise DeprecationWarning('Set Scene.phi and Scene.theta manually!')

        return self.data_items


__all__ = """
Matlab
Gamma
""".split()
