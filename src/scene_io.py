import importlib
import numpy as num
import os
import glob
import logging


__all__ = ['Matlab', 'Gamma', 'ISCE']

logger = logging.getLogger(name='SceneIO')


class SceneIO(object):
    """ Prototype class for SARIO objects """
    def __init__(self):
        self.container = {
            'phi': None,
            'theta': None,
            'displacement': None,
            'lllon': None,
            'lllat': None,
            'dlat': None,
            'dlon': None
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

    **Matlab**

    Variable naming conventions for variables in Matlab ``.mat`` file:

    ================== ====================
    Property           Matlab ``.mat`` name
    ================== ====================
    Scene.displacement ``ig_``
    Scene.phi          ``phi``
    Scene.theta        ``theta``
    Scene.utm.x        ``xx``
    Scene.utm.x        ``yy``
    ================== ====================
    """
    def __init__(self):
        self.io = importlib.import_module('scipy.io')
        SceneIO.__init__(self)

    def validate(self, filename):
        try:
            self.io.loadmat(filename)
            return True
        except ValueError:
            return False

    def read(self, filename):
        import utm

        mat = self.io.loadmat(filename)
        for mat_k, v in mat.iteritems():
            for io_k in self.container.iterkeys():
                if io_k in mat_k:
                    self.container[io_k] = mat[mat_k]
                elif 'ig_' in mat_k:
                    self.container['displacement'] = mat[mat_k]
                elif 'xx' in mat_k:
                    utm_e = mat[mat_k].flatten()
                elif 'yy' in mat_k:
                    utm_n = mat[mat_k].flatten()

        utm_zone = 32
        logger.warning('UTM zone for easting/northing not defined, using %d' %
                       utm_zone)
        self.container['lllat'], self.container['lllat'] =\
            utm.to_latlon(utm_e.min(), utm_n.min(), utm_zone)

        urlat, urlon = utm.to_latlon(utm_e.max(), utm_n.max(), utm_zone)
        self.container['dlat'] =\
            (urlat - self.container['lllat']) /\
                self.container['displacement'].shape[0]

        self.container['dlon'] =\
            (urlat - self.container['lllon']) /\
                self.container['displacement'].shape[1]

        return self.container


class Gamma(SceneIO):
    """Reads Gamma binary files

    A ``.par`` file is expected in the import folder
    """
    @staticmethod
    def _getParameterFile(filename):
        path = os.path.dirname(os.path.realpath(filename))
        try:
            return glob.glob('%s/*.gc_par' % path)[0]
        except IndexError:
            raise ImportError('Could not find Gamma parameter file (.gc_par)')

    @staticmethod
    def _parseParameterFile(parameter_file):
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
        logger.info('')
        if parameter_file is None:
            parameter_file = self._getParameterFile(filename)
        par = self._parseParameterFile(parameter_file)

        try:
            par['width'] = int(par['width'])
            par['nlines'] = int(par['nlines'])
        except:
            raise ValueError('Error parsing width and nlines from %s' %
                             parameter_file)

        self.container['displacement'] = num.fromfile(filename, dtype='>f4')

        # Resize array last line is not scanned completely
        _fill = num.empty(par['width'] -
                          self.container['displacement'].size % par['width'])
        _fill.fill(num.nan)
        self.container['displacement'] = num.append(
                          self.container['displacement'], _fill)

        # Reshape displacement vector
        self.container['displacement'] = \
            self.container['displacement'].reshape(par['nlines'], par['width'])
        self.container['displacement'][self.container['displacement']
                                        == -0.] = num.nan

        # LatLon UTM Conversion
        self.container['lllat'] = par['corner_lat'] +\
                                    par['post_lat'] * par['width']
        self.container['lllon'] = par['corner_lon']

        # Theta and Phi
        logger.warning('Using static phi and theta!')
        self.container['theta'] = 0.
        self.container['phi'] = 0.

        return self.container


class GMTSAR(SceneIO):
    def validate(self, filename, **kwargs):
        pass


class ISCEXMLParser(object):
    def __init__(self, filename):
        import xml.etree.ElementTree as ET
        self.root = ET.parse(filename).getroot()

    @staticmethod
    def type_convert(value):
        for t in (float, int, str):
            try:
                return t(value)
            except ValueError:
                continue
        raise ValueError('Could not convert value')

    def getProperty(self, name):
        for child in self.root.iter():
            if child.get('name') == name:
                if child.tag == 'property':
                    return self.type_convert(child.find('value').text)
                elif child.tag == 'component':
                    values = {}
                    for prop in child.iter('property'):
                        values[prop.get('name')] =\
                            self.type_convert(prop.find('value').text)
                    return values
        return None


class ISCE(SceneIO):
    def __init__(self):
        SceneIO.__init__(self)

    def validate(self, filename, **kwargs):
        try:
            self._getDisplacementFile(filename)
            self._getLOSFile(filename)
            return True
        except ImportError:
            return False

    @staticmethod
    def _getLOSFile(path):
        if not os.path.isdir(path):
            path = os.path.abspath(path)
        rdr_files = glob.glob(os.path.join(path, '*.rdr.geo'))

        if len(rdr_files) == 0:
            raise ImportError('Could not find LOS file (.rdr.geo)')
        if not os.path.isfile('%s.xml' % rdr_files[0]):
            raise ImportError('Could not find LOS XML file (.rdr.geo.xml)')
        return rdr_files[0]

    @staticmethod
    def _getDisplacementFile(path):
        if os.path.isfile(path):
            disp_file = path
        else:
            files = glob.glob(os.path.join(path, '*.unw.geo'))
            if len(files) == 0:
                raise ImportError('Could not find displacement file '
                                  '(.unw.geo) at %s', path)
            disp_file = files[0]

        if not os.path.isfile('%s.xml' % disp_file):
            raise ImportError('Could not find displacement XML file '
                              '(%s.unw.geo.xml)' % os.path.basename(disp_file))
        return disp_file

    def read(self, filename, **kwargs):
        isce_xml = ISCEXMLParser(self._getDisplacementFile() + '.xml')

        coordinate1 = isce_xml.getProperty('coordinate1')
        coordinate2 = isce_xml.getProperty('coordinate2')

        self.displacement = num.fromfile(self._getDisplacementFile(filename),
                                         dtype='<f4')
        los_data = num.fromfile(self._getLOSFile(filename), dtype='<f4')


