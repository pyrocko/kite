import os
import glob
import scipy.io
import numpy as num

__all__ = ['Gamma', 'Matlab', 'ISCE', 'GMTSAR']


class SceneIO(object):
    """ Prototype class for SARIO objects """
    def __init__(self, scene=None):
        if scene is not None:
            self._log = scene._log.getChild('IO/%s' % self.__class__.__name__)
        else:
            import logging
            self._log = logging.getLogger('SceneIO/%s'
                                          % self.__class__.__name__)

        self.container = {
            'phi': 0.,    # Look incident angle from vertical in degree
            'theta': 0.,  # Look orientation angle from east; 0deg East,
                            # 90deg North
            'displacement': None,  # Displacement towards LOS
            'frame': {
                'llLon': None,  # Lower left corner latitude
                'llLat': None,  # Lower left corner londgitude
                'dLat': None,   # Pixel delta latitude
                'dLon': None,   # Pixel delta longitude
            },
            # Meta information
            'meta': {
                'title': None,
                'orbit_direction': None,
                'satellite_name': None,
                'time_master': None,
                'time_slave': None
            },
            # All extra information
            'extra': {}
        }

    def read(self, filename, **kwargs):
        """ Read function of the file format

        :param filename: file to read
        :type filename: string
        :param **kwargs: Keyword arguments
        :type **kwargs: {dict}
        """
        raise NotImplementedError('read not implemented')

    def write(self, filename, **kwargs):
        """ Write method for IO

        :param filename: file to write to
        :type filename: string
        :param **kwargs: Keyword arguments
        :type **kwargs: {dict}
        """
        raise NotImplementedError('write not implemented')

    def validate(self, filename, **kwargs):
        """ Validate file format

        :param filename: file to validate
        :type filename: string
        :returns: Validation
        :rtype: {bool}
        """
        pass
        raise NotImplementedError('validate not implemented')


class Matlab(SceneIO):
    """
    Variable naming conventions for Matlab :file:`.mat` container:

        ================== ====================
        Property           Matlab ``.mat`` name
        ================== ====================
        Scene.displacement ``ig_``
        Scene.phi          ``phi``
        Scene.theta        ``theta``
        Scene.frame.x      ``xx``
        Scene.frame.y      ``yy``
        ================== ====================
    """
    def validate(self, filename, **kwargs):
        if filename[-4:] == '.mat':
            return True
        else:
            return False
        try:
            variables = self.io.whosmat(filename)
            if len(variables) > 50:
                return False
            return True
        except ValueError:
            return False

    def read(self, filename, **kwargs):
        import utm
        c = self.container

        mat = scipy.io.loadmat(filename)

        utm_e = utm_n = None

        for mat_k, v in mat.iteritems():
            for io_k in c.iterkeys():
                if io_k in mat_k:
                    c[io_k] = mat[mat_k]
                elif 'ig_' in mat_k:
                    c['displacement'] = mat[mat_k]
                elif 'xx' in mat_k:
                    utm_e = mat[mat_k].flatten()
                elif 'yy' in mat_k:
                    utm_n = mat[mat_k].flatten()

        if not (num.all(utm_e) or num.all(utm_n)):
            self._log.warning(
                'Could not find referencing UTM vectors in .mat file')
            utm_e = num.linspace(100000, 110000, c['displacement'].shape[0])
            utm_n = num.linspace(1100000, 1110000, c['displacement'].shape[1])

        if utm_e.min() < 1e4 or utm_n.min() < 1e4:
            utm_e *= 1e3
            utm_n *= 1e3
        utm_zone = 32
        utm_zone_letter = 'N'
        try:
            c['frame']['llLat'], c['frame']['llLon'] =\
                utm.to_latlon(utm_e.min(), utm_n.min(),
                              utm_zone, utm_zone_letter)
            urlat, urlon = utm.to_latlon(utm_e.max(), utm_n.max(),
                                         utm_zone, utm_zone_letter)
            c['frame']['dLat'] =\
                (urlat - c['frame']['llLat']) /\
                c['displacement'].shape[0]

            c['frame']['dLon'] =\
                (urlon - c['frame']['llLon']) /\
                c['displacement'].shape[1]
        except utm.error.OutOfRangeError:
            c['frame']['llLat'], c['frame']['llLon'] = (0., 0.)
            c['frame']['dLat'] = (utm_e[1] - utm_e[0]) / 110e3
            c['frame']['dLon'] = (utm_n[1] - utm_n[0]) / 110e3
            self._log.warning('Could not interpret spatial vectors, '
                              'referencing to 0, 0 (lat, lon)')
        return c


class Gamma(SceneIO):
    """
    .. warning :: Data has to be georeferenced to latitude/longitude!

    Expects two files:

        * Binary file from gamma (:file:`*`)
        * Parameter file (:file:`*par`), including ``corner_lat, corner_lon,
          nlines, width, post_lat, post_lon``
    """
    def _getParameterFile(self, path):
        path = os.path.dirname(os.path.realpath(path))
        par_files = glob.glob('%s/*par' % path)

        for file in par_files:
            try:
                self._parseParameterFile(file)
                self._log.debug('Found parameter file %s' % file)
                return file
            except ImportError:
                continue
        raise ImportError('Could not find suiting Gamma parameter file (*par)')

    @staticmethod
    def _parseParameterFile(par_file):
        import re

        params = {}
        required = ['corner_lat', 'corner_lon', 'nlines', 'width',
                    'post_lat', 'post_lon']
        rc = re.compile(r'^(\w*):\s*([a-zA-Z0-9+-.*]*\s[a-zA-Z0-9_]*).*')

        with open(par_file, mode='r') as par:
            for line in par:
                parsed = rc.match(line)
                if parsed is None:
                    continue

                groups = parsed.groups()
                try:
                    params[groups[0]] = float(groups[1])
                except ValueError:
                    params[groups[0]] = groups[1].strip()

        for r in required:
            if r not in params:
                raise ImportError(
                    'Parameter file does not hold required parameter %s' % r)

        return params

    def validate(self, filename, **kwargs):
        try:
            par_file = kwargs.pop('par_file',
                                  self._getParameterFile(filename))
            self._parseParameterFile(par_file)
            return True
        except ImportError:
            return False

    def _getAngle(self, filename, pattern):
        path = os.path.dirname(os.path.realpath(filename))
        phi_files = glob.glob('%s/%s' % (path, pattern))
        if len(phi_files) == 0:
            self._log.warning('Could not find %s file, defaulting to 0.'
                              % pattern)
            return 0.
        elif len(phi_files) > 1:
            self._log.warning('Found multiple %s files, defaulting to 0.'
                              % pattern)
            return 0.

        filename = phi_files[0]
        self._log.debug('Found %s in %s' % (pattern, filename))
        return num.memmap(filename, mode='r', dtype='>f4')

    def read(self, filename, **kwargs):
        """
        :param filename: Gamma software binary file
        :type filename: str
        :param par_file: Corresponding parameter (:file:`*par`) file.
                         (optional)
        :type par_file: str
        :returns: Import dictionary
        :rtype: dict
        :raises: ImportError
        """
        par_file = kwargs.pop('par_file',
                              self._getParameterFile(filename))
        par = self._parseParameterFile(par_file)
        fill = None

        try:
            nrows = int(par['width'])
            nlines = int(par['nlines'])
        except:
            raise ImportError('Error parsing width and nlines from %s' %
                              par_file)

        displ = num.fromfile(filename, dtype='>f4')
        # Resize array if last line is not scanned completely
        if (displ.size % nrows) != 0:
            fill = num.empty(nrows - displ.size % nrows)
            fill.fill(num.nan)
            displ = num.append(displ, fill)

        displ = displ.reshape(nlines, nrows)
        displ[displ == -0.] = num.nan

        phi = self._getAngle(filename, '*phi*')
        theta = self._getAngle(filename, '*theta*')
        theta = num.cos(theta)

        if fill is not None:
            theta = num.append(theta, fill)
            phi = num.append(phi, fill)

        theta = theta.reshape(nlines, nrows)
        phi = phi.reshape(nlines, nrows)

        # LatLon UTM Conversion
        c = self.container
        c['displacement'] = displ*1e-2
        c['frame']['llLat'] = par['corner_lat'] + par['post_lat'] * nrows
        c['frame']['llLon'] = par['corner_lon']
        c['frame']['dLon'] = par['post_lon']
        c['frame']['dLat'] = par['post_lat']
        c['theta'] = theta
        c['phi'] = phi

        c['meta']['title'] = par.get('title', None)
        c['bin_file'] = filename
        c['par_file'] = par_file
        return self.container


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
    """
    Expects three files in the same folder:

        * Unwrapped displacement binary (:file:`*.unw.geo`)
        * Metadata XML (:file:`*.unw.geo.xml`)
        * LOS binary data (:file:`*.rdr.geo`)
    """
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
            path = os.path.dirname(path)
        rdr_files = glob.glob(os.path.join(path, '*.rdr.geo'))

        if len(rdr_files) == 0:
            raise ImportError('Could not find LOS file (.rdr.geo)')
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

    def read(self, path, **kwargs):
        path = os.path.abspath(path)
        c = self.container
        isce_xml = ISCEXMLParser(self._getDisplacementFile(path) + '.xml')

        coord_lon = isce_xml.getProperty('coordinate1')
        coord_lat = isce_xml.getProperty('coordinate2')
        c['frame']['dLat'] = num.abs(coord_lat['delta'])
        c['frame']['dLon'] = num.abs(coord_lon['delta'])
        nlon = int(coord_lon['size'])
        nlat = int(coord_lat['size'])

        c['frame']['llLat'] = coord_lat['startingvalue'] +\
            (nlat * coord_lat['delta'])
        c['frame']['llLon'] = coord_lon['startingvalue']

        displacement = num.memmap(self._getDisplacementFile(path),
                                  dtype='<f4')\
            .reshape(nlat, nlon*2)[:, nlon:]
        displacement[displacement == 0.] = num.nan
        c['displacement'] = displacement*1e-2

        los_data = num.fromfile(self._getLOSFile(path), dtype='<f4')\
            .reshape(nlat, nlon*2)
        c['phi'] = los_data[:, :nlon]
        c['theta'] = los_data[:, nlon:] + 90.

        return c


class GMTSAR(SceneIO):
    """
    Use gmt5sar ``SAT_look`` to calculate the corresponding unit look vectors:

    .. code-block:: sh

        gmt grd2xyz unwrap_ll.grd | gmt grdtrack -Gdem.grd |
        awk {'print $1, $2, $4'} |
        SAT_look 20050731.PRM -bos > 20050731.los.enu

    Expects two binary files:

        * Displacement grid (NetCDF, :file:`*los_ll.grd`)
        * LOS binary data (see instruction, :file:`*los.enu`)

    """
    def validate(self, filename, **kwargs):
        try:
            if self._getDisplacementFile(filename)[-4:] == '.grd':
                return True
        except ImportError:
            return False
        return False

    def _getLOSFile(self, path):
        if not os.path.isdir(path):
            path = os.path.dirname(path)
        los_files = glob.glob(os.path.join(path, '*.los.*'))
        if len(los_files) == 0:
            self._log.warning(GMTSAR.__doc__)
            raise ImportError('Could not find LOS file (*.los.*)')
        return los_files[0]

    @staticmethod
    def _getDisplacementFile(path):
        if os.path.isfile(path):
            return path
        else:
            files = glob.glob(os.path.join(path, '*.grd'))
            if len(files) == 0:
                raise ImportError('Could not find displacement file '
                                  '(*.grd) at %s', path)
            disp_file = files[0]
        return disp_file

    def read(self, path, **kwargs):
        from scipy.io import netcdf
        path = os.path.abspath(path)
        c = self.container

        grd = netcdf.netcdf_file(self._getDisplacementFile(path),
                                 mode='r', version=2)
        c['displacement'] = grd.variables['z'][:].copy()
        c['displacement'] *= 1e-2
        shape = c['displacement'].shape
        # LatLon
        c['frame']['llLat'] = grd.variables['lat'][:].min()
        c['frame']['llLon'] = grd.variables['lon'][:].min()

        c['frame']['dLat'] = (grd.variables['lat'][:].max() -
                              c['frame']['llLat'])/shape[1]
        c['frame']['dLon'] = (grd.variables['lon'][:].max() -
                              c['frame']['llLon'])/shape[0]

        # Theta and Phi
        try:
            los = num.memmap(self._getLOSFile(path), dtype='<f4')
            e = los[3::6].copy().reshape(shape)
            n = los[4::6].copy().reshape(shape)
            u = los[5::6].copy().reshape(shape)

            theta = num.rad2deg(num.arctan(n/e))
            phi = num.rad2deg(num.arccos(u))
            theta[n < 0] += 180.

            c['phi'] = phi
            c['theta'] = theta
        except ImportError:
            self._log.warning('Defaulting theta and phi to 0')
            c['theta'] = 0.
            c['phi'] = 0.

        return c
