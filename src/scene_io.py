import os
import glob
import scipy.io
import numpy as num

__all__ = ['Gamma', 'Matlab', 'ISCE', 'GMTSAR', 'ROI_PAC']


def check_required(required, params):
    for r in required:
        if r not in params:
            return False
    return True


def safe_cast(val, to_type, default=None):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default


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
                'wavelength': None,
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
                    c[io_k] = num.rot90(mat[mat_k])
                elif 'ig_' in mat_k:
                    c['displacement'] = num.rot90(mat[mat_k])
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
        utm_zone = 47
        utm_zone_letter = 'Q'
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
    .. warning :: Data has to be georeferenced to latitude/longitude or UTM!

    Expects two files:

        * Binary file from Gamma (:file:`*`)
        * Parameter file (:file:`*par`) describing `corner_lat, corner_lon,
          nlines, width, post_lat, post_lon`
          or `'post_north', 'post_east', 'corner_east',
          'corner_north', 'nlines', 'width'`
        * when `radar_frequency` in [Hz] is given in the :file:`*.par` file,
          the displacement is expected to be in radians and will be scaled.
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
        required_utm = ['post_north', 'post_east', 'corner_east',
                        'corner_north', 'nlines', 'width']
        required_lat_lon = ['corner_lat', 'corner_lon', 'nlines',
                            'width', 'post_lat', 'post_lon']
        rc = re.compile(r'^(\w*):\s*([a-zA-Z0-9+-.*]*\s[a-zA-Z0-9_]*).*')

        with open(par_file, mode='r') as par:
            for line in par:
                parsed = rc.match(line)
                if parsed is None:
                    continue

                groups = parsed.groups()
                params[groups[0]] = safe_cast(groups[1], float,
                                              default=groups[1].strip())

        if check_required(required_utm, params)\
           or check_required(required_lat_lon, params):
            return params

        raise ImportError(
                    'Parameter file does not hold required parameters')

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
        self._log.info('Found %s in %s' % (pattern, filename))
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

        nrows = int(par['width'])
        nlines = int(par['nlines'])
        radar_frequency = par.get('radar_frequency', None)

        displ = num.fromfile(filename, dtype='>f4')
        # Resize array if last line is not scanned completely
        if (displ.size % nrows) != 0:
            fill = num.empty(nrows - displ.size % nrows)
            fill.fill(num.nan)
            displ = num.append(displ, fill)

        displ = displ.reshape(nlines, nrows)
        displ[displ == -0.] = num.nan
        displ = num.fliplr(displ)

        phi = self._getAngle(filename, '*phi*')
        theta = self._getAngle(filename, '*theta*')
        theta = num.cos(theta)

        if isinstance(phi, num.ndarray):
            phi.reshape(nlines, nrows)
        if isinstance(theta, num.ndarray):
            theta.reshape(nlines, nrows)

        if fill is not None:
            theta = num.append(theta, fill)
            phi = num.append(phi, fill)

        c = self.container

        if radar_frequency is not None:
            self._log.info('Scaling radian displacement by radar_frequency')
            wavelength = 299792458. / radar_frequency
            displ = (displ / (4.*num.pi)) * wavelength
            c['meta']['wavelength'] = wavelength

        c['displacement'] = displ
        c['theta'] = theta
        c['phi'] = phi

        c['meta']['title'] = par.get('title', 'None')
        c['bin_file'] = filename
        c['par_file'] = par_file

        if par['DEM_projection'] == 'UTM':
            self._log.info('Parameter file provides UTM reference')
            import utm

            utm_zone = par['projection_zone']
            try:
                utm_zone_letter = utm.latitude_to_zone_letter(
                    par['center_latitude'])
            except ValueError:
                self._log.warning('Could not parse UTM Zone letter,'
                                  ' defaulting to N!')
                utm_zone_letter = 'N'

            utm_e = utm_n = None
            dN = par['post_north']
            dE = par['post_east']
            utm_corn_e = par['corner_east']
            utm_corn_n = par['corner_north']

            utm_corn_eo = utm_corn_e + dE * displ.shape[1]
            utm_corn_no = utm_corn_n + dN * displ.shape[0]

            utm_e = num.linspace(utm_corn_e, utm_corn_eo, displ.shape[1])
            utm_n = num.linspace(utm_corn_n, utm_corn_no, displ.shape[0])

            c['frame']['llLat'], c['frame']['llLon'] =\
                utm.to_latlon(utm_e.min(), utm_n.min(),
                              utm_zone, utm_zone_letter)
            urlat, urlon = utm.to_latlon(utm_e.max(), utm_n.max(),
                                         utm_zone, utm_zone_letter)
            c['frame']['dLat'] =\
                (urlat - c['frame']['llLat']) / displ.shape[0]

            c['frame']['dLon'] =\
                (urlon - c['frame']['llLon']) / displ.shape[1]
        else:
            self._log.info('Parameter file provides Lat/Lon reference')
            c['frame']['llLat'] = par['corner_lat'] + par['post_lat'] * nrows
            c['frame']['llLon'] = par['corner_lon']
            c['frame']['dLon'] = par['post_lon']
            c['frame']['dLat'] = par['post_lat']
        return self.container


class ROI_PAC(SceneIO):
    """
    .. warning :: Data has to be georeferenced to latitude/longitude!

    The unwrapped displacement is expected in radians and will be scaled by
    `WAVELENGTH` parsed from the :file:`*.rsc` file.

    Expects two files:

        * Binary file from ROI_PAC (:file:`*`)
        * Parameter file (:file:`<binary_file>.rsc`)
          describing ``'WIDTH', 'FILE_LENGTH', 'X_FIRST', 'Y_FIRST', 'X_STEP',
          'Y_STEP', 'WAVELENGTH``
    """

    def validate(self, filename, **kwargs):
        try:
            par_file = kwargs.pop('par_file',
                                  self._getParameterFile(filename))
            self._parseParameterFile(par_file)
            return True
        except ImportError:
            return False

    def _getParameterFile(self, bin_file):
        par_file = os.path.realpath(bin_file) + '.rsc'
        try:
            self._parseParameterFile(par_file)
            self._log.debug('Found parameter file %s' % file)
            return par_file
        except (ImportError, IOError):
            raise ImportError('Could not find ROI_PAC parameter file (%s)'
                              % par_file)

    @staticmethod
    def _parseParameterFile(par_file):
        import re

        params = {}
        required = ['WIDTH', 'FILE_LENGTH', 'X_FIRST', 'Y_FIRST', 'X_STEP',
                    'Y_STEP', 'WAVELENGTH']

        rc = re.compile(r'([\w]*)\s*([\w.+-]*)')
        with open(par_file, 'r') as par:
            for line in par:
                parsed = rc.match(line)
                if parsed is None:
                    continue
                groups = parsed.groups()
                params[groups[0]] = safe_cast(groups[1], float,
                                              default=groups[1].strip())

        if check_required(required, params):
            return params

        raise ImportError(
            'Parameter file does not hold required parameters')

    def read(self, filename, **kwargs):
        """
        :param filename: ROI_PAC binary file
        :type filename: str
        :param par_file: Corresponding parameter (:file:`*rsc`) file.
                         (optional)
        :type par_file: str
        :returns: Import dictionary
        :rtype: dict
        :raises: ImportError
        """
        par_file = kwargs.pop('par_file',
                              self._getParameterFile(filename))
        par = self._parseParameterFile(par_file)

        nlines = int(par['FILE_LENGTH'])
        nrows = int(par['WIDTH'])
        wavelength = par['WAVELENGTH']

        data = num.memmap(filename, dtype='<f4')
        data = data.reshape(nlines, nrows*2)

        displ = data[:, nrows:]
        displ[displ == -0.] = num.nan
        displ = displ / (4.*num.pi) * wavelength

        z_scale = par.get('Z_SCALE', 1.)
        z_offset = par.get('Z_OFFSET', 0.)
        displ += z_offset
        displ *= z_scale

        c = self.container
        c['displacement'] = displ
        c['theta'] = 2 * num.pi
        c['phi'] = 0.
        self._log.warning('NOT IMPLEMENTED - '
                          'Theta and phi are defaulting to vertical incident!')

        c['meta']['title'] = par.get('TITLE', 'None')
        c['meta']['wavelength'] = par['WAVELENGTH']
        c['bin_file'] = filename
        c['par_file'] = par_file

        c['frame']['llLat'] = par['Y_FIRST'] + par['Y_STEP'] * nrows
        c['frame']['llLon'] = par['X_FIRST']
        c['frame']['dLon'] = par['X_STEP']
        c['frame']['dLat'] = par['Y_STEP']
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
            raise ImportError('Could not find LOS file (*.rdr.geo)')
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

        displ = num.memmap(self._getDisplacementFile(path),
                           dtype='<f4')\
            .reshape(nlat, nlon*2)[:, nlon:]
        displ[displ == 0.] = num.nan
        c['displacement'] = displ

        los_data = num.fromfile(self._getLOSFile(path), dtype='<f4')\
            .reshape(nlat, nlon*2)
        c['phi'] = los_data[:, :nlon]
        c['theta'] = los_data[:, nlon:] + num.pi/2

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
        displ = grd.variables['z'][:].copy()
        displ /= 1e2  # los_ll.grd files come in cm
        c['displacement'] = displ
        shape = c['displacement'].shape
        # LatLon
        c['frame']['llLat'] = grd.variables['lat'][:].min()
        c['frame']['llLon'] = grd.variables['lon'][:].min()

        c['frame']['dLat'] = (grd.variables['lat'][:].max() -
                              c['frame']['llLat'])/shape[0]
        c['frame']['dLon'] = (grd.variables['lon'][:].max() -
                              c['frame']['llLon'])/shape[1]

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
            self._log.warning(self.__doc__)
            self._log.warning('Defaulting theta and phi to 0./2*pi')
            c['theta'] = num.pi/2
            c['phi'] = 0.

        return c
