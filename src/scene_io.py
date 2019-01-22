import re
import utm
import os.path as op
import glob
import scipy.io
import numpy as num
from kite import util

__all__ = ['Gamma', 'Matlab', 'ISCE', 'GMTSAR', 'ROI_PAC', 'SARscape']

d2r = num.pi/180.


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


class HeaderError(Exception):
    pass


class AttribDict(dict):

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, item, value):
        self[item] = value


class SceneIO(object):
    """ Prototype class for SARIO objects. """
    def __init__(self, scene=None):
        if scene is not None:
            self._log = scene._log.getChild('IO/%s' % self.__class__.__name__)
        else:
            import logging
            self._log = logging.getLogger('SceneIO/%s'
                                          % self.__class__.__name__)

        self.container = AttribDict(
            phi=0.,    # Look orientation counter-clockwise angle from east
            theta=0.,  # Look elevation angle (up from horizontal in degree)
                       # 90deg North
            displacement=None,  # Displacement towards LOS
            frame=AttribDict(
                llLon=None,  # Lower left corner latitude
                llLat=None,  # Lower left corner londgitude
                dN=None,   # Pixel delta in north, meter or degree
                dE=None,   # Pixel delta in east, meter or degree
                spacing='meter',  # Pixel spacing unit
            ),
            # Meta information
            meta=AttribDict(
                title=None,
                orbital_node=None,
                satellite_name=None,
                wavelength=None,
                time_master=None,
                time_slave=None
            ),
            # All extra information
            extra={}
        )

    def read(self, filename, **kwargs):
        """ Read function of the file format

        :param filename: file to read
        :type filename: string
        :param kwargs: Keyword arguments
        :type kwargs: {dict}
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

        ================== ==================== ===================== =====
        Property           Matlab ``.mat`` name type                  unit
        ================== ==================== ===================== =====
        Scene.displacement ``ig_``              n x m array           [m]
        Scene.phi          ``phi``              float or n x m array  [rad]
        Scene.theta        ``theta``            float or n x m array  [rad]
        Scene.frame.x      ``xx``               n x 1 vector          [m]
        Scene.frame.y      ``yy``               m x 1 vector          [m]
        Scene.utm_zone     ``utm_zone``         str ('33T')
        ================== ==================== ===================== =====

    Displacement is expected to be in meters. Note that the displacement maps
    could also be pixel offset maps rather than unwrapped SAR interferograms.
    For SAR azimuth pixel offset maps calculate ``phi`` from the heading
    direction and set ``theta=0.``. For SAR range pixel offsets use the same
    LOS angles as for InSAR.
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
        c = self.container

        mat = scipy.io.loadmat(filename)
        utm_e = None
        utm_n = None
        utm_zone = None
        utm_zone_letter = None
        phi0 = None
        theta0 = None

        for mat_k, v in mat.items():
            for io_k in c.keys():
                if io_k in mat_k:
                    c[io_k] = num.rot90(mat[mat_k])
                elif 'ig_' in mat_k:
                    c.displacement = num.rot90(mat[mat_k])
                elif 'xx' in mat_k:
                    utm_e = mat[mat_k].flatten()
                elif 'yy' in mat_k:
                    utm_n = mat[mat_k].flatten()
                elif 'utm_zone' in mat_k:
                    utm_zone = int(mat['utm_zone'][0][:-1])
                    utm_zone_letter = str(mat['utm_zone'][0][-1])
                elif 'phi' in mat_k:
                    phi0 = mat[mat_k].flatten()
                elif 'theta' in mat_k:
                    theta0 = mat[mat_k].flatten()

        if len(theta0) == 1:
            c.theta = num.ones(num.shape(c.displacement)) * theta0

        if len(theta0) == 1:
            c.phi = num.ones(num.shape(c.displacement)) * phi0

        if utm_zone is None:
            utm_zone = 33
            utm_zone_letter = 'N'
            self._log.warning(
                'Variable utm_zone not defined. Defaulting to UTM Zone %d%s!'
                % (utm_zone, utm_zone_letter))

        if not (num.all(utm_e) or num.all(utm_n)):
            self._log.warning(
                'Could not find referencing UTM vectors in .mat file!')
            utm_e = num.linspace(100000, 110000, c.displacement.shape[0])
            utm_n = num.linspace(1100000, 1110000, c.displacement.shape[1])

        if utm_e.min() < 1e4 or utm_n.min() < 1e4:
            utm_e *= 1e3
            utm_n *= 1e3

        c.frame.dE = num.abs(utm_e[1] - utm_e[0])
        c.frame.dN = num.abs(utm_n[1] - utm_n[0])
        try:
            c.frame.llLat, c.frame.llLon =\
                utm.to_latlon(utm_e.min(), utm_n.min(),
                              utm_zone, utm_zone_letter)

        except utm.error.OutOfRangeError:
            self._log.warning(
                'Could not interpret spatial vectors,'
                ' referencing to 0, 0 (lat, lon)')
            c.frame.llLat, c.frame.llLon = (0., 0.)
        return c


class Gamma(SceneIO):
    """

    Reading geocoded displacement maps (unwrapped igs) originating
        from GAMMA software.

    .. note :: Expects:

        * [:file:`*`] Binary file from Gamma with displacement in radians
        * [:file:`*.slc.par`] If you want to translate radians to
          meters using the `radar_frequency`.
        * [:file:`*par`] Parameter file, describing ``corner_lat, corner_lon,
          nlines, width, post_lat, post_lon`` or ``post_north, post_east,
          corner_east, corner_north, nlines, width``.
        * [:file:`*theta*`, :file:`*phi*`] Two look vector files,
          generated by GAMMA command ``look_vector``.

    .. warning ::

        * Data has to be georeferenced to latitude/longitude or UTM!
        * Look vector files - expected to have a particular name
    """
    @staticmethod
    def _parseParameterFile(filename):
        params = {}
        rc = re.compile(r'^(\w*):\s*([a-zA-Z0-9+-.*]*\s[a-zA-Z0-9_]*).*')

        with open(filename, mode='r') as par:
            for line in par:
                parsed = rc.match(line)
                if parsed is None:
                    continue

                groups = parsed.groups()
                params[groups[0]] = safe_cast(groups[1], float,
                                              default=groups[1].strip())
        return params

    def _getParameters(self, path, log=False):
        required_utm = ['post_north', 'post_east', 'corner_east',
                        'corner_north', 'nlines', 'width']
        required_lat_lon = ['corner_lat', 'corner_lon', 'nlines',
                            'width', 'post_lat', 'post_lon']

        path = op.dirname(op.realpath(path))
        par_files = glob.glob('%s/*par' % path)

        for file in par_files:
            params = self._parseParameterFile(file)

            if check_required(required_utm, params)\
               or check_required(required_lat_lon, params):
                if not log:
                    self._log.info('Found parameter file %s' % file)
                return params

        raise ImportError(
                    'Parameter file does not hold required parameters')

    def _getSLCParameters(self, path):
        required_params = ['radar_frequency']
        path = op.dirname(op.realpath(path))
        par_files = glob.glob('%s/*slc.par' % path)

        for file in par_files:
            params = self._parseParameterFile(file)

            if check_required(required_params, params):
                self._log.info('Found SLC parameter file %s' % file)
                return params

        raise ImportError('Could not find SLC parameter file *.slc.par'
                          ' with parameters %s' % required_params)

    def validate(self, filename, **kwargs):
        try:
            par_file = kwargs.pop('par_file', filename)
            self._getParameters(par_file)
            return True
        except ImportError:
            return False

    def _getLOSAngles(self, filename, pattern):
        path = op.dirname(op.realpath(filename))
        phi_files = glob.glob('%s/%s' % (path, pattern))
        if len(phi_files) == 0:
            self._log.warning('Could not find LOS file %s, '
                              'defaulting to angle to 0. [rad]' % pattern)
            return 0.
        elif len(phi_files) > 1:
            self._log.warning('Found multiple LOS files %s, '
                              'defaulting to angle 0. [rad]' % pattern)
            return 0.

        filename = phi_files[0]
        self._log.info('Loading LOS %s from %s' % (pattern, filename))
        return num.memmap(filename, mode='r', dtype='>f4')

    def read(self, filename, **kwargs):
        """
        :param filename: Gamma software parameter file
        :type filename: str
        :param par_file: Corresponding parameter (:file:`*par`) file.
                         (optional)
        :type par_file: str
        :returns: Import dictionary
        :rtype: dict
        :raises: ImportError
        """
        par_file = kwargs.pop('par_file', filename)

        params = self._getParameters(par_file, log=True)

        try:
            params_slc = self._getSLCParameters(par_file)
        except ImportError as e:
            raise e

        fill = None

        nrows = int(params['width'])
        nlines = int(params['nlines'])
        radar_frequency = params_slc.get('radar_frequency', None)

        displ = num.fromfile(filename, dtype='>f4')
        # Resize array if last line is not scanned completely
        if (displ.size % nrows) != 0:
            fill = num.empty(nrows - displ.size % nrows)
            fill.fill(num.nan)
            displ = num.append(displ, fill)

        displ = displ.reshape(nlines, nrows)
        displ[displ == -0.] = num.nan
        displ = num.flipud(displ)

        if radar_frequency is not None:
            radar_frequency = float(radar_frequency)
            self._log.info('Scaling displacement by radar_frequency %f GHz'
                           % (radar_frequency/1e9))
            wavelength = util.C / radar_frequency
            displ /= -4*num.pi
            displ *= wavelength

        else:
            wavelength = 'None'
            self._log.warning(
                'Could not determine radar_frequency from *.slc.par file!'
                ' Leaving displacement to radians.')

        phi = self._getLOSAngles(filename, '*phi*')
        theta = self._getLOSAngles(filename, '*theta*')
        theta = theta

        if isinstance(phi, num.ndarray):
            phi = phi.reshape(nlines, nrows)
            phi = num.flipud(phi)
        if isinstance(theta, num.ndarray):
            theta = theta.reshape(nlines, nrows)
            theta = num.flipud(theta)

        if fill is not None:
            theta = num.append(theta, fill)
            phi = num.append(phi, fill)

        c = self.container

        c.displacement = displ
        c.theta = theta
        c.phi = phi

        c.meta.wavelength = wavelength
        c.meta.title = params.get('title', 'None')

        c.bin_file = filename
        c.par_file = par_file

        if params['DEM_projection'] == 'UTM':
            utm_zone = params['projection_zone']
            try:
                utm_zone_letter = utm.latitude_to_zone_letter(
                    params['center_latitude'])
            except ValueError:
                self._log.warning('Could not parse UTM Zone letter,'
                                  ' defaulting to N!')
                utm_zone_letter = 'N'

            self._log.info('Using UTM reference: Zone %d%s'
                           % (utm_zone, utm_zone_letter))

            dN = params['post_north']
            dE = params['post_east']

            utm_corn_e = params['corner_east']
            utm_corn_n = params['corner_north']

            utm_corn_eo = utm_corn_e + dE * displ.shape[1]
            utm_corn_no = utm_corn_n + dN * displ.shape[0]

            utm_e = num.linspace(utm_corn_e, utm_corn_eo, displ.shape[1])
            utm_n = num.linspace(utm_corn_n, utm_corn_no, displ.shape[0])

            llLat, llLon = utm.to_latlon(utm_e.min(), utm_n.min(),
                                         utm_zone, utm_zone_letter)

            c.frame.llLat = llLat
            c.frame.llLon = llLon

            c.frame.dE = abs(dE)
            c.frame.dN = abs(dN)

        else:
            self._log.info('Using Lat/Lon reference')
            c.frame.llLat = params['corner_lat'] \
                + params['post_lat'] * nrows
            c.frame.llLon = params['corner_lon']
            c.frame.dLon = abs(params['post_lon'])
            c.frame.dLat = abs(params['post_lat'])

        return self.container


class ROI_PAC(SceneIO):
    """
    .. note:: Expects:

        * Binary file from ROI_PAC (:file:`*`)
        * Parameter file (:file:`<binary_file>.rsc`),
          describing ``WIDTH, FILE_LENGTH, X_FIRST, Y_FIRST, X_STEP,
          Y_STEP, WAVELENGTH``

    .. warning ::
        Data has to be georeferenced to latitude/longitude!

        The unwrapped displacement is expected in radians and will be scaled
        to meters by ``WAVELENGTH`` parsed from the :file:`*.rsc` file.

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
        par_file = op.realpath(bin_file) + '.rsc'
        try:
            self._parseParameterFile(par_file)
            self._log.info('Found parameter file %s' % par_file)
            return par_file
        except (ImportError, IOError):
            raise ImportError('Could not find ROI_PAC parameter file (%s)'
                              % par_file)

    @staticmethod
    def _parseParameterFile(par_file):
        params = {}
        required = ['WIDTH', 'FILE_LENGTH', 'X_FIRST', 'Y_FIRST', 'X_STEP',
                    'Y_STEP', 'WAVELENGTH', 'LAT_REF1', 'LON_REF1']

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
            'Parameter file %s does not hold required parameters' % par_file)

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
        par_file = kwargs.pop('par_file', self._getParameterFile(filename))

        par = self._parseParameterFile(par_file)
        nlines = int(par['FILE_LENGTH'])
        nrows = int(par['WIDTH'])
        wavelength = par['WAVELENGTH']
        heading = par['HEADING_DEG']
        lat_ref = par['LAT_REF1']
        lon_ref = par['LON_REF1']
        look_ref1 = par['LOOK_REF1']
        look_ref2 = par['LOOK_REF2']
        look_ref3 = par['LOOK_REF3']
        look_ref4 = par['LOOK_REF4']

        utm_zone_letter = utm.latitude_to_zone_letter(
                    par['LAT_REF1'])
        utm_zone = utm.latlon_to_zone_number(par['LAT_REF1'], par['LON_REF1'])

        look = num.mean(
            num.array([look_ref1, look_ref2, look_ref3, look_ref4]))

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

        c.displacement = displ
        c.theta = 90. - look
        c.phi = -heading - 180

        c.meta.title = par.get('TITLE', 'None')
        c.meta.wavelength = par['WAVELENGTH']
        c.bin_file = filename
        c.par_file = par_file

        c.frame.llLat = par['Y_FIRST'] + par['Y_STEP'] * nrows
        c.frame.llLon = par['X_FIRST']
        c.frame.dLon = par['X_STEP']
        c.frame.dLat = par['Y_STEP']
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
    Reading geocoded, unwraped displacement maps
        processed with ISCE software (https://winsar.unavco.org/isce.html).

    .. note :: Expects:

        * Unwrapped displacement binary (:file:`*.unw.geo`)
        * Metadata XML (:file:`*.unw.geo.xml`)
        * LOS binary data (:file:`*.rdr.geo`)

    .. warning::

        Data are in radians but no transformation to
        meters yet, as ``wavelength`` or at least sensor name is not
        provided in the XML file.
    """
    def validate(self, filename, **kwargs):
        try:
            self._getDisplacementFile(filename)
            self._getLOSFile(filename)
            return True
        except ImportError:
            return False

    def _getLOSFile(self, path):
        if not op.isdir(path):
            path = op.dirname(path)
        rdr_files = glob.glob(op.join(path, '*.rdr.geo'))

        if len(rdr_files) == 0:
            raise ImportError('Could not find LOS file (*.rdr.geo)')

        rdr_file = rdr_files[0]
        self._log.info('Found LOS file: %s', rdr_file)
        return rdr_file

    def _getDisplacementFile(self, path):
        if op.isfile(path):
            disp_file = path
        else:
            files = glob.glob(op.join(path, '*.unw.geo'))
            if len(files) == 0:
                raise ImportError('Could not find displacement file '
                                  '(.unw.geo) at %s', path)
            disp_file = files[0]

        if not op.isfile('%s.xml' % disp_file):
            raise ImportError('Could not find displacement XML file '
                              '(%s.unw.geo.xml)' % op.basename(disp_file))
        self._log.info('Found Displacement file: %s', disp_file)
        return disp_file

    def read(self, path, **kwargs):
        path = op.abspath(path)
        c = self.container

        xml_file = self._getDisplacementFile(path) + '.xml'
        self._log.info('Parsing ISCE XML file %s' % xml_file)
        isce_xml = ISCEXMLParser(xml_file)

        coord_lon = isce_xml.getProperty('coordinate1')
        coord_lat = isce_xml.getProperty('coordinate2')
        c.frame.dN = num.abs(coord_lat['delta'])
        c.frame.dE = num.abs(coord_lon['delta'])

        nlon = int(coord_lon['size'])
        nlat = int(coord_lat['size'])

        c.frame.spacing = 'degree'
        c.frame.llLat = coord_lat['startingvalue'] +\
            (nlat * coord_lat['delta'])
        c.frame.llLon = coord_lon['startingvalue']

        displ = num.memmap(self._getDisplacementFile(path),
                           dtype='<f4')\
            .reshape(nlat, nlon*2)[:, nlon:]
        displ[displ == 0.] = num.nan
        c.displacement = displ

        los_file = self._getLOSFile(path)
        los_data = num.fromfile(los_file, dtype='<f4')\
            .reshape(nlat, nlon*2)

        theta = los_data[:, :nlon]
        phi = los_data[:, nlon:]

        def los_is_degree():
            return num.abs(theta).max() > num.pi or num.abs(phi).max() > num.pi

        if not los_is_degree():
            raise ImportError(
                'The LOS file (%s) seems to be in radians! '
                'Change it to degree!' % op.basename(los_file))

        phi *= d2r
        theta *= d2r

        phi = num.pi/2 - phi

        c.phi = phi
        c.theta = theta

        return c


class GMTSAR(SceneIO):
    """

    .. note ::

        Expects:

        * Displacement grid (NetCDF, :file:`*los_ll.grd`) in cm
          (gets transformed to meters)
        * LOS binary data (see instruction, :file:`*los.enu`)

    Calculate the corresponding unit look vectors with GMT5SAR ``SAT_look``:

    .. code-block:: sh

        gmt grd2xyz unwrap_ll.grd | gmt grdtrack -Gdem.grd |
        awk {'print $1, $2, $4'} |
        SAT_look 20050731.PRM -bos > 20050731.los.enu
    """
    def validate(self, filename, **kwargs):
        try:
            if self._getDisplacementFile(filename)[-4:] == '.grd':
                return True
        except ImportError:
            return False
        return False

    def _getLOSFile(self, path):
        if not op.isdir(path):
            path = op.dirname(path)
        los_files = glob.glob(op.join(path, '*.los.*'))
        if len(los_files) == 0:
            self._log.warning(GMTSAR.__doc__)
            raise ImportError('Could not find LOS file (*.los.*)')
        los_file = los_files[0]
        self._log.debug('Found LOS file: %s', los_file)
        return los_file

    def _getDisplacementFile(self, path):
        if op.isfile(path):
            return path
        else:
            files = glob.glob(op.join(path, '*.grd'))
            if len(files) == 0:
                raise ImportError('Could not find displacement file '
                                  '(*.grd) at %s', path)
            disp_file = files[0]
        self._log.debug('Found Displacement file: %s', disp_file)
        return disp_file

    def read(self, path, **kwargs):
        from scipy.io import netcdf
        path = op.abspath(path)
        c = self.container

        grd = netcdf.netcdf_file(self._getDisplacementFile(path),
                                 mode='r', version=2)
        displ = grd.variables['z'][:].copy()
        displ /= 1e2  # los_ll.grd files come in cm
        c.displacement = displ
        shape = c.displacement.shape
        # LatLon
        c.frame.llLat = grd.variables['lat'][:].min()
        c.frame.llLon = grd.variables['lon'][:].min()

        c.frame.dLat = (grd.variables['lat'][:].max() -
                        c.frame.llLat) / shape[0]
        c.frame.dLon = (grd.variables['lon'][:].max() -
                        c.frame.llLon) / shape[1]

        # Theta and Phi
        try:
            los = num.memmap(self._getLOSFile(path), dtype='<f4')
            e = los[3::6].copy().reshape(shape)
            n = los[4::6].copy().reshape(shape)
            u = los[5::6].copy().reshape(shape)

            theta = num.rad2deg(num.arctan(n/e))
            phi = num.rad2deg(num.arccos(u))
            theta[n < 0] += 180.

            c.phi = phi
            c.theta = theta
        except ImportError:
            self._log.warning(self.__doc__)
            self._log.warning('Defaulting theta and phi to 0./2*pi [rad]')
            c.theta = num.pi/2
            c.phi = 0.

        return c


class SARscape(SceneIO):
    """

    .. note ::

        Expects:

        * Header file in :file:`*_disp.hdr`
        * Displacement data in cm in :file:`*_disp`
        * LOS data in :file:`*disp_ILOS` and :file:`*disp_ALOS` files.
    """
    def read(self, filename, **kwargs):
        header = self.parseHeaderFile(filename)

        def load_data(filename):
            self._log.debug('Loading %s' % filename)
            return num.flipud(
                num.fromfile(filename, dtype=num.float32)
                .reshape((header.lines, header.samples)))

        displacement = load_data(filename)
        theta_file, phi_file = self.getLOSFiles(filename)

        if not theta_file:
            theta = num.full_like(displacement, 0.)
        else:
            theta = load_data(theta_file)
            theta = num.deg2rad(theta)

        if not phi_file:
            phi = num.full_like(displacement, num.pi/2)
        else:
            phi = load_data(phi_file)
            phi = num.pi/2 - num.rad2deg(phi)

        c = self.container
        c.displacement = displacement
        c.phi = phi
        c.theta = theta

        map_info = header.map_info
        c.frame.dE = float(map_info[5])
        c.frame.dN = dN = float(map_info[6])
        c.frame.spacing = 'meter'

        c.frame.llLat, c.frame.llLon = utm.to_latlon(
            float(map_info[3]) - header.lines * dN,
            float(map_info[4]),
            zone_number=int(map_info[7]),
            northern=True if map_info[8] == 'Northern' else False)

        return c

    def parseHeaderFile(self, filename):
        hdr_file = self._getHDRFile(filename)
        conf = re.compile(r'^(.+)\s+=\s+(.+)\n', re.MULTILINE)

        header = AttribDict()
        with open(hdr_file) as f:
            s = f.read()

            linebreaks = re.compile(r'{(.+)\n?(.+)}')
            s = linebreaks.sub(r'{ \g<1> \g<2> }', s)

            for match in conf.finditer(s):
                groups = match.groups()
                key = groups[0].strip().replace(' ', '_')
                value = groups[1].strip()
                try:
                    value = int(value)
                except ValueError:
                    pass

                header[key] = value

            header.map_info = header.map_info.strip('{} ').split(', ')
            if not len(header.map_info) == 11:
                raise HeaderError('`map info` header is not consistent!')
            if header.map_info[0] != 'UTM':
                raise HeaderError('`map info` is not UTM!')

        return header

    def getLOSFiles(self, filename):
        ilos_file = op.abspath(filename + '_ILOS')
        if not op.exists(ilos_file):
            self._log.warning('Could not find ILOS file! (%s)' % ilos_file)
            ilos_file = False

        alos_file = op.abspath(filename + '_ALOS')
        if not op.exists(alos_file):
            self._log.warning('Could not find ALOS file! (%s)' % alos_file)
            alos_file = False
        return ilos_file, alos_file

    def _getHDRFile(self, filename):
        hdr_file = op.abspath(op.splitext(filename)[0] + '.hdr')
        if not op.exists(hdr_file):
            raise OSError('SARscape .hdr file not found (%s)' % hdr_file)
        return hdr_file

    def validate(self, filename, **kwargs):
        val = re.compile(r'SARscape|ENVI Standard', re.MULTILINE)
        try:
            hdr_file = self._getHDRFile(filename)
        except OSError as e:
            return False

        with open(hdr_file) as f:
            res = val.search(f.read())
            if res is not None:
                return True
            return False

        raise NotImplementedError('validate not implemented')
