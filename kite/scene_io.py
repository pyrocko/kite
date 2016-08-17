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
        pass

    def write(self, filename, **kwargs):
        """Write method for IO

        :param filename: file to write to
        :type filename: string
        :param **kwargs: Keyword arguments
        :type **kwargs: {dict}
        """
        pass

    def validate(self, filename, **kwargs):
        """Validate file format

        :param filename: file to validate
        :type filename: string
        :returns: Validation
        :rtype: {bool}
        """
        pass
        return False


class MatlabData(SceneIO):
    """Reads Matlab .mat files into :py:class:`kite.scene.Scene`
    
    Variable naming convenctions in .mat file
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


__all__ = """
MatlabData
""".split()
