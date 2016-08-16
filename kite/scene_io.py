class SceneIO(object):
    """ Prototype class for SARIO objects """

    def __init__(self):
        self.theta = None
        self.phi = None
        self.displacement = None
        self.x = None
        self.y = None

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
    pass

__all__ = """
MatlabData
""".split()
