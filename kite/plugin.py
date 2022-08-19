from hashlib import sha1

from pyrocko.guts import Bool, Object


class PluginConfig(Object):

    applied = Bool.T(default=False)

    def get_hash(self):
        sha = sha1()
        sha.update(str(self).encode())
        return sha.hexdigest()


class Plugin(object):
    def __init__(self, scene, config=None):
        self.scene = scene
        self.config = config or PluginConfig()

    def get_state_hash(self):
        return self.config.get_hash()

    def set_enabled(self, enable):
        assert isinstance(enable, bool)
        self.config.applied = enable
        self.update()

    def is_enabled(self):
        return self.config.applied

    def update(self):
        return self.scene.evChanged()

    def apply(self, displacement):
        raise NotImplementedError
        return displacement
