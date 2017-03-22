from pyrocko.guts import Object, Bool
from os import path


class TalpaConfig(Object):
    show_cursor = Bool.T(
        default=True)


def getConfig():
    config_file = path.expanduser('~/.kite/talpa_config.yml')
    if not path.isfile(config_file):
        import os
        os.makedirs(path.dirname(config_file))

        config = TalpaConfig()
        config.dump(filename=config_file)

    try:
        config = TalpaConfig.load(filename=config_file)
    except:
        config = TalpaConfig()
    return config
