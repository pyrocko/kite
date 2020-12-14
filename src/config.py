# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
from __future__ import absolute_import
import os
import os.path as op
from copy import deepcopy

from pyrocko import util
from pyrocko.guts import Object, Float, String, load, dump, TBase, Int

guts_prefix = 'pf'

kite_dir_tmpl = os.environ.get(
    'KITE_DIR',
    os.path.join('~', '.kite'))


def make_conf_path_tmpl(name='config'):
    return op.join(kite_dir_tmpl, '%s.pf' % name)


default_phase_key_mapping = {
    'F1': 'P', 'F2': 'S', 'F3': 'R', 'F4': 'Q', 'F5': '?'}


class BadConfig(Exception):
    pass


class PathWithPlaceholders(String):
    '''Path, possibly containing placeholders.'''
    pass


class VisibleLengthSetting(Object):
    class __T(TBase):
        def regularize_extra(self, val):
            if isinstance(val, list):
                return self.cls(key=val[0], value=val[1])

            return val

        def to_save(self, val):
            return (val.key, val.value)

        def to_save_xml(self, val):
            raise NotImplementedError()

    key = String.T()
    value = Float.T()


class ConfigBase(Object):
    @classmethod
    def default(cls):
        return cls()


class KiteConfig(ConfigBase):

    quadtree_min_tile_factor = Int.T(
        default=6,
        help='Exponent to the basis of 2. '
             'The larger the smaller the potential quadtree.')
    epsilon_min_factor = Float.T(
        default=0.1,
        help='The smaller the smaller the potential epsilon min')


config_cls = {
    'config': KiteConfig,
}


def fill_template(tmpl, config_type):
    tmpl = tmpl .format(
        module=('.' + config_type) if config_type != 'kite' else '')
    return tmpl


def expand(x):
    x = op.expanduser(op.expandvars(x))
    return x


def rec_expand(x):
    for prop, val in x.T.ipropvals(x):
        if prop.multivalued:
            if val is not None:
                for i, ele in enumerate(val):
                    if isinstance(prop.content_t, PathWithPlaceholders.T):
                        newele = expand(ele)
                        if newele != ele:
                            val[i] = newele

                    elif isinstance(ele, Object):
                        rec_expand(ele)
        else:
            if isinstance(prop, PathWithPlaceholders.T):
                newval = expand(val)
                if newval != val:
                    setattr(x, prop.name, newval)

            elif isinstance(val, Object):
                rec_expand(val)


def processed(config):
    config = deepcopy(config)
    rec_expand(config)
    return config


def mtime(p):
    return os.stat(p).st_mtime


g_conf_mtime = {}
g_conf = {}


def raw_config(config_name='config'):

    conf_path = expand(make_conf_path_tmpl(config_name))

    if not op.exists(conf_path):
        g_conf[config_name] = config_cls[config_name].default()
        write_config(g_conf[config_name], config_name)

    conf_mtime_now = mtime(conf_path)
    if conf_mtime_now != g_conf_mtime.get(config_name, None):
        g_conf[config_name] = load(filename=conf_path)
        if not isinstance(g_conf[config_name], config_cls[config_name]):
            raise BadConfig('config file does not contain a '
                            'valid "%s" section.' %
                            config_cls[config_name].__name__)

        g_conf_mtime[config_name] = conf_mtime_now

    return g_conf[config_name]


def config(config_name='config'):
    return processed(raw_config(config_name))


def write_config(conf, config_name='config'):
    conf_path = expand(make_conf_path_tmpl(config_name))
    util.ensuredirs(conf_path)
    dump(conf, filename=conf_path)


override_gui_toolkit = None


def effective_gui_toolkit():
    return override_gui_toolkit or config().gui_toolkit
