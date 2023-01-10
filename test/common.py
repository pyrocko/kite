import logging
import os
import re
import time

import requests
from requests.compat import urljoin

op = os.path

data_uri = "https://data.pyrocko.org/testing/kite/"
data_dir = op.join(op.dirname(op.abspath(__file__)), "data/")

logger = logging.getLogger("kite.testing")


class DownloadError(Exception):
    ...


def _makedir(path):
    try:
        os.makedirs(path)
    except OSError:
        ...


def get_test_data(fn):
    def _dir_content(url):
        r = requests.get(url)
        r.raise_for_status()

        entries = re.findall(r'href="([a-zA-Z0-9_.-]+/?)"', r.text)

        files = sorted(set(fn for fn in entries if not fn.endswith("/")))
        return files

    def _file_size(url):
        r = requests.head(url, headers={"Accept-Encoding": "identity"})
        r.raise_for_status()
        return int(r.headers["content-length"])

    def _download_file(url, fn_local):
        if op.exists(fn_local):
            if os.stat(fn_local).st_size == _file_size(url):
                logger.info("Using cached file %s" % fn_local)
                return fn_local

        logger.info("Downloading %s..." % url)
        fsize = _file_size(url)

        r = requests.get(url, stream=True)
        r.raise_for_status()

        dl_bytes = 0
        with open(fn_local, "wb") as f:
            for d in r.iter_content(chunk_size=1024):
                dl_bytes += len(d)
                f.write(d)

        if dl_bytes != fsize:
            raise DownloadError(
                f"Download {url} incomplete! Got {fsize} bytes, expected {dl_bytes}"
            )
        logger.info("Download completed.")
        return fn_local

    url = urljoin(data_uri, fn)

    dl_dir = data_dir
    _makedir(dl_dir)

    if fn.endswith("/"):
        dl_dir = op.join(data_dir, fn)
        _makedir(dl_dir)
        dl_files = _dir_content(url)

        dl_files = zip(
            [urljoin(url, u) for u in dl_files], [op.join(dl_dir, f) for f in dl_files]
        )
    else:
        dl_files = (url, op.join(data_dir, fn))
        return _download_file(*dl_files)

    return [_download_file(*f) for f in dl_files]


class Benchmark(object):
    def __init__(self, prefix=None):
        self.prefix = prefix or ""
        self.results = []

    def __call__(self, func):
        def stopwatch(*args):
            t0 = time.time()
            name = self.prefix + func.__name__
            result = func(*args)
            elapsed = time.time() - t0
            self.results.append((name, elapsed))
            return result

        return stopwatch

    def __str__(self):
        rstr = ["Benchmark results"]
        if self.prefix != "":
            rstr[-1] += " - %s" % self.prefix

        if len(self.results) > 0:
            indent = max([len(name) for name, _ in self.results])
        else:
            indent = 0
        rstr.append("=" * (indent + 17))
        rstr.insert(0, rstr[-1])
        for res in self.results:
            rstr.append("{0:<{indent}}{1:.8f} s".format(*res, indent=indent + 5))
        if len(self.results) == 0:
            rstr.append("None ran!")
        return "\n".join(rstr)


def setLogLevel(level):
    level = getattr(logging, level, "DEBUG")
    logging.basicConfig(level=level)
