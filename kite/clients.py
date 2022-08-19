import logging
import os
import re

import requests

op = os.path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kite.clients")


def _download_file(url, outfile):
    logger.debug("Downloading %s to %s", url, outfile)
    r = requests.get(url)

    with open(outfile, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return outfile


def download_licsar(unw_url, destination="."):
    if not unw_url.endswith(".unw.tif"):
        raise ValueError("%s does not end with .unw.tif!" % unw_url)

    scene_name = op.basename(unw_url)
    url_dir = op.dirname(unw_url)
    product_name = op.basename(unw_url)

    os.makedirs(destination, exist_ok=True)

    logger.info("Downloading surface displacement data from LiCSAR: %s", product_name)
    unw_file = op.join(destination, scene_name)
    _download_file(unw_url, unw_file)

    logger.info("Downloading LOS angles...")
    scene_id = url_dir.split("/")[-3]
    meta_url = op.normpath(op.join(url_dir, "../../metadata"))

    for unit in ("E", "N", "U"):
        fn = "%s.geo.%s.tif" % (scene_id, unit)
        los_url = op.join(meta_url, fn)
        los_url = re.sub(r"^(https:/)\b", r"\1/", los_url, 0)
        outfn = op.normpath(op.join(destination, fn))

        _download_file(los_url, outfn)

    logger.info("Download complete! Open with\n\n\tspool --load=%s", unw_file)


if __name__ == "__main__":
    import sys

    download_licsar(*sys.argv[1:])
