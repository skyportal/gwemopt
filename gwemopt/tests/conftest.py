import base64
import hashlib
from pathlib import Path
from typing import Any

from pytest import fixture

from gwemopt.paths import CONFIG_DIR
from gwemopt.utils.param_utils import readParamsFromFile


@fixture
def skymap_path():
    test_dir = Path(__file__).parent.absolute()
    return Path(test_dir, "data", "S190814bv_5_LALInference.v1.fits.gz")


@fixture
def ztf_config():
    return readParamsFromFile(CONFIG_DIR.joinpath("ZTF.config"))


@fixture
def mxt_config():
    return readParamsFromFile(CONFIG_DIR.joinpath("SVOM-MXT.config"))


@fixture
def decam_config():
    return readParamsFromFile(CONFIG_DIR.joinpath("DECam.config"))


def make_hash_sha256(o: tuple | list | dict | set | frozenset | Any) -> str:
    """
    Generate a hash from an arbitrary python types

    Parameters
    ----------
    o : tuple | list | dict | set | frozenset | Any
        an object to hash

    Returns
    -------
    str
        the hash of the object o
    """
    hasher = hashlib.sha256()
    hasher.update(repr(make_hashable(o)).encode())
    return base64.b64encode(hasher.digest()).decode()


def make_hashable(o: tuple | list | dict | set | frozenset | Any) -> tuple:
    """
    Make the object o hashable

    Parameters
    ----------
    o : tuple | list | dict | set | frozenset | Any
        an object to transform to be hashable

    Returns
    -------
    tuple
        the object is now ready to be hashable
    """
    if isinstance(o, (tuple, list)):
        return tuple((make_hashable(e) for e in o))

    if isinstance(o, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in o.items()))

    if isinstance(o, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in o))

    return o
