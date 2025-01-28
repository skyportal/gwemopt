from pathlib import Path

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
