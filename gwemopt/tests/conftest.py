from pytest import fixture
from pathlib import Path
from importlib.resources import files
from gwemopt.paths import CONFIG_DIR
from gwemopt.utils import readParamsFromFile

@fixture
def skymap_path():
    test_dir = Path(__file__).parent.absolute()
    return Path(test_dir, "data", "S190814bv_5_LALInference.v1.fits.gz")


@fixture
def ztf_config():
    return readParamsFromFile(CONFIG_DIR.joinpath("ZTF.config"))