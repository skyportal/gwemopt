from pytest import fixture
from pathlib import Path

@fixture
def skymap_path():
    test_dir = Path(__file__).parent.absolute()
    return Path(test_dir, "data", "S190814bv_5_LALInference.v1.fits.gz")
