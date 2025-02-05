import hashlib
from math import isclose
from pathlib import Path

from numpy import all as np_all
from numpy import array, float64, int64, load
from numpy.testing import assert_allclose
from pytest import raises

from gwemopt.telescope import Telescope
from gwemopt.tests.conftest import make_hash_sha256


def test_attribute_telescope(ztf_config: dict, mxt_config: dict, decam_config: dict):
    ztf = Telescope("ZTF", ztf_config)
    mxt = Telescope("MXT", mxt_config)
    decam = Telescope("DECam", decam_config)
    assert ztf.telescope_mag == 20.4
    assert ztf.telescope_exptime == 30.0
    assert ztf.telescope_name == "ZTF"
    assert ztf.filters == "r"
    assert ztf.fov == 6.86
    assert ztf.fov_type == "square"
    assert isclose(ztf.mag_exposure(1 / 60), 16.33090936)
    assert ztf.sat_sun_restriction == 0.0
    assert mxt.sat_sun_restriction == 30.0
    assert ztf.overhead_per_exposure == 10.0
    assert mxt.overhead_per_exposure == 0.0
    assert ztf.filt_change_time == 60.0
    assert mxt.filt_change_time == 0.0
    assert ztf.min_observability_duration == 0.0
    assert mxt.min_observability_duration == 0.0
    assert ztf.ha_constraint == (-24.0, 24.0)
    assert decam.ha_constraint == (-5.0, 5.0)
    assert mxt.moon_constraint == 0.0
    assert ztf.moon_constraint == 20.0


def test_tesselation(ztf_config: dict):
    ztf = Telescope("ZTF", ztf_config)
    tess = ztf.tesselation
    assert tess.shape == (1778, 3)

    fake_telescope = Telescope(
        "fake",
        {
            "FOV_type": "circle",
            "FOV": 1,
            "longitude": 0,
            "latitude": 0,
            "elevation": 1,
            "tesselationFile": "no",
        },
    )
    fake_tess = fake_telescope.tesselation

    test_dir = Path(__file__).parent.absolute()
    save_path = Path(test_dir, "data", "expected_results", "telescope_tesselation")
    with open(Path(save_path, "circle.tess"), "rb") as fp:
        ref_circle_tess = load(fp)

    assert fake_tess.shape == (16415, 3)
    assert_allclose(
        fake_tess,
        ref_circle_tess,
        rtol=1e-5,
        err_msg="circle tesselation are not equal (rtol=1e-5)",
    )

    fake_telescope = Telescope(
        "fake",
        {
            "FOV_type": "square",
            "FOV": 30 / 60,
            "longitude": 0,
            "latitude": 0,
            "elevation": 1,
            "tesselationFile": "no",
        },
    )
    fake_tess = fake_telescope.tesselation

    with open(Path(save_path, "square.tess"), "rb") as fp:
        ref_square_tess = load(fp)

    assert fake_tess.shape == (175139, 3)
    assert_allclose(
        fake_tess,
        ref_square_tess,
        rtol=1e-5,
        err_msg="circle tesselation are not equal (rtol=1e-5)",
    )

    assert (
        hashlib.sha512(fake_tess).hexdigest()
        == "fec3845b0c8321a39b8c57074982766901e8da2bd12c348b87cebc87605cf820547e23b44cfcaebde1201bedfa6c046cb2a782b8ed7916fc029b6648270e6dae"
    )

    with raises(RuntimeError):
        fake_telescope = Telescope(
            "fake",
            {
                "FOV_type": "triangle",
                "FOV": 30 / 60,
                "longitude": 0,
                "latitude": 0,
                "elevation": 1,
                "tesselationFile": "no",
            },
        )
        fake_telescope.tesselation


def test_referenceField(ztf_config: dict):
    ztf = Telescope("ZTF", ztf_config)
    assert len(ztf.referenceImages) == 930
    assert (
        make_hash_sha256(ztf.referenceImages)
        == "8/mZfifCETVG5X5TH/5L90D4KJnhsgc9acMS0qpYUHs="
    )

    fake = Telescope(
        "Fake",
        {
            "FOV_type": "triangle",
            "FOV": 30 / 60,
            "longitude": 0,
            "latitude": 0,
            "elevation": 1,
            "tesselationFile": "no",
        },
    )
    assert fake.referenceImages is None


def test_tesselation_setter(ztf_config: dict):
    ztf = Telescope("ZTF", ztf_config)
    tile_structs = {
        "ZTF": {
            int64(0): {"ra": float64(0.0), "dec": float64(0.0)},
            int64(1): {"ra": float64(10.0), "dec": float64(-2.0)},
        }
    }
    ztf.tesselation = tile_structs
    assert np_all(ztf.tesselation == array([[0.0, 0.0, 0.0], [1.0, 10.0, -2.0]]))
