from gwemopt.telescope import Telescope
from math import isclose
from pytest import raises
from numpy import int64, float64, all as np_all, array


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
    assert isclose(tess[:, 1].mean(), 178.7753416)
    assert isclose(tess[:, 2].mean(), -0.879678942)

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
    assert fake_tess.shape == (16415, 3)
    assert isclose(fake_tess[:, 1].mean(), 179.98501, rel_tol=1e-5)
    assert isclose(fake_tess[:, 2].mean(), 1.91151e-15, rel_tol=1e-5)

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
    assert fake_tess.shape == (175139, 3)
    assert isclose(fake_tess[:, 1].mean(), 179.6187029)
    assert isclose(fake_tess[:, 2].mean(), -0.0005138775485)

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
