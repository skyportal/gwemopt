from gwemopt.telescope import Telescope
from math import isclose
from pytest import raises

def test_ztf_telescope(ztf_config: dict):
    ztf = Telescope("ZTF", ztf_config)
    assert ztf.telescope_mag == 20.4
    assert ztf.telescope_exptime == 30.0
    assert ztf.telescope_name == "ZTF"
    assert ztf.filters == "r"
    assert ztf.fov == 6.86
    assert ztf.fov_type == "square"
    assert isclose(ztf.mag_exposure(1/60), 16.33090936)

def test_tesselation(ztf_config: dict):
    ztf = Telescope("ZTF", ztf_config)
    tess = ztf.tesselation
    assert tess.shape == (1778, 3)
    assert isclose(tess[:, 1].mean(), 178.7753416)
    assert isclose(tess[:, 2].mean(), -0.879678942)
    
    fake_telescope = Telescope("fake", {
        "FOV_type": "circle",
        "FOV": 1,
        "longitude": 0,
        "latitude": 0,
        "elevation": 1,
        "tesselationFile": "no"
    })
    fake_tess = fake_telescope.tesselation
    assert fake_tess.shape == (16415, 3)
    assert isclose(fake_tess[:, 1].mean(), 179.9850171)
    assert isclose(fake_tess[:, 2].mean(), 1.939221111e-15)

    fake_telescope = Telescope("fake", {
        "FOV_type": "square",
        "FOV": 30/60,
        "longitude": 0,
        "latitude": 0,
        "elevation": 1,
        "tesselationFile": "no"
    })
    fake_tess = fake_telescope.tesselation
    assert fake_tess.shape == (175139, 3)
    assert isclose(fake_tess[:, 1].mean(), 179.6187029)
    assert isclose(fake_tess[:, 2].mean(), -0.0005138775485)

    with raises(RuntimeError):
        fake_telescope = Telescope("fake", {
            "FOV_type": "triangle",
            "FOV": 30/60,
            "longitude": 0,
            "latitude": 0,
            "elevation": 1,
            "tesselationFile": "no"
        })
        fake_telescope.tesselation


def test_referenceField(ztf_config: dict):
    ztf = Telescope("ZTF", ztf_config)
    assert len(ztf.referenceFields)== 930

    fake = Telescope("Fake", {
            "FOV_type": "triangle",
            "FOV": 30/60,
            "longitude": 0,
            "latitude": 0,
            "elevation": 1,
            "tesselationFile": "no"
        })
    assert fake.referenceFields is None