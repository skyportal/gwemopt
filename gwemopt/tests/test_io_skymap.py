from math import isclose

from numpy import median

from gwemopt.io.skymap import read_skymap


def test_read_skymap(skymap_path):
    map_struct, gpstime = read_skymap(
        skymap_path=skymap_path,
        nside_raster=512,
        galactic_limit=0.0,
        confidence_level=1.0,
    )

    assert map_struct["skymap"].colnames == [
        "UNIQ",
        "PROBDENSITY",
        "DISTMU",
        "DISTSIGMA",
        "DISTNORM",
        "ra",
        "dec",
    ]

    assert isclose(gpstime, 1249852256.995869)

    assert len(map_struct["skymap"]["PROBDENSITY"]) == 16896

    assert isclose(median(map_struct["skymap"]["PROBDENSITY"]), 0.0028802321135086907)

    assert isclose(map_struct["center"].ra.value, 12.83203125)
    assert isclose(map_struct["center"].dec.value, -25.2001275)
