import numpy as np
from astropy.time import Time

from gwemopt.utils import angular_distance, greenwich_sidereal_time, hour_angle


def test_greenwich_sidereal_time():
    """
    Test greenwich_sidereal_time function
    """

    # Test with a known date and time
    time = Time("2022-01-01T00:00:00", scale="utc")

    gst = greenwich_sidereal_time(time.jd, time.gps, 0) % (2 * np.pi)

    assert np.isclose(
        gst, 1.7563325, atol=1e-6
    ), f"Expected gst: 1.7536325, but got: {gst}"


def test_hour_angle():
    """
    Test hour_angle function
    """

    # Test with known values
    time = Time("2022-01-01T00:00:00", scale="utc")
    longitude = 45.0
    ra = 10.0
    ha = hour_angle(time.jd, time.gps, longitude, ra, 0)

    assert np.isclose(
        ha, 9.042003, atol=1e-6
    ), f"Expected hour angle: 1.7563325, but got: {ha}"


def test_angular_distance():
    """
    Test angular_distance function
    """

    # Test with known coordinates
    ra1, dec1 = 0.0, 0.0  # in degrees
    ra2, dec2 = 90.0, 0.0  # in degrees

    distance = angular_distance(ra1, dec1, ra2, dec2)

    assert np.isclose(
        distance, 90.0, atol=1e-6
    ), f"Expected distance: 90.0, but got: {distance}"
