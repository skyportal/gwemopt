import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy.time import Time

from gwemopt.utils import angular_distance, greenwich_sidereal_time, hour_angle
from gwemopt.utils.misc import integrationTime
from gwemopt.utils.param_utils import readParamsFromFile


class TestGreenwichSiderealTime:
    def test_known_date(self):
        time = Time("2022-01-01T00:00:00", scale="utc")
        gst = greenwich_sidereal_time(time.jd, time.gps, 0) % (2 * np.pi)
        assert np.isclose(gst, 1.7563325, atol=1e-6)

    def test_different_date(self):
        time = Time("2022-06-21T12:00:00", scale="utc")
        gst = greenwich_sidereal_time(time.jd, time.gps, 0) % (2 * np.pi)
        # GST should be between 0 and 2*pi
        assert 0 <= gst < 2 * np.pi

    def test_equation_of_equinoxes(self):
        time = Time("2022-01-01T00:00:00", scale="utc")
        gst0 = greenwich_sidereal_time(time.jd, time.gps, 0)
        gst1 = greenwich_sidereal_time(time.jd, time.gps, 1.0)
        # equation_of_equinoxes is scaled by pi/43200 in the function
        expected_delta = 1.0 * np.pi / 43200.0
        assert np.isclose(gst1 - gst0, expected_delta, atol=1e-10)


class TestHourAngle:
    def test_known_values(self):
        time = Time("2022-01-01T00:00:00", scale="utc")
        ha = hour_angle(time.jd, time.gps, 45.0, 10.0, 0)
        assert np.isclose(ha, 9.042003, atol=1e-6)

    def test_zero_longitude_and_ra(self):
        time = Time("2022-01-01T00:00:00", scale="utc")
        ha = hour_angle(time.jd, time.gps, 0.0, 0.0, 0)
        assert np.isfinite(ha)


class TestAngularDistance:
    def test_90_degree_separation(self):
        distance = angular_distance(0.0, 0.0, 90.0, 0.0)
        assert np.isclose(distance, 90.0, atol=1e-6)

    def test_zero_distance(self):
        distance = angular_distance(10.0, 20.0, 10.0, 20.0)
        assert np.isclose(distance, 0.0, atol=1e-6)

    def test_antipodal_points(self):
        distance = angular_distance(0.0, 0.0, 180.0, 0.0)
        assert np.isclose(distance, 180.0, atol=1e-6)

    def test_poles(self):
        distance = angular_distance(0.0, 90.0, 0.0, -90.0)
        assert np.isclose(distance, 180.0, atol=1e-6)

    def test_vectorized(self):
        ra1 = np.array([0.0, 0.0])
        dec1 = np.array([0.0, 0.0])
        ra2 = np.array([90.0, 180.0])
        dec2 = np.array([0.0, 0.0])
        distances = angular_distance(ra1, dec1, ra2, dec2)
        np.testing.assert_allclose(distances, [90.0, 180.0], atol=1e-6)


class TestIntegrationTime:
    def test_uniform_probability(self):
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        t_obs = 3600.0
        t_tiles = integrationTime(t_obs, probs)
        # With uniform probability, each tile gets equal time
        assert np.allclose(t_tiles, t_tiles[0])
        # Total time should not exceed T_obs (allowing for rounding)
        assert np.sum(t_tiles) <= t_obs + 60.0

    def test_single_tile_gets_all_time(self):
        probs = np.array([1.0, 0.0, 0.0])
        t_obs = 3600.0
        t_tiles = integrationTime(t_obs, probs)
        assert t_tiles[0] == t_obs
        assert t_tiles[1] == 0.0
        assert t_tiles[2] == 0.0

    def test_custom_integration_time(self):
        probs = np.array([0.5, 0.5])
        t_obs = 3600.0
        t_tiles = integrationTime(t_obs, probs, T_int=120.0)
        # Results should be multiples of T_int
        assert all(t % 120.0 == 0 for t in t_tiles)

    def test_zero_probabilities_handled(self):
        probs = np.array([0.0, 0.0, 0.0])
        t_obs = 3600.0
        t_tiles = integrationTime(t_obs, probs)
        # All zero probabilities should give zero time
        assert np.all(t_tiles == 0.0)


class TestReadParamsFromFile:
    def test_read_numeric_params(self, tmp_path):
        f = tmp_path / "params.dat"
        f.write_text("FOV 3.0\nexposuretime 30.0\nlatitude -30.169\n")
        params = readParamsFromFile(str(f))
        assert params["FOV"] == 3.0
        assert params["exposuretime"] == 30.0
        assert np.isclose(params["latitude"], -30.169)

    def test_read_string_params(self, tmp_path):
        f = tmp_path / "params.dat"
        f.write_text("FOV_type circle\ntelescope ZTF\n")
        params = readParamsFromFile(str(f))
        assert params["FOV_type"] == "circle"
        assert params["telescope"] == "ZTF"

    def test_missing_file(self):
        params = readParamsFromFile("/nonexistent/path/file.dat")
        assert params == {}

    def test_empty_file(self, tmp_path):
        f = tmp_path / "params.dat"
        f.write_text("")
        params = readParamsFromFile(str(f))
        assert params == {}

    def test_mixed_params(self, tmp_path):
        f = tmp_path / "params.dat"
        f.write_text("FOV 3.0\nFOV_type square\nslew_rate 2.5\n")
        params = readParamsFromFile(str(f))
        assert params["FOV"] == 3.0
        assert params["FOV_type"] == "square"
        assert params["slew_rate"] == 2.5
