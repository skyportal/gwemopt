import pytest

from gwemopt.io import get_skymap
from gwemopt.tests.test_schedule import test_data_dir


@pytest.mark.flaky(reruns=3)
def test_scheduler():
    """
    Test scheduler

    :return: None
    """

    get_skymap("S190814bv", output_dir=test_data_dir)
