from gwemopt.gracedb import get_event
from gwemopt.tests.test_schedule import test_data_dir


def test_scheduler():
    """
    Test scheduler

    :return: None
    """

    get_event("S190814bv", output_dir=test_data_dir)
