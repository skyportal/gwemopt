from pathlib import Path
import pandas as pd


def read_schedule(schedule_path: str | Path):
    """
    Reads a schedule file and returns a pandas dataframe

    :param schedule_path: path to schedule file
    :return: pandas dataframe
    """

    schedule = pd.read_csv(
        schedule_path,
        sep=" ",
        names=[
            "field",
            "ra",
            "dec",
            "tobs",
            "limmag",
            "texp",
            "prob",
            "airmass",
            "filter",
            "pid",
        ],
    )

    return schedule
