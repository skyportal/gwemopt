"""
Module for defining paths to data directories.
"""
from pathlib import Path

DATA_DIR = Path(__file__).parent.absolute().joinpath("data")

DEFAULT_LIGHTCURVE_DIR = DATA_DIR.joinpath("lightcurves")
DEFAULT_BASE_OUTPUT_DIR = Path.home().joinpath("Data/gwemopt/")

TESSELATION_DIR = DATA_DIR.joinpath("tesselations")
REFS_DIR = DATA_DIR.joinpath("refs")
CONFIG_DIR = DATA_DIR.joinpath("config")
TILING_DIR = DATA_DIR.joinpath("tiling")

SKYMAP_DIR = DEFAULT_BASE_OUTPUT_DIR.joinpath("skymaps")
SKYMAP_DIR.mkdir(exist_ok=True, parents=True)

CATALOG_DIR = DEFAULT_BASE_OUTPUT_DIR.joinpath("catalogs")
CATALOG_DIR.mkdir(exist_ok=True, parents=True)
