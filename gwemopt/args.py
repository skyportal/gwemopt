"""
Module for parsing command line arguments
"""
import argparse

from gwemopt.paths import (
    CATALOG_DIR,
    DEFAULT_BASE_OUTPUT_DIR,
    DEFAULT_CONFIG_DIR,
    DEFAULT_LIGHTCURVE_DIR,
    DEFAULT_TILING_DIR,
    SKYMAP_DIR,
)


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--configDirectory",
        help="GW-EM config file directory.",
        default=DEFAULT_CONFIG_DIR,
    )
    parser.add_argument("-s", "--skymap", help="GW skymap.", default=None)
    parser.add_argument("-g", "--gpstime", help="GPS time.", default=None)
    parser.add_argument("--do3D", action="store_true", default=False)
    parser.add_argument("--do2D", action="store_true", default=False)

    parser.add_argument(
        "-o", "--outputDir", help="output directory", default=DEFAULT_BASE_OUTPUT_DIR
    )
    parser.add_argument("-n", "--event", help="event name", default="G268556")
    parser.add_argument(
        "--tilingDir", help="tiling directory", default=DEFAULT_TILING_DIR
    )

    parser.add_argument("--doSkymap", action="store_true", default=False)
    parser.add_argument("--doSamples", action="store_true", default=False)

    parser.add_argument("--doCoverage", action="store_true", default=False)

    parser.add_argument("--doSchedule", action="store_true", default=False)
    parser.add_argument("--scheduleType", help="schedule type", default="greedy_slew")
    parser.add_argument(
        "--timeallocationType", help="time allocation type", default="powerlaw"
    )

    parser.add_argument("--doPlots", action="store_true", default=False)
    parser.add_argument("--doDatabase", action="store_true", default=False)
    parser.add_argument("--doMovie", action="store_true", default=False)
    parser.add_argument("--doTiles", action="store_true", default=False)
    parser.add_argument("--tilesType", help="tiling type", default="moc")
    parser.add_argument("--doMindifFilt", action="store_true", default=False)

    parser.add_argument("--doIterativeTiling", action="store_true", default=False)
    parser.add_argument("--doMinimalTiling", action="store_true", default=False)
    parser.add_argument("--doOverlappingScheduling", action="store_true", default=False)
    parser.add_argument("--doPerturbativeTiling", action="store_true", default=False)
    parser.add_argument("--doOrderByObservability", action="store_true", default=False)

    parser.add_argument("--doCatalog", action="store_true", default=False)
    parser.add_argument("--doUseCatalog", action="store_true", default=False)
    parser.add_argument("--doCatalogDatabase", action="store_true", default=False)
    parser.add_argument("--catalogDir", help="catalog directory", default=CATALOG_DIR)
    parser.add_argument("--galaxy_catalog", help="Source catalog", default="GLADE")
    parser.add_argument(
        "--galaxy_grade",
        help="grade to use ('S', 'Sloc' or 'Smass')",
        type=str,
        default="S",
    )
    parser.add_argument("--writeCatalog", action="store_true", default=False)
    parser.add_argument("--catalog_n", default=1.0, type=float)
    parser.add_argument("--AGN_flag", action="store_true", default=False)
    parser.add_argument("--doObservability", action="store_true", default=False)
    parser.add_argument("--doObservabilityExit", action="store_true", default=False)
    parser.add_argument("--observability_thresh", default=0.05, type=float)
    parser.add_argument("--doSkybrightness", action="store_true", default=False)

    parser.add_argument("--doEfficiency", action="store_true", default=False)
    parser.add_argument(
        "-e",
        "--efficiencyOutput",
        help="Output file of the efficiency.",
        default="efficiency.txt",
    )
    parser.add_argument(
        "-m", "--modelType", help="(file, Bulla, Tophat, afterglow)", default="Tophat"
    )
    parser.add_argument("--mag", help="mag.", default=-16, type=float)
    parser.add_argument("--dmag", help="dmag.", default=0.0, type=float)
    parser.add_argument("-t", "--telescopes", help="Telescope names.", default="ATLAS")
    parser.add_argument(
        "-d",
        "--coverageFiles",
        help="Telescope coverage files.",
        default="../data/ATLAS_GW170104.dat",
    )
    parser.add_argument(
        "-l",
        "--lightcurveFiles",
        help="Lightcurve files.",
        default=DEFAULT_LIGHTCURVE_DIR.joinpath("Me2017_H4M050V20.dat"),
    )
    parser.add_argument(
        "--observedTiles", help="Tiles that have already been observed.", default=""
    )
    parser.add_argument("--Ninj", default=10000, type=int)
    parser.add_argument("--Ntiles", default=10, type=int)
    parser.add_argument("--doCalcTiles", action="store_true", default=False)
    parser.add_argument("--Ntiles_cr", default=0.70, type=float)
    parser.add_argument("--Ndet", default=1, type=int)
    parser.add_argument("--nside", default=256, type=int)
    parser.add_argument("--DScale", default=1.0, type=float)
    parser.add_argument("--Tobs", default="0.0,1.0")

    parser.add_argument("--mindiff", default=0.0, type=float)

    parser.add_argument("--powerlaw_cl", default=0.9, type=float)
    parser.add_argument("--powerlaw_n", default=1.0, type=float)
    parser.add_argument("--powerlaw_dist_exp", default=0, type=float)

    parser.add_argument("--galaxies_FoV_sep", default=1.0, type=float)

    parser.add_argument("--doFootprint", action="store_true", default=False)
    parser.add_argument("--footprint_ra", default=30.0, type=float)
    parser.add_argument("--footprint_dec", default=60.0, type=float)
    parser.add_argument("--footprint_radius", default=10.0, type=float)
    parser.add_argument("--doTreasureMap", action="store_true", default=False)
    parser.add_argument(
        "--treasuremap_token", help="Treasure Map API Token.", default=""
    )
    parser.add_argument(
        "--treasuremap_status",
        help="Status of Treasure Map observations to be queried.",
        default="planned,completed",
    )
    parser.add_argument("--start_time", default=None)
    parser.add_argument("--end_time", default=None)

    parser.add_argument("--doTrueLocation", action="store_true", default=False)
    parser.add_argument("--true_ra", default=30.0, type=float)
    parser.add_argument("--true_dec", default=60.0, type=float)
    parser.add_argument("--true_distance", default=100.0, type=float)

    parser.add_argument("--dt", default=14.0, type=float)

    parser.add_argument("-a", "--airmass", default=2.5, type=float)

    parser.add_argument("--doSingleExposure", action="store_true", default=False)
    parser.add_argument("--filters", default="r,g,r")
    parser.add_argument("--doAlternatingFilters", action="store_true", default=False)
    parser.add_argument("--doRASlices", action="store_true", default=False)
    parser.add_argument("--nside_down", default=2, type=int)
    parser.add_argument("--max_filter_sets", default=4, type=int)
    parser.add_argument("--iterativeOverlap", default=0.0, type=float)
    parser.add_argument("--maximumOverlap", default=1.0, type=float)
    parser.add_argument("--doBalanceExposure", action="store_true", default=False)

    parser.add_argument("--exposuretimes", default="30.0,30.0,30.0")

    parser.add_argument("--doMaxTiles", action="store_true", default=False)
    parser.add_argument("--max_nb_tiles", default="-1,-1,-1")
    parser.add_argument("--doReferences", action="store_true", default=False)

    parser.add_argument("--doChipGaps", action="store_true", default=False)
    parser.add_argument("--doUsePrimary", action="store_true", default=False)
    parser.add_argument("--doUseSecondary", action="store_true", default=False)

    parser.add_argument("--doSplit", action="store_true", default=False)
    parser.add_argument("--splitType", default="regional")
    parser.add_argument("--Nregions", default=768, type=int)

    parser.add_argument("--doParallel", action="store_true", default=False)
    parser.add_argument("--Ncores", default=4, type=int)

    parser.add_argument("--doBlocks", action="store_true", default=False)
    parser.add_argument("--Nblocks", default=4, type=int)

    parser.add_argument("--doRASlice", action="store_true", default=False)
    parser.add_argument("--raslice", default="0.0,24.0")
    parser.add_argument("--program_id", default=-1, type=int)

    parser.add_argument("--absmag", default=-15.0, type=float)

    parser.add_argument("--doRotate", action="store_true", default=False)
    parser.add_argument("--theta", help="theta rotation.", default=0.0, type=float)
    parser.add_argument("--phi", help="phi rotation.", default=0.0, type=float)

    parser.add_argument("--doAvoidGalacticPlane", action="store_true", default=False)
    parser.add_argument(
        "--galactic_limit", help="Galactic limit.", default=15.0, type=float
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Run verbosely. (Default: False)",
    )

    return parser.parse_args(args=args)
