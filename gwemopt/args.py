"""
Module for parsing command line arguments
"""
import argparse

from gwemopt.paths import CATALOG_DIR, DEFAULT_LIGHTCURVE_DIR


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--gpstime", help="GPS time.", default=None)

    parser.add_argument("--geometry", help="2d, 3d, or None (=auto)", default=None)

    parser.add_argument("-o", "--outputDir", help="output directory", default=None)
    parser.add_argument("-e", "--event", help="event name", default=None)

    parser.add_argument("--doCoverage", action="store_true", default=False)

    parser.add_argument("--doSchedule", action="store_true", default=False)
    parser.add_argument("--scheduleType", help="schedule type", default="greedy_slew")
    parser.add_argument(
        "--timeallocationType", help="time allocation type", default="powerlaw"
    )

    parser.add_argument("--doPlots", action="store_true", default=False)
    parser.add_argument("--doMovie", action="store_true", default=False)
    parser.add_argument("--doTiles", action="store_true", default=False)
    parser.add_argument("--tilesType", help="tiling type", default="moc")
    parser.add_argument("--doMindifFilt", action="store_true", default=False)

    parser.add_argument("--doIterativeTiling", action="store_true", default=False)
    parser.add_argument("--doMinimalTiling", action="store_true", default=False)
    parser.add_argument("--doOverlappingScheduling", action="store_true", default=False)
    parser.add_argument("--doPerturbativeTiling", action="store_true", default=False)
    parser.add_argument("--doOrderByObservability", action="store_true", default=False)

    parser.add_argument("--doUseCatalog", action="store_true", default=False)
    parser.add_argument("--catalogDir", help="catalog directory", default=CATALOG_DIR)
    parser.add_argument(
        "--catalog", help="Galaxy catalog name (e.g GLADE)", default=None
    )
    parser.add_argument(
        "--galaxy_grade",
        help="grade to use ('S', 'Sloc' or 'Smass')",
        type=str,
        default="S",
    )
    parser.add_argument(
        "--galaxy_limit",
        help="Limit on the number of returned galaxies",
        type=int,
        default=2000,
    )
    parser.add_argument("--doObservability", action="store_true", default=False)

    parser.add_argument("--doEfficiency", action="store_true", default=False)
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
        default=None,
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

    parser.add_argument("--Ndet", default=1, type=int)
    parser.add_argument("--nside", default=256, type=int)
    parser.add_argument("--DScale", default=1.0, type=float)
    parser.add_argument("--Tobs", default="0.0,1.0")

    parser.add_argument("--mindiff", default=0.0, type=float)

    parser.add_argument("--powerlaw_cl", default=0.9, type=float)
    parser.add_argument("--powerlaw_n", default=1.0, type=float)
    parser.add_argument("--powerlaw_dist_exp", default=0, type=float)

    parser.add_argument("--galaxies_FoV_sep", default=1.0, type=float)

    parser.add_argument(
        "--treasuremap_token", help="Treasure Map API Token.", default=None
    )
    parser.add_argument(
        "--treasuremap_status",
        help="Status of Treasure Map observations to be queried.",
        default="planned,completed",
    )

    parser.add_argument("--start_time", default=None)
    parser.add_argument("--end_time", default=None)

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

    parser.add_argument("--max_nb_tiles", default=None, type=int)
    parser.add_argument("--doReferences", action="store_true", default=False)

    parser.add_argument("--doChipGaps", action="store_true", default=False)
    parser.add_argument("--doUsePrimary", action="store_true", default=False)
    parser.add_argument("--doUseSecondary", action="store_true", default=False)

    parser.add_argument("--splitType", help="e.g: regional", default=None)

    parser.add_argument("--Nregions", default=768, type=int)

    parser.add_argument("--doParallel", action="store_true", default=False)
    parser.add_argument("--Ncores", default=4, type=int)

    parser.add_argument("--doBlocks", action="store_true", default=False)
    parser.add_argument("--Nblocks", default=4, type=int)

    parser.add_argument("--doRASlice", action="store_true", default=False)
    parser.add_argument("--raslice", default="0.0,24.0")

    parser.add_argument("--absmag", default=-15.0, type=float)

    parser.add_argument(
        "--galactic_limit", help="Galactic limit.", default=0.0, type=float
    )

    return parser.parse_args(args=args)
