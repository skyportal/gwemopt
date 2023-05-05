import glob
import os

import astroplan
import astropy
import numpy as np
from astropy import table


def readParamsFromFile(file):
    """@read gwemopt params file

    @param file
        gwemopt params file
    """

    params = {}
    if os.path.isfile(file):
        with open(file, "r") as f:
            for line in f:
                line_without_return = line.split("\n")
                line_split = line_without_return[0].split(" ")
                line_split = list(filter(None, line_split))
                if line_split:
                    try:
                        params[line_split[0]] = float(line_split[1])
                    except:
                        params[line_split[0]] = line_split[1]
    return params


def params_checker(params):
    """ "Assigns defaults to params."""
    do_Parameters = [
        "do3D",
        "doEvent",
        "doSuperSched",
        "doMovie_supersched",
        "doSkymap",
        "doSamples",
        "doCoverage",
        "doSchedule",
        "doPlots",
        "doDatabase",
        "doMovie",
        "doTiles",
        "doIterativeTiling",
        "doMinimalTiling",
        "doOverlappingScheduling",
        "doPerturbativeTiling",
        "doOrderByObservability",
        "doCatalog",
        "doUseCatalog",
        "doCatalogDatabase",
        "doObservability",
        "doSkybrightness",
        "doEfficiency",
        "doCalcTiles",
        "doTransients",
        "doSingleExposure",
        "doAlternatingFilters",
        "doMaxTiles",
        "doReferences",
        "doChipGaps",
        "doUsePrimary",
        "doUseSecondary",
        "doSplit",
        "doParallel",
        "writeCatalog",
        "doFootprint",
        "doBalanceExposure",
        "doBlocks",
        "doUpdateScheduler",
        "doTreasureMap",
        "doRASlice",
        "doRASlices",
        "doRotate",
        "doMindifFilt",
        "doTrueLocation",
        "doAvoidGalacticPlane",
    ]

    for parameter in do_Parameters:
        if parameter not in params.keys():
            params[parameter] = False

    if "skymap" not in params.keys():
        params["skymap"] = "../output/skymaps/G268556.fits"

    if "gpstime" not in params.keys():
        params["gpstime"] = 1167559936.0

    if "outputDir" not in params.keys():
        params["outputDir"] = "../output"

    if "tilingDir" not in params.keys():
        params["tilingDir"] = "../tiling"

    if "catalogDir" not in params.keys():
        params["catalogDir"] = "../catalogs"

    if "event" not in params.keys():
        params["event"] = "G268556"

    if "coverageFiles" not in params.keys():
        params["coverageFiles"] = "../data/ATLAS_GW170104.dat"

    if "telescopes" not in params.keys():
        params["telescopes"] = "ATLAS"

    if type(params["telescopes"]) == str:
        params["telescopes"] = params["telescopes"].split(",")

    if "lightcurveFiles" not in params.keys():
        params["lightcurveFiles"] = "../lightcurves/Me2017_H4M050V20.dat"

    if "tilesType" not in params.keys():
        params["tilesType"] = "moc"

    if "scheduleType" not in params.keys():
        params["scheduleType"] = "greedy"

    if "timeallocationType" not in params.keys():
        params["timeallocationType"] = "powerlaw"

    if "Ninj" not in params.keys():
        params["Ninj"] = 1000

    if "Ndet" not in params.keys():
        params["Ndet"] = 1

    if "Ntiles" not in params.keys():
        params["Ntiles"] = 10

    if "Ntiles_cr" not in params.keys():
        params["Ntiles_cr"] = 0.70

    if "Dscale" not in params.keys():
        params["Dscale"] = 1.0

    if "nside" not in params.keys():
        params["nside"] = 256

    if "Tobs" not in params.keys():
        params["Tobs"] = np.array([0.0, 1.0])

    if "powerlaw_cl" not in params.keys():
        params["powerlaw_cl"] = 0.9

    if "powerlaw_n" not in params.keys():
        params["powerlaw_n"] = 1.0

    if "powerlaw_dist_exp" not in params.keys():
        params["powerlaw_dist_exp"] = 0

    if "galaxies_FoV_sep" not in params.keys():
        params["galaxies_FoV_sep"] = 1.0

    if "footprint_ra" not in params.keys():
        params["footprint_ra"] = 30.0

    if "footprint_dec" not in params.keys():
        params["footprint_dec"] = 60.0

    if "footprint_radius" not in params.keys():
        params["footprint_radius"] = 10.0

    if "transientsFile" not in params.keys():
        params["transientsFile"] = "../data/GW190425/transients.dat"

    if "transients_to_catalog" not in params.keys():
        params["transients_to_catalog"] = 0.8

    if "dt" not in params.keys():
        params["dt"] = 14.0

    if "galaxy_catalog" not in params.keys():
        params["galaxy_catalog"] = "GLADE"

    if "filters" not in params.keys():
        params["filters"] = ["r", "g", "r"]

    if "exposuretimes" not in params.keys():
        params["exposuretimes"] = np.array([30.0, 30.0, 30.0])

    if "max_nb_tiles" not in params.keys():
        params["max_nb_tiles"] = np.array([-1, -1, -1])

    if "mindiff" not in params.keys():
        params["mindiff"] = 0.0

    if "airmass" not in params.keys():
        params["airmass"] = 2.5

    if "iterativeOverlap" not in params.keys():
        params["iterativeOverlap"] = 0.0

    if "maximumOverlap" not in params.keys():
        params["maximumOverlap"] = 1.0

    if "catalog_n" not in params.keys():
        params["catalog_n"] = 1.0

    if "galaxy_grade" not in params.keys():
        params["galaxy_grade"] = "S"

    if "AGN_flag" not in params.keys():
        params["AGN_flag"] = False

    if "splitType" not in params.keys():
        params["splitType"] = "regional"

    if "Nregions" not in params.keys():
        params["Nregions"] = 768

    if "configDirectory" not in params.keys():
        params["configDirectory"] = "../config/"

    if "Ncores" not in params.keys():
        params["Ncores"] = 4

    if "Nblocks" not in params.keys():
        params["Nblocks"] = 4

    if "unbalanced_tiles" not in params.keys():
        params["unbalanced_tiles"] = None

    if "treasuremap_token" not in params.keys():
        params["treasuremap_token"] = ""

    if "treasuremap_status" not in params.keys():
        params["treasuremap_status"] = ["planned", "completed"]

    if "graceid" not in params.keys():
        params["graceid"] = "S190426c"

    if "raslice" not in params.keys():
        params["raslice"] = [0.0, 24.0]

    if "nside_down" not in params.keys():
        params["nside_down"] = 2

    if "max_filter_sets" not in params.keys():
        params["max_filter_sets"] = 4

    if "absmag" not in params.keys():
        params["absmag"] = -15

    if "phi" not in params.keys():
        params["phi"] = 0

    if "theta" not in params.keys():
        params["theta"] = 0

    if "program_id" not in params.keys():
        params["program_id"] = -1

    if "galactic_limit" not in params.keys():
        params["galactic_limit"] = 15.0

    if "true_ra" not in params.keys():
        params["true_ra"] = 30.0
    if "true_dec" not in params.keys():
        params["true_dec"] = 60.0
    if "true_distance" not in params.keys():
        params["true_distance"] = 100.0

    if "config" not in params.keys():
        params["config"] = {}
        configFiles = glob.glob("%s/*.config" % params["configDirectory"])
        for configFile in configFiles:
            telescope = configFile.split("/")[-1].replace(".config", "")
            if not telescope in params["telescopes"]:
                continue
            params["config"][telescope] = readParamsFromFile(configFile)
            params["config"][telescope]["telescope"] = telescope
            if params["doSingleExposure"]:
                exposuretime = np.array(opts.exposuretimes.split(","), dtype=float)[0]

                nmag = -2.5 * np.log10(
                    np.sqrt(params["config"][telescope]["exposuretime"] / exposuretime)
                )
                params["config"][telescope]["magnitude"] = (
                    params["config"][telescope]["magnitude"] + nmag
                )
                params["config"][telescope]["exposuretime"] = exposuretime
            if "tesselationFile" in params["config"][telescope]:
                if not os.path.isfile(params["config"][telescope]["tesselationFile"]):
                    if params["config"][telescope]["FOV_type"] == "circle":
                        gwemopt.tiles.tesselation_spiral(params["config"][telescope])
                    elif params["config"][telescope]["FOV_type"] == "square":
                        gwemopt.tiles.tesselation_packing(params["config"][telescope])
                if params["tilesType"] == "galaxy":
                    params["config"][telescope]["tesselation"] = np.empty((3,))
                else:
                    params["config"][telescope]["tesselation"] = np.loadtxt(
                        params["config"][telescope]["tesselationFile"],
                        usecols=(0, 1, 2),
                        comments="%",
                    )

            if "referenceFile" in params["config"][telescope]:
                refs = table.unique(
                    table.Table.read(
                        params["config"][telescope]["referenceFile"],
                        format="ascii",
                        data_start=2,
                        data_end=-1,
                    )["field", "fid"]
                )
                reference_images = {
                    group[0]["field"]: group["fid"].astype(int).tolist()
                    for group in refs.group_by("field").groups
                }
                reference_images_map = {1: "g", 2: "r", 3: "i"}
                for key in reference_images:
                    reference_images[key] = [
                        reference_images_map.get(n, n) for n in reference_images[key]
                    ]
                params["config"][telescope]["reference_images"] = reference_images

            location = astropy.coordinates.EarthLocation(
                params["config"][telescope]["longitude"],
                params["config"][telescope]["latitude"],
                params["config"][telescope]["elevation"],
            )
            observer = astroplan.Observer(location=location)
            params["config"][telescope]["observer"] = observer

    return params
