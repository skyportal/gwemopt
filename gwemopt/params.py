import numpy as np
import gwemopt
import os
import astropy
from astropy import table, time
from astropy import units as u
import astroplan
from gwemopt.paths import DEFAULT_CONFIG_DIR, TESSELATION_DIR, REFS_DIR


def params_struct(opts):
    """@Creates gwemopt params structure
    @param opts
        gwemopt command line options
    """

    telescopes = str(opts.telescopes).split(",")

    params = dict()

    params["config"] = {}
    for telescope in telescopes:

        print(telescope)

        config_file = DEFAULT_CONFIG_DIR.joinpath(telescope + ".config")
        params["config"][telescope] = gwemopt.utils.readParamsFromFile(config_file)
        params["config"][telescope]["telescope"] = telescope
        if opts.doSingleExposure:
            exposuretime = np.array(opts.exposuretimes.split(","), dtype=float)[0]

            params["config"][telescope]["magnitude_orig"] = params["config"][telescope][
                "magnitude"
            ]
            params["config"][telescope]["exposuretime_orig"] = params["config"][
                telescope
            ]["exposuretime"]

            nmag = -2.5 * np.log10(
                np.sqrt(params["config"][telescope]["exposuretime"] / exposuretime)
            )
            params["config"][telescope]["magnitude"] = (
                params["config"][telescope]["magnitude"] + nmag
            )
            params["config"][telescope]["exposuretime"] = exposuretime
        if "tesselationFile" in params["config"][telescope]:
            tessfile = TESSELATION_DIR.joinpath(
                params["config"][telescope]["tesselationFile"]
            )
            if not os.path.isfile(tessfile):
                if params["config"][telescope]["FOV_type"] == "circle":
                    gwemopt.tiles.tesselation_spiral(params["config"][telescope])
                elif params["config"][telescope]["FOV_type"] == "square":
                    gwemopt.tiles.tesselation_packing(params["config"][telescope])
            if opts.tilesType == "galaxy":
                params["config"][telescope]["tesselation"] = np.empty((3,))
            else:

                params["config"][telescope]["tesselation"] = np.loadtxt(
                    tessfile, usecols=(0, 1, 2), comments="%"
                )

        if "referenceFile" in params["config"][telescope]:
            reffile = REFS_DIR.joinpath(params["config"][telescope]["referenceFile"])

            refs = table.unique(
                table.Table.read(reffile, format="ascii", data_start=2, data_end=-1)[
                    "field", "fid"
                ]
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

        # observer = ephem.Observer()
        # observer.lat = str(params["config"][telescope]["latitude"])
        # observer.lon = str(params["config"][telescope]["longitude"])
        # observer.horizon = str(-12.0)
        # observer.elevation = params["config"][telescope]["elevation"]
        # params["config"][telescope]["observer"] = observer

    params["mag"] = opts.mag
    params["dmag"] = opts.dmag
    params["modelType"] = opts.modelType
    params["skymap"] = opts.skymap
    params["gpstime"] = opts.gpstime
    params["outputDir"] = opts.outputDir
    params["tilingDir"] = opts.tilingDir
    params["catalogDir"] = opts.catalogDir
    params["event"] = opts.event
    params["coverageFiles"] = opts.coverageFiles.split(",")
    params["telescopes"] = telescopes
    params["lightcurveFiles"] = str(opts.lightcurveFiles).split(",")
    params["tilesType"] = opts.tilesType
    params["scheduleType"] = opts.scheduleType
    params["timeallocationType"] = opts.timeallocationType
    params["Ninj"] = opts.Ninj
    params["Ndet"] = opts.Ndet
    params["Ntiles"] = opts.Ntiles
    params["Ntiles_cr"] = opts.Ntiles_cr
    params["DScale"] = opts.DScale
    params["nside"] = opts.nside
    params["Tobs"] = np.array(opts.Tobs.split(","), dtype=float)
    params["powerlaw_cl"] = opts.powerlaw_cl
    params["powerlaw_n"] = opts.powerlaw_n
    params["powerlaw_dist_exp"] = opts.powerlaw_dist_exp
    params["galaxies_FoV_sep"] = opts.galaxies_FoV_sep
    params["observedTiles"] = opts.observedTiles.split(",")
    params["doPlots"] = opts.doPlots
    params["doMovie"] = opts.doMovie
    params["doMindifFilt"] = opts.doMindifFilt
    params["doObservability"] = opts.doObservability
    params["do3D"] = opts.do3D
    params["doDatabase"] = opts.doDatabase
    params["doTreasureMap"] = opts.doTreasureMap
    params["treasuremap_token"] = opts.treasuremap_token
    params["treasuremap_status"] = opts.treasuremap_status.split(",")
    params["graceid"] = opts.graceid
    params["program_id"] = opts.program_id
    params["doCalcTiles"] = opts.doCalcTiles
    params["doFootprint"] = opts.doFootprint
    params["footprint_ra"] = opts.footprint_ra
    params["footprint_dec"] = opts.footprint_dec
    params["footprint_radius"] = opts.footprint_radius

    params["doRASlice"] = opts.doRASlice
    params["raslice"] = np.array(opts.raslice.split(","), dtype=float)

    params["dt"] = opts.dt

    params["galaxy_catalog"] = opts.galaxy_catalog

    params["doSingleExposure"] = opts.doSingleExposure
    params["doBalanceExposure"] = opts.doBalanceExposure
    params["unbalanced_tiles"] = None
    params["filters"] = opts.filters.split(",")
    params["exposuretimes"] = np.array(opts.exposuretimes.split(","), dtype=float)
    params["doMovie_supersched"] = False
    params["doSuperSched"] = False
    params["doUpdateScheduler"] = False
    params["doMaxTiles"] = opts.doMaxTiles
    params["max_nb_tiles"] = np.array(opts.max_nb_tiles.split(","), dtype=float)
    params["mindiff"] = opts.mindiff
    params["doAlternatingFilters"] = opts.doAlternatingFilters
    params["doReferences"] = opts.doReferences

    params["airmass"] = opts.airmass

    params["doIterativeTiling"] = opts.doIterativeTiling
    params["iterativeOverlap"] = opts.iterativeOverlap
    params["maximumOverlap"] = opts.maximumOverlap
    params["doMinimalTiling"] = opts.doMinimalTiling
    params["doOverlappingScheduling"] = opts.doOverlappingScheduling
    params["doPerturbativeTiling"] = opts.doPerturbativeTiling
    params["doOrderByObservability"] = opts.doOrderByObservability
    params["doRASlices"] = opts.doRASlices
    params["max_filter_sets"] = opts.max_filter_sets
    params["nside_down"] = opts.nside_down

    params["doCatalog"] = opts.doCatalog
    params["catalog_n"] = opts.catalog_n
    params["doUseCatalog"] = opts.doUseCatalog
    params["doCatalogDatabase"] = opts.doCatalogDatabase
    params["galaxy_grade"] = opts.galaxy_grade
    params["writeCatalog"] = opts.writeCatalog
    params["AGN_flag"] = opts.AGN_flag

    params["doChipGaps"] = opts.doChipGaps
    params["doUsePrimary"] = opts.doUsePrimary
    params["doUseSecondary"] = opts.doUseSecondary

    params["doSplit"] = opts.doSplit
    params["splitType"] = opts.splitType
    params["Nregions"] = opts.Nregions

    params["doParallel"] = opts.doParallel
    params["Ncores"] = opts.Ncores

    params["doBlocks"] = opts.doBlocks
    params["Nblocks"] = opts.Nblocks

    params["absmag"] = opts.absmag

    params["doRotate"] = opts.doRotate
    params["phi"] = opts.phi
    params["theta"] = opts.theta

    params["doTrueLocation"] = opts.doTrueLocation
    params["true_ra"] = opts.true_ra
    params["true_dec"] = opts.true_dec
    params["true_distance"] = opts.true_distance

    params["doAvoidGalacticPlane"] = opts.doAvoidGalacticPlane
    params["galactic_limit"] = opts.galactic_limit

    if opts.start_time is None:
        params["start_time"] = time.Time.now() - time.TimeDelta(1.0 * u.day)
    else:
        params["start_time"] = time.Time(opts.start_time, format="isot", scale="utc")

    if opts.end_time is None:
        params["end_time"] = time.Time.now()
    else:
        params["end_time"] = time.Time(opts.end_time, format="isot", scale="utc")

    return params
