import copy

import ligo.segments as segments
import numpy as np
import regions
from astropy import units as u
from astropy.coordinates import SkyCoord, get_sun
from astropy.time import Time
from mocpy import MOC
from tqdm import tqdm

import gwemopt.plotting
import gwemopt.scheduler
import gwemopt.tiles
from gwemopt.io.schedule import read_schedule
from gwemopt.tiles import (
    balance_tiles,
    check_overlapping_tiles,
    eject_tiles,
    optimize_max_tiles,
    order_by_observability,
    perturb_tiles,
    slice_galaxy_tiles,
    slice_map_tiles,
    slice_number_tiles,
)
from gwemopt.utils.treasuremap import get_treasuremap_pointings


def combine_coverage_structs(coverage_structs):
    coverage_struct_combined = {}
    coverage_struct_combined["data"] = np.empty((0, 8))
    coverage_struct_combined["filters"] = np.empty((0, 1))
    coverage_struct_combined["moc"] = []
    coverage_struct_combined["telescope"] = np.empty((0, 1))
    coverage_struct_combined["galaxies"] = []
    coverage_struct_combined["exposureused"] = []

    for coverage_struct in coverage_structs:
        coverage_struct_combined["data"] = np.append(
            coverage_struct_combined["data"], coverage_struct["data"], axis=0
        )
        coverage_struct_combined["filters"] = np.append(
            coverage_struct_combined["filters"], coverage_struct["filters"]
        )
        coverage_struct_combined["moc"] = (
            coverage_struct_combined["moc"] + coverage_struct["moc"]
        )
        coverage_struct_combined["telescope"] = np.append(
            coverage_struct_combined["telescope"], coverage_struct["telescope"]
        )
        coverage_struct_combined["exposureused"] += list(
            coverage_struct["exposureused"]
        )
        if "galaxies" in coverage_struct:
            coverage_struct_combined["galaxies"] = (
                coverage_struct_combined["galaxies"] + coverage_struct["galaxies"]
            )

    return coverage_struct_combined


def read_coverage(params, telescope, filename, moc_struct=None):
    nside = params["nside"]
    config_struct = params["config"][telescope]
    schedule_table = read_schedule(filename)

    coverage_struct = {}
    coverage_struct["data"] = np.empty((0, 8))
    coverage_struct["filters"] = []
    coverage_struct["moc"] = []

    for ii, row in schedule_table.iterrows():
        ra, dec = row["ra"], row["dec"]
        tobs = row["tobs"]
        limmag = row["limmag"]
        texp = row["texp"]
        field = row["field"]
        prob = row["prob"]
        airmass = row["airmass"]
        filt = row["filter"]

        coverage_struct["data"] = np.append(
            coverage_struct["data"],
            np.array([[ra, dec, tobs, limmag, texp, field, prob, airmass]]),
            axis=0,
        )
        coverage_struct["filters"].append(filt)

        if moc_struct is None:
            if config_struct["FOV_coverage_type"] == "square":
                center = SkyCoord(ra, dec, unit="deg", frame="icrs")
                region = regions.RectangleSkyRegion(
                    center,
                    config_struct["FOV_coverage"] * u.deg,
                    config_struct["FOV_coverage"] * u.deg,
                )
                moc = MOC.from_astropy_regions(region, max_depth=10)
            elif config_struct["FOV_coverage_type"] == "circle":
                center = SkyCoord(ra, dec, unit="deg", frame="icrs")
                region = regions.CircleSkyRegion(
                    center, radius=config_struct["FOV"] * u.deg
                )
                moc = MOC.from_astropy_regions(region, max_depth=10)
        else:
            moc = moc_struct[field]["moc"]

        coverage_struct["moc"].append(moc)

    coverage_struct["filters"] = np.array(coverage_struct["filters"])
    coverage_struct["FOV"] = config_struct["FOV_coverage"] * np.ones(
        (len(coverage_struct["filters"]),)
    )
    coverage_struct["telescope"] = [config_struct["telescope"]] * len(
        coverage_struct["filters"]
    )
    coverage_struct["exposureused"] = []

    return coverage_struct


def read_coverage_files(params, moc_structs):
    if params["coverageFiles"]:
        if not (len(params["telescopes"]) == len(params["coverageFiles"])):
            return ValueError("Need same number of coverageFiles as telescopes")
    else:
        return ValueError("Need coverageFiles if doCoverage is enabled.")

    coverage_structs = []
    for telescope, coverageFile in zip(params["telescopes"], params["coverageFiles"]):
        coverage_struct = read_coverage(
            params, telescope, coverageFile, moc_struct=moc_structs[telescope]
        )
        coverage_structs.append(coverage_struct)

    return combine_coverage_structs(coverage_structs)


def powerlaw(params, map_struct, tile_structs, previous_coverage_struct=None):
    map_struct_hold = copy.deepcopy(map_struct)

    coverage_structs = []
    n_scope = 0
    full_prob_map = map_struct["skymap"]

    for jj, telescope in enumerate(params["telescopes"]):
        config_struct = params["config"][telescope]
        tile_struct = tile_structs[telescope]

        # Try to load the minimum duration of time from telescope config file
        # Otherwise set it to zero
        try:
            min_obs_duration = config_struct["min_observability_duration"] / 24
        except:
            min_obs_duration = 0.0

        if params["doIterativeTiling"] and (params["tilesType"] == "galaxy"):
            tile_struct = slice_galaxy_tiles(
                params, tile_struct, combine_coverage_structs(coverage_structs)
            )

        if (
            params["doPerturbativeTiling"]
            and (jj > 0)
            and (not params["tilesType"] == "galaxy")
        ):
            tile_struct = perturb_tiles(
                params, config_struct, telescope, map_struct_hold, tile_struct
            )

        if params["doOverlappingScheduling"]:
            tile_struct = check_overlapping_tiles(
                params, tile_struct, combine_coverage_structs(coverage_structs)
            )

        if params["doAlternatingFilters"]:
            params_hold = copy.copy(params)
            config_struct_hold = copy.copy(config_struct)

            coverage_struct, tile_struct = gwemopt.scheduler.schedule_alternating(
                params_hold,
                config_struct_hold,
                telescope,
                map_struct_hold,
                tile_struct,
                previous_coverage_struct,
            )

            if params["doBalanceExposure"]:
                if (
                    params["max_nb_tiles"] is None
                ):  # optimize max tiles (iff max tiles not already specified)
                    optimized_max, coverage_struct, tile_struct = optimize_max_tiles(
                        params,
                        tile_struct,
                        coverage_struct,
                        config_struct,
                        telescope,
                        map_struct_hold,
                    )
                    params["max_nb_tiles"] = np.array([optimized_max], dtype=float)
                else:
                    params_hold = copy.copy(params)
                    config_struct_hold = copy.copy(config_struct)

                    (
                        coverage_struct,
                        tile_struct,
                    ) = gwemopt.scheduler.schedule_alternating(
                        params_hold,
                        config_struct_hold,
                        telescope,
                        map_struct_hold,
                        tile_struct,
                        previous_coverage_struct,
                    )
                    doReschedule, balanced_fields = balance_tiles(
                        params_hold, tile_struct, coverage_struct
                    )

                    if doReschedule:
                        config_struct_hold = copy.copy(config_struct)
                        (
                            coverage_struct,
                            tile_struct,
                        ) = gwemopt.scheduler.schedule_alternating(
                            params_hold,
                            config_struct_hold,
                            telescope,
                            map_struct_hold,
                            tile_struct,
                            previous_coverage_struct,
                        )

        #                coverage_struct = gwemopt.utils.erase_unbalanced_tiles(params_hold,coverage_struct)
        else:
            # load the sun retriction for a satelite
            try:
                sat_sun_restriction = config_struct["sat_sun_restriction"]
            except:
                sat_sun_restriction = 0.0

            if not params["tilesType"] == "galaxy":
                tile_struct = gwemopt.tiles.powerlaw_tiles_struct(
                    params, config_struct, telescope, map_struct_hold, tile_struct
                )

                if sat_sun_restriction != 0.0:
                    # Check that a given tile is not to close to the sun for the satelite
                    # If it's to close set the proba associated to the tile to zero

                    time = params["gpstime"]
                    time = Time(time, format="gps", scale="utc")
                    sun_position = get_sun(time)

                    # astropy don't like the output of get sun in the following separator function, need here to redefine the skycoord
                    sun_position = SkyCoord(
                        sun_position.ra, sun_position.dec, frame="gcrs"
                    )

                    for key in tile_struct.keys():
                        if (
                            "segmentlist"
                            and "prob" in tile_struct[key]
                            and tile_struct[key]["segmentlist"]
                        ):
                            for counter in range(len(tile_struct[key]["segmentlist"])):
                                tile_position = SkyCoord(
                                    ra=tile_struct[key]["ra"] * u.degree,
                                    dec=tile_struct[key]["dec"] * u.degree,
                                    frame="icrs",
                                )
                                ang_dist = sun_position.separation(tile_position).deg
                            if ang_dist < sat_sun_restriction:
                                tile_struct[key]["prob"] = 0.0

            elif sat_sun_restriction == 0.0:
                for key in tile_struct.keys():
                    # Check that a given tile is observable a minimum amount of time
                    # If not set the proba associated to the tile to zero
                    if (
                        "segmentlist"
                        and "prob" in tile_struct[key]
                        and tile_struct[key]["segmentlist"]
                        and min_obs_duration > 0.0
                    ):
                        observability_duration = 0.0
                        for counter in range(len(tile_struct[key]["segmentlist"])):
                            observability_duration += (
                                tile_struct[key]["segmentlist"][counter][1]
                                - tile_struct[key]["segmentlist"][counter][0]
                            )
                        if (
                            tile_struct[key]["prob"] > 0.0
                            and observability_duration < min_obs_duration
                        ):
                            tile_struct[key]["prob"] = 0.0

            else:
                # Check that a given tile is not to close to the sun for the satelite
                # If it's to close set the proba associated to the tile to zero
                time = params["gpstime"]
                time = Time(time, format="gps", scale="utc")
                sun_position = get_sun(time)

                # astropy don't like the output of get sun in the following separator function, need here to redefine the skycoord
                sun_position = SkyCoord(sun_position.ra, sun_position.dec, frame="gcrs")

                for key in tile_struct.keys():
                    if (
                        "segmentlist"
                        and "prob" in tile_struct[key]
                        and tile_struct[key]["segmentlist"]
                    ):
                        for counter in range(len(tile_struct[key]["segmentlist"])):
                            tile_position = SkyCoord(
                                ra=tile_struct[key]["ra"] * u.degree,
                                dec=tile_struct[key]["dec"] * u.degree,
                                frame="icrs",
                            )
                            ang_dist = sun_position.separation(tile_position).deg

                        if ang_dist < sat_sun_restriction:
                            tile_struct[key]["prob"] = 0.0

            if (
                params["timeallocationType"] == "manual"
            ):  # only works if using same telescope
                try:
                    for field_id in params["observedTiles"]:
                        field_id = int(field_id)
                        if field_id in tile_struct:
                            tile_struct[field_id]["prob"] = 0.0

                except:
                    raise ValueError(
                        "need to specify tiles that have been observed using --observedTiles"
                    )

            coverage_struct = gwemopt.scheduler.scheduler(
                params, config_struct, tile_struct
            )

            if params["doBalanceExposure"]:
                cnt, ntrials = 0, 10
                while cnt < ntrials:
                    doReschedule, balanced_fields = balance_tiles(
                        params, tile_struct, coverage_struct
                    )
                    if doReschedule:
                        for key in params["unbalanced_tiles"]:
                            tile_struct[key]["prob"] = 0.0
                        coverage_struct = gwemopt.scheduler.scheduler(
                            params, config_struct, tile_struct
                        )
                        cnt = cnt + 1
                    else:
                        break

            #                coverage_struct = gwemopt.utils.erase_unbalanced_tiles(params,coverage_struct)

            if params["max_nb_tiles"] is not None:
                tile_struct, doReschedule = slice_number_tiles(
                    params, telescope, tile_struct, coverage_struct
                )
                if doReschedule:
                    coverage_struct = gwemopt.scheduler.scheduler(
                        params, config_struct, tile_struct
                    )

        tile_structs[telescope] = tile_struct
        coverage_structs.append(coverage_struct)

        if params["doIterativeTiling"]:
            map_struct_hold = slice_map_tiles(params, map_struct_hold, coverage_struct)

    map_struct["skymap"] = full_prob_map

    return tile_structs, combine_coverage_structs(coverage_structs)


def timeallocation(params, map_struct, tile_structs, previous_coverage_struct=None):
    if len(params["telescopes"]) > 1 and params["doOrderByObservability"]:
        order_by_observability(params, tile_structs)

    if params["timeallocationType"] == "powerlaw":
        print("Generating powerlaw schedule...")

        if params["treasuremap_token"] is not None:
            treasuremap_coverage = get_treasuremap_pointings(params)

            if previous_coverage_struct and treasuremap_coverage["data"]:
                previous_coverage_struct["data"] = np.append(
                    previous_coverage_struct["data"],
                    treasuremap_coverage["data"],
                    axis=0,
                )
                previous_coverage_struct["filters"] = (
                    previous_coverage_struct["filters"]
                    + treasuremap_coverage["filters"]
                )
                previous_coverage_struct["ipix"] = (
                    previous_coverage_struct["ipix"] + treasuremap_coverage["ipix"]
                )
            elif treasuremap_coverage["data"]:
                previous_coverage_struct = treasuremap_coverage

        if params["treasuremap_token"] is not None and not previous_coverage_struct:
            print("\nNo previous observations were ingested.\n")

        tile_structs, coverage_struct = gwemopt.coverage.powerlaw(
            params, map_struct, tile_structs, previous_coverage_struct
        )

    elif params["timeallocationType"] == "manual":
        print("Generating manual schedule...")
        tile_structs, coverage_struct = gwemopt.coverage.powerlaw(
            params, map_struct, tile_structs
        )

    return tile_structs, coverage_struct
