import copy

import ligo.segments as segments
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, get_sun
from astropy.time import Time
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
from gwemopt.utils.pixels import getCirclePixels, getSquarePixels
from gwemopt.utils.treasuremap import get_treasuremap_pointings


def combine_coverage_structs(coverage_structs):
    coverage_struct_combined = {}
    coverage_struct_combined["data"] = np.empty((0, 8))
    coverage_struct_combined["filters"] = np.empty((0, 1))
    coverage_struct_combined["ipix"] = []
    coverage_struct_combined["patch"] = []
    coverage_struct_combined["FOV"] = np.empty((0, 1))
    coverage_struct_combined["area"] = np.empty((0, 1))
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
        coverage_struct_combined["ipix"] = (
            coverage_struct_combined["ipix"] + coverage_struct["ipix"]
        )
        coverage_struct_combined["patch"] = (
            coverage_struct_combined["patch"] + coverage_struct["patch"]
        )
        coverage_struct_combined["FOV"] = np.append(
            coverage_struct_combined["FOV"], coverage_struct["FOV"]
        )
        coverage_struct_combined["area"] = np.append(
            coverage_struct_combined["area"], coverage_struct["area"]
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
    coverage_struct["ipix"] = []
    coverage_struct["patch"] = []
    coverage_struct["area"] = []

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
            if telescope == "ATLAS":
                alpha = 0.2
                color = "#6c71c4"
            elif telescope == "PS1":
                alpha = 0.1
                color = "#859900"
            else:
                alpha = 0.2
                color = "#6c71c4"

            if config_struct["FOV_coverage_type"] == "square":
                ipix, radecs, patch, area = getSquarePixels(
                    ra,
                    dec,
                    config_struct["FOV_coverage"],
                    nside,
                    alpha=alpha,
                    color=color,
                )
            elif config_struct["FOV_coverage_type"] == "circle":
                ipix, radecs, patch, area = getCirclePixels(
                    ra,
                    dec,
                    config_struct["FOV_coverage"],
                    nside,
                    alpha=alpha,
                    color=color,
                )
        else:
            ipix = moc_struct[field]["ipix"]
            patch = moc_struct[field]["patch"]
            area = moc_struct[field]["area"]

        coverage_struct["patch"].append(patch)
        coverage_struct["ipix"].append(ipix)
        coverage_struct["area"].append(area)

    coverage_struct["filters"] = np.array(coverage_struct["filters"])
    coverage_struct["area"] = np.array(coverage_struct["area"])
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
    full_prob_map = map_struct["prob"]

    for jj, telescope in enumerate(params["telescopes"]):
        if params["splitType"] is not None:
            if "observability" in map_struct:
                map_struct["observability"][telescope]["prob"] = map_struct["groups"][
                    n_scope
                ]
            else:
                map_struct["prob"] = map_struct["groups"][n_scope]
            if n_scope < len(map_struct["groups"]) - 1:
                n_scope += 1
            else:
                n_scope = 0

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
            if params["doBlocks"]:
                tile_struct = eject_tiles(params, telescope, tile_struct)

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

            if params["doRASlices"]:
                coverage_struct = gwemopt.scheduler.schedule_ra_splits(
                    params,
                    config_struct,
                    map_struct_hold,
                    tile_struct,
                    telescope,
                    previous_coverage_struct,
                )
            elif params["doBalanceExposure"]:
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

            if params["treasuremap_token"] is not None and previous_coverage_struct:
                tile_struct = update_observed_tiles(
                    params, tile_struct, previous_coverage_struct
                )  # coverage_struct of the previous round

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

    map_struct["prob"] = full_prob_map

    return tile_structs, combine_coverage_structs(coverage_structs)


def absmag(params, map_struct, tile_structs, previous_coverage_struct=None):
    if "distmu" not in map_struct:
        print("timeallocationType absmag requires 3d geometry")
        exit(0)

    map_struct_hold = copy.deepcopy(map_struct)

    coverage_structs = []
    n_scope = 0
    full_prob_map = map_struct["prob"]

    for jj, telescope in enumerate(params["telescopes"]):
        if params["splitType"] is not None:
            if "observability" in map_struct:
                map_struct["observability"][telescope]["prob"] = map_struct["groups"][
                    n_scope
                ]
            else:
                map_struct["prob"] = map_struct["groups"][n_scope]
            if n_scope < len(map_struct["groups"]) - 1:
                n_scope += 1
            else:
                n_scope = 0

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
            if params["doBlocks"]:
                tile_struct = eject_tiles(params, telescope, tile_struct)

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
            if params["doRASlices"]:
                coverage_struct = gwemopt.scheduler.schedule_ra_splits(
                    params,
                    config_struct,
                    map_struct_hold,
                    tile_struct,
                    telescope,
                    previous_coverage_struct,
                )
            elif params["doBalanceExposure"]:
                if params["max_nb_tiles"] is None:
                    # optimize max tiles (iff max tiles not already specified)
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
                    do_reschedule, balanced_fields = balance_tiles(
                        params_hold, tile_struct, coverage_struct
                    )

                    if do_reschedule:
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
                tile_struct = gwemopt.tiles.absmag_tiles_struct(
                    params, config_struct, telescope, map_struct_hold, tile_struct
                )

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

                time = map_struct["trigtime"]
                time = Time(time, format="isot", scale="utc")
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

            if params["treasuremap_token"] is not None and previous_coverage_struct:
                tile_struct = update_observed_tiles(
                    params, tile_struct, previous_coverage_struct
                )  # coverage_struct of the previous round

            coverage_struct = gwemopt.scheduler.scheduler(
                params, config_struct, tile_struct
            )

            if params["doBalanceExposure"]:
                ntrials = 10
                for _ in tqdm(range(ntrials)):
                    do_reschedule, balanced_fields = balance_tiles(
                        params, tile_struct, coverage_struct
                    )
                    if do_reschedule:
                        for key in params["unbalanced_tiles"]:
                            tile_struct[key]["prob"] = 0.0
                        coverage_struct = gwemopt.scheduler.scheduler(
                            params, config_struct, tile_struct
                        )

            if params["max_nb_tiles"] is not None:
                tile_struct, do_reschedule = slice_number_tiles(
                    params, telescope, tile_struct, coverage_struct
                )
                if do_reschedule:
                    coverage_struct = gwemopt.scheduler.scheduler(
                        params, config_struct, tile_struct
                    )

        tile_structs[telescope] = tile_struct
        coverage_structs.append(coverage_struct)

        if params["doIterativeTiling"]:
            map_struct_hold = slice_map_tiles(params, map_struct_hold, coverage_struct)

    map_struct["prob"] = full_prob_map

    return tile_structs, combine_coverage_structs(coverage_structs)


def pem(params, map_struct, tile_structs):
    map_struct_hold = copy.deepcopy(map_struct)

    coverage_structs = []
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]
        tile_struct = tile_structs[telescope]

        if params["doAlternatingFilters"]:
            filters, exposuretimes = params["filters"], params["exposuretimes"]
            tile_struct_hold = copy.copy(tile_struct)
            coverage_structs_hold = []
            for filt, exposuretime in zip(filters, exposuretimes):
                params["filters"] = [filt]
                params["exposuretimes"] = [exposuretime]
                tile_struct_hold = gwemopt.tiles.pem_tiles_struct(
                    params, config_struct, telescope, map_struct_hold, tile_struct_hold
                )
                coverage_struct_hold = gwemopt.scheduler.scheduler(
                    params, config_struct, tile_struct_hold
                )
                coverage_structs_hold.append(coverage_struct_hold)
            coverage_struct = combine_coverage_structs(coverage_structs_hold)
        else:
            tile_struct = gwemopt.tiles.pem_tiles_struct(
                params, config_struct, telescope, map_struct_hold, tile_struct
            )
            coverage_struct = gwemopt.scheduler.scheduler(
                params, config_struct, tile_struct
            )
        coverage_structs.append(coverage_struct)

        if params["doIterativeTiling"]:
            map_struct_hold = slice_map_tiles(map_struct_hold, coverage_struct)

    return combine_coverage_structs(coverage_structs)


def update_observed_tiles(params, tile_struct, previous_coverage_struct):
    if not params["doAlternatingFilters"]:
        tile_struct = check_overlapping_tiles(
            params, tile_struct, previous_coverage_struct
        )  # maps field ids to tile_struct

    for key in tile_struct.keys():  # sets tile to 0 if previously observed
        if "epochs" not in tile_struct[key]:
            continue
        ipix = tile_struct[key]["ipix"]

        tot_overlap = sum(
            tile_struct[key]["epochs_overlap"]
        )  # sums over list of overlapping ipix lengths

        if params["doAlternatingFilters"]:
            # only takes into account fields with same filters for total overlap
            for ii, filt in enumerate(tile_struct[key]["epochs_filters"]):
                if filt != params["filters"][0]:
                    tot_overlap -= tile_struct[key]["epochs_overlap"][ii]

        rat = tot_overlap / len(ipix)

        if rat > 0.3:
            tile_struct[key]["prob"] = 0.0

    return tile_struct


def timeallocation(params, map_struct, tile_structs, previous_coverage_struct=None):
    if len(params["telescopes"]) > 1 and params["doOrderByObservability"]:
        order_by_observability(params, tile_structs)

    if (params["timeallocationType"] == "powerlaw") or (
        params["timeallocationType"] == "absmag"
    ):
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

        if params["doBlocks"]:
            exposurelists = {}
            scheduled_fields = {}
            for jj, telescope in enumerate(params["telescopes"]):
                config_struct = params["config"][telescope]
                exposurelist_split = np.array_split(
                    config_struct["exposurelist"], params["Nblocks"]
                )
                exposurelists[telescope] = exposurelist_split
                scheduled_fields[telescope] = []
            tile_structs_hold = copy.copy(tile_structs)
            coverage_structs = []

            for ii in range(params["Nblocks"]):
                params_hold = copy.copy(params)
                params_hold["scheduled_fields"] = scheduled_fields
                for jj, telescope in enumerate(params["telescopes"]):
                    exposurelist = segments.segmentlist()
                    for seg in exposurelists[telescope][ii]:
                        exposurelist.append(segments.segment(seg[0], seg[1]))
                    params_hold["config"][telescope]["exposurelist"] = exposurelist

                    if params["timeallocationType"] == "absmag":
                        tile_structs_hold[
                            telescope
                        ] = gwemopt.tiles.absmag_tiles_struct(
                            params_hold,
                            config_struct,
                            telescope,
                            map_struct,
                            tile_structs_hold[telescope],
                        )
                    elif params["timeallocationType"] == "powerlaw":
                        tile_structs_hold[
                            telescope
                        ] = gwemopt.tiles.powerlaw_tiles_struct(
                            params_hold,
                            config_struct,
                            telescope,
                            map_struct,
                            tile_structs_hold[telescope],
                        )

                if params["timeallocationType"] == "absmag":
                    tile_structs_hold, coverage_struct = gwemopt.coverage.absmag(
                        params_hold,
                        map_struct,
                        tile_structs_hold,
                        previous_coverage_struct,
                    )
                elif params["timeallocationType"] == "powerlaw":
                    tile_structs_hold, coverage_struct = gwemopt.coverage.powerlaw(
                        params_hold,
                        map_struct,
                        tile_structs_hold,
                        previous_coverage_struct,
                    )

                coverage_structs.append(coverage_struct)
                for ii in range(len(coverage_struct["ipix"])):
                    telescope = coverage_struct["telescope"][ii]
                    scheduled_fields[telescope].append(
                        coverage_struct["data"][ii, 5]
                    )  # appends all scheduled fields to appropriate list

            coverage_struct = combine_coverage_structs(coverage_structs)
        else:
            if params["timeallocationType"] == "absmag":
                tile_structs, coverage_struct = gwemopt.coverage.absmag(
                    params, map_struct, tile_structs, previous_coverage_struct
                )
            elif params["timeallocationType"] == "powerlaw":
                tile_structs, coverage_struct = gwemopt.coverage.powerlaw(
                    params, map_struct, tile_structs, previous_coverage_struct
                )

    elif params["timeallocationType"] == "manual":
        print("Generating manual schedule...")
        tile_structs, coverage_struct = gwemopt.coverage.powerlaw(
            params, map_struct, tile_structs
        )

    elif params["timeallocationType"] == "pem":
        print("Generating PEM schedule...")
        coverage_struct = gwemopt.coverage.pem(params, map_struct, tile_structs)

    return tile_structs, coverage_struct
