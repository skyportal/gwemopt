import copy

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from astropy.time import Time
import ligo.segments as segments
from gwemopt.tiles import absmag_tiles_struct, powerlaw_tiles_struct
import gwemopt
from gwemopt.segments import get_segments_tiles


def slice_map_tiles(params, map_struct, coverage_struct):
    prob = copy.deepcopy(map_struct["prob"])
    prob[prob < 0] = 0.0

    sort_idx = np.argsort(prob)[::-1]
    csm = np.empty(len(prob))
    csm[sort_idx] = np.cumsum(prob[sort_idx])
    ipix_keep = np.where(csm <= params["iterativeOverlap"])[0]

    for ii in range(len(coverage_struct["ipix"])):
        data = coverage_struct["data"][ii, :]
        filt = coverage_struct["filters"][ii]
        ipix = coverage_struct["ipix"][ii]
        patch = coverage_struct["patch"][ii]
        FOV = coverage_struct["FOV"][ii]
        area = coverage_struct["area"][ii]

        observ_time, exposure_time, field_id, prob, airmass = (
            data[2],
            data[4],
            data[5],
            data[6],
            data[7],
        )

        ipix_slice = np.setdiff1d(ipix, ipix_keep)
        if len(ipix_slice) == 0:
            continue
        map_struct["prob"][ipix_slice] = -1

    return map_struct


def slice_number_tiles(params, telescope, tile_struct, coverage_struct):
    idx = params["telescopes"].index(telescope)
    max_nb_tile = params["max_nb_tiles"][idx]
    if max_nb_tile < 0:
        return tile_struct, False

    keys = tile_struct.keys()
    keys_scheduled = np.unique(coverage_struct["data"][:, 5])

    if len(keys_scheduled) <= max_nb_tile:
        return tile_struct, False

    prob = np.zeros((len(keys),))
    for ii, key in enumerate(keys):
        if key in keys_scheduled:
            prob[ii] = prob[ii] + tile_struct[key]["prob"]

    sort_idx = np.argsort(prob)[::-1]
    idx_keep = sort_idx[: int(max_nb_tile)]

    for ii, key in enumerate(keys):
        # in the golden tile set
        if ii in idx_keep:
            continue
        tile_struct[key]["prob"] = 0.0

    return tile_struct, True


def eject_tiles(params, telescope, tile_struct):
    for field_id in params["scheduled_fields"][telescope]:
        tile_struct[field_id]["prob"] = 0.0

    return tile_struct


def balance_tiles(params, tile_struct, coverage_struct):
    filters, exposuretimes = params["filters"], params["exposuretimes"]

    keys_scheduled = coverage_struct["data"][:, 5]

    doReschedule = False
    unique, freq = np.unique(keys_scheduled, return_counts=True)
    balanced_fields = np.sum(freq == len(filters))

    params["unbalanced_tiles"] = [
        key for i, key in enumerate(unique) if freq[i] != len(filters)
    ]

    if len(params["unbalanced_tiles"]) != 0:
        doReschedule = True

    return doReschedule, balanced_fields


def erase_unbalanced_tiles(params, coverage_struct):
    idxs = np.isin(coverage_struct["data"][:, 5], params["unbalanced_tiles"])
    out = np.nonzero(idxs)[0]

    coverage_struct["data"] = np.delete(coverage_struct["data"], out, axis=0)
    coverage_struct["filters"] = np.delete(coverage_struct["filters"], out)
    coverage_struct["area"] = np.delete(coverage_struct["area"], out)
    coverage_struct["FOV"] = np.delete(coverage_struct["FOV"], out)
    coverage_struct["telescope"] = np.delete(coverage_struct["telescope"], out)

    for i in out[::-1]:
        del coverage_struct["patch"][i]
        del coverage_struct["ipix"][i]

    return coverage_struct


def slice_galaxy_tiles(params, tile_struct, coverage_struct):
    coverage_ras = coverage_struct["data"][:, 0]
    coverage_decs = coverage_struct["data"][:, 1]

    if len(coverage_ras) == 0:
        return tile_struct

    keys = tile_struct.keys()
    ras, decs = [], []
    for key in keys:
        ras.append(tile_struct[key]["ra"])
        decs.append(tile_struct[key]["dec"])
    ras, decs = np.array(ras), np.array(decs)

    prob = np.zeros((len(keys),))
    for ii, key in enumerate(keys):
        prob[ii] = prob[ii] + tile_struct[key]["prob"]

    sort_idx = np.argsort(prob)[::-1]
    csm = np.empty(len(prob))
    csm[sort_idx] = np.cumsum(prob[sort_idx])
    ipix_keep = np.where(csm <= params["iterativeOverlap"])[0]

    catalog1 = SkyCoord(
        ra=coverage_ras * u.degree, dec=coverage_decs * u.degree, frame="icrs"
    )

    for ii, key in enumerate(keys):
        # in the golden tile set
        if ii in ipix_keep:
            continue

        catalog2 = SkyCoord(
            ra=tile_struct[key]["ra"] * u.degree,
            dec=tile_struct[key]["dec"] * u.degree,
            frame="icrs",
        )
        sep = catalog1.separation(catalog2)
        galaxies = tile_struct[key]["galaxies"]
        for jj, s in enumerate(sep):
            if s.deg > 1:
                continue
            galaxies2 = coverage_struct["galaxies"][jj]
            overlap = np.intersect1d(galaxies, galaxies2)
            if len(overlap) == 0:
                continue

            rat = np.array(
                [
                    float(len(overlap)) / float(len(galaxies)),
                    float(len(overlap)) / float(len(galaxies2)),
                ]
            )

            if np.max(rat) > params["maximumOverlap"]:
                tile_struct[key]["prob"] = 0.0
                break

    return tile_struct


def optimize_max_tiles(
    params,
    opt_tile_struct,
    opt_coverage_struct,
    config_struct,
    telescope,
    map_struct_hold,
):
    """Returns value of max_tiles_nb optimized for number of scheduled fields with 'balanced' exposures in each filter."""
    prob = {}
    for key in opt_tile_struct.keys():
        prob[key] = opt_tile_struct[key]["prob"]

    tile_struct_hold = copy.copy(opt_tile_struct)
    keys_scheduled = opt_coverage_struct["data"][:, 5]
    unique, freq = np.unique(keys_scheduled, return_counts=True)
    n_equal = np.sum(freq == len(params["filters"]))
    n_dif = np.sum(freq != len(params["filters"]))

    optimized_max = -1  # assigns baseline optimized maxtiles
    n_dif_og = np.sum(freq != len(params["filters"]))
    params["doMaxTiles"] = True
    countervals = []

    coarse_bool = False
    repeating = False
    if (
        config_struct["FOV_type"] == "circle"
        and config_struct["FOV"] <= 2.0
        or config_struct["FOV_type"] == "square"
        and config_struct["FOV"] <= 4.0
    ):
        max_trials = np.linspace(10, 210, 9)
        coarse_bool = True
    else:
        max_trials = np.linspace(10, 200, 20)

    for ii, max_trial in enumerate(max_trials):
        for key in tile_struct_hold.keys():
            tile_struct_hold[key]["prob"] = prob[key]
            if "epochs" in tile_struct_hold[key]:
                tile_struct_hold[key]["epochs"] = np.empty((0, 9))
        params["max_nb_tiles"] = np.array([max_trial], dtype=float)
        params_hold = copy.copy(params)
        config_struct_hold = copy.copy(config_struct)
        coverage_struct_hold, tile_struct_hold = schedule_alternating(
            params_hold,
            config_struct_hold,
            telescope,
            map_struct_hold,
            tile_struct_hold,
        )

        keys_scheduled = coverage_struct_hold["data"][:, 5]
        unique, freq = np.unique(keys_scheduled, return_counts=True)
        counter = np.sum(freq == len(params["filters"]))

        countervals.append(counter)

        # check for breaking conditions
        if counter >= n_equal:
            n_dif = np.sum(freq != len(params["filters"]))
            if (
                counter > n_equal
                or counter == n_equal
                and optimized_max == -1
                and n_dif <= n_dif_og
            ):
                n_equal, optimized_max = counter, max_trial
                n_dif = np.sum(freq != len(params["filters"]))
                opt_coverage_struct, opt_tile_struct = (
                    coverage_struct_hold,
                    tile_struct_hold,
                )
        if ii > 2:
            repeating = countervals[ii] == countervals[ii - 1] == countervals[ii - 2]

        if ii > 0 and counter < countervals[ii - 1] or repeating:
            break

    # optimize within narrower range for more precision
    if coarse_bool == True:
        max_trials = np.linspace(optimized_max, optimized_max + 24, 4)
    else:
        if optimized_max < 100:
            max_trials = np.linspace(optimized_max - 3, optimized_max + 9, 7)
        elif optimized_max == 200:
            max_trials = np.linspace(optimized_max, optimized_max + 60, 4)
        else:
            max_trials = np.linspace(optimized_max, optimized_max + 9, 4)

    countervals = []
    repeating = False
    for ii, max_trial in enumerate(max_trials):
        if optimized_max == -1:
            break  # breaks if no max tiles restriction should be imposed
        for key in tile_struct_hold.keys():
            tile_struct_hold[key]["prob"] = prob[key]
            if "epochs" in tile_struct_hold[key]:
                tile_struct_hold[key]["epochs"] = np.empty((0, 9))
        params["max_nb_tiles"] = np.array([max_trial], dtype=float)
        params_hold = copy.copy(params)
        config_struct_hold = copy.copy(config_struct)

        coverage_struct_hold, tile_struct_hold = schedule_alternating(
            params_hold,
            config_struct_hold,
            telescope,
            map_struct_hold,
            tile_struct_hold,
        )

        keys_scheduled = coverage_struct_hold["data"][:, 5]
        unique, freq = np.unique(keys_scheduled, return_counts=True)
        counter = np.sum(freq == len(params["filters"]))

        countervals.append(counter)

        if counter > n_equal:
            n_equal, optimized_max = counter, max_trial
            n_dif = np.sum(freq != len(params["filters"]))
            opt_coverage_struct, opt_tile_struct = (
                coverage_struct_hold,
                tile_struct_hold,
            )
        if counter == n_equal and ii > 1:
            repeating = countervals[ii] == countervals[ii - 1] == countervals[ii - 2]

        if ii > 0 and counter < countervals[ii - 1] or repeating:
            break

    # check percent difference between # of fields scheduled in each filter
    n_1_og = n_equal
    n_2 = n_equal + n_dif
    p_dif = n_dif / ((n_1_og + n_2) * 0.5)

    if p_dif >= 0.1:
        count = 0
        n_difs, n_equals, p_difs = [n_dif], [n_equal], [p_dif]
        unbalanced_tiles = []
        params_hold = copy.copy(params)

        while count < 20:
            count += 1

            for key in tile_struct_hold.keys():
                tile_struct_hold[key]["prob"] = prob[key]
                if "epochs" in tile_struct_hold[key]:
                    tile_struct_hold[key]["epochs"] = np.empty((0, 9))
            doReschedule, balanced_fields = balance_tiles(
                params_hold, opt_tile_struct, opt_coverage_struct
            )
            params_hold["unbalanced_tiles"] = (
                unbalanced_tiles + params_hold["unbalanced_tiles"]
            )
            if not doReschedule:
                break
            config_struct_hold = copy.copy(config_struct)
            params_hold["max_nb_tiles"] = np.array(
                [np.ceil(optimized_max)], dtype=float
            )
            coverage_struct, tile_struct_hold = schedule_alternating(
                params_hold,
                config_struct_hold,
                telescope,
                map_struct_hold,
                tile_struct_hold,
            )
            keys_scheduled = coverage_struct["data"][:, 5]
            unique, freq = np.unique(keys_scheduled, return_counts=True)
            n_equal = np.sum(freq == len(params["filters"]))
            n_dif = np.sum(freq != len(params["filters"]))
            n_1 = n_equal
            n_2 = n_equal + n_dif
            p_dif = n_dif / ((n_1 + n_2) * 0.5)

            if (
                p_dif > p_difs[-1] and p_dif >= 0.15 and optimized_max > 0
            ):  # try decreasing max-tiles if n_difs increase
                optimized_max -= 0.1 * optimized_max
                continue
            elif (p_dif > p_difs[-1]) or (p_difs[-1] < 0.15 and n_equal < n_equals[-1]):
                break
            opt_coverage_struct, opt_tile_struct = coverage_struct, tile_struct_hold
            n_difs.append(n_dif)
            n_equals.append(n_equal)
            p_difs.append(p_dif)
            unbalanced_tiles = unbalanced_tiles + params_hold["unbalanced_tiles"]

            if (
                count == 19 and np.min(p_difs) > 0.15
            ):  # try setting it to original n_equal as final resort
                optimized_max = n_1_og

    for key in tile_struct_hold.keys():
        tile_struct_hold[key]["prob"] = prob[key]

    return optimized_max, opt_coverage_struct, opt_tile_struct


def check_overlapping_tiles(params, tile_struct, coverage_struct):
    coverage_ras = coverage_struct["data"][:, 0]
    coverage_decs = coverage_struct["data"][:, 1]
    #   coverage_mjds = coverage_struct["data"][:,2]
    coverage_ipixs = coverage_struct["ipix"]
    if len(coverage_ras) == 0:
        return tile_struct

    keys = list(tile_struct.keys())
    ras, decs = [], []
    for key in keys:
        ras.append(tile_struct[key]["ra"])
        decs.append(tile_struct[key]["dec"])
    ras, decs = np.array(ras), np.array(decs)

    catalog1 = SkyCoord(
        ra=coverage_ras * u.degree, dec=coverage_decs * u.degree, frame="icrs"
    )
    if params["tilesType"] == "galaxy":
        for ii, key in enumerate(keys):
            catalog2 = SkyCoord(
                ra=tile_struct[key]["ra"] * u.degree,
                dec=tile_struct[key]["dec"] * u.degree,
                frame="icrs",
            )
            sep = catalog1.separation(catalog2)
            galaxies = tile_struct[key]["galaxies"]
            for jj, s in enumerate(sep):
                if s.deg > 1:
                    continue
                galaxies2 = coverage_struct["galaxies"][jj]
                overlap = np.setdiff1d(galaxies, galaxies2)
                if len(overlap) == 0:
                    if not "epochs" in tile_struct[key]:
                        tile_struct[key]["epochs"] = np.empty((0, 9))
                    tile_struct[key]["epochs"] = np.append(
                        tile_struct[key]["epochs"],
                        np.atleast_2d(coverage_struct["data"][jj, :]),
                        axis=0,
                    )
    else:
        for ii, key in enumerate(keys):
            catalog2 = SkyCoord(
                ra=tile_struct[key]["ra"] * u.degree,
                dec=tile_struct[key]["dec"] * u.degree,
                frame="icrs",
            )
            sep = catalog1.separation(catalog2)
            ipix = tile_struct[key]["ipix"]
            for jj, s in enumerate(sep):
                if s.deg > 25:
                    continue
                ipix2 = coverage_struct["ipix"][jj]
                overlap = np.intersect1d(ipix, ipix2)

                rat = np.array(
                    [
                        float(len(overlap)) / float(len(ipix)),
                        float(len(overlap)) / float(len(ipix2)),
                    ]
                )

                if len(overlap) == 0 or (params["doSuperSched"] and np.max(rat) < 0.50):
                    continue
                if params["doSuperSched"]:
                    if "epochs_telescope" not in tile_struct[key]:
                        tile_struct[key]["epochs_telescope"] = []
                    tile_struct[key]["epochs_telescope"].append(
                        coverage_struct["telescope"][jj]
                    )

                if not "epochs" in tile_struct[key]:
                    tile_struct[key]["epochs"] = np.empty((0, 9))
                    tile_struct[key]["epochs_overlap"] = []
                    tile_struct[key]["epochs_filters"] = []

                tile_struct[key]["epochs"] = np.append(
                    tile_struct[key]["epochs"],
                    np.atleast_2d(coverage_struct["data"][jj, :]),
                    axis=0,
                )
                tile_struct[key]["epochs_overlap"].append(len(overlap))
                tile_struct[key]["epochs_filters"].append(
                    coverage_struct["filters"][jj]
                )

    return tile_struct


def append_tile_epochs(tile_struct, coverage_struct):
    for key in tile_struct.keys():
        if key not in coverage_struct["data"][:, 5]:
            continue
        if not "epochs" in tile_struct[key]:
            tile_struct[key]["epochs"] = np.empty((0, 9))
        idx = np.where(coverage_struct["data"][:, 5] == key)[0]
        for jj in idx:
            tile_struct[key]["epochs"] = np.append(
                tile_struct[key]["epochs"],
                np.atleast_2d(coverage_struct["data"][jj, :]),
                axis=0,
            )

    return tile_struct


def order_by_observability(params, tile_structs):
    observability = []
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]
        tiles_struct = tile_structs[telescope]
        exposurelist = config_struct["exposurelist"]
        observability_prob = 0.0
        keys = tiles_struct.keys()
        for jj, key in enumerate(keys):
            tilesegmentlist = tiles_struct[key]["segmentlist"]
            if tiles_struct[key]["prob"] == 0:
                continue
            if tilesegmentlist.intersects_segment(exposurelist[0]):
                observability_prob = observability_prob + tiles_struct[key]["prob"]
        observability.append(observability_prob)
    idx = np.argsort(observability)[::-1]
    params["telescopes"] = [params["telescopes"][ii] for ii in idx]


def perturb_tiles(params, config_struct, telescope, map_struct, tile_struct):
    map_struct_hold = copy.deepcopy(map_struct)
    ipix_keep = map_struct_hold["ipix_keep"]
    nside = params["nside"]

    if config_struct["FOV_type"] == "square":
        width = config_struct["FOV"] * 0.5
    elif config_struct["FOV_type"] == "circle":
        width = config_struct["FOV"] * 1.0

    moc_struct = {}
    keys = list(tile_struct.keys())
    for ii, key in enumerate(keys):
        if tile_struct[key]["prob"] == 0.0:
            continue

        if np.mod(ii, 100) == 0:
            print("Optimizing tile %d/%d" % (ii, len(keys)))

        x0 = [tile_struct[key]["ra"], tile_struct[key]["dec"]]
        FOV = config_struct["FOV"]
        bounds = [
            [tile_struct[key]["ra"] - width, tile_struct[key]["ra"] + width],
            [tile_struct[key]["dec"] - width, tile_struct[key]["dec"] + width],
        ]

        ras = np.linspace(
            tile_struct[key]["ra"] - width, tile_struct[key]["ra"] + width, 5
        )
        decs = np.linspace(
            tile_struct[key]["dec"] - width, tile_struct[key]["dec"] + width, 5
        )
        RAs, DECs = np.meshgrid(ras, decs)
        ras, decs = RAs.flatten(), DECs.flatten()

        vals = []
        for ra, dec in zip(ras, decs):
            if np.abs(dec) > 90:
                vals.append(0)
                continue

            moc_struct_temp = gwemopt.moc.Fov2Moc(
                params, config_struct, telescope, ra, dec, nside
            )
            idx = np.where(map_struct_hold["prob"][moc_struct_temp["ipix"]] == -1)[0]
            idx = np.setdiff1d(idx, ipix_keep)
            if len(map_struct_hold["prob"][moc_struct_temp["ipix"]]) == 0:
                rat = 0.0
            else:
                rat = float(len(idx)) / float(
                    len(map_struct_hold["prob"][moc_struct_temp["ipix"]])
                )
            if rat > params["maximumOverlap"]:
                val = 0.0
            else:
                ipix = moc_struct_temp["ipix"]
                if len(ipix) == 0:
                    val = 0.0
                else:
                    vals_to_sum = map_struct_hold["prob"][ipix]
                    vals_to_sum[vals_to_sum < 0] = 0
                    val = np.sum(vals_to_sum)
            vals.append(val)
        idx = np.argmax(vals)
        ra, dec = ras[idx], decs[idx]
        moc_struct[key] = gwemopt.moc.Fov2Moc(
            params, config_struct, telescope, ra, dec, nside
        )

        map_struct_hold["prob"][moc_struct[key]["ipix"]] = -1
        ipix_keep = np.setdiff1d(ipix_keep, moc_struct[key]["ipix"])

    if params["timeallocationType"] == "absmag":
        tile_struct = absmag_tiles_struct(
            params, config_struct, telescope, map_struct, moc_struct
        )
    elif params["timeallocationType"] == "powerlaw":
        tile_struct = powerlaw_tiles_struct(
            params, config_struct, telescope, map_struct, moc_struct
        )
    tile_struct = gwemopt.segments.get_segments_tiles(
        params, config_struct, tile_struct
    )

    return tile_struct


def schedule_alternating(params, config_struct, telescope, map_struct, tile_struct,
                         previous_coverage_struct=None):
    if "filt_change_time" in config_struct.keys():
        filt_change_time = config_struct["filt_change_time"]
    else:
        filt_change_time = 0
    if (params["doUpdateScheduler"] or params[
        "doTreasureMap"]) and previous_coverage_struct:
        tile_struct_hold = check_overlapping_tiles(params, tile_struct,
                                                   previous_coverage_struct)  # maps field ids to tile_struct

    filters, exposuretimes = params["filters"], params["exposuretimes"]
    coverage_structs = []
    maxidx = 0

    for i in range(len(exposuretimes)):
        params["filters"] = [filters[i]]
        params["exposuretimes"] = [exposuretimes[i]]
        config_struct["exposurelist"] = segments.segmentlist(
            config_struct["exposurelist"][maxidx:])
        total_nexps = len(config_struct["exposurelist"])

        # if the duration of a single block is less than 30 min, shift by additional time to add up to 30 min
        if i > 0:
            start = Time(coverage_struct["data"][0][2], format='mjd')
            end = Time(coverage_struct["data"][-1][2], format='mjd')

            delta = end - start
            delta.format = 'sec'
            duration = delta.value + exposuretimes[i] + filt_change_time
            extra_time = (30 * 60) - duration
            if extra_time > 0:
                extra_time = extra_time + filt_change_time
            elif extra_time <= 0:
                extra_time = filt_change_time
            config_struct["exposurelist"] = config_struct["exposurelist"].shift(
                extra_time / 86400.)

        if not params["tilesType"] == "galaxy":
            if params["timeallocationType"] == "absmag":
                tile_struct = absmag_tiles_struct(params, config_struct,
                                                                telescope, map_struct,
                                                                tile_struct)
            else:
                tile_struct = powerlaw_tiles_struct(params, config_struct,
                                                                  telescope, map_struct,
                                                                  tile_struct)

        if (params["doUpdateScheduler"] or params[
            "doTreasureMap"]) and previous_coverage_struct:  # erases tiles from a previous round
            tile_struct = gwemopt.coverage.update_observed_tiles(params,
                                                                 tile_struct_hold,
                                                                 previous_coverage_struct)

        # set unbalanced fields to 0
        if params["doBalanceExposure"] and params["unbalanced_tiles"]:
            for key in params["unbalanced_tiles"]:
                tile_struct[key]['prob'] = 0.0

        if coverage_structs and params["mindiff"]:
            if len(coverage_structs) > 1:
                tile_struct = append_tile_epochs(tile_struct,
                                                 gwemopt.coverage.combine_coverage_structs(
                                                     coverage_structs))
            elif len(coverage_structs) == 1:
                tile_struct = append_tile_epochs(tile_struct, coverage_structs[0])

        coverage_struct = gwemopt.scheduler.scheduler(params, config_struct,
                                                      tile_struct)
        if params["doMaxTiles"]:
            tile_struct, doReschedule = slice_number_tiles(params, telescope, tile_struct, coverage_struct)

            if doReschedule:
                coverage_struct = gwemopt.scheduler.scheduler(params, config_struct,
                                                              tile_struct)

        if len(coverage_struct["exposureused"]) > 0:
            maxidx = int(coverage_struct["exposureused"][-1])
            deltaL = total_nexps - maxidx
        elif len(coverage_struct["exposureused"]) == 0:
            deltaL = 0

        coverage_structs.append(coverage_struct)

        if deltaL <= 1: break
    params["filters"], params["exposuretimes"] = filters, exposuretimes

    return gwemopt.coverage.combine_coverage_structs(coverage_structs), tile_struct