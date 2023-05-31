import copy

import healpy as hp
import ligo.segments as segments
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from scipy.stats import norm
from shapely.geometry import MultiPoint

import gwemopt
import gwemopt.moc
import gwemopt.segments
from gwemopt.utils.geometry import angular_distance

TILE_TYPES = ["moc", "galaxy"]


def slice_map_tiles(params, map_struct, coverage_struct):
    prob = copy.deepcopy(map_struct["prob"])
    prob[prob < 0] = 0.0

    sort_idx = np.argsort(prob)[::-1]
    csm = np.empty(len(prob))
    csm[sort_idx] = np.cumsum(prob[sort_idx])
    ipix_keep = np.where(csm <= params["iterativeOverlap"])[0]

    for ii in range(len(coverage_struct["ipix"])):
        data = coverage_struct["data"][ii, :]
        ipix = coverage_struct["ipix"][ii]
        FOV = coverage_struct["FOV"][ii]

        ipix_slice = np.setdiff1d(ipix, ipix_keep)
        if len(ipix_slice) == 0:
            continue
        map_struct["prob"][ipix_slice] = -1

    return map_struct


def slice_number_tiles(params, telescope, tile_struct, coverage_struct):
    max_nb_tile = params["max_nb_tiles"]
    if max_nb_tile is None:
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
                tile_struct_hold[key]["epochs"] = np.empty((0, 8))
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
                tile_struct_hold[key]["epochs"] = np.empty((0, 8))
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
                    tile_struct_hold[key]["epochs"] = np.empty((0, 8))
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
                        tile_struct[key]["epochs"] = np.empty((0, 8))
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

                if len(overlap) == 0:
                    continue

                if not "epochs" in tile_struct[key]:
                    tile_struct[key]["epochs"] = np.empty((0, 8))
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
        if "epochs" not in tile_struct[key]:
            tile_struct[key]["epochs"] = np.empty((0, 8))
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


def schedule_alternating(
    params,
    config_struct,
    telescope,
    map_struct,
    tile_struct,
    previous_coverage_struct=None,
):
    if "filt_change_time" in config_struct.keys():
        filt_change_time = config_struct["filt_change_time"]
    else:
        filt_change_time = 0
    if params["treasuremap_token"] is not None and previous_coverage_struct:
        tile_struct_hold = check_overlapping_tiles(
            params, tile_struct, previous_coverage_struct
        )  # maps field ids to tile_struct

    filters, exposuretimes = params["filters"], params["exposuretimes"]
    coverage_structs = []
    maxidx = 0

    for i in range(len(exposuretimes)):
        params["filters"] = [filters[i]]
        params["exposuretimes"] = [exposuretimes[i]]
        config_struct["exposurelist"] = segments.segmentlist(
            config_struct["exposurelist"][maxidx:]
        )
        total_nexps = len(config_struct["exposurelist"])

        # if the duration of a single block is less than 30 min, shift by additional time to add up to 30 min
        if i > 0:
            start = Time(coverage_struct["data"][0][2], format="mjd")
            end = Time(coverage_struct["data"][-1][2], format="mjd")

            delta = end - start
            delta.format = "sec"
            duration = delta.value + exposuretimes[i] + filt_change_time
            extra_time = (30 * 60) - duration
            if extra_time > 0:
                extra_time = extra_time + filt_change_time
            elif extra_time <= 0:
                extra_time = filt_change_time
            config_struct["exposurelist"] = config_struct["exposurelist"].shift(
                extra_time / 86400.0
            )

        if not params["tilesType"] == "galaxy":
            if params["timeallocationType"] == "absmag":
                tile_struct = absmag_tiles_struct(
                    params, config_struct, telescope, map_struct, tile_struct
                )
            else:
                tile_struct = powerlaw_tiles_struct(
                    params, config_struct, telescope, map_struct, tile_struct
                )

        if params["treasuremap_token"] is not None and previous_coverage_struct:
            # erases tiles from a previous round
            tile_struct = gwemopt.coverage.update_observed_tiles(
                params, tile_struct_hold, previous_coverage_struct
            )

        # set unbalanced fields to 0
        if params["doBalanceExposure"] and params["unbalanced_tiles"]:
            for key in params["unbalanced_tiles"]:
                tile_struct[key]["prob"] = 0.0

        if coverage_structs and params["mindiff"]:
            if len(coverage_structs) > 1:
                tile_struct = append_tile_epochs(
                    tile_struct,
                    gwemopt.coverage.combine_coverage_structs(coverage_structs),
                )
            elif len(coverage_structs) == 1:
                tile_struct = append_tile_epochs(tile_struct, coverage_structs[0])

        coverage_struct = gwemopt.scheduler.scheduler(
            params, config_struct, tile_struct
        )
        if params["max_nb_tiles"] is not None:
            tile_struct, doReschedule = slice_number_tiles(
                params, telescope, tile_struct, coverage_struct
            )

            if doReschedule:
                coverage_struct = gwemopt.scheduler.scheduler(
                    params, config_struct, tile_struct
                )

        if len(coverage_struct["exposureused"]) > 0:
            maxidx = int(coverage_struct["exposureused"][-1])
            deltaL = total_nexps - maxidx
        elif len(coverage_struct["exposureused"]) == 0:
            deltaL = 0

        coverage_structs.append(coverage_struct)

        if deltaL <= 1:
            break
    params["filters"], params["exposuretimes"] = filters, exposuretimes

    return gwemopt.coverage.combine_coverage_structs(coverage_structs), tile_struct


def get_rectangle(ras, decs, ra_size, dec_size):
    ras[ras > 180.0] = ras[ras > 180.0] - 360.0

    poly = MultiPoint([(x, y) for x, y in zip(ras, decs)]).envelope
    minx, miny, maxx, maxy = poly.bounds
    width = maxx - minx
    height = maxy - miny

    while (width < ra_size) or (height < dec_size):
        ra_mean, dec_mean = np.mean(ras), np.mean(decs)
        dist = angular_distance(ra_mean, dec_mean, ras, decs)
        idx = np.setdiff1d(np.arange(len(ras)), np.argmax(dist))
        ras, decs = ras[idx], decs[idx]

        if len(ras) == 1:
            return np.mod(ras[0], 360.0), decs[0]

        poly = MultiPoint([(x, y) for x, y in zip(ras, decs)]).envelope
        minx, miny, maxx, maxy = poly.bounds
        width = maxx - minx
        height = maxy - miny

    return np.mod((minx + maxx) / 2.0, 360.0), (miny + maxy) / 2.0


def galaxy(params, map_struct, catalog_struct: pd.DataFrame):
    """
    Creates a tile_struct for a galaxy survey
    """
    nside = params["nside"]

    tile_structs = {}
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]

        # Combine in a single pointing, galaxies that are distant by
        # less than fov * params['galaxies_FoV_sep']
        # Take galaxy with highest proba at the center of new pointing
        fov = params["config"][telescope]["FOV"] * params["galaxies_FoV_sep"]
        if "FOV_center" in params["config"][telescope]:
            fov_center = (
                params["config"][telescope]["FOV_center"] * params["galaxies_FoV_sep"]
            )
        else:
            fov_center = params["config"][telescope]["FOV"] * params["galaxies_FoV_sep"]

        new_cat = []

        idx_remaining = np.arange(len(catalog_struct["ra"])).astype(int)

        while len(idx_remaining) > 0:
            ii = idx_remaining[0]

            row = catalog_struct.iloc[ii]
            remaining = catalog_struct.iloc[idx_remaining]

            ra, dec = row["ra"], row["dec"]

            if config_struct["FOV_type"] == "square":
                dec_corners = (dec - fov, dec + fov)
                # assume small enough to use average dec for corners
                ra_corners = (
                    ra - fov / np.cos(np.deg2rad(dec)),
                    ra + fov / np.cos(np.deg2rad(dec)),
                )

                mask = (
                    (remaining["ra"] >= ra_corners[0])
                    & (remaining["ra"] <= ra_corners[1])
                    & (remaining["dec"] >= dec_corners[0])
                    & (remaining["dec"] <= dec_corners[1])
                )

                if np.sum(mask) > 1:
                    nearby_gals = remaining[mask]

                    ra_center, dec_center = get_rectangle(
                        nearby_gals["ra"].to_numpy(),
                        nearby_gals["dec"].to_numpy(),
                        fov / np.cos(np.deg2rad(dec)),
                        fov,
                    )

                    dec_corners = (dec_center - fov / 2.0, dec_center + fov / 2.0)
                    ra_corners = (
                        ra_center - fov / (2.0 * np.cos(np.deg2rad(dec))),
                        ra_center + fov / (2.0 * np.cos(np.deg2rad(dec))),
                    )

                    mask2 = (
                        (remaining["ra"] >= ra_corners[0])
                        & (remaining["ra"] <= ra_corners[1])
                        & (remaining["dec"] >= dec_corners[0])
                        & (remaining["dec"] <= dec_corners[1])
                    )

                    dec_corners = (
                        dec_center - fov_center / 2.0,
                        dec_center + fov_center / 2.0,
                    )
                    ra_corners = (
                        ra_center - fov_center / (2.0 * np.cos(np.deg2rad(dec))),
                        ra_center + fov_center / (2.0 * np.cos(np.deg2rad(dec))),
                    )
                    mask3 = (
                        (remaining["ra"] >= ra_corners[0])
                        & (remaining["ra"] <= ra_corners[1])
                        & (remaining["dec"] >= dec_corners[0])
                        & (remaining["dec"] <= dec_corners[1])
                    )

                    # did the optimization help?
                    if (np.sum(mask2) > 2) and (np.sum(mask3) > 0):
                        mask = mask2

                else:
                    ra_center, dec_center = row["ra"], row["dec"]

            elif config_struct["FOV_type"] == "circle":
                dist = angular_distance(
                    ra,
                    dec,
                    remaining["ra"].to_numpy(),
                    remaining["dec"].to_numpy(),
                )
                mask = dist <= (2 * fov)
                if len(mask) > 1:
                    nearby_gals = remaining[mask]

                    ra_center, dec_center = get_rectangle(
                        nearby_gals["ra"].to_numpy(),
                        nearby_gals["dec"].to_numpy(),
                        (fov / np.sqrt(2)) / np.cos(np.deg2rad(dec)),
                        fov / np.sqrt(2),
                    )

                    dist = angular_distance(
                        ra_center,
                        dec_center,
                        nearby_gals["ra"].to_numpy(),
                        nearby_gals["dec"].to_numpy(),
                    )
                    mask2 = np.where(fov >= dist)[0]
                    # did the optimization help?
                    if len(mask2) > 2:
                        mask = mask2
                else:
                    ra_center, dec_center = row["ra"], row["dec"]
            else:
                raise ValueError("FOV_type not recognized")

            new_cat.append(
                {
                    "ra": ra_center,
                    "dec": dec_center,
                    "Sloc": np.sum(remaining["Sloc"].to_numpy()[mask]),
                    "S": np.sum(remaining["S"].to_numpy()[mask]),
                    "Smass": np.sum(remaining["Smass"].to_numpy()[mask]),
                    "galaxies": idx_remaining[mask],
                }
            )

            idx_remaining = np.setdiff1d(idx_remaining, idx_remaining[mask])

        # redefine catalog_struct
        catalog_struct_new = pd.DataFrame(new_cat)

        moc_struct = {}
        cnt = 0
        for _, row in catalog_struct_new.iterrows():
            moc_struct[cnt] = gwemopt.moc.Fov2Moc(
                params, config_struct, telescope, row["ra"], row["dec"], nside
            )
            moc_struct[cnt]["galaxies"] = row["galaxies"]
            cnt = cnt + 1

        if params["timeallocationType"] == "absmag":
            tile_struct = absmag_tiles_struct(
                params, config_struct, telescope, map_struct, moc_struct
            )
        elif params["timeallocationType"] == "powerlaw":
            tile_struct = powerlaw_tiles_struct(
                params,
                config_struct,
                telescope,
                map_struct,
                moc_struct,
                catalog_struct=catalog_struct,
            )
        else:
            raise ValueError("timeallocationType not recognized")

        tile_struct = gwemopt.segments.get_segments_tiles(
            params, config_struct, tile_struct
        )

        cnt = 0
        for _, row in catalog_struct_new.iterrows():
            tile_struct[cnt]["prob"] = row[params["galaxy_grade"]]

            tile_struct[cnt]["galaxies"] = row["galaxies"]

            if config_struct["FOV_type"] == "square":
                tile_struct[cnt]["area"] = params["config"][telescope]["FOV"] ** 2
            elif config_struct["FOV_type"] == "circle":
                tile_struct[cnt]["area"] = (
                    4 * np.pi * params["config"][telescope]["FOV"] ** 2
                )
            else:
                raise ValueError("FOV_type not recognized")
            cnt = cnt + 1

        tile_structs[telescope] = tile_struct

    return tile_structs


def absmag_tiles_struct(params, config_struct, telescope, map_struct, tile_struct):
    keys = tile_struct.keys()
    ntiles = len(keys)
    if ntiles == 0:
        return tile_struct

    if "observability" in map_struct:
        prob = map_struct["observability"][telescope]["prob"]
    else:
        prob = map_struct["prob"]

    n, cl, dist_exp = (
        params["powerlaw_n"],
        params["powerlaw_cl"],
        params["powerlaw_dist_exp"],
    )

    if params["tilesType"] == "galaxy":
        tile_probs = compute_tiles_map(
            params, tile_struct, prob, func="center", ipix_keep=map_struct["ipix_keep"]
        )
    else:
        tile_probs = compute_tiles_map(
            params,
            tile_struct,
            prob,
            func="np.sum(x)",
            ipix_keep=map_struct["ipix_keep"],
        )

    tile_probs[tile_probs < np.max(tile_probs) * 0.01] = 0.0

    prob_scaled = copy.deepcopy(prob)
    prob_sorted = np.sort(prob_scaled)[::-1]
    prob_indexes = np.argsort(prob_scaled)[::-1]
    prob_cumsum = np.cumsum(prob_sorted)
    index = np.argmin(np.abs(prob_cumsum - cl)) + 1
    # prob_scaled[prob_indexes[index:]] = 1e-10
    prob_scaled[prob_indexes[index:]] = 0.0
    prob_scaled = prob_scaled**n
    prob_scaled = prob_scaled / np.nansum(prob_scaled)

    if params["tilesType"] == "galaxy":
        ranked_tile_probs = compute_tiles_map(
            params,
            tile_struct,
            prob_scaled,
            func="center",
            ipix_keep=map_struct["ipix_keep"],
        )
    else:
        ranked_tile_probs = compute_tiles_map(
            params,
            tile_struct,
            prob_scaled,
            func="np.sum(x)",
            ipix_keep=map_struct["ipix_keep"],
        )

    ranked_tile_probs[np.isnan(ranked_tile_probs)] = 0.0
    ranked_tile_probs_thresh = np.max(ranked_tile_probs) * 0.01
    ranked_tile_probs[ranked_tile_probs <= ranked_tile_probs_thresh] = 0.0
    ranked_tile_probs = ranked_tile_probs / np.nansum(ranked_tile_probs)

    if "distmed" in map_struct:
        distmed = map_struct["distmed"] + 2 * map_struct["diststd"]
        distmed[distmed <= 0] = np.nan
        distmed[~np.isfinite(distmed)] = np.nan
        # distmed[distmed<np.nanmedian(distmed)/4.0] = np.nanmedian(distmed)/4.0

        if params["tilesType"] == "galaxy":
            ranked_tile_distances = compute_tiles_map(
                params,
                tile_struct,
                distmed,
                func="center",
                ipix_keep=map_struct["ipix_keep"],
            )
        else:
            ranked_tile_distances = compute_tiles_map(
                params,
                tile_struct,
                distmed,
                func="np.nanmedian(x)",
                ipix_keep=map_struct["ipix_keep"],
            )

        distmod = 5 * np.log10(ranked_tile_distances * 1e6) - 5.0
        absmag = params["absmag"]
        appmag = distmod + absmag
        nmag = appmag - config_struct["magnitude_orig"]

        nexp = np.round(np.exp(nmag * np.log(2.5)))
        nexp[nexp < 0] = 0.0
        nexp = nexp + 1

    keys = tile_struct.keys()
    ranked_tile_times = np.zeros((len(ranked_tile_probs), len(params["exposuretimes"])))
    for ii in range(len(params["exposuretimes"])):
        idx = np.where(ranked_tile_probs > 0)[0]
        ranked_tile_times[idx, ii] = config_struct["exposuretime_orig"] * nexp[idx]
        if "max_exposure" in config_struct.keys():
            idy = np.where(ranked_tile_times[idx, ii] > config_struct["max_exposure"])[
                0
            ]
            ranked_tile_times[idx[idy], ii] = config_struct["max_exposure"]

    for key, prob, exposureTime, tileprob in zip(
        keys, ranked_tile_probs, ranked_tile_times, tile_probs
    ):
        # Try to load the minimum duration of time from telescope config file
        # Otherwise set it to zero
        try:
            min_obs_duration = config_struct["min_observability_duration"] / 24
        except:
            min_obs_duration = 0.0

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
                tileprob = 0.0

        tile_struct[key]["prob"] = tileprob
        if prob == 0.0:
            tile_struct[key]["exposureTime"] = 0.0
            tile_struct[key]["nexposures"] = 0
            tile_struct[key]["filt"] = []
        else:
            if params["doReferences"] and (telescope in ["ZTF", "DECam"]):
                tile_struct[key]["exposureTime"] = []
                tile_struct[key]["nexposures"] = []
                tile_struct[key]["filt"] = []
                if key in config_struct["reference_images"]:
                    for ii in range(len(params["filters"])):
                        if (
                            params["filters"][ii]
                            in config_struct["reference_images"][key]
                        ):
                            tile_struct[key]["exposureTime"].append(exposureTime[ii])
                            tile_struct[key]["filt"].append(params["filters"][ii])
                    tile_struct[key]["nexposures"] = len(
                        tile_struct[key]["exposureTime"]
                    )
                else:
                    tile_struct[key]["exposureTime"] = 0.0
                    tile_struct[key]["nexposures"] = 0
                    tile_struct[key]["filt"] = []
            else:
                tile_struct[key]["exposureTime"] = exposureTime
                tile_struct[key]["nexposures"] = len(params["exposuretimes"])
                tile_struct[key]["filt"] = params["filters"]

    return tile_struct


def powerlaw_tiles_struct(
    params, config_struct, telescope, map_struct, tile_struct, catalog_struct=None
):
    keys = tile_struct.keys()
    ntiles = len(keys)
    if ntiles == 0:
        return tile_struct

    tot_obs_time = config_struct["tot_obs_time"]

    if "observability" in map_struct:
        prob = map_struct["observability"][telescope]["prob"]
    else:
        prob = map_struct["prob"]

    n, cl, dist_exp = (
        params["powerlaw_n"],
        params["powerlaw_cl"],
        params["powerlaw_dist_exp"],
    )

    if params["tilesType"] == "galaxy":
        tile_probs = compute_tiles_map(
            params,
            tile_struct,
            prob,
            func="galaxy",
            ipix_keep=map_struct["ipix_keep"],
            catalog_struct=catalog_struct,
        )
    else:
        tile_probs = compute_tiles_map(
            params,
            tile_struct,
            prob,
            func="np.sum(x)",
            ipix_keep=map_struct["ipix_keep"],
        )

    tile_probs[tile_probs < np.max(tile_probs) * 0.01] = 0.0

    prob_scaled = copy.deepcopy(prob)
    prob_sorted = np.sort(prob_scaled)[::-1]
    prob_indexes = np.argsort(prob_scaled)[::-1]
    prob_cumsum = np.cumsum(prob_sorted)
    index = np.argmin(np.abs(prob_cumsum - cl)) + 1
    # prob_scaled[prob_indexes[index:]] = 1e-10
    prob_scaled[prob_indexes[index:]] = 0.0
    prob_scaled = prob_scaled**n
    prob_scaled = prob_scaled / np.nansum(prob_scaled)

    if params["tilesType"] == "galaxy":
        ranked_tile_probs = compute_tiles_map(
            params,
            tile_struct,
            prob_scaled,
            func="galaxy",
            ipix_keep=map_struct["ipix_keep"],
            catalog_struct=catalog_struct,
        )
    else:
        ranked_tile_probs = compute_tiles_map(
            params,
            tile_struct,
            prob_scaled,
            func="np.sum(x)",
            ipix_keep=map_struct["ipix_keep"],
        )

    ranked_tile_probs[np.isnan(ranked_tile_probs)] = 0.0
    ranked_tile_probs_thresh = np.max(ranked_tile_probs) * 0.01
    ranked_tile_probs[ranked_tile_probs <= ranked_tile_probs_thresh] = 0.0
    ranked_tile_probs = ranked_tile_probs / np.nansum(ranked_tile_probs)

    if "distmed" in map_struct:
        distmed = map_struct["distmed"]
        distmed[distmed <= 0] = np.nan
        distmed[~np.isfinite(distmed)] = np.nan
        # distmed[distmed<np.nanmedian(distmed)/4.0] = np.nanmedian(distmed)/4.0

        if params["tilesType"] == "galaxy":
            ranked_tile_distances = compute_tiles_map(
                params,
                tile_struct,
                distmed,
                func="center",
                ipix_keep=map_struct["ipix_keep"],
            )
        else:
            ranked_tile_distances = compute_tiles_map(
                params,
                tile_struct,
                distmed,
                func="np.nanmedian(x)",
                ipix_keep=map_struct["ipix_keep"],
            )
        ranked_tile_distances_median = ranked_tile_distances / np.nanmedian(
            ranked_tile_distances
        )
        ranked_tile_distances_median = ranked_tile_distances_median**dist_exp
        ranked_tile_probs = ranked_tile_probs * ranked_tile_distances_median
        ranked_tile_probs = ranked_tile_probs / np.nansum(ranked_tile_probs)
        ranked_tile_probs[np.isnan(ranked_tile_probs)] = 0.0

    if params["doSingleExposure"]:
        keys = tile_struct.keys()
        ranked_tile_times = np.zeros(
            (len(ranked_tile_probs), len(params["exposuretimes"]))
        )
        for ii in range(len(params["exposuretimes"])):
            ranked_tile_times[ranked_tile_probs > 0, ii] = params["exposuretimes"][ii]

        for key, prob, exposureTime, tileprob in zip(
            keys, ranked_tile_probs, ranked_tile_times, tile_probs
        ):
            # Try to load the minimum duration of time from telescope config file
            # Otherwise set it to zero
            try:
                min_obs_duration = config_struct["min_observability_duration"] / 24
            except:
                min_obs_duration = 0.0

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
                    tileprob = 0.0

            tile_struct[key]["prob"] = tileprob
            if prob == 0.0:
                tile_struct[key]["exposureTime"] = 0.0
                tile_struct[key]["nexposures"] = 0
                tile_struct[key]["filt"] = []
            else:
                if params["doReferences"] and (telescope in ["ZTF", "DECam"]):
                    tile_struct[key]["exposureTime"] = []
                    tile_struct[key]["nexposures"] = []
                    tile_struct[key]["filt"] = []
                    if key in config_struct["reference_images"]:
                        for ii in range(len(params["filters"])):
                            if (
                                params["filters"][ii]
                                in config_struct["reference_images"][key]
                            ):
                                tile_struct[key]["exposureTime"].append(
                                    exposureTime[ii]
                                )
                                tile_struct[key]["filt"].append(params["filters"][ii])
                        tile_struct[key]["nexposures"] = len(
                            tile_struct[key]["exposureTime"]
                        )
                    else:
                        tile_struct[key]["exposureTime"] = 0.0
                        tile_struct[key]["nexposures"] = 0
                        tile_struct[key]["filt"] = []
                else:
                    tile_struct[key]["exposureTime"] = exposureTime
                    tile_struct[key]["nexposures"] = len(params["exposuretimes"])
                    tile_struct[key]["filt"] = params["filters"]
    else:
        ranked_tile_times = gwemopt.utils.integrationTime(
            tot_obs_time,
            ranked_tile_probs,
            func=None,
            T_int=config_struct["exposuretime"],
        )

        keys = tile_struct.keys()

        for key, prob, exposureTime, tileprob in zip(
            keys, ranked_tile_probs, ranked_tile_times, tile_probs
        ):
            # Try to load the minimum duration of time from telescope config file
            # Otherwise set it to zero
            try:
                min_obs_duration = config_struct["min_observability_duration"] / 24
            except:
                min_obs_duration = 0.0

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
                    prob = 0.0

            tile_struct[key]["prob"] = prob
            tile_struct[key]["exposureTime"] = exposureTime
            tile_struct[key]["nexposures"] = int(
                np.floor(exposureTime / config_struct["exposuretime"])
            )
            tile_struct[key]["filt"] = [config_struct["filt"]] * tile_struct[key][
                "nexposures"
            ]

    return tile_struct


def moc(params, map_struct, moc_structs, doSegments=True):
    tile_structs = {}
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]
        moc_struct = moc_structs[telescope]

        if params["timeallocationType"] == "absmag":
            tile_struct = gwemopt.tiles.absmag_tiles_struct(
                params, config_struct, telescope, map_struct, moc_struct
            )
        elif params["timeallocationType"] == "powerlaw":
            tile_struct = powerlaw_tiles_struct(
                params, config_struct, telescope, map_struct, moc_struct
            )

        if doSegments:
            tile_struct = gwemopt.segments.get_segments_tiles(
                params, config_struct, tile_struct
            )
        tile_structs[telescope] = tile_struct

    return tile_structs


def pem_tiles_struct(params, config_struct, telescope, map_struct, tile_struct):
    tot_obs_time = config_struct["tot_obs_time"]

    if "observability" in map_struct:
        prob = map_struct["observability"][telescope]["prob"]
    else:
        prob = map_struct["prob"]

    ranked_tile_probs = compute_tiles_map(
        params, tile_struct, prob, func="np.sum(x)", ipix_keep=map_struct["ipix_keep"]
    )

    lim_mag = config_struct["magnitude"]
    lim_time = config_struct["exposuretime"]

    tau = np.arange(lim_time, 3600.0, lim_time)
    Loftau = None

    N_ref = 9.7847e9
    L_min = 4.9370e31
    L_max = 4.9370e33

    if "distmu" in map_struct:
        prob = map_struct["prob"]
        distnorm = map_struct["distnorm"]
        distmu = map_struct["distmu"]
        distsigma = map_struct["distsigma"]

        D_min = 1.0e7
        D_max = 1.0e10
        R = np.linspace(D_min / 1e6, D_max / 1e6)
        p_R = [
            np.sum(prob * rr**2 * distnorm * norm(distmu, distsigma).pdf(rr))
            for rr in R
        ]
        p_R = np.array(p_R)
        p_R = p_R / len(prob)

        R = R * 1e6
        D_mu = None
        D_sig = None
    else:
        D_mu = 200.0e6
        D_sig = 60.0e6
        R = None
        p_R = None

    tau, prob = gwemopt.pem.Pem(
        lim_mag,
        lim_time,
        N_ref=N_ref,
        L_min=L_min,
        L_max=L_max,
        tau=tau,
        Loftau=Loftau,
        D_mu=D_mu,
        D_sig=D_sig,
        R=R,
        p_R=p_R,
    )

    tprob, time_allocation = gwemopt.pem.Main(
        tot_obs_time, 0, 0, ranked_tile_probs, tau, prob, "Eq"
    )

    if params["doPlots"]:
        gwemopt.plotting.tauprob(params, tau, prob)

    keys = tile_struct.keys()
    for key, prob, exposureTime in zip(keys, ranked_tile_probs, time_allocation):
        tile_struct[key]["prob"] = prob
        tile_struct[key]["exposureTime"] = exposureTime
        tile_struct[key]["nexposures"] = int(
            np.floor(exposureTime / config_struct["exposuretime"])
        )
        tile_struct[key]["filt"] = [config_struct["filt"]] * tile_struct[key][
            "nexposures"
        ]

    return tile_struct


def compute_tiles_map(
    params, tile_struct, skymap, func=None, ipix_keep=[], catalog_struct=None
):
    if func is None:
        f = lambda x: np.sum(x)
    elif func == "center":
        keys = tile_struct.keys()
        ntiles = len(keys)
        vals = np.nan * np.ones((ntiles,))
        nside = hp.npix2nside(len(skymap))
        for ii, key in enumerate(tile_struct.keys()):
            pix_center = hp.ang2pix(
                nside, tile_struct[key]["ra"], tile_struct[key]["dec"], lonlat=True
            )
            val = skymap[pix_center]
            vals[ii] = val
        return vals
    elif func == "galaxy":
        keys = tile_struct.keys()
        ntiles = len(keys)
        vals = np.nan * np.ones((ntiles,))
        nside = hp.npix2nside(len(skymap))
        for ii, key in enumerate(tile_struct.keys()):
            galaxies = tile_struct[key]["galaxies"]
            val = np.sum(catalog_struct[params["galaxy_grade"]][galaxies])
            vals[ii] = val
        return vals
    else:
        f = lambda x: eval(func)

    prob = copy.deepcopy(skymap)

    keys = tile_struct.keys()
    ntiles = len(keys)
    vals = np.nan * np.ones((ntiles,))
    for ii, key in enumerate(tile_struct.keys()):
        idx = np.where(prob[tile_struct[key]["ipix"]] == -1)[0]
        idx = np.setdiff1d(idx, ipix_keep)
        if len(prob[tile_struct[key]["ipix"]]) == 0:
            rat = 0.0
        else:
            rat = float(len(idx)) / float(len(prob[tile_struct[key]["ipix"]]))
        if rat > params["maximumOverlap"]:
            vals[ii] = 0.0
        else:
            ipix = tile_struct[key]["ipix"]
            if len(ipix) == 0:
                vals[ii] = 0.0
            else:
                vals_to_sum = prob[ipix]
                if func == "np.nanmedian(x)":
                    vals_to_sum[vals_to_sum < 0] = np.nan
                else:
                    vals_to_sum[vals_to_sum < 0] = 0
                vals[ii] = f(vals_to_sum)

        ipix_keep = np.setdiff1d(ipix_keep, tile_struct[key]["ipix"])
        ipix_slice = np.setdiff1d(tile_struct[key]["ipix"], ipix_keep)
        if len(ipix_slice) > 0:
            prob[ipix_slice] = -1

    return vals


def tesselation_spiral(config_struct, scale=0.80):
    if config_struct["FOV_type"] == "square":
        FOV = config_struct["FOV"] * config_struct["FOV"] * scale
    elif config_struct["FOV_type"] == "circle":
        FOV = np.pi * config_struct["FOV"] * config_struct["FOV"] * scale

    area_of_sphere = 4 * np.pi * (180 / np.pi) ** 2
    n = int(np.ceil(area_of_sphere / FOV))
    print("Using %d points to tile the sphere..." % n)

    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)

    points = np.zeros((n, 3))
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = z

    ra, dec = hp.pixelfunc.vec2ang(points, lonlat=True)
    fid = open(config_struct["tesselationFile"], "w")
    for ii in range(len(ra)):
        fid.write("%d %.5f %.5f\n" % (ii, ra[ii], dec[ii]))
    fid.close()


def tesselation_packing(config_struct, scale=0.97):
    sphere_radius = 1.0
    if config_struct["FOV_type"] == "square":
        circle_radius = np.deg2rad(config_struct["FOV"] / 2.0) * scale
    elif config_struct["FOV_type"] == "circle":
        circle_radius = np.deg2rad(config_struct["FOV"]) * scale
    vertical_count = int((np.pi * sphere_radius) / (2 * circle_radius))

    phis = []
    thetas = []

    phi = -0.5 * np.pi
    phi_step = np.pi / vertical_count
    while phi < 0.5 * np.pi:
        horizontal_count = int(
            (2 * np.pi * np.cos(phi) * sphere_radius) / (2 * circle_radius)
        )
        if horizontal_count == 0:
            horizontal_count = 1
        theta = 0
        theta_step = 2 * np.pi / horizontal_count
        while theta < 2 * np.pi - 1e-8:
            phis.append(phi)
            thetas.append(theta)
            theta += theta_step
        phi += phi_step
    dec = np.array(np.rad2deg(phis))
    ra = np.array(np.rad2deg(thetas))

    fid = open(config_struct["tesselationFile"], "w")
    for ii in range(len(ra)):
        fid.write("%d %.5f %.5f\n" % (ii, ra[ii], dec[ii]))
    fid.close()
