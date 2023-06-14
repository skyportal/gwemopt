import copy

import astropy.coordinates
import astropy.units as u
import ligo.segments as segments
import numpy as np
from astropy.time import Time
from munkres import Munkres, make_cost_matrix

from gwemopt.tiles import balance_tiles, optimize_max_tiles, schedule_alternating
from gwemopt.utils import angular_distance


def get_altaz_tiles(ras, decs, observatory, obstime):
    # Convert to RA, Dec.
    radecs = astropy.coordinates.SkyCoord(
        ra=np.array(ras) * u.degree, dec=np.array(decs) * u.degree, frame="icrs"
    )

    # Alt/az reference frame at observatory, now
    frame = astropy.coordinates.AltAz(obstime=obstime, location=observatory)

    # Transform grid to alt/az coordinates at observatory, now
    altaz = radecs.transform_to(frame)

    return altaz


def find_tile(
    exposureids_tile,
    exposureids,
    probs,
    idxs=None,
    exptimecheckkeys=[],
    current_ra=np.nan,
    current_dec=np.nan,
    slew_rate=1,
    readout=1,
):
    # exposureids_tile: {expo id}-> list of the tiles available for observation
    # exposureids: list of tile ids for every exposure it is allocated to observe

    if idxs is not None:
        for idx in idxs:
            if len(exposureids_tile["exposureids"]) - 1 < idx:
                continue
            idx2 = exposureids_tile["exposureids"][idx]
            if idx2 in exposureids and not idx2 in exptimecheckkeys:
                idx = exposureids.index(idx2)
                exposureids.pop(idx)
                probs.pop(idx)
                return idx2, exposureids, probs

    findTile = True
    while findTile:
        if not exposureids_tile["probs"]:
            idx2 = -1
            findTile = False
            break
        if (not np.isnan(current_ra)) and (not np.isnan(current_dec)):
            dist = angular_distance(
                current_ra,
                current_dec,
                np.array(exposureids_tile["ras"]),
                np.array(exposureids_tile["decs"]),
            )
            slew_readout = readout / (dist / slew_rate)
            slew_readout[slew_readout > 1] = 1.0
            score = np.array(exposureids_tile["probs"]) * slew_readout
            idx = np.argmax(score)
        else:
            idx = np.argmax(exposureids_tile["probs"])
        idx2 = exposureids_tile["exposureids"][idx]
        if exposureids:
            if idx2 in exposureids and not idx2 in exptimecheckkeys:
                idx = exposureids.index(idx2)
                exposureids.pop(idx)
                probs.pop(idx)
                findTile = False
            else:
                exposureids_tile["exposureids"].pop(idx)
                exposureids_tile["probs"].pop(idx)
                exposureids_tile["ras"].pop(idx)
                exposureids_tile["decs"].pop(idx)
        else:
            findTile = False

    return idx2, exposureids, probs


def get_order(
    params, tile_struct, tilesegmentlists, exposurelist, observatory, config_struct
):
    """
    tile_struct: dictionary. key -> struct info.
    tilesegmentlists: list of lists. Segments for each tile in tile_struct
        that are available for observation.
    exposurelist: list of segments that the telescope is supposed to be working.
        consecutive segments from the start to the end, with each segment size
        being the exposure time.
    Returns a list of tile indices in the order of observation.
    """
    keys = tile_struct.keys()

    exposureids_tiles = {}
    first_exposure = np.inf * np.ones((len(keys),))
    last_exposure = -np.inf * np.ones((len(keys),))
    tileprobs = np.zeros((len(keys),))
    tilenexps = np.zeros((len(keys),))
    tileexptime = np.zeros((len(keys),))
    tileexpdur = np.zeros((len(keys),))
    tilefilts = {}
    tileavailable = np.zeros((len(keys),))
    tileavailable_tiles = {}
    keynames = []

    nexps = 0
    for jj, key in enumerate(keys):
        tileprobs[jj] = tile_struct[key]["prob"]
        tilenexps[jj] = tile_struct[key]["nexposures"]
        try:
            tileexpdur[jj] = tile_struct[key]["exposureTime"]
        except:
            try:
                tileexpdur[jj] = tile_struct[key]["exposureTime"][0]
            except:
                tileexpdur[jj] = 0.0

        tilefilts[key] = copy.deepcopy(tile_struct[key]["filt"])
        tileavailable_tiles[jj] = []
        keynames.append(key)

        nexps = nexps + tile_struct[key]["nexposures"]

    if "dec_constraint" in config_struct:
        dec_constraint = config_struct["dec_constraint"].split(",")
        dec_min = float(dec_constraint[0])
        dec_max = float(dec_constraint[1])

    for ii in range(len(exposurelist)):
        exposureids_tiles[ii] = {}
        exposureids = []
        probs = []
        ras, decs = [], []
        for jj, key in enumerate(keys):
            tilesegmentlist = tilesegmentlists[jj]
            if tile_struct[key]["prob"] == 0:
                continue
            if "dec_constraint" in config_struct:
                if (tile_struct[key]["dec"] < dec_min) or (
                    tile_struct[key]["dec"] > dec_max
                ):
                    continue
            if "epochs" in tile_struct[key]:
                if params.get("doMindifFilt", False):
                    if "epochs_filters" not in tile_struct[key]:
                        tile_struct[key]["epochs_filters"] = []
                    # take into account filter for mindiff
                    idx = np.where(
                        np.asarray(tile_struct[key]["epochs_filters"])
                        == params["filters"][0]
                    )[0]
                    if np.any(
                        np.abs(exposurelist[ii][0] - tile_struct[key]["epochs"][idx, 2])
                        < params["mindiff"] / 86400.0
                    ):
                        continue
                elif np.any(
                    np.abs(exposurelist[ii][0] - tile_struct[key]["epochs"][:, 2])
                    < params["mindiff"] / 86400.0
                ):
                    continue
            if tilesegmentlist.intersects_segment(exposurelist[ii]):
                exposureids.append(key)
                probs.append(tile_struct[key]["prob"])
                ras.append(tile_struct[key]["ra"])
                decs.append(tile_struct[key]["dec"])

                first_exposure[jj] = np.min([first_exposure[jj], ii])
                last_exposure[jj] = np.max([last_exposure[jj], ii])
                tileavailable_tiles[jj].append(ii)
                tileavailable[jj] = tileavailable[jj] + 1
        # in every exposure, the tiles available for observation
        exposureids_tiles[ii]["exposureids"] = exposureids  # list of tile ids
        exposureids_tiles[ii]["probs"] = probs  # the corresponding probs
        exposureids_tiles[ii]["ras"] = ras
        exposureids_tiles[ii]["decs"] = decs

    exposureids = []
    probs = []
    ras, decs = [], []
    for ii, key in enumerate(keys):
        # tile_struct[key]["nexposures"]: the number of exposures assigned to this tile
        for jj in range(tile_struct[key]["nexposures"]):
            exposureids.append(
                key
            )  # list of tile ids for every exposure it is allocated to observe
            probs.append(tile_struct[key]["prob"])
            ras.append(tile_struct[key]["ra"])
            decs.append(tile_struct[key]["dec"])

    idxs = -1 * np.ones((len(exposureids_tiles.keys()),))
    filts = ["n"] * len(exposureids_tiles.keys())

    if nexps == 0:
        return idxs, filts

    if params["scheduleType"] == "airmass_weighted":
        # # first step is to sort the array in order of descending probability
        indsort = np.argsort(-np.array(probs))
        probs = np.array(probs)[indsort]
        ras = np.array(ras)[indsort]
        decs = np.array(decs)[indsort]
        exposureids = np.array(exposureids)[indsort]

        tilematrix = np.zeros((len(exposurelist), len(ras)))
        probmatrix = np.zeros((len(exposurelist), len(ras)))

        for ii in np.arange(len(exposurelist)):
            # first, create an array of airmass-weighted probabilities
            t = Time(exposurelist[ii][0], format="mjd")
            altaz = get_altaz_tiles(ras, decs, observatory, t)
            alts = altaz.alt.degree
            horizon = config_struct["horizon"]
            horizon_mask = alts <= horizon
            airmass = 1 / np.cos((90.0 - alts) * np.pi / 180.0)
            below_horizon_mask = horizon_mask * 10.0**100
            airmass = airmass + below_horizon_mask
            airmass_weight = 10 ** (0.4 * 0.1 * (airmass - 1))
            tilematrix[ii, :] = np.array(probs / airmass_weight)
            probmatrix[ii, :] = np.array(probs * (True ^ horizon_mask))

    dt = int(np.ceil((exposurelist[1][0] - exposurelist[0][0]) * 86400))
    if params["scheduleType"] == "greedy":
        for ii in np.arange(len(exposurelist)):
            if idxs[ii] > 0:
                continue

            exptimecheck = np.where(
                exposurelist[ii][0] - tileexptime < params["mindiff"] / 86400.0
            )[0]
            exptimecheckkeys = [keynames[x] for x in exptimecheck]

            # restricted by availability of tile and timeallocation
            idx2, exposureids, probs = find_tile(
                exposureids_tiles[ii],
                exposureids,
                probs,
                exptimecheckkeys=exptimecheckkeys,
            )
            if idx2 in keynames:
                idx = keynames.index(idx2)
                tilenexps[idx] = tilenexps[idx] - 1
                tileexptime[idx] = exposurelist[ii][0]

                num = int(np.ceil(tileexpdur[idx] / dt))
                tilenexps[idx] = tilenexps[idx] - 1
                tileexptime[idx] = exposurelist[ii][0]
                if len(tilefilts[idx2]) > 0:
                    filt = tilefilts[idx2].pop(0)
                    for jj in range(num):
                        try:
                            filts[ii + jj] = filt
                        except:
                            pass
                for jj in range(num):
                    try:
                        idxs[ii + jj] = idx2
                    except:
                        pass
            else:
                idxs[ii] = idx2

            if not exposureids:
                break
    elif params["scheduleType"] == "greedy_slew":
        current_ra, current_dec = np.nan, np.nan
        for ii in np.arange(len(exposurelist)):
            exptimecheck = np.where(
                exposurelist[ii][0] - tileexptime < params["mindiff"] / 86400.0
            )[0]
            exptimecheckkeys = [keynames[x] for x in exptimecheck]

            # find_tile finds the tile that covers the largest probablity
            # restricted by availability of tile and timeallocation
            idx2, exposureids, probs = find_tile(
                exposureids_tiles[ii],
                exposureids,
                probs,
                exptimecheckkeys=exptimecheckkeys,
                current_ra=current_ra,
                current_dec=current_dec,
                slew_rate=config_struct["slew_rate"],
                readout=config_struct["readout"],
            )
            if idx2 in keynames:
                idx = keynames.index(idx2)
                tilenexps[idx] = tilenexps[idx] - 1
                tileexptime[idx] = exposurelist[ii][0]

                num = int(np.ceil(tileexpdur[idx] / dt))
                tilenexps[idx] = tilenexps[idx] - 1
                tileexptime[idx] = exposurelist[ii][0]
                if len(tilefilts[idx2]) > 0:
                    filt = tilefilts[idx2].pop(0)
                    for jj in range(num):
                        try:
                            filts[ii + jj] = filt
                        except:
                            pass
                for jj in range(num):
                    try:
                        idxs[ii + jj] = idx2
                    except:
                        pass
                current_ra = tile_struct[idx2]["ra"]
                current_dec = tile_struct[idx2]["dec"]
            else:
                idxs[ii] = idx2

            if not exposureids:
                break
    elif params["scheduleType"] == "sear":
        # for ii in np.arange(len(exposurelist)):
        iis = np.arange(len(exposurelist)).tolist()
        while len(iis) > 0:
            ii = iis[0]
            mask = np.where((ii == last_exposure) & (tilenexps > 0))[0]

            exptimecheck = np.where(
                exposurelist[ii][0] - tileexptime < params["mindiff"] / 86400.0
            )[0]
            exptimecheckkeys = [keynames[x] for x in exptimecheck]

            if len(mask) > 0:
                idxsort = mask[np.argsort(tileprobs[mask])]
                idx2, exposureids, probs = find_tile(
                    exposureids_tiles[ii],
                    exposureids,
                    probs,
                    idxs=idxsort,
                    exptimecheckkeys=exptimecheckkeys,
                )
                last_exposure[mask] = last_exposure[mask] + 1
            else:
                idx2, exposureids, probs = find_tile(
                    exposureids_tiles[ii],
                    exposureids,
                    probs,
                    exptimecheckkeys=exptimecheckkeys,
                )
            if idx2 in keynames:
                idx = keynames.index(idx2)
                tilenexps[idx] = tilenexps[idx] - 1
                tileexptime[idx] = exposurelist[ii][0]
                if len(tilefilts[idx2]) > 0:
                    filt = tilefilts[idx2].pop(0)
                    filts[ii] = filt
            idxs[ii] = idx2
            iis.pop(0)

            if not exposureids:
                break

    elif params["scheduleType"] == "weighted":
        for ii in np.arange(len(exposurelist)):
            jj = exposureids_tiles[ii]["exposureids"]
            weights = tileprobs[jj] * tilenexps[jj] / tileavailable[jj]
            weights[~np.isfinite(weights)] = 0.0

            exptimecheck = np.where(
                exposurelist[ii][0] - tileexptime < params["mindiff"] / 86400.0
            )[0]
            weights[exptimecheck] = 0.0

            if np.any(weights >= 0):
                idxmax = np.argmax(weights)
                idx2 = jj[idxmax]
                if idx2 in keynames:
                    idx = keynames.index(idx2)
                    tilenexps[idx] = tilenexps[idx] - 1
                    tileexptime[idx] = exposurelist[ii][0]
                    if len(tilefilts[idx2]) > 0:
                        filt = tilefilts[idx2].pop(0)
                        filts[ii] = filt
                idxs[ii] = idx2
            tileavailable[jj] = tileavailable[jj] - 1

    elif params["scheduleType"] == "airmass_weighted":
        # then use the Hungarian algorithm (munkres) to schedule high prob tiles at low airmass
        tilematrix_mask = tilematrix > 10 ** (-10)

        if tilematrix_mask.any():
            print("Calculating Hungarian solution...")
            total_cost = 0
            cost_matrix = make_cost_matrix(tilematrix)
            m = Munkres()
            optimal_points = m.compute(cost_matrix)
            print("Hungarian solution calculated...")
            max_no_observ = min(tilematrix.shape)
            for jj in range(max_no_observ):
                idx0, idx1 = optimal_points[jj]
                # idx0 indexes over the time windows, idx1 indexes over the probabilities
                # idx2 gets the exposure id of the tile, used to assign tileexptime and tilenexps
                try:
                    idx2 = exposureids[idx1]
                    pamw = tilematrix[idx0][idx1]
                    total_cost += pamw
                    if len(tilefilts[idx2]) > 0:
                        filt = tilefilts[idx2].pop(0)
                        filts[idx0] = filt
                except:
                    continue

                idxs[idx0] = idx2

        else:
            print("The localization is not visible from the site.")

    else:
        raise ValueError(
            "Scheduling options are greedy/sear/weighted/airmass_weighted, or with _slew."
        )

    return idxs, filts


def scheduler(params, config_struct, tile_struct):
    """
    config_struct: the telescope configurations
    tile_struct: the tiles, contains time allocation information
    """
    import gwemopt.segments

    # import gwemopt.segments_astroplan
    coverage_struct = {}
    coverage_struct["data"] = np.empty((0, 8))
    coverage_struct["filters"] = []
    coverage_struct["ipix"] = []
    coverage_struct["patch"] = []
    coverage_struct["area"] = []
    if params["tilesType"] == "galaxy":
        coverage_struct["galaxies"] = []

    observatory = astropy.coordinates.EarthLocation(
        lat=config_struct["latitude"] * u.deg,
        lon=config_struct["longitude"] * u.deg,
        height=config_struct["elevation"] * u.m,
    )

    exposurelist = config_struct["exposurelist"]
    # tilesegmentlists = gwemopt.segments_astroplan.get_segments_tiles(config_struct, tile_struct, observatory, segmentlist)
    tilesegmentlists = []
    keys = tile_struct.keys()
    for key in keys:
        # segments.py: tile_struct[key]["segmentlist"] is a list of segments when the tile is available for observation
        tilesegmentlists.append(tile_struct[key]["segmentlist"])
    print("Generating schedule order...")
    keys, filts = get_order(
        params, tile_struct, tilesegmentlists, exposurelist, observatory, config_struct
    )

    if params["doPlots"]:
        from gwemopt.plotting import make_schedule_plots

        make_schedule_plots(params, exposurelist, keys)

    exposureused = np.where(np.array(keys) >= 0)[0]
    coverage_struct["exposureused"] = exposureused
    while len(exposurelist) > 0:
        key, filt = keys[0], filts[0]
        if key == -1:
            keys = keys[1:]
            filts = filts[1:]
            exposurelist = exposurelist[1:]
        else:
            tile_struct_hold = tile_struct[key]
            mjd_exposure_start = exposurelist[0][0]
            nkeys = len(keys)
            for jj in range(nkeys):
                if (keys[jj] == key) and (filts[jj] == filt) and not (nkeys == jj + 1):
                    if np.abs(exposurelist[jj][1] - mjd_exposure_start) > 5.0 / 24:
                        mjd_exposure_end = exposurelist[jj - 1][1]
                        keys = keys[jj:]
                        filts = filts[jj:]
                        exposurelist = exposurelist[jj:]
                        break
                    else:
                        mjd_exposure_end = exposurelist[jj][1]

                elif (keys[jj] == key) and (filts[jj] == filt) and (nkeys == jj + 1):
                    mjd_exposure_end = exposurelist[jj][1]

                    exposureTime = (mjd_exposure_end - mjd_exposure_start) * 86400.0

                    keys = []
                    filts = []
                    exposurelist = []
                else:
                    keys = keys[jj:]
                    filts = filts[jj:]
                    exposurelist = exposurelist[jj:]
                    break

            # calculate airmass for each tile at the start of its exposure:
            t = Time(mjd_exposure_start, format="mjd")
            altaz = get_altaz_tiles(
                tile_struct_hold["ra"], tile_struct_hold["dec"], observatory, t
            )
            alt = altaz.alt.degree
            airmass = 1 / np.cos((90.0 - alt) * np.pi / 180)

            # total duration of the observation (?)
            exposureTime = (mjd_exposure_end - mjd_exposure_start) * 86400.0

            nmag = -2.5 * np.log10(
                np.sqrt(config_struct["exposuretime"] / exposureTime)
            )
            mag = config_struct["magnitude"] + nmag

            coverage_struct["data"] = np.append(
                coverage_struct["data"],
                np.array(
                    [
                        [
                            tile_struct_hold["ra"],
                            tile_struct_hold["dec"],
                            mjd_exposure_start,
                            mag,
                            exposureTime,
                            int(key),
                            tile_struct_hold["prob"],
                            airmass,
                        ]
                    ]
                ),
                axis=0,
            )

            coverage_struct["filters"].append(filt)
            coverage_struct["patch"].append(tile_struct_hold["patch"])
            coverage_struct["ipix"].append(tile_struct_hold["ipix"])
            coverage_struct["area"].append(tile_struct_hold["area"])
            if params["tilesType"] == "galaxy":
                coverage_struct["galaxies"].append(tile_struct_hold["galaxies"])

    coverage_struct["area"] = np.array(coverage_struct["area"])
    coverage_struct["filters"] = np.array(coverage_struct["filters"])
    coverage_struct["FOV"] = [config_struct["FOV"]] * len(coverage_struct["filters"])
    coverage_struct["telescope"] = [config_struct["telescope"]] * len(
        coverage_struct["filters"]
    )

    return coverage_struct


def computeSlewReadoutTime(config_struct, coverage_struct):
    slew_rate = config_struct["slew_rate"]
    readout = config_struct["readout"]
    prev_ra = config_struct["latitude"]
    prev_dec = config_struct["longitude"]
    acc_time = 0
    for dat in coverage_struct["data"]:
        dist = angular_distance(prev_ra, prev_dec, dat[0], dat[1])
        slew_readout_time = np.max([dist / slew_rate, readout])
        acc_time += slew_readout_time
        prev_dec = dat[0]
        prev_ra = dat[1]
    return acc_time


def schedule_ra_splits(
    params,
    config_struct,
    map_struct_hold,
    tile_struct,
    telescope,
    previous_coverage_struct,
):
    location = astropy.coordinates.EarthLocation(
        config_struct["longitude"],
        config_struct["latitude"],
        config_struct["elevation"],
    )

    raslices = gwemopt.utils.utils.auto_rasplit(
        params, map_struct_hold, params["nside_down"]
    )

    maxidx = 0
    coverage_structs = []
    skip = False
    while len(raslices) != 0:
        params["unbalanced_tiles"] = []
        config_struct["exposurelist"] = segments.segmentlist(
            config_struct["exposurelist"][maxidx:]
        )
        if len(config_struct["exposurelist"]) < 2:
            break

        map_struct_slice = copy.deepcopy(map_struct_hold)

        exposurelist = np.array_split(config_struct["exposurelist"], len(raslices))[0]
        minhas = []
        minhas_late = []
        try_end = False
        if len(raslices) == 1:
            raslice = raslices[0]
            del raslices[0]
        else:
            for raslice in raslices:
                has = []
                has_late = []
                for seg in exposurelist:
                    mjds = np.linspace(seg[0], seg[1], 100)
                    tt = Time(mjds, format="mjd", scale="utc", location=location)
                    lst = tt.sidereal_time("mean") / u.hourangle
                    ha = np.abs(lst - raslice[0])
                    ha_late = np.abs(lst - raslice[1])

                    idx = np.where(ha > 12.0)[0]
                    ha[idx] = 24.0 - ha[idx]
                    idx_late = np.where(ha_late > 12.0)[0]
                    ha_late[idx_late] = 24.0 - ha_late[idx_late]
                    has += list(ha)
                    has_late += list(ha_late)
                if len(has) > 0:
                    minhas.append(np.min(has))
                if len(has_late) > 0:
                    minhas_late.append(np.min(has_late))

            if (len(minhas_late) > 0) and (len(has_late) > 0):
                # conditions for trying to schedule end of slice
                if np.min(minhas_late) <= 5.0 and np.min(has) > 4.0 and not skip:
                    try_end = True
                    min = np.argmin(minhas_late)
                    raslice = raslices[min]
                else:
                    min = np.argmin(minhas)
                    raslice = raslices[min]
                    del raslices[min]
            else:
                min = np.argmin(minhas)
                raslice = raslices[min]
                del raslices[min]

        # do RA slicing
        ra_low, ra_high = raslice[0], raslice[1]
        ra = map_struct_slice["ra"]
        if ra_low <= ra_high:
            ipix = np.where(
                (ra_high * 360.0 / 24.0 < ra) | (ra_low * 360.0 / 24.0 > ra)
            )[0]
        else:
            ipix = np.where(
                (ra_high * 360.0 / 24.0 < ra) & (ra_low * 360.0 / 24.0 > ra)
            )[0]

        map_struct_slice["prob"][ipix] = 0.0

        if params["timeallocationType"] == "absmag":
            tile_struct = gwemopt.tiles.absmag_tiles_struct(
                params, config_struct, telescope, map_struct_slice, tile_struct
            )
        else:
            tile_struct = gwemopt.tiles.powerlaw_tiles_struct(
                params, config_struct, telescope, map_struct_slice, tile_struct
            )

        config_struct_hold = copy.copy(config_struct)
        coverage_struct, tile_struct = schedule_alternating(
            params,
            config_struct_hold,
            telescope,
            map_struct_slice,
            tile_struct,
            previous_coverage_struct,
        )
        if len(coverage_struct["ipix"]) == 0:
            continue
        optimized_max, coverage_struct, tile_struct = optimize_max_tiles(
            params,
            tile_struct,
            coverage_struct,
            config_struct,
            telescope,
            map_struct_slice,
        )
        params["max_nb_tiles"] = np.array([optimized_max], dtype=float)
        balanced_fields = 0
        coverage_struct, tile_struct = schedule_alternating(
            params,
            config_struct,
            telescope,
            map_struct_slice,
            tile_struct,
            previous_coverage_struct,
        )

        doReschedule, balanced_fields = balance_tiles(
            params, tile_struct, coverage_struct
        )
        config_struct_hold = copy.copy(config_struct)

        if balanced_fields == 0:
            if try_end:
                skip = True
            continue
        elif try_end:
            del raslices[min]
        skip = False

        if len(coverage_struct["exposureused"]) > 0:
            maxidx = int(coverage_struct["exposureused"][-1])

        coverage_struct = gwemopt.utils.utils.erase_unbalanced_tiles(
            params, coverage_struct
        )

        # limit to max number of filter sets
        if len(coverage_structs) < params["max_filter_sets"]:
            coverage_structs.append(coverage_struct)
        else:
            prob_structs = [
                np.sum(prev_struct["data"][:, 6]) for prev_struct in coverage_structs
            ]
            if np.any(np.array(prob_structs) < np.sum(coverage_struct["data"][:, 6])):
                argmin = np.argmin(prob_structs)
                del coverage_structs[argmin]
                coverage_structs.append(coverage_struct)

    return gwemopt.coverage.combine_coverage_structs(coverage_structs)
