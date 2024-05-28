import copy

import astropy.coordinates
import astropy.units as u
import ephem
import ligo.segments as segments
import numpy as np
from astropy.time import Time
from ortools.linear_solver import pywraplp

from gwemopt.tiles import balance_tiles, optimize_max_tiles, schedule_alternating
from gwemopt.utils import angular_distance, solve_milp


def get_altaz_tile(ra, dec, observer, obstime):

    observer.date = ephem.Date(obstime.iso)

    fxdbdy = ephem.FixedBody()
    fxdbdy._ra = ephem.degrees(str(ra))
    fxdbdy._dec = ephem.degrees(str(dec))
    fxdbdy.compute(observer)

    return float(repr(fxdbdy.alt)) * (360 / (2 * np.pi)), float(repr(fxdbdy.az)) * (
        360 / (2 * np.pi)
    )


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


def get_order_heuristic(
    params, tile_struct, tilesegmentlists, exposurelist, observer, config_struct
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

        if type(tile_struct[key]["exposureTime"]) in [float, np.float64]:
            tileexpdur[jj] = tile_struct[key]["exposureTime"]
        elif type(tile_struct[key]["exposureTime"]) in [list, np.ndarray]:
            tileexpdur[jj] = tile_struct[key]["exposureTime"][0]
        else:
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
            altazs = [
                get_altaz_tile(ra, dec, observer, t) for ra, dec in zip(ras, decs)
            ]
            alts = np.array([altaz[0] for altaz in altazs])
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
            print("Calculating MILP solution...")
            total_cost = 0

            maximum = max(max(row) for row in tilematrix)
            inversion_function = lambda x: maximum - x

            cost_matrix = []
            for row in tilematrix:
                cost_matrix.append([inversion_function(value) for value in row])

            # Create a linear solver
            solver = pywraplp.Solver.CreateSolver("CBC")

            # Define variables
            num_workers = len(cost_matrix)
            num_tasks = len(cost_matrix[0])
            x = {}

            for i in range(num_workers):
                for j in range(num_tasks):
                    x[i, j] = solver.BoolVar("x[%i,%i]" % (i, j))

            # Define constraints: each worker is assigned to at most one task
            for i in range(num_workers):
                solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) <= 1)

            # Define constraints: each task is assigned to exactly one worker
            for j in range(num_tasks):
                solver.Add(solver.Sum([x[i, j] for i in range(num_workers)]) == 1)

            # Define objective function: minimize the total cost
            objective = solver.Objective()
            for i in range(num_workers):
                for j in range(num_tasks):
                    objective.SetCoefficient(x[i, j], cost_matrix[i][j])
            objective.SetMinimization()

            # Solve the problem
            status = solver.Solve()

            optimal_points = []
            if status == pywraplp.Solver.OPTIMAL:
                print("Total cost =", solver.Objective().Value())
                for i in range(num_workers):
                    for j in range(num_tasks):
                        if x[i, j].solution_value() > 0:
                            optimal_points.append((i, j))
            else:
                print("The problem does not have an optimal solution.")

            max_no_observ = min(tilematrix.shape)
            for jj in range(max_no_observ):
                # idx0 indexes over the time windows, idx1 indexes over the probabilities
                # idx2 gets the exposure id of the tile, used to assign tileexptime and tilenexps
                try:
                    idx0, idx1 = optimal_points[jj]
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


def get_order_milp(params, tile_struct, exposurelist, observer, config_struct):
    """
    tile_struct: dictionary. key -> struct info.
    exposurelist: list of segments that the telescope is supposed to be working.
        consecutive segments from the start to the end, with each segment size
        being the exposure time.
    Returns a list of tile indices in the order of observation.
    """

    if "dec_constraint" in config_struct:
        dec_constraint = config_struct["dec_constraint"].split(",")
        dec_min = float(dec_constraint[0])
        dec_max = float(dec_constraint[1])

    exposureids = []
    probs = []
    ras, decs, filts, keys = [], [], [], []
    for ii, key in enumerate(list(tile_struct.keys())):
        if tile_struct[key]["prob"] == 0:
            continue
        if "dec_constraint" in config_struct:
            if (tile_struct[key]["dec"] < dec_min) or (
                tile_struct[key]["dec"] > dec_max
            ):
                continue

        exposureids.append(key)
        probs.append(tile_struct[key]["prob"])
        ras.append(tile_struct[key]["ra"])
        decs.append(tile_struct[key]["dec"])
        filts.append(tile_struct[key]["filt"])
        keys.append(key)

    fields = -1 * np.ones(
        (len(exposurelist)),
    )
    filters = ["n"] * len(exposurelist)

    if len(probs) == 0:
        return fields, filters

    probs = np.array(probs)
    ras = np.array(ras)
    decs = np.array(decs)
    exposureids = np.array(exposureids)
    keys = np.array(keys)
    tilematrix = np.zeros((len(exposurelist), len(ras)))
    for ii in np.arange(len(exposurelist)):
        # first, create an array of airmass-weighted probabilities
        t = Time(exposurelist[ii][0], format="mjd")
        altazs = [get_altaz_tile(ra, dec, observer, t) for ra, dec in zip(ras, decs)]
        alts = np.array([altaz[0] for altaz in altazs])
        horizon = config_struct["horizon"]
        horizon_mask = alts <= horizon
        airmass = 1 / np.cos((90.0 - alts) * np.pi / 180.0)
        airmass_mask = airmass > params["airmass"]

        airmass_weight = 10 ** (0.4 * 0.1 * (airmass - 1))

        if params["scheduleType"] in ["greedy", "greedy_slew"]:
            tilematrix[ii, :] = np.array(probs)
        elif params["scheduleType"] == ["airmass_weighted", "airmass_weighted_slew"]:
            tilematrix[ii, :] = np.array(probs / airmass_weight)
        tilematrix[ii, horizon_mask] = np.nan
        tilematrix[ii, airmass_mask] = np.nan

        for jj, key in enumerate(keys):
            tilesegmentlist = tile_struct[key]["segmentlist"]
            if not tilesegmentlist.intersects_segment(exposurelist[ii]):
                tilematrix[ii, jj] = np.nan

    # which fields are never observable
    ind = np.where(np.nansum(tilematrix, axis=0) > 0)[0]

    probs = np.array(probs)[ind]
    ras = np.array(ras)[ind]
    decs = np.array(decs)[ind]
    exposureids = np.array(exposureids)[ind]
    filts = [filts[i] for i in ind]
    tilematrix = tilematrix[:, ind]

    # which times do not have any observability
    ind = np.where(np.nansum(tilematrix, axis=1) > 0)[0]
    tilematrix = tilematrix[ind, :]

    cost_matrix = tilematrix
    cost_matrix[np.isnan(cost_matrix)] = -np.inf

    distmatrix = np.zeros((len(ras), len(ras)))
    for ii, (r, d) in enumerate(zip(ras, decs)):
        dist = angular_distance(r, d, ras, decs)
        if "slew" in params["scheduleType"]:
            dist = dist / config_struct["slew_rate"]
            dist = dist - config_struct["readout"]
            dist[dist < 0] = 0
        else:
            distmatrix[ii, :] = dist
    distmatrix = distmatrix / np.max(distmatrix)

    dt = int(np.ceil((exposurelist[1][0] - exposurelist[0][0]) * 86400))
    optimal_points = solve_milp(
        cost_matrix,
        dist_matrix=distmatrix,
        useDistance=False,
        max_tasks_per_worker=len(params["filters"]),
        useTaskSepration=False,
        min_task_separation=int(np.ceil(dt / params["mindiff"])),
    )

    for optimal_point in optimal_points:
        idx = ind[optimal_point[0]]
        idy = optimal_point[1]
        if len(filts[idy]) > 0:
            fields[idx] = exposureids[idy]
            filters[idx] = filts[idy][0]
            filt = filts[idy][1:]
            filts[idy] = filt

    return fields, filters


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
    coverage_struct["moc"] = []
    if params["tilesType"] == "galaxy":
        coverage_struct["galaxies"] = []

    observer = ephem.Observer()
    observer.lat = str(config_struct["latitude"])
    observer.lon = str(config_struct["longitude"])
    observer.horizon = str(config_struct["horizon"])
    observer.elevation = config_struct["elevation"]
    observer.horizon = ephem.degrees(
        str(90 - np.arccos(1 / params["airmass"]) * 180 / np.pi)
    )

    exposurelist = config_struct["exposurelist"]
    # tilesegmentlists = gwemopt.segments_astroplan.get_segments_tiles(config_struct, tile_struct, observatory, segmentlist)
    tilesegmentlists = []
    keys = tile_struct.keys()
    for key in keys:
        # segments.py: tile_struct[key]["segmentlist"] is a list of segments when the tile is available for observation
        tilesegmentlists.append(tile_struct[key]["segmentlist"])

    if params["solverType"] == "heuristic":
        keys, filts = get_order_heuristic(
            params, tile_struct, tilesegmentlists, exposurelist, observer, config_struct
        )
    elif params["solverType"] == "milp":
        keys, filts = get_order_milp(
            params, tile_struct, exposurelist, observer, config_struct
        )
    else:
        raise ValueError(f'Unknown solverType {params["solverType"]}')

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
            alt, az = get_altaz_tile(
                tile_struct_hold["ra"], tile_struct_hold["dec"], observer, t
            )
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
            coverage_struct["moc"].append(tile_struct_hold["moc"])
            if params["tilesType"] == "galaxy":
                coverage_struct["galaxies"].append(tile_struct_hold["galaxies"])

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
