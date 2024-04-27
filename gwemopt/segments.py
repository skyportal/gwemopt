import copy

import astropy.coordinates
import astropy.units as u
import ephem
import ligo.segments as segments
import numpy as np
from astropy.time import Time
from joblib import Parallel, delayed
from tqdm import tqdm

from gwemopt.utils.geometry import angular_distance
from gwemopt.utils.misc import get_exposures
from gwemopt.utils.sidereal_time import greenwich_sidereal_time

# conversion between MJD (tt) and DJD (what ephem uses)
MJD_TO_DJD = -2400000.5 + 2415020.0


def get_telescope_segments(params):
    for telescope in params["telescopes"]:
        params["config"][telescope]["segmentlist"] = get_segments(
            params, params["config"][telescope]
        )
        params["config"][telescope]["exposurelist"] = get_exposures(
            params,
            params["config"][telescope],
            params["config"][telescope]["segmentlist"],
        )
        if len(params["config"][telescope]["exposurelist"]) == 0:
            params["config"][telescope]["n_windows"] = 0
            params["config"][telescope]["tot_obs_time"] = 0.0
            continue

        nexp, junk = np.array(params["config"][telescope]["exposurelist"]).shape
        params["config"][telescope]["n_windows"] = nexp
        tot_obs_time = (
            np.sum(np.diff(np.array(params["config"][telescope]["exposurelist"])))
            * 86400.0
        )
        params["config"][telescope]["tot_obs_time"] = tot_obs_time

    return params


def get_moon_radecs(segmentlist, observer):
    dt = 1.0 / 24.0
    tt = np.arange(segmentlist[0][0], segmentlist[-1][1] + dt, dt)
    tt_DJD = tt - MJD_TO_DJD

    moon_radecs = []
    # Where is the moon?
    moon = ephem.Moon()
    for ii in range(len(tt) - 1):
        observer.date = ephem.Date(tt_DJD[ii])
        moon.compute(observer)

        # Coverting both target and moon ra and dec to radians
        ra1 = float(repr(moon.ra))
        d1 = float(repr(moon.dec))

        moon_radecs.append([ra1, d1])

    return moon_radecs


def get_moon_segments(config_struct, segmentlist, moon_radecs, radec):
    if "moon_constraint" in config_struct:
        moon_constraint = float(config_struct["moon_constraint"])
    else:
        moon_constraint = 20.0

    moonsegments = []
    moonsegmentlist = segments.segmentlist()
    dt = 1.0 / 24.0
    tt = np.arange(segmentlist[0][0], segmentlist[-1][1] + dt, dt)

    ra2 = np.deg2rad(radec[0])
    d2 = np.deg2rad(radec[1])

    # Where is the moon?
    for ii in range(len(tt) - 1):
        # Coverting both target and moon ra and dec to radians
        ra1 = moon_radecs[ii][0]
        d1 = moon_radecs[ii][1]

        # Calculate angle between target and moon
        cosA = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(ra1 - ra2)
        angle = np.arccos(cosA) * (360 / (2 * np.pi))
        # print("Angle between moon and target: %.5f"%(angle))

        # if angle >= 50.0*moon.moon_phase**2:
        if angle >= moon_constraint:
            moonsegments.append([tt[ii], tt[ii + 1]])

    moonsegmentlist = segments.segmentlist()
    for seg in moonsegments:
        moonsegmentlist.append(segments.segment(seg))
    moonsegmentlist.coalesce()

    moonsegmentlistdic = segments.segmentlistdict()
    moonsegmentlistdic["observations"] = segmentlist
    moonsegmentlistdic["moon"] = moonsegmentlist
    moonsegmentlist = moonsegmentlistdic.intersection(["observations", "moon"])
    moonsegmentlist.coalesce()

    return moonsegmentlist


def get_ha_segments(config_struct, segmentlist, observer, radec):
    if "ha_constraint" in config_struct:
        ha_constraint = config_struct["ha_constraint"].split(",")
        ha_min = float(ha_constraint[0])
        ha_max = float(ha_constraint[1])
    else:
        ha_min, ha_max = -24.0, 24.0

    if config_struct["telescope"] == "DECam":
        if radec[1] <= -30.0:
            ha_min, ha_max = -5.2, 5.2
        else:
            ha_min, ha_max = -0.644981 * np.sqrt(35.0 - radec[1]), 0.644981 * np.sqrt(
                35.0 - radec[1]
            )

    halist = segments.segmentlist()
    for seg in segmentlist:
        mjds = np.linspace(seg[0], seg[1], 100)
        tt = Time(mjds, format="mjd", scale="utc")
        lst = np.mod(
            np.deg2rad(config_struct["longitude"]) + greenwich_sidereal_time(tt),
            2 * np.pi,
        )
        ha = (lst - np.deg2rad(radec[0])) * 24 / (2 * np.pi)
        idx = np.where((ha >= ha_min) & (ha <= ha_max))[0]
        if len(idx) >= 2:
            halist.append(segments.segment(mjds[idx[0]], mjds[idx[-1]]))

    return halist


def get_segments(params, config_struct):
    gpstime = params["gpstime"]
    event_mjd = Time(gpstime, format="gps", scale="utc").mjd

    segmentlist = segments.segmentlist()
    n_windows = len(params["Tobs"]) // 2
    start_segments = event_mjd + params["Tobs"][::2]
    end_segments = event_mjd + params["Tobs"][1::2]
    for start_segment, end_segment in zip(start_segments, end_segments):
        segmentlist.append(segments.segment(start_segment, end_segment))

    observer = ephem.Observer()
    observer.lat = str(config_struct["latitude"])
    observer.lon = str(config_struct["longitude"])
    observer.horizon = str(-12.0)
    observer.elevation = config_struct["elevation"]

    date_start = ephem.Date(segmentlist[0][0] - MJD_TO_DJD)
    date_end = ephem.Date(segmentlist[-1][1] - MJD_TO_DJD)
    observer.date = ephem.Date(segmentlist[0][0] - MJD_TO_DJD)

    sun = ephem.Sun()
    nightsegmentlist = segments.segmentlist()
    while date_start < date_end:
        date_rise = observer.next_rising(sun, start=date_start)
        date_set = observer.next_setting(sun, start=date_start)
        if date_set > date_rise:
            date_set = observer.previous_setting(sun, start=date_start)

        segment = segments.segment(date_rise + MJD_TO_DJD, date_set + MJD_TO_DJD)
        nightsegmentlist = nightsegmentlist + segments.segmentlist([segment])
        nightsegmentlist.coalesce()

        date_start = date_rise
        observer.date = date_rise

    segmentlistdic = segments.segmentlistdict()
    segmentlistdic["observations"] = segmentlist
    segmentlistdic["night"] = nightsegmentlist

    # load the sun retriction for a satelite
    try:
        sat_sun_restriction = config_struct["sat_sun_restriction"]
    except:
        sat_sun_restriction = 0.0

    # in the case of satellite use don't intersect with night segment and take all observation time available
    if sat_sun_restriction:
        segmentlist.coalesce()

        return segmentlist

    segmentlist = segmentlistdic.intersection(["observations", "night"])
    segmentlist.coalesce()

    return segmentlist


def get_segments_tile(config_struct, radec, segmentlist, moon_radecs, airmass):

    # check for empty segmentlist and immediately return
    if len(segmentlist) == 0:
        return segments.segmentlistdict()

    observer = ephem.Observer()
    observer.lat = str(config_struct["latitude"])
    observer.lon = str(config_struct["longitude"])
    observer.horizon = str(config_struct["horizon"])
    observer.elevation = config_struct["elevation"]
    observer.horizon = ephem.degrees(str(90 - np.arccos(1 / airmass) * 180 / np.pi))
    observer.date = ephem.Date(segmentlist[0][0] - MJD_TO_DJD)

    fxdbdy = ephem.FixedBody()
    fxdbdy._ra = ephem.degrees(str(radec[0]))
    fxdbdy._dec = ephem.degrees(str(radec[1]))
    fxdbdy.compute(observer)

    date_start = ephem.Date(segmentlist[0][0] - MJD_TO_DJD)
    date_end = ephem.Date(segmentlist[-1][1] - MJD_TO_DJD)
    tilesegmentlist = segments.segmentlist()
    while date_start < date_end:
        try:
            date_rise = observer.next_rising(fxdbdy, start=observer.date)
            date_set = observer.next_setting(fxdbdy, start=observer.date)
            if date_rise > date_set:
                date_rise = observer.previous_rising(fxdbdy, start=observer.date)
        except ephem.AlwaysUpError:
            date_rise = date_start
            date_set = date_end
        except ephem.NeverUpError:
            date_rise = ephem.Date(0.0)
            date_set = ephem.Date(0.0)
            break

        segment = segments.segment(date_rise + MJD_TO_DJD, date_set + MJD_TO_DJD)
        tilesegmentlist = tilesegmentlist + segments.segmentlist([segment])
        tilesegmentlist.coalesce()

        date_start = date_set
        observer.date = date_set

    # moonsegmentlist = get_skybrightness(\
    #    config_struct,segmentlist,observer,fxdbdy,radec)

    halist = get_ha_segments(config_struct, segmentlist, observer, radec)

    moonsegmentlist = get_moon_segments(config_struct, segmentlist, moon_radecs, radec)

    tilesegmentlistdic = segments.segmentlistdict()
    tilesegmentlistdic["observations"] = segmentlist
    tilesegmentlistdic["tile"] = tilesegmentlist
    tilesegmentlistdic["moon"] = moonsegmentlist
    tilesegmentlistdic["halist"] = halist
    tilesegmentlist = tilesegmentlistdic.intersection(
        ["observations", "tile", "moon", "halist"]
    )
    # tilesegmentlist = tilesegmentlistdic.intersection(["observations","tile"])
    tilesegmentlist.coalesce()

    return tilesegmentlist


def get_segments_tiles(params, config_struct, tile_struct):
    segmentlist = config_struct["segmentlist"]

    print("Generating segments for tiles...")

    radecs = []
    keys = tile_struct.keys()
    for key in keys:
        radecs.append([tile_struct[key]["ra"], tile_struct[key]["dec"]])

    if params["ignore_observability"]:
        for ii, key in enumerate(keys):
            tile_struct[key]["segmentlist"] = copy.deepcopy(segmentlist)
        return tile_struct

    observer = ephem.Observer()
    observer.lat = str(config_struct["latitude"])
    observer.lon = str(config_struct["longitude"])
    observer.horizon = str(config_struct["horizon"])
    observer.elevation = config_struct["elevation"]
    observer.horizon = ephem.degrees(
        str(90 - np.arccos(1 / params["airmass"]) * 180 / np.pi)
    )

    moon_radecs = get_moon_radecs(segmentlist, observer)

    if params["doParallel"]:
        tilesegmentlists = Parallel(
            n_jobs=params["Ncores"],
            backend="multiprocessing",
            batch_size=int(len(radecs) / params["Ncores"]) + 1,
        )(
            delayed(get_segments_tile)(
                config_struct, radec, segmentlist, moon_radecs, params["airmass"]
            )
            for radec in tqdm(radecs)
        )
        for ii, key in enumerate(keys):
            tile_struct[key]["segmentlist"] = tilesegmentlists[ii]
    else:
        for ii, key in tqdm(enumerate(keys), total=len(keys)):
            radec = radecs[ii]

            if params["doMinimalTiling"]:
                if ii == 0:
                    keys_computed = [key]
                    radecs_computed = np.atleast_2d(radec)
                    tilesegmentlist = get_segments_tile(
                        config_struct,
                        radec,
                        segmentlist,
                        moon_radecs,
                        params["airmass"],
                    )
                    tile_struct[key]["segmentlist"] = tilesegmentlist
                else:
                    seps = angular_distance(
                        radec[0],
                        radec[1],
                        radecs_computed[:, 0],
                        radecs_computed[:, 1],
                    )
                    sepmin = np.min(seps)
                    sepamin = np.argmin(seps)
                    if sepmin <= 5.0:
                        key_computed = keys_computed[sepamin]
                        tile_struct[key]["segmentlist"] = copy.deepcopy(
                            tile_struct[key_computed]["segmentlist"]
                        )
                    else:
                        keys_computed.append(key)
                        radecs_computed = np.vstack((radecs_computed, radec))
                        tilesegmentlist = get_segments_tile(
                            config_struct,
                            radec,
                            segmentlist,
                            moon_radecs,
                            params["airmass"],
                        )
                        tile_struct[key]["segmentlist"] = tilesegmentlist

            else:
                tilesegmentlist = get_segments_tile(
                    config_struct, radec, segmentlist, moon_radecs, params["airmass"]
                )
                tile_struct[key]["segmentlist"] = tilesegmentlist

    return tile_struct
