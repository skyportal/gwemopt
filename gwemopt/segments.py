import copy

import ephem
import ligo.segments as segments
import numpy as np
from astropy.time import Time
from joblib import Parallel, delayed
from tqdm import tqdm

from gwemopt.telescope import Telescope
from gwemopt.utils.misc import get_exposures
from gwemopt.utils.sidereal_time import hour_angle

# conversion between MJD (tt) and DJD (what ephem uses)
MJD_TO_DJD = -2400000.5 + 2415020.0


def get_telescope_segments(
    telescopes: list[Telescope], gpstime, Tobs, exposuretimes, doAlternatingFilters
) -> tuple[segments.segmentlist, segments.segmentlist, int, float]:
    for telescope in telescopes:
        telescope_segments = get_segments(telescope, gpstime, Tobs)
        exposurelist = get_exposures(
            telescope,
            telescope_segments,
            exposuretimes,
            doAlternatingFilters,
        )
        if len(exposurelist) == 0:
            nwindows = 0
            tot_obs_time = 0.0
            continue

        nexp, _ = np.array(exposurelist).shape
        nwindows = nexp
        tot_obs_time = np.sum(np.diff(np.array(exposurelist))) * 86400.0
        tot_obs_time = tot_obs_time

    return telescope_segments, exposurelist, nwindows, tot_obs_time


def get_moon_radecs(segmentlist, observer):
    dt = 1.0 / 24.0
    if len(segmentlist) == 0:
        return []

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


def get_moon_segments(telescope: Telescope, segmentlist, moon_radecs, radec):
    moon_constraint = telescope.moon_constraint

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


def get_ha_segments(telescope: Telescope, segmentlist, radec):
    ha_min, ha_max = telescope.ha_constraint

    if telescope.telescope_name == "DECam":
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
        ha = hour_angle(tt.jd, tt.gps, telescope.longitude.value, radec[0], 0)
        idx = np.where((ha >= ha_min) & (ha <= ha_max))[0]
        if len(idx) >= 2:
            halist.append(segments.segment(mjds[idx[0]], mjds[idx[-1]]))

    return halist


def get_segments(
    telescope: Telescope, gpstime, Tobs: np.ndarray
) -> segments.segmentlist:
    event_mjd = Time(gpstime, format="gps", scale="utc").mjd

    segmentlist = segments.segmentlist()
    start_segments = event_mjd + Tobs[::2]
    end_segments = event_mjd + Tobs[1::2]
    for start_segment, end_segment in zip(start_segments, end_segments):
        segmentlist.append(segments.segment(start_segment, end_segment))

    observer = ephem.Observer()
    # FIXME latitude and longitude are converted into string because
    # the test_coverage fail if not. Should not be the case as
    # the value in float is the same.
    observer.lat = str(telescope.latitude.value)
    observer.lon = str(telescope.longitude.value)
    observer.horizon = str(-12.0)
    observer.elevation = telescope.elevation.value

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
    sat_sun_restriction = telescope.sat_sun_restriction
    # in the case of satellite use don't intersect with night segment and take all observation time available
    if sat_sun_restriction:
        segmentlist.coalesce()

        return segmentlist

    segmentlist = segmentlistdic.intersection(["observations", "night"])
    segmentlist.coalesce()

    return segmentlist


def get_segments_tile(telescope: Telescope, radec, segmentlist, moon_radecs, airmass):
    # check for empty segmentlist and immediately return
    if len(segmentlist) == 0:
        return segments.segmentlistdict()

    observer = ephem.Observer()
    observer.lat = str(telescope.latitude.value)
    observer.lon = str(telescope.longitude.value)
    observer.elevation = telescope.elevation.value
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

    halist = get_ha_segments(telescope, segmentlist, radec)

    moonsegmentlist = get_moon_segments(telescope, segmentlist, moon_radecs, radec)

    tilesegmentlistdic = segments.segmentlistdict()
    tilesegmentlistdic["observations"] = segmentlist
    tilesegmentlistdic["tile"] = tilesegmentlist
    tilesegmentlistdic["moon"] = moonsegmentlist
    tilesegmentlistdic["halist"] = halist
    tilesegmentlist = tilesegmentlistdic.intersection(
        ["observations", "tile", "moon", "halist"]
    )
    tilesegmentlist.coalesce()

    return tilesegmentlist


def get_segments_tiles(
    params,
    segmentList: segments.segmentlist,
    telescope: Telescope,
    airmass: float,
    tile_struct,
):

    print("Generating segments for tiles...")

    radecs = []
    keys = tile_struct.keys()
    for key in keys:
        radecs.append([tile_struct[key]["ra"], tile_struct[key]["dec"]])

    if params["ignore_observability"]:
        for ii, key in enumerate(keys):
            tile_struct[key]["segmentlist"] = copy.deepcopy(segmentList)
        return tile_struct

    observer = ephem.Observer()
    observer.lat = str(telescope.latitude.value)
    observer.lon = str(telescope.longitude.value)
    observer.elevation = telescope.elevation.value
    observer.horizon = ephem.degrees(str(90 - np.arccos(1 / airmass) * 180 / np.pi))

    moon_radecs = get_moon_radecs(segmentList, observer)

    if params["doParallel"]:
        tilesegmentlists = Parallel(
            n_jobs=params["Ncores"],
            backend=params["parallelBackend"],
            batch_size=int(len(radecs) / params["Ncores"]) + 1,
        )(
            delayed(get_segments_tile)(
                telescope, radec, segmentList, moon_radecs, airmass
            )
            for radec in radecs
        )
        for ii, key in enumerate(keys):
            tile_struct[key]["segmentlist"] = tilesegmentlists[ii]
    else:
        for ii, key in tqdm(enumerate(keys), total=len(keys)):
            radec = radecs[ii]
            tilesegmentlist = get_segments_tile(
                telescope, radec, segmentList, moon_radecs, airmass
            )
            tile_struct[key]["segmentlist"] = tilesegmentlist

    return tile_struct
