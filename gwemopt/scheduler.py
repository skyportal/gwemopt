
import os, sys
import numpy as np
import healpy as hp

import astropy.coordinates
from astropy.time import Time, TimeDelta
import astropy.units as u

import ephem

import glue.segments

import gwemopt.utils
import gwemopt.rankedTilesGenerator
import gwemopt.moc, gwemopt.pem

def get_altaz_tiles(radecs, observatory, obstime):

    # Alt/az reference frame at observatory, now
    frame = astropy.coordinates.AltAz(obstime=obstime, location=observatory)
            
    # Transform grid to alt/az coordinates at observatory, now
    altaz = radecs.transform_to(frame)    

    return altaz

def get_segments_tile(config_struct, radec, segmentlist):

    observer = ephem.Observer()
    observer.lat = str(config_struct["latitude"])
    observer.lon = str(config_struct["longitude"])
    observer.horizon = str(-12.0)
    observer.elevation = config_struct["elevation"]
    observer.date = ephem.Date(Time(segmentlist[0][0], format='mjd', scale='utc').iso)

    date_start = ephem.Date(Time(segmentlist[0][0], format='mjd', scale='utc').iso)
    date_end = ephem.Date(Time(segmentlist[-1][1], format='mjd', scale='utc').iso)

    fxdbdy = ephem.FixedBody()
    fxdbdy._ra = ephem.degrees(str(radec.ra.degree))
    fxdbdy._dec = ephem.degrees(str(radec.dec.degree))
    fxdbdy.compute(observer)

    tilesegmentlist = glue.segments.segmentlist()
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

        astropy_rise = Time(date_rise.datetime(), scale='utc').mjd
        astropy_set  = Time(date_set.datetime(), scale='utc').mjd

        segment = glue.segments.segment(astropy_rise,astropy_set)
        tilesegmentlist = tilesegmentlist + glue.segments.segmentlist([segment])
        tilesegmentlist.coalesce()

        date_start = date_set
        observer.date = date_set

    tilesegmentlistdic = glue.segments.segmentlistdict()
    tilesegmentlistdic["observations"] = segmentlist
    tilesegmentlistdic["tile"] = tilesegmentlist
    tilesegmentlist = tilesegmentlistdic.intersection(["observations","tile"])
    tilesegmentlist.coalesce()

    return tilesegmentlist

def get_segments_tiles(config_struct, radecs, segmentlist):
  
    tilesegmentlists = []
    for radec in radecs:
        tilesegmentlist = get_segments_tile(config_struct, radec, segmentlist)
        tilesegmentlists.append(tilesegmentlist)

    return tilesegmentlists

def sort_tiles(tile_struct):

    keys = []
    probs = []
    ras = []
    decs = []
    for key in tile_struct.iterkeys():
        keys.append(key)
        probs.append(tile_struct[key]["prob"])
        ras.append(tile_struct[key]["ra"])
        decs.append(tile_struct[key]["dec"])
    keys = np.array(keys)
    probs = np.array(probs)
    ras = np.array(ras)
    decs = np.array(decs)
    idx = np.argsort(probs)[::-1]

    keys = keys[idx].tolist()
    ras = ras[idx]
    decs = decs[idx]
    probs = probs[idx]

    # Convert to RA, Dec.
    radecs = astropy.coordinates.SkyCoord(
            ra=ras*u.degree, dec=decs*u.degree, frame='icrs')

    return keys, radecs, probs

def get_exposures(params, config_struct, segmentlist):

    exposurelist = np.empty((0,1))
    for ii in xrange(len(segmentlist)):
        start_segment, end_segment = segmentlist[ii][0], segmentlist[ii][1]
        exposures = np.arange(start_segment, end_segment, config_struct["exposuretime"]/86400.0)
        exposurelist = np.append(exposurelist,exposures)
    return exposurelist

def get_segments(params, config_struct):

    gpstime = params["gpstime"]
    event_mjd = Time(gpstime, format='gps', scale='utc').mjd

    segmentlist = glue.segments.segmentlist()
    n_windows = len(params["Tobs"]) // 2
    start_segments = event_mjd + params["Tobs"][::2]
    end_segments = event_mjd + params["Tobs"][1::2]
    for start_segment, end_segment in zip(start_segments,end_segments):
        segmentlist.append(glue.segments.segment(start_segment,end_segment))

    observer = ephem.Observer()
    observer.lat = str(config_struct["latitude"])
    observer.lon = str(config_struct["longitude"])
    observer.horizon = str(-12.0)
    observer.elevation = config_struct["elevation"]

    date_start = ephem.Date(Time(segmentlist[0][0], format='mjd', scale='utc').iso)
    date_end = ephem.Date(Time(segmentlist[-1][1], format='mjd', scale='utc').iso)
    observer.date = ephem.Date(Time(segmentlist[0][0], format='mjd', scale='utc').iso)

    sun = ephem.Sun()
    nightsegmentlist = glue.segments.segmentlist()
    while date_start < date_end:
        date_rise = observer.next_rising(sun, start = date_start)
        date_set = observer.next_setting(sun, start = date_start)
        if date_set > date_rise:
            date_set = observer.previous_setting(sun, start = date_start)

        astropy_rise = Time(date_rise.datetime(), scale='utc').mjd
        astropy_set  = Time(date_set.datetime(), scale='utc').mjd

        segment = glue.segments.segment(astropy_set,astropy_rise)
        nightsegmentlist = nightsegmentlist + glue.segments.segmentlist([segment])
        nightsegmentlist.coalesce()

        date_start = date_rise
        observer.date = date_rise

    segmentlistdic = glue.segments.segmentlistdict()
    segmentlistdic["observations"] = segmentlist
    segmentlistdic["night"] = nightsegmentlist
    segmentlist = segmentlistdic.intersection(["observations","night"])
    segmentlist.coalesce()

    return segmentlist

def scheduler(params, config_struct, tile_struct):

    coverage_struct = {}
    coverage_struct["data"] = np.empty((0,5))
    coverage_struct["filters"] = []
    coverage_struct["ipix"] = []
    coverage_struct["patch"] = []
    coverage_struct["area"] = []

    keys, radecs, probs = sort_tiles(tile_struct)
    observatory = astropy.coordinates.EarthLocation(
        lat=config_struct["latitude"]*u.deg, lon=config_struct["longitude"]*u.deg, height=config_struct["elevation"]*u.m)

    segmentlist = get_segments(params, config_struct)
    exposurelist = get_exposures(params, config_struct, segmentlist)
    tilesegmentlists = get_segments_tiles(config_struct, radecs, segmentlist)

    while len(segmentlist) > 0:
        obstime = Time(segmentlist[0][0], format='mjd', scale='utc')
        altaz = get_altaz_tiles(radecs, observatory, obstime)

        if params["scheduleType"] == "greedy":
            idx = np.where((altaz.alt >= 30.0*u.deg) & (altaz.secz <= 2.5))[0]
        elif params["scheduleType"] == "sear":
            idx1 = np.where((altaz.alt >= 30.0*u.deg) & (altaz.secz <= 2.5))[0]
            # Is anyone going to set soon?
            idx2 = np.where(set_time[idx1] <= segmentlist[0][0]+config_struct["exposuretime"]/86400.0)[0]
            if len(idx2) > 0:
                print idx2
            else:
                idx = idx1   
        elif params["scheduleType"] == "optimal":
            idx = np.where((altaz.alt >= 30.0*u.deg) & (altaz.secz <= 2.5))[0]

        if len(idx) == 0:
            segment = glue.segments.segment(segmentlist[0][0],segmentlist[0][0]+config_struct["exposuretime"]/86400.0)
            segmentlist = segmentlist - glue.segments.segmentlist([segment])
            segmentlist.coalesce()
        else:
            key = keys[idx[0]]
            tile_struct_hold = tile_struct[key] 
            exposureTime = tile_struct_hold["exposureTime"]

            mjd_exposure_start = segmentlist[0][0]
            mjd_exposure_end = mjd_exposure_start + exposureTime/86400.0
            if mjd_exposure_end > segmentlist[0][1]:
                mjd_exposure_end = segmentlist[0][1]
                exposureTime = (mjd_exposure_end - mjd_exposure_start)*86400.0
                tile_struct[key]["exposureTime"] = tile_struct[key]["exposureTime"] - exposureTime 
            else:
                del tile_struct[key]
                keys.pop(idx[0])

            segment = glue.segments.segment(mjd_exposure_start,mjd_exposure_end)
            segmentlist = segmentlist - glue.segments.segmentlist([segment])
            segmentlist.coalesce()

            mjd_exposure_mid = (mjd_exposure_start+mjd_exposure_end)/2.0
            nexp = np.round(exposureTime/config_struct["exposuretime"])
            nmag = np.log(nexp) / np.log(2.5)
            mag = config_struct["magnitude"] + nmag

            coverage_struct["data"] = np.append(coverage_struct["data"],np.array([[tile_struct_hold["ra"],tile_struct_hold["dec"],mjd_exposure_mid,mag,exposureTime]]),axis=0)

            coverage_struct["filters"].append(config_struct["filter"])
            coverage_struct["patch"].append(tile_struct_hold["patch"])
            coverage_struct["ipix"].append(tile_struct_hold["ipix"])
            coverage_struct["area"].append(tile_struct_hold["area"])

    coverage_struct["area"] = np.array(coverage_struct["area"])
    coverage_struct["filters"] = np.array(coverage_struct["filters"])
    coverage_struct["FOV"] = config_struct["FOV"]*np.ones((len(coverage_struct["filters"]),))

    return coverage_struct

def summary(params, map_struct, coverage_struct):

    summaryfile = os.path.join(params["outputDir"],'summary.dat')
    fid = open(summaryfile,'w')

    gpstime = params["gpstime"]
    event_mjd = Time(gpstime, format='gps', scale='utc').mjd

    tts = np.array([1,7,60])
    for tt in tts:

        radecs = np.empty((0,2))
        mjds_floor = []
        mjds = []
        ipixs = np.empty((0,2))
        cum_prob = 0.0
        cum_area = 0.0

        for ii in xrange(len(coverage_struct["ipix"])):
            data = coverage_struct["data"][ii,:]
            filt = coverage_struct["filters"][ii]
            ipix = coverage_struct["ipix"][ii]
            patch = coverage_struct["patch"][ii]
            FOV = coverage_struct["FOV"][ii]
            area = coverage_struct["area"][ii]

            prob = np.sum(map_struct["prob"][ipix])

            if data[2] > event_mjd+tt:
                continue

            ipixs = np.append(ipixs,ipix)
            ipixs = np.unique(ipixs).astype(int)

            cum_prob = np.sum(map_struct["prob"][ipixs])
            cum_area = len(ipixs) * map_struct["pixarea_deg2"]
            mjds.append(data[2])
            mjds_floor.append(int(np.floor(data[2])))

        if len(mjds_floor) == 0:
            print "No images after %.1f days..."%tt
            fid.write('%.1f,-1,-1,-1,-1\n'%(tt))
        else:

            mjds = np.unique(mjds)
            mjds_floor = np.unique(mjds_floor)

            print "After %.1f days..."%tt
            print "Number of hours after first image: %.5f"%(24*(np.min(mjds)-event_mjd))
            print "MJDs covered: %s"%(" ".join(str(x) for x in mjds_floor))
            print "Cumultative probability: %.5f"%cum_prob
            print "Cumultative area: %.5f degrees"%cum_area

            fid.write('%.1f,%.5f,%.5f,%.5f,%s\n'%(tt,24*(np.min(mjds)-event_mjd),cum_prob,cum_area," ".join(str(x) for x in mjds_floor)))

    fid.close()

