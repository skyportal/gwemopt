
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

def get_altaz_tiles(ras, decs, observatory, obstime):

    # Convert to RA, Dec.
    radecs = astropy.coordinates.SkyCoord(
            ra=np.array(ras)*u.degree, dec=np.array(decs)*u.degree, frame='icrs')

    # Alt/az reference frame at observatory, now
    frame = astropy.coordinates.AltAz(obstime=obstime, location=observatory)
            
    # Transform grid to alt/az coordinates at observatory, now
    altaz = radecs.transform_to(frame)    

    return altaz

def get_segments_tile(config_struct, observatory, radec, segmentlist):

    observer = ephem.Observer()
    observer.lat = str(config_struct["latitude"])
    observer.lon = str(config_struct["longitude"])
    observer.horizon = str(30.0)
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
            break

        astropy_rise = Time(date_rise.datetime(), scale='utc')
        astropy_set  = Time(date_set.datetime(), scale='utc')

        astropy_rise_mjd = astropy_rise.mjd
        astropy_set_mjd  = astropy_set.mjd

        # Alt/az reference frame at observatory, now
        #frame_rise = astropy.coordinates.AltAz(obstime=astropy_rise, location=observatory)
        #frame_set = astropy.coordinates.AltAz(obstime=astropy_set, location=observatory)    
        # Transform grid to alt/az coordinates at observatory, now
        #altaz_rise = radec.transform_to(frame_rise)
        #altaz_set = radec.transform_to(frame_set)        

        segment = glue.segments.segment(astropy_rise_mjd,astropy_set_mjd)
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

def get_segments_tiles(config_struct, observatory, ras, decs, segmentlist):

    print "Generating segments for tiles..."

    # Convert to RA, Dec.
    radecs = astropy.coordinates.SkyCoord(
            ra=np.array(ras)*u.degree, dec=np.array(decs)*u.degree, frame='icrs') 
    tilesegmentlists = []
    for ii,radec in enumerate(radecs):
        #print "Generating segments for tile %d/%d"%(ii,len(radecs))
        tilesegmentlist = get_segments_tile(config_struct, observatory, radec, segmentlist)
        tilesegmentlists.append(tilesegmentlist)

    return tilesegmentlists

def sort_tiles(tile_struct):

    probs = []
    ras = []
    decs = []
    keys = tile_struct.keys()

    for key in keys:
        probs.append(tile_struct[key]["prob"])
        ras.append(tile_struct[key]["ra"])
        decs.append(tile_struct[key]["dec"])
    keys = np.array(keys)
    probs = np.array(probs)
    ras = np.array(ras)
    decs = np.array(decs)
    idx = np.argsort(probs)[::-1]

    keys = keys[idx].tolist()
    ras = ras[idx].tolist()
    decs = decs[idx].tolist()
    probs = probs[idx].tolist()

    return keys, ras, decs, probs

def find_tile(exposureids_tile,exposureids,probs, idxs = None):

    if not idxs == None:
        for idx in idxs:
            idx2 = exposureids_tile["exposureids"][idx]
            if idx2 in exposureids:
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
        idx = np.argmax(exposureids_tile["probs"])
        idx2 = exposureids_tile["exposureids"][idx]

        if exposureids:
            if idx2 in exposureids:
                idx = exposureids.index(idx2)
                exposureids.pop(idx)
                probs.pop(idx)
                findTile = False
            else:
                exposureids_tile["exposureids"].pop(idx)
                exposureids_tile["probs"].pop(idx)
        else:
            findTile = False

    return idx2, exposureids, probs

def get_order(params, tile_struct, keys, tilesegmentlists, exposurelist):    
 
    exposureids_tiles = {}
    first_exposure = np.inf*np.ones((len(keys),))
    last_exposure = -np.inf*np.ones((len(keys),))
    tileprobs = np.zeros((len(keys),))
    tilenexps = np.zeros((len(keys),))

    for ii in xrange(len(exposurelist)):
        exposureids_tiles[ii] = {}
        exposureids = []
        probs = []
        for jj, key in enumerate(keys):
            tilesegmentlist = tilesegmentlists[jj]
            if tilesegmentlist.intersects_segment(exposurelist[ii]):
                exposureids.append(key)
                probs.append(tile_struct[key]["prob"])

                first_exposure[jj] = np.min([first_exposure[jj],ii])
                last_exposure[jj] = np.max([last_exposure[jj],ii])
                tileprobs[jj] = tile_struct[key]["prob"]
                tilenexps[jj] = tile_struct[key]["nexposures"]

        exposureids_tiles[ii]["exposureids"] = exposureids
        exposureids_tiles[ii]["probs"] = probs

    exposureids = []
    probs = []
    for ii, key in enumerate(keys):
        for jj in xrange(tile_struct[key]["nexposures"]):
            exposureids.append(key)
            probs.append(tile_struct[key]["prob"])

    idxs = -1*np.ones((len(exposureids_tiles.keys()),))
    if params["scheduleType"] == "greedy":
        for ii in np.arange(len(exposurelist)): 
            idx2, exposureids, probs = find_tile(exposureids_tiles[ii],exposureids,probs)
            tilenexps[idx2] = tilenexps[idx2] - 1
            idxs[ii] = idx2

    elif params["scheduleType"] == "sear":
        #for ii in np.arange(len(exposurelist)):
        iis = np.arange(len(exposurelist)).tolist()
        while len(iis) > 0: 
            ii = iis[0]
            mask = np.where((ii == last_exposure) & (tilenexps > 0))[0]
            if len(mask) > 0:
                idxsort = mask[np.argsort(tileprobs[mask])]
                idx2, exposureids, probs = find_tile(exposureids_tiles[ii],exposureids,probs,idxs=idxsort) 
                last_exposure[mask] = last_exposure[mask] + 1
            else:
                idx2, exposureids, probs = find_tile(exposureids_tiles[ii],exposureids,probs)
            tilenexps[idx2] = tilenexps[idx2] - 1
            idxs[ii] = idx2 
            iis.pop(0)
    else:
        print "Only greedy implemented..."
        exit(0)

    return idxs

def scheduler(params, config_struct, tile_struct):

    coverage_struct = {}
    coverage_struct["data"] = np.empty((0,5))
    coverage_struct["filters"] = []
    coverage_struct["ipix"] = []
    coverage_struct["patch"] = []
    coverage_struct["area"] = []

    keys, ras, decs, probs = sort_tiles(tile_struct)
    observatory = astropy.coordinates.EarthLocation(
        lat=config_struct["latitude"]*u.deg, lon=config_struct["longitude"]*u.deg, height=config_struct["elevation"]*u.m)

    segmentlist = config_struct["segmentlist"]
    exposurelist = config_struct["exposurelist"]
    tilesegmentlists = get_segments_tiles(config_struct, observatory, ras, decs, segmentlist)
    keys = get_order(params,tile_struct,keys,tilesegmentlists,exposurelist)

    if params["doPlots"]:
        gwemopt.plotting.scheduler(params,exposurelist,keys)

    while len(exposurelist) > 0:
        key = keys[0]
        if key == -1:
            keys = keys[1:]
            exposurelist = exposurelist[1:]
        else:
            tile_struct_hold = tile_struct[key]

            mjd_exposure_start = exposurelist[0][0]
            for jj in xrange(len(keys)):
                if keys[jj] == key:
                    mjd_exposure_end = exposurelist[jj][1]
                else:
                    nexp = jj + 1
                    keys = keys[jj:]
                    exposurelist = exposurelist[jj:]
                    break    
        
            mjd_exposure_mid = (mjd_exposure_start+mjd_exposure_end)/2.0
            nmag = np.log(nexp) / np.log(2.5)
            mag = config_struct["magnitude"] + nmag
            exposureTime = (mjd_exposure_end-mjd_exposure_start)*86400.0

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

