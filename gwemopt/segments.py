import numpy as np
import copy

import astropy.coordinates
from astropy.time import Time
import astropy.units as u
from joblib import Parallel, delayed
import ephem

import ligo.segments as segments
from gwemopt.utils.misc import get_exposures
from gwemopt.utils.rotate import angular_distance


def get_telescope_segments(params):

    for telescope in params["telescopes"]:

        params["config"][telescope]["segmentlist"] = get_segments(params, params["config"][telescope])
        params["config"][telescope]["exposurelist"] = get_exposures(params, params["config"][telescope], params["config"][telescope]["segmentlist"])
        if len(params["config"][telescope]["exposurelist"]) == 0:
            params["config"][telescope]["n_windows"] = 0
            params["config"][telescope]["tot_obs_time"] = 0.0
            continue

        nexp, junk = np.array(params["config"][telescope]["exposurelist"]).shape        
        params["config"][telescope]["n_windows"] = nexp
        tot_obs_time = np.sum(np.diff(np.array(params["config"][telescope]["exposurelist"]))) * 86400.
        params["config"][telescope]["tot_obs_time"] = tot_obs_time

    return params


def get_moon_segments(config_struct,segmentlist,observer,fxdbdy,radec):

    if "moon_constraint" in config_struct:
        moon_constraint = float(config_struct["moon_constraint"])
    else:
        moon_constraint = 20.0

    moonsegmentlist = segments.segmentlist()
    dt = 1.0/24.0
    tt = np.arange(segmentlist[0][0],segmentlist[-1][1]+dt,dt)
    conv = - 2400000.5 + 2415020.0 # conversion between MJD (tt) and DJD (what ephem uses)
    tt_DJD = tt - conv

    ra2 = radec.ra.radian
    d2 = radec.dec.radian

    # Where is the moon?
    moon = ephem.Moon()
    for ii in range(len(tt)-1):
        observer.date = ephem.Date(tt_DJD[ii])
        moon.compute(observer)
        fxdbdy.compute(observer)

        alt_target = float(repr(fxdbdy.alt)) * (360/(2*np.pi))
        az_target = float(repr(fxdbdy.az)) * (360/(2*np.pi))
        #print("Altitude / Azimuth of target: %.5f / %.5f"%(alt_target,az_target))

        alt_moon = float(repr(moon.alt)) * (360/(2*np.pi))
        az_moon = float(repr(moon.az)) * (360/(2*np.pi))
        #print("Altitude / Azimuth of moon: %.5f / %.5f"%(alt_moon,az_moon))

        ra_moon = (180/np.pi)*float(repr(moon.ra))
        dec_moon = (180/np.pi)*float(repr(moon.dec))

        # Coverting both target and moon ra and dec to radians
        ra1 = float(repr(moon.ra))
        d1 = float(repr(moon.dec))

        # Calculate angle between target and moon
        cosA = np.sin(d1)*np.sin(d2) + np.cos(d1)*np.cos(d2)*np.cos(ra1-ra2)
        angle = np.arccos(cosA)*(360/(2*np.pi))
        #print("Angle between moon and target: %.5f"%(angle))

        #if angle >= 50.0*moon.moon_phase**2:
        if angle >= moon_constraint:
            segment = segments.segment(tt[ii],tt[ii+1])
            moonsegmentlist = moonsegmentlist + segments.segmentlist([segment])
            moonsegmentlist.coalesce()

    moonsegmentlistdic = segments.segmentlistdict()
    moonsegmentlistdic["observations"] = segmentlist
    moonsegmentlistdic["moon"] = moonsegmentlist
    moonsegmentlist = moonsegmentlistdic.intersection(["observations","moon"])
    moonsegmentlist.coalesce()

    return moonsegmentlist


def get_skybrightness(config_struct,segmentlist,observer,fxdbdy,radec):

    moonsegmentlist = segments.segmentlist()
    if config_struct["filt"] == "c":
        passband = "g"
    else:
        passband = config_struct["filt"]

    # Moon phase data (from Coughlin, Stubbs, and Claver Table 2) 
    moon_phases = [2,10,45,90]
    moon_data = {'u':[2.7,3.1,4.2,5.7],
                 'g':[2.4,2.8,3.8,5.2],
                 'r':[2.1,2.5,3.4,4.9],
                 'i':[1.9,2.3,3.3,4.7],
                 'z':[1.9,2.2,3.2,4.6],
                 'y':[1.8,2.2,3.1,4.5]}

    # Determine moon data for this phase
    moon_data_passband = moon_data[passband]

    # Fits to solar sky brightness (from Coughlin, Stubbs, and Claver Table 4) 
    sun_data = {'u':[88.5,-0.5,-0.5,0.4],
                'g':[386.5,-2.2,-2.4,0.8],
                'r':[189.0,-1.4,-1.1,0.8],
                'i':[164.8,-1.5,-0.7,0.6],
                'z':[231.2,-2.8,-0.7,1.4],
                'zs':[131.1,-1.4,-0.5,0.2],
                'y':[92.0,-1.3,-0.2,0.9]}

    sun_data_error = {'u':[6.2,0.1,0.1,0.1],
                'g':[34.0,0.2,0.2,0.5],
                'r':[32.7,0.2,0.2,0.5],
                'i':[33.1,0.2,0.2,0.5],
                'z':[62.3,0.3,0.4,0.9],
                'zs':[45.6,0.2,0.3,0.6],
                'y':[32.7,0.2,0.2,0.5]}

    # Determine sun data for this phase
    sun_data_passband = sun_data[passband]

    dt = 1.0/24.0
    tt = np.arange(segmentlist[0][0],segmentlist[-1][1]+dt,dt)

    ra2 = radec.ra.radian
    d2 = radec.dec.radian

    # Where is the moon?
    moon = ephem.Moon()
    for ii in range(len(tt)-1):
        observer.date = ephem.Date(Time(tt[ii], format='mjd', scale='utc').iso)
        moon.compute(observer)
        fxdbdy.compute(observer)

        alt_target = float(repr(fxdbdy.alt)) * (360/(2*np.pi))
        az_target = float(repr(fxdbdy.az)) * (360/(2*np.pi))
        #print("Altitude / Azimuth of target: %.5f / %.5f"%(alt_target,az_target))

        alt_moon = float(repr(moon.alt)) * (360/(2*np.pi))
        az_moon = float(repr(moon.az)) * (360/(2*np.pi))
        #print("Altitude / Azimuth of moon: %.5f / %.5f"%(alt_moon,az_moon))

        if (alt_target < 30.0) or (alt_moon < 30.0):
            total_mag, total_mag_error, flux_mag, flux_mag_error = np.inf, np.inf, np.inf, np.inf
        else:
            ra_moon = (180/np.pi)*float(repr(moon.ra))
            dec_moon = (180/np.pi)*float(repr(moon.dec))

            # Coverting both target and moon ra and dec to radians
            ra1 = float(repr(moon.ra))
            d1 = float(repr(moon.dec))

            # Calculate angle between target and moon
            cosA = np.sin(d1)*np.sin(d2) + np.cos(d1)*np.cos(d2)*np.cos(ra1-ra2)
            angle = np.arccos(cosA)*(360/(2*np.pi))
            #print("Angle between moon and target: %.5f"%(angle))

            delta_mag = np.interp(moon.moon_phase*100.0,moon_phases,moon_data_passband)
            delta_mag_error = 0.1*delta_mag

            flux = sun_data_passband[0] + sun_data_passband[1]*angle +\
                sun_data_passband[2]*alt_target + sun_data_passband[3]*alt_moon
            flux_zp = sun_data_passband[0] + sun_data_passband[1]*90.0 +\
                sun_data_passband[2]*90.0 + sun_data_passband[3]*45.0

            # check if flux < 0: too small to fit
            if flux < 0:
                flux = 1e-10

            flux = flux* (10**11)
            flux_zp = flux_zp* (10**11)
            flux_mag = -2.5 * (np.log10(flux) - np.log10(flux_zp))

            sun_data_passband_error = sun_data_error[passband]
            flux_error = np.sqrt(sun_data_passband_error[0]**2 + sun_data_passband_error[1]**2 * angle**2 +\
                sun_data_passband_error[2]**2 * alt_target**2 + sun_data_passband_error[3]**2 * alt_moon**2)
            flux_error = flux_error * (10**11)

            flux_mag_error = 1.08574 * flux_error / flux

            # Determine total magnitude contribution
            total_mag = delta_mag + flux_mag
            total_mag_error = np.sqrt(delta_mag_error**2 + flux_mag_error**2)
            #print(tt[ii], angle, alt_target, alt_moon, total_mag, total_mag_error)
        if total_mag > 0.0:
            segment = segments.segment(tt[ii],tt[ii+1])
            moonsegmentlist = moonsegmentlist + segments.segmentlist([segment])
            moonsegmentlist.coalesce()
        #else:
        #    print(tt[ii], angle, alt_target, alt_moon, total_mag, total_mag_error)

    moonsegmentlistdic = segments.segmentlistdict()
    moonsegmentlistdic["observations"] = segmentlist
    moonsegmentlistdic["moon"] = moonsegmentlist
    moonsegmentlist = moonsegmentlistdic.intersection(["observations","moon"])
    moonsegmentlist.coalesce()

    #print("Keeping %.2f %% of data"%(100.0*np.sum(np.diff(moonsegmentlist))/np.sum(np.diff(segmentlist))))

    return moonsegmentlist


def get_ha_segments(config_struct,segmentlist,observer,fxdbdy,radec):

    if "ha_constraint" in config_struct:
        ha_constraint = config_struct["ha_constraint"].split(",")
        ha_min = float(ha_constraint[0])
        ha_max = float(ha_constraint[1])
    else:
        ha_min, ha_max = -24.0, 24.0

    if config_struct["telescope"] == "DECam":
        if radec.dec.deg <= -30.0:
            ha_min, ha_max = -5.2, 5.2
        else:
            ha_min, ha_max = -0.644981*np.sqrt(35.0-radec.dec.deg), 0.644981*np.sqrt(35.0-radec.dec.deg)
            
    location = astropy.coordinates.EarthLocation(config_struct["longitude"],
                                                 config_struct["latitude"],
                                                 config_struct["elevation"])

    halist = segments.segmentlist()
    for seg in segmentlist:
        mjds = np.linspace(seg[0], seg[1], 100)
        tt = Time(mjds, format='mjd', scale='utc', location=location)
        lst = tt.sidereal_time('mean')
        ha = (lst - radec.ra).hour
        idx = np.where((ha >= ha_min) & (ha <= ha_max))[0]
        if len(idx) >= 2:
            halist.append(segments.segment(mjds[idx[0]],mjds[idx[-1]]))
 
    return halist

def get_segments(params, config_struct):

    gpstime = params["gpstime"]
    event_mjd = Time(gpstime, format='gps', scale='utc').mjd

    segmentlist = segments.segmentlist()
    n_windows = len(params["Tobs"]) // 2
    start_segments = event_mjd + params["Tobs"][::2]
    end_segments = event_mjd + params["Tobs"][1::2]
    for start_segment, end_segment in zip(start_segments,end_segments):
        segmentlist.append(segments.segment(start_segment,end_segment))

    observer = ephem.Observer()
    observer.lat = str(config_struct["latitude"])
    observer.lon = str(config_struct["longitude"])
    observer.horizon = str(-12.0)
    observer.elevation = config_struct["elevation"]

    date_start = ephem.Date(Time(segmentlist[0][0], format='mjd', scale='utc').iso)
    date_end = ephem.Date(Time(segmentlist[-1][1], format='mjd', scale='utc').iso)
    observer.date = ephem.Date(Time(segmentlist[0][0], format='mjd', scale='utc').iso)

    sun = ephem.Sun()
    nightsegmentlist = segments.segmentlist()
    while date_start < date_end:
        date_rise = observer.next_rising(sun, start = date_start)
        date_set = observer.next_setting(sun, start = date_start)
        if date_set > date_rise:
            date_set = observer.previous_setting(sun, start = date_start)

        astropy_rise = Time(date_rise.datetime(), scale='utc').mjd
        astropy_set  = Time(date_set.datetime(), scale='utc').mjd

        segment = segments.segment(astropy_set,astropy_rise)
        nightsegmentlist = nightsegmentlist + segments.segmentlist([segment])
        nightsegmentlist.coalesce()

        date_start = date_rise
        observer.date = date_rise

    segmentlistdic = segments.segmentlistdict()
    segmentlistdic["observations"] = segmentlist
    segmentlistdic["night"] = nightsegmentlist

    #load the sun retriction for a satelite
    try:   
        sat_sun_restriction = config_struct["sat_sun_restriction"]
    except:
        sat_sun_restriction = 0.0

    #in the case of satellite use don't intersect with night segment and take all observation time available  
    if sat_sun_restriction:
        
        segmentlist.coalesce()

        return segmentlist    
    
    segmentlist = segmentlistdic.intersection(["observations","night"])
    segmentlist.coalesce()

    return segmentlist

def get_segments_tile(config_struct, observatory, radec, segmentlist, airmass):
    observer = ephem.Observer()
    observer.lat = str(config_struct["latitude"])
    observer.lon = str(config_struct["longitude"])
    observer.horizon = str(config_struct["horizon"])
    observer.elevation = config_struct["elevation"]
    observer.horizon = ephem.degrees(str(90-np.arccos(1/airmass)*180/np.pi))

    fxdbdy = ephem.FixedBody()
    fxdbdy._ra = ephem.degrees(str(radec.ra.degree))
    fxdbdy._dec = ephem.degrees(str(radec.dec.degree))

    observer.date = ephem.Date(Time(segmentlist[0][0], format='mjd', scale='utc').iso)
    fxdbdy.compute(observer)

    date_start = ephem.Date(Time(segmentlist[0][0], format='mjd', scale='utc').iso)
    date_end = ephem.Date(Time(segmentlist[-1][1], format='mjd', scale='utc').iso)
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

        segment = segments.segment(astropy_rise_mjd,astropy_set_mjd)
        tilesegmentlist = tilesegmentlist + segments.segmentlist([segment])
        tilesegmentlist.coalesce()

        date_start = date_set
        observer.date = date_set

    #moonsegmentlist = get_skybrightness(\
    #    config_struct,segmentlist,observer,fxdbdy,radec)

    halist = get_ha_segments(\
        config_struct,segmentlist,observer,fxdbdy,radec)    

    moonsegmentlist = get_moon_segments(\
        config_struct,segmentlist,observer,fxdbdy,radec)

    tilesegmentlistdic = segments.segmentlistdict()
    tilesegmentlistdic["observations"] = segmentlist
    tilesegmentlistdic["tile"] = tilesegmentlist
    tilesegmentlistdic["moon"] = moonsegmentlist
    tilesegmentlistdic["halist"] = halist
    tilesegmentlist = tilesegmentlistdic.intersection(["observations","tile","moon","halist"])
    #tilesegmentlist = tilesegmentlistdic.intersection(["observations","tile"])
    tilesegmentlist.coalesce()

    return tilesegmentlist

def get_segments_tiles(params, config_struct, tile_struct):

    observatory = astropy.coordinates.EarthLocation(
        lat=config_struct["latitude"]*u.deg, lon=config_struct["longitude"]*u.deg, height=config_struct["elevation"]*u.m)

    segmentlist = config_struct["segmentlist"]

    print("Generating segments for tiles...")

    ras = []
    decs = []
    keys = tile_struct.keys()
    for key in keys:
        ras.append(tile_struct[key]["ra"])
        decs.append(tile_struct[key]["dec"])

    # Convert to RA, Dec.
    radecs = astropy.coordinates.SkyCoord(
            ra=np.array(ras)*u.degree, dec=np.array(decs)*u.degree, frame='icrs')

    if params["doParallel"]:
        tilesegmentlists = Parallel(n_jobs=params["Ncores"])(delayed(get_segments_tile)(config_struct, observatory, radec, segmentlist) for radec in radecs)
        for ii,key in enumerate(keys):
            tile_struct[key]["segmentlist"] = tilesegmentlists[ii]
    else:

        for ii,key in enumerate(keys):
            #if np.mod(ii,100) == 0: 
            #    print("Generating segments for tile %d/%d"%(ii+1,len(radecs)))
            radec = radecs[ii]

            if params["doMinimalTiling"]:
                if ii == 0:
                    keys_computed = [key]
                    radecs_computed = np.atleast_2d([radec.ra.value, radec.dec.value])
                    tilesegmentlist = get_segments_tile(config_struct, observatory, radec, segmentlist, params["airmass"])
                    tile_struct[key]["segmentlist"] = tilesegmentlist
                else:
                    seps = angular_distance(radec.ra.value, radec.dec.value,
                                            radecs_computed[:,0],
                                            radecs_computed[:,1])
                    sepmin = np.min(seps)
                    sepamin = np.argmin(seps)
                    if sepmin <= 5.0:
                        key_computed = keys_computed[sepamin]
                        tile_struct[key]["segmentlist"] = copy.deepcopy(tile_struct[key_computed]["segmentlist"])
                    else:
                        keys_computed.append(key)
                        radecs_computed = np.vstack((radecs_computed,[radec.ra.value, radec.dec.value]))
                        tilesegmentlist = get_segments_tile(config_struct, observatory, radec, segmentlist, params["airmass"])
                        tile_struct[key]["segmentlist"] = tilesegmentlist

            else:
                tilesegmentlist = get_segments_tile(config_struct, observatory, radec, segmentlist, params["airmass"])
                tile_struct[key]["segmentlist"] = tilesegmentlist

    return tile_struct
