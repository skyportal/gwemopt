
import os, sys
import numpy as np
import healpy as hp
import copy

import astropy.coordinates
from astropy.time import Time, TimeDelta
import astropy.units as u
import astroplan

import ligo.segments as segments
import gwemopt.utils

def get_telescope_segments(params):
    # Calculate the observation availabilities and save them into params
    # segmentlist: list of times that the observation is available
    # exposurelist: segmentlist broken down into chunks with the length of telescope exposures
    # tot_obs_time: total time covered by the exposurelist
    for telescope in params["telescopes"]:

        params["config"][telescope]["segmentlist"] = get_segments(params, params["config"][telescope])
        params["config"][telescope]["exposurelist"] = gwemopt.utils.get_exposures(params, params["config"][telescope], params["config"][telescope]["segmentlist"])

        nexp, junk = np.array(params["config"][telescope]["exposurelist"]).shape        
        params["config"][telescope]["n_windows"] = nexp
        tot_obs_time = np.sum(np.diff(np.array(params["config"][telescope]["exposurelist"]))) * 86400.
        params["config"][telescope]["tot_obs_time"] = tot_obs_time

    return params

def get_moon_segments(config_struct,segmentlist,observer,fxdbdy,radec):

    moonsegmentlist = segments.segmentlist()
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

        ra_moon = (180/np.pi)*float(repr(moon.ra))
        dec_moon = (180/np.pi)*float(repr(moon.dec))

        # Coverting both target and moon ra and dec to radians
        ra1 = float(repr(moon.ra))
        d1 = float(repr(moon.dec))

        # Calculate angle between target and moon
        cosA = np.sin(d1)*np.sin(d2) + np.cos(d1)*np.cos(d2)*np.cos(ra1-ra2)
        angle = np.arccos(cosA)*(360/(2*np.pi))
        #print("Angle between moon and target: %.5f"%(angle))

        if angle >= 50.0*moon.moon_phase**2:
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

    dt = 6.0/24.0
    tt = np.arange(segmentlist[0][0],segmentlist[-1][1]+dt,dt)

    fxdbdy = astroplan.FixedTarget(coord=radec)

    ra2 = radec.ra.radian
    d2 = radec.dec.radian

    # Where is the moon?
    for ii in range(len(tt)-1):

        date_start = Time(tt[ii], format='mjd', scale='utc')

        alt_target = observer.altaz(date_start,fxdbdy).alt.deg
        az_target = observer.altaz(date_start,fxdbdy).az.deg
        #print("Altitude / Azimuth of target: %.5f / %.5f"%(alt_target,az_target))

        alt_moon = observer.moon_altaz(date_start).alt.deg
        az_moon = observer.moon_altaz(date_start).az.deg 

        #print("Altitude / Azimuth of moon: %.5f / %.5f"%(alt_moon,az_moon))

        if (alt_target < 30.0) or (alt_moon < 30.0):
            total_mag, total_mag_error, flux_mag, flux_mag_error = np.inf, np.inf, np.inf, np.inf
        else:
            # Coverting both target and moon ra and dec to radians
            ra1 = astropy.coordinates.get_moon(date_start).ra.radian
            d1 = astropy.coordinates.get_moon(date_start).dec.radian

            # Calculate angle between target and moon
            cosA = np.sin(d1)*np.sin(d2) + np.cos(d1)*np.cos(d2)*np.cos(ra1-ra2)
            angle = np.arccos(cosA)*(360/(2*np.pi))
            #print("Angle between moon and target: %.5f"%(angle))

            moon_phase = np.mod(observer.moon_phase(date_start).value*(360/(2*np.pi)),90)

            delta_mag = np.interp(moon_phase,moon_phases,moon_data_passband)
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

def get_segments(params, config_struct):

    gpstime = params["gpstime"]
    event_mjd = Time(gpstime, format='gps', scale='utc').mjd

    segmentlist = segments.segmentlist()
    n_windows = len(params["Tobs"]) // 2
    start_segments = event_mjd + params["Tobs"][::2]
    end_segments = event_mjd + params["Tobs"][1::2]
    for start_segment, end_segment in zip(start_segments,end_segments):
        segmentlist.append(segments.segment(start_segment,end_segment))

    location = astropy.coordinates.EarthLocation(config_struct["longitude"],config_struct["latitude"],config_struct["elevation"])
    observer = astroplan.Observer(location=location)

    date_start = Time(segmentlist[0][0], format='mjd', scale='utc')
    date_end = Time(segmentlist[-1][1], format='mjd', scale='utc')

    nightsegmentlist = segments.segmentlist()
    while date_start < date_end:
        date_rise = observer.twilight_morning_astronomical(date_start)
        date_set = observer.twilight_evening_astronomical(date_start)
        if date_set.mjd > date_rise.mjd:
            date_set = observer.twilight_evening_astronomical(date_start-TimeDelta(24*u.hour))

        segment = segments.segment(date_set.mjd,date_rise.mjd)
        nightsegmentlist = nightsegmentlist + segments.segmentlist([segment])
        nightsegmentlist.coalesce()

        date_start = date_rise + TimeDelta(24*u.hour)

    segmentlistdic = segments.segmentlistdict()
    segmentlistdic["observations"] = segmentlist
    segmentlistdic["night"] = nightsegmentlist
    segmentlist = segmentlistdic.intersection(["observations","night"])
    segmentlist.coalesce()

    return segmentlist

def get_segments_tile(config_struct, observatory, radec, segmentlist):

    observer = astroplan.Observer(location=observatory)

    fxdbdy = astroplan.FixedTarget(coord=radec)

    date_start = Time(segmentlist[0][0], format='mjd', scale='utc')
    date_end = Time(segmentlist[-1][1], format='mjd', scale='utc')

    tilesegmentlist = segments.segmentlist()
    while date_start.mjd < date_end.mjd:
        date_rise = observer.target_rise_time(date_start,fxdbdy)
        date_set = observer.target_set_time(date_start,fxdbdy)

        print(date_rise.mjd,date_set.mjd)
        if (date_rise.mjd<0) and (date_set.mjd<0): break

        print(date_rise.mjd,date_set.mjd)

        if date_rise > date_set:
            date_rise = observer.target_rise_time(date_start-TimeDelta(24*u.hour),fxdbdy)
        print(date_rise.mjd,date_set.mjd)

        segment = segments.segment(date_rise.mjd,date_set.mjd)
        tilesegmentlist = tilesegmentlist + segments.segmentlist([segment])
        tilesegmentlist.coalesce()

        date_start = date_set+TimeDelta(24*u.hour)

    #moonsegmentlist = get_skybrightness(\
    #    config_struct,segmentlist,observer,fxdbdy,radec)

    moonsegmentlist = get_moon_segments(\
        config_struct,segmentlist,observer,fxdbdy,radec)

    tilesegmentlistdic = segments.segmentlistdict()
    tilesegmentlistdic["observations"] = segmentlist
    tilesegmentlistdic["tile"] = tilesegmentlist
    tilesegmentlistdic["moon"] = moonsegmentlist
    tilesegmentlist = tilesegmentlistdic.intersection(["observations","tile","moon"])
    tilesegmentlist.coalesce()

    return tilesegmentlist

def get_segments_tiles(config_struct, tile_struct):

    print("Generating segments for tiles...")

    observatory = astropy.coordinates.EarthLocation(
        lat=config_struct["latitude"]*u.deg, lon=config_struct["longitude"]*u.deg, height=config_struct["elevation"]*u.m)

    segmentlist = config_struct["segmentlist"]

    ras = []
    decs = []
    keys = tile_struct.keys()
    for key in keys:
        ras.append(tile_struct[key]["ra"])
        decs.append(tile_struct[key]["dec"])

    # Convert to RA, Dec.
    radecs = astropy.coordinates.SkyCoord(
            ra=np.array(ras)*u.degree, dec=np.array(decs)*u.degree, frame='icrs')
    tilesegmentlists = []
    for ii,key in enumerate(keys):
        print(ii,radecs[ii])
        #if np.mod(ii,100) == 0: 
        #    print("Generating segments for tile %d/%d"%(ii+1,len(radecs)))
        radec = radecs[ii]
        tilesegmentlist = get_segments_tile(config_struct, observatory, radec, segmentlist)
        tilesegmentlists.append(tilesegmentlist)
        tile_struct[key]["segmentlist"] = tilesegmentlist
        print(tile_struct[key]["segmentlist"])

    return tilesegmentlists
