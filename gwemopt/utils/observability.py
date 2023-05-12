import copy

import astropy.coordinates
import healpy as hp
import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta
from tqdm import tqdm


def calculate_observability(params, map_struct):
    """
    Calculate observability of a skymap for a given set of telescopes.
    """
    airmass = params["airmass"]
    nside = params["nside"]
    npix = hp.nside2npix(nside)
    gpstime = params["gpstime"]
    event_time = Time(gpstime, format="gps", scale="utc")
    dts = np.arange(0, 1, 0.1)

    observatory_struct = {}

    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]

        observatory = astropy.coordinates.EarthLocation(
            lat=config_struct["latitude"] * u.deg,
            lon=config_struct["longitude"] * u.deg,
            height=config_struct["elevation"] * u.m,
        )

        # Look up (celestial) spherical polar coordinates of HEALPix grid.
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        # Convert to RA, Dec.
        radecs = astropy.coordinates.SkyCoord(
            ra=phi * u.rad, dec=(0.5 * np.pi - theta) * u.rad
        )

        observatory_struct[telescope] = {}
        observatory_struct[telescope]["prob"] = copy.deepcopy(map_struct["prob"])
        observatory_struct[telescope]["observability"] = np.zeros((npix,))
        observatory_struct[telescope]["dts"] = {}

        for dt in tqdm(dts):
            time = event_time + TimeDelta(dt * u.day)

            # Alt/az reference frame at observatory, now
            frame = astropy.coordinates.AltAz(obstime=time, location=observatory)
            # Transform grid to alt/az coordinates at observatory, now
            altaz = radecs.transform_to(frame)

            # Where is the sun, now?
            sun_altaz = astropy.coordinates.get_sun(time).transform_to(altaz)

            # How likely is it that the (true, unknown) location of the source
            # is within the area that is visible, now? Demand that sun is at
            # least 18 degrees below the horizon and that the airmass
            # (secant of zenith angle approximation) is at most 2.5.
            idx = np.where(
                (altaz.alt >= 30 * u.deg)
                & (sun_altaz.alt <= -18 * u.deg)
                & (altaz.secz <= airmass)
            )[0]
            observatory_struct[telescope]["dts"][dt] = np.zeros((npix,))
            observatory_struct[telescope]["dts"][dt][idx] = 1
            observatory_struct[telescope]["observability"][idx] = 1
        observatory_struct[telescope]["prob"] = (
            observatory_struct[telescope]["prob"]
            * observatory_struct[telescope]["observability"]
        )

    return observatory_struct
