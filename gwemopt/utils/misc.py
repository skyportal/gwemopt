import copy

import astropy
import astropy.coordinates
import astropy.units as u
import healpy as hp
import ligo.segments as segments
import ligo.skymap.distance as ligodist
import numpy as np
from astropy.time import Time, TimeDelta


def auto_rasplit(params, map_struct, nside_down):
    if params["do_3d"]:
        prob_down, distmu_down, distsigma_down, distnorm_down = ligodist.ud_grade(
            map_struct["prob"],
            map_struct["distmu"],
            map_struct["distsigma"],
            nside_down,
        )
    else:
        prob_down = hp.ud_grade(map_struct["prob"], nside_down, power=-2)

    npix = hp.nside2npix(nside_down)
    theta, phi = hp.pix2ang(nside_down, np.arange(npix))
    ra = np.rad2deg(phi) * 24.0 / 360.0
    dec = np.rad2deg(0.5 * np.pi - theta)

    ra_unique = np.unique(ra)
    prob_sum = np.zeros(ra_unique.shape)
    for ii, r in enumerate(ra_unique):
        idx = np.where(r == ra)[0]
        prob_sum[ii] = np.sum(prob_down[idx])

    sort_idx = np.argsort(prob_sum)[::-1]
    csm = np.empty(len(prob_sum))
    csm[sort_idx] = np.cumsum(prob_sum[sort_idx])

    idx = np.where(csm <= params["powerlaw_cl"])[0]

    if (0 in idx) and (len(ra_unique) - 1 in idx):
        wrap = True
    else:
        wrap = False

    dr = ra_unique[1] - ra_unique[0]
    segmentlist = segments.segmentlist()
    for ii in idx:
        ra1, ra2 = ra_unique[ii], ra_unique[ii] + dr
        segment = segments.segment(ra1, ra2)
        segmentlist = segmentlist + segments.segmentlist([segment])
        segmentlist.coalesce()

    if wrap:
        idxremove = []
        for ii, seg in enumerate(segmentlist):
            if np.isclose(seg[0], 0.0) and wrap:
                seg1 = seg
                idxremove.append(ii)
                continue
            if np.isclose(seg[1], 24.0) and wrap:
                seg2 = seg
                idxremove.append(ii)
                continue

        for ele in sorted(idxremove, reverse=True):
            del segmentlist[ele]

        raslices = []
        for segment in segmentlist:
            raslices.append([segment[0], segment[1]])
        raslices.append([seg2[0], seg1[1]])
    else:
        raslices = []
        for segment in segmentlist:
            raslices.append([segment[0], segment[1]])

    return raslices


def integrationTime(T_obs, pValTiles, func=None, T_int=60.0):
    """
    METHOD :: This method accepts the probability values of the ranked tiles, the
              total observation time and the rank of the source tile. It returns
              the array of time to be spent in each tile which is determined based
              on the localizaton probability of the tile. How the weight factor is
              computed can also be supplied in functional form. Default is linear.

    pValTiles :: The probability value of the ranked tiles. Obtained from ZTF_RT
                             output
    T_obs     :: Total observation time available for the follow-up.
    func      :: functional form of the weight. Default is linear.
                             For example, use x**2 to use a quadratic function.
    """

    if func is None:
        f = lambda x: x
    else:
        f = lambda x: eval(func)
    fpValTiles = f(pValTiles)
    modified_prob = fpValTiles / np.sum(fpValTiles)
    modified_prob[np.isnan(modified_prob)] = 0.0
    t_tiles = modified_prob * T_obs  ### Time spent in each tile if not constrained
    # t_tiles[t_tiles > 1200.0] = 1200.0 ### Upper limit of exposure time
    # t_tiles[t_tiles < 60] = 60.0 ### Lower limit of exposure time
    t_tiles = T_int * np.round(t_tiles / T_int)
    # Obs = np.cumsum(t_tiles) <= T_obs ### Tiles observable in T_obs seconds
    # time_per_tile = t_tiles[Obs] ### Actual time spent per tile

    return t_tiles


def observability(params, map_struct):
    airmass = params["airmass"]
    nside = params["nside"]
    npix = hp.nside2npix(nside)
    gpstime = params["gpstime"]
    event_time = Time(gpstime, format="gps", scale="utc")
    dts = np.arange(0, 7, 1.0 / 24.0)
    dts = np.arange(0, 7, 1.0 / 4.0)

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

        for dt in dts:
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


def get_exposures(params, config_struct, segmentlist):
    """
    Convert the availability times to a list segments with the length of telescope exposures.
    segmentlist: the segments that the telescope can do the follow-up.
    """
    exposurelist = segments.segmentlist()
    if "overhead_per_exposure" in config_struct.keys():
        overhead = config_struct["overhead_per_exposure"]
    else:
        overhead = 0.0

    # add the filter change time to the total overheads for integrated
    if not params["doAlternatingFilters"]:
        overhead = overhead + config_struct["filt_change_time"]

    exposure_time = np.max(params["exposuretimes"])

    for ii in range(len(segmentlist)):
        start_segment, end_segment = segmentlist[ii][0], segmentlist[ii][1]
        exposures = np.arange(
            start_segment, end_segment, (overhead + exposure_time) / 86400.0
        )

        for jj in range(len(exposures)):
            exposurelist.append(
                segments.segment(exposures[jj], exposures[jj] + exposure_time / 86400.0)
            )

    return exposurelist
