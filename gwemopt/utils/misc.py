import copy

import astropy
import astropy.coordinates
import astropy.units as u
import healpy as hp
import ligo.segments as segments
import ligo.skymap.distance as ligodist
import numpy as np
from astropy.time import Time, TimeDelta


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
        overhead = overhead + config_struct.get("filt_change_time", 0)

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
