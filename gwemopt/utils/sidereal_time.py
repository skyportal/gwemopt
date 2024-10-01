from datetime import datetime
from math import pi

import numpy as np
from numba import njit

GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)
EPOCH_J2000_0_JD = 2451545.0


@njit
def greenwich_sidereal_time(jd, gps, equation_of_equinoxes=0):
    """
    Compute the Greenwich mean sidereal time from the GPS time and equation of
    equinoxes.

    Based on XLALGreenwichSiderealTime in lalsuite/lal/lib/XLALSiderealTime.c.

    Parameters
    ----------
    jd : float
        Julian date.
    gps : float
        GPS time (in seconds).
    equation_of_equinoxes : float, optional
        Equation of equinoxes (default is 0).
    """
    t_hi = (jd - EPOCH_J2000_0_JD) / 36525.0
    t_lo = (gps % 1) / (36525.0 * 86400.0)

    t = t_hi + t_lo

    sidereal_time = (
        equation_of_equinoxes + (-6.2e-6 * t + 0.093104) * t**2 + 67310.54841
    )
    sidereal_time += 8640184.812866 * t_lo
    sidereal_time += 3155760000.0 * t_lo
    sidereal_time += 8640184.812866 * t_hi
    sidereal_time += 3155760000.0 * t_hi

    return sidereal_time * pi / 43200.0


@njit
def hour_angle(jd, gps, long, ra, equation_of_equinoxes=0):
    """
    Compute the hour angle from the Julian date, GPS time, observer's longitude,
    and equation of equinoxes.

    Parameters
    ----------
    jd : float
        Julian date.
    gps : float
        GPS time (in seconds).
    long : float
        Observer's longitude (in degrees).
    ra : float
        Target's right ascension (in degrees).
    equation_of_equinoxes : float, optional
        Equation of equinoxes (default is 0).
    """
    lst = np.mod(
        np.deg2rad(long) + greenwich_sidereal_time(jd, gps, equation_of_equinoxes),
        2 * np.pi,
    )
    return (lst - np.deg2rad(ra)) * 24 / (2 * np.pi)
