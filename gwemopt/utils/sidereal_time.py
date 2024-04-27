from datetime import datetime, timedelta
from math import pi

import numpy as np

GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)
EPOCH_J2000_0_JD = 2451545.0
_LEAP_SECONDS = np.asarray(
    [
        46828800,
        78364801,
        109900802,
        173059203,
        252028804,
        315187205,
        346723206,
        393984007,
        425520008,
        457056009,
        504489610,
        551750411,
        599184012,
        820108813,
        914803214,
        1025136015,
        1119744016,
        1167264017,
    ]
)


def greenwich_sidereal_time(tt, equation_of_equinoxes=0):
    """
    Compute the Greenwich mean sidereal time from the GPS time and equation of
    equinoxes.

    Based on XLALGreenwichSiderealTime in lalsuite/lal/lib/XLALSiderealTime.c.

    Parameters
    ----------
    tt: astropy.time.Time
        The astropy time to convert
    """
    t_hi = (tt.jd - EPOCH_J2000_0_JD) / 36525.0
    t_lo = (tt.gps % 1) / (36525.0 * 86400.0)

    t = t_hi + t_lo

    sidereal_time = (
        equation_of_equinoxes + (-6.2e-6 * t + 0.093104) * t**2 + 67310.54841
    )
    sidereal_time += 8640184.812866 * t_lo
    sidereal_time += 3155760000.0 * t_lo
    sidereal_time += 8640184.812866 * t_hi
    sidereal_time += 3155760000.0 * t_hi

    return sidereal_time * pi / 43200.0
