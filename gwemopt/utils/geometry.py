import numpy as np


def angular_distance(ra1, dec1, ra2, dec2):
    """
    Calculate the angular distance between two points on the sky

    :param ra1: Right ascension of first point (in degrees)
    :param dec1: Declination of first point (in degrees)
    :param ra2: Right ascension of second point (in degrees)
    :param dec2: Declination of second point (in degrees)
    :return: Angular distance between the two points (in degrees)
    """
    delt_lon = (ra1 - ra2) * np.pi / 180.0
    delt_lat = (dec1 - dec2) * np.pi / 180.0
    dist = 2.0 * np.arcsin(
        np.sqrt(
            np.sin(delt_lat / 2.0) ** 2
            + np.cos(dec1 * np.pi / 180.0)
            * np.cos(dec2 * np.pi / 180.0)
            * np.sin(delt_lon / 2.0) ** 2
        )
    )

    return dist / np.pi * 180.0
