import healpy as hp
import numpy as np


def rotate_map(hmap, rot_theta, rot_phi):
    """
    Take hmap (a healpix map array) and return another healpix map array
    which is ordered such that it has been rotated in (theta, phi) by the
    amounts given.
    """
    nside = hp.npix2nside(len(hmap))

    # Get theta, phi for non-rotated map
    t, p = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))  # theta, phi

    # Define a rotator
    r = hp.Rotator(deg=False, rot=[rot_phi, rot_theta])

    # Get theta, phi under rotated co-ordinates
    trot, prot = r(t, p)

    # Interpolate map onto these co-ordinates
    rot_map = hp.get_interp_val(hmap, trot, prot)

    return rot_map


def angular_distance(ra1, dec1, ra2, dec2):
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
