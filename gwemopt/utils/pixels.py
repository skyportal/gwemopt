import astropy.units as u
import healpy as hp
import matplotlib
import numpy as np
from astropy import coordinates
from astropy_healpix import HEALPix
from mocpy import MOC


def getRegionPixels(
    ra_pointing,
    dec_pointing,
    regions,
    nside,
):
    theta = 0.5 * np.pi - np.deg2rad(dec_pointing)
    phi = np.deg2rad(ra_pointing)

    HPX = HEALPix(nside=nside, order="nested", frame=coordinates.ICRS())

    skyoffset_frames = coordinates.SkyCoord(
        ra_pointing, dec_pointing, unit=u.deg
    ).skyoffset_frame()

    moc = None

    # security for the periodic limit conditions
    for reg in regions:
        ra_tmp = reg.vertices.ra
        dec_tmp = reg.vertices.dec

        coords = np.stack([np.array(ra_tmp), np.array(dec_tmp)])

        # Copy the tile coordinates such that there is one per field
        # in the grid
        coords_icrs = coordinates.SkyCoord(
            *np.tile(coords[:, np.newaxis, ...], (1, 1, 1)),
            unit=u.deg,
            frame=skyoffset_frames[:, np.newaxis, np.newaxis],
        ).transform_to(coordinates.ICRS)
        coords = coords_icrs.transform_to(HPX.frame)

        if moc is None:
            moc = MOC.from_polygon_skycoord(coords)
        else:
            moc = moc + MOC.from_polygon_skycoord(coords)

    return moc
