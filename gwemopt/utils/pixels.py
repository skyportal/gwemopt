import astropy.units as u
import healpy as hp
import matplotlib
import numpy as np
from astropy import coordinates
from astropy_healpix import HEALPix
from mocpy import MOC


def get_ellipse_coords(a=0.0, b=0.0, x=0.0, y=0.0, angle=0.0, npts=10):
    """Draws an ellipse using (360*k + 1) discrete points; based on pseudo code
    given at http://en.wikipedia.org/wiki/Ellipse
    k = 1 means 361 points (degree by degree)
    a = major axis distance,
    b = minor axis distance,
    x = offset along the x-axis
    y = offset along the y-axis
    angle = clockwise rotation [in degrees] of the ellipse;
        * angle=0  : the ellipse is aligned with the positive x-axis
        * angle=30 : rotated 30 degrees clockwise from positive x-axis
    """
    pts = np.zeros((npts, 2))

    beta = -angle * np.pi / 180.0
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    alpha = np.linspace(0, 2 * np.pi, npts)

    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)

    pts[:, 0] = x + (a * cos_alpha * cos_beta - b * sin_alpha * sin_beta)
    pts[:, 1] = y + (a * cos_alpha * sin_beta + b * sin_alpha * cos_beta)

    return pts


def getRegionPixels(
    ra_pointing,
    dec_pointing,
    regions,
    nside,
    alpha=0.4,
    color="k",
    edgecolor="k",
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


def getCirclePixels(
    ra_pointing,
    dec_pointing,
    radius,
    alpha=0.4,
    color="k",
    edgecolor="k",
):

    radecs = get_ellipse_coords(
        a=radius / np.cos(np.deg2rad(dec_pointing)),
        b=radius,
        x=ra_pointing,
        y=dec_pointing,
        angle=0.0,
        npts=25,
    )
    idx = np.where(radecs[:, 1] > 90.0)[0]
    radecs[idx, 1] = 180.0 - radecs[idx, 1]
    idx = np.where(radecs[:, 1] < -90.0)[0]
    radecs[idx, 1] = -180.0 - radecs[idx, 1]
    idx = np.where(radecs[:, 0] > 360.0)[0]
    radecs[idx, 0] = 720.0 - radecs[idx, 0]
    idx = np.where(radecs[:, 0] < 0.0)[0]
    radecs[idx, 0] = 360.0 + radecs[idx, 0]

    radecs = np.array(radecs)
    coords = coordinates.SkyCoord(radecs[:, 0] * u.deg, radecs[:, 1] * u.deg)
    moc = MOC.from_polygon_skycoord(coords)

    return moc


def getRectanglePixels(
    ra_pointing,
    dec_pointing,
    raSide,
    decSide,
    alpha=0.4,
    color="k",
    edgecolor="k",
):
    area = raSide * decSide

    decCorners = (dec_pointing - decSide / 2.0, dec_pointing + decSide / 2.0)

    # security for the periodic limit conditions
    radecs = []
    for d in decCorners:
        if d > 90.0:
            d = 180.0 - d
        elif d < -90.0:
            d = -180 - d

        raCorners = (
            ra_pointing - (raSide / 2.0) / np.cos(np.deg2rad(d)),
            ra_pointing + (raSide / 2.0) / np.cos(np.deg2rad(d)),
        )
        # security for the periodic limit conditions
        for r in raCorners:
            if r > 360.0:
                r = r - 360.0
            elif r < 0.0:
                r = 360.0 + r
            radecs.append([r, d])

    radecs = np.array(radecs)
    idx1 = np.where(radecs[:, 0] >= 180.0)[0]
    idx2 = np.where(radecs[:, 0] < 180.0)[0]
    idx3 = np.where(radecs[:, 0] > 300.0)[0]
    idx4 = np.where(radecs[:, 0] < 60.0)[0]
    if (len(idx1) > 0 and len(idx2) > 0) and not (len(idx3) > 0 and len(idx4) > 0):
        alpha = 0.0

    idx1 = np.where(np.abs(radecs[:, 1]) >= 87.0)[0]
    if len(idx1) == 4:
        return MOC.new_empty(max_depth=10)

    idx1 = np.where((radecs[:, 1] >= 87.0) | (radecs[:, 1] <= -87.0))[0]
    if len(idx1) > 0:
        radecs = np.delete(radecs, idx1[0], 0)

    npts, junk = radecs.shape
    if npts == 4:
        idx = [0, 1, 3, 2]
        radecs = radecs[idx, :]

    coords = coordinates.SkyCoord(radecs[:, 0] * u.deg, radecs[:, 1] * u.deg)
    moc = MOC.from_polygon_skycoord(coords)

    return moc


def getSquarePixels(
    ra_pointing,
    dec_pointing,
    tileSide,
    alpha=0.4,
    color="k",
    edgecolor="k",
):
    area = tileSide * tileSide

    decCorners = (dec_pointing - tileSide / 2.0, dec_pointing + tileSide / 2.0)

    # security for the periodic limit conditions
    radecs = []
    for d in decCorners:
        if d > 90.0:
            d = 180.0 - d
        elif d < -90.0:
            d = -180 - d

        raCorners = (
            ra_pointing - (tileSide / 2.0) / np.cos(np.deg2rad(d)),
            ra_pointing + (tileSide / 2.0) / np.cos(np.deg2rad(d)),
        )

        # security for the periodic limit conditions
        for r in raCorners:
            if r > 360.0:
                r = r - 360.0
            elif r < 0.0:
                r = 360.0 + r
            radecs.append([r, d])

    radecs = np.array(radecs)
    idx1 = np.where(radecs[:, 0] >= 180.0)[0]
    idx2 = np.where(radecs[:, 0] < 180.0)[0]
    idx3 = np.where(radecs[:, 0] > 300.0)[0]
    idx4 = np.where(radecs[:, 0] < 60.0)[0]
    if (len(idx1) > 0 and len(idx2) > 0) and not (len(idx3) > 0 and len(idx4) > 0):
        alpha = 0.0

    idx1 = np.where(np.abs(radecs[:, 1]) >= 87.0)[0]
    if len(idx1) == 4:
        return MOC.new_empty(max_depth=10)

    idx1 = np.where((radecs[:, 1] >= 87.0) | (radecs[:, 1] <= -87.0))[0]
    if len(idx1) > 0:
        radecs = np.delete(radecs, idx1[0], 0)

    npts, junk = radecs.shape
    if npts == 4:
        idx = [0, 1, 3, 2]
        radecs = radecs[idx, :]

    coords = coordinates.SkyCoord(radecs[:, 0] * u.deg, radecs[:, 1] * u.deg)
    moc = MOC.from_polygon_skycoord(coords)

    return moc
