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
    rotation=None,
):
    theta = 0.5 * np.pi - np.deg2rad(dec_pointing)
    phi = np.deg2rad(ra_pointing)

    HPX = HEALPix(nside=nside, order="nested", frame=coordinates.ICRS())

    skyoffset_frames = coordinates.SkyCoord(
        ra_pointing, dec_pointing, unit=u.deg
    ).skyoffset_frame()

    proj = hp.projector.MollweideProj(rot=rotation, coord=None)

    # security for the periodic limit conditions
    ipix = []
    radecs = np.empty((0, 2))
    patch = []
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

        moc = MOC.from_polygon_skycoord(coords)

        pix_id = moc.degrade_to_order(int(np.log2(nside))).flatten()
        if len(pix_id) > 0:
            ipix.extend(hp.nest2ring(int(nside), pix_id.tolist()))

        ras = [coord.ra.deg for coord in coords]
        decs = [coord.dec.deg for coord in coords]
        coords = np.vstack([ras, decs]).T

        xyz = hp.ang2vec(coords[:, 0], coords[:, 1], lonlat=True)
        x, y = proj.vec2xy(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        xy = np.vstack([x, y]).T
        path = matplotlib.path.Path(xy)

        radecs = np.vstack([radecs, coords])

        patch.append(
            matplotlib.patches.PathPatch(
                path,
                alpha=alpha,
                facecolor=color,
                fill=True,
                zorder=3,
                edgecolor=edgecolor,
            )
        )

    ipix = list(set(ipix))
    area = len(ipix) * hp.nside2pixarea(nside, degrees=True)

    return ipix, radecs, patch, area


def getCirclePixels(
    ra_pointing,
    dec_pointing,
    radius,
    nside,
    alpha=0.4,
    color="k",
    edgecolor="k",
    rotation=None,
):
    theta = 0.5 * np.pi - np.deg2rad(dec_pointing)
    phi = np.deg2rad(ra_pointing)

    xyz = hp.ang2vec(theta, phi)
    ipix = hp.query_disc(nside, xyz, np.deg2rad(radius))

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
    idx1 = np.where(radecs[:, 0] >= 180.0)[0]
    idx2 = np.where(radecs[:, 0] < 180.0)[0]
    idx3 = np.where(radecs[:, 0] > 300.0)[0]
    idx4 = np.where(radecs[:, 0] < 60.0)[0]
    if (len(idx1) > 0 and len(idx2) > 0) and not (len(idx3) > 0 and len(idx4) > 0):
        alpha = 0.0

    xyz = hp.ang2vec(radecs[:, 0], radecs[:, 1], lonlat=True)

    proj = hp.projector.MollweideProj(rot=rotation, coord=None)
    x, y = proj.vec2xy(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    xy = np.zeros(radecs.shape)
    xy[:, 0] = x
    xy[:, 1] = y
    # path = matplotlib.path.Path(xyz[:,1:3])
    path = matplotlib.path.Path(xy)
    patch = matplotlib.patches.PathPatch(
        path, alpha=alpha, facecolor=color, fill=True, zorder=3, edgecolor=edgecolor
    )

    area = np.pi * radius**2

    return ipix, radecs, patch, area


def getRectanglePixels(
    ra_pointing,
    dec_pointing,
    raSide,
    decSide,
    nside,
    alpha=0.4,
    color="k",
    edgecolor="k",
    rotation=None,
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
        return [], [], [], []

    idx1 = np.where((radecs[:, 1] >= 87.0) | (radecs[:, 1] <= -87.0))[0]
    if len(idx1) > 0:
        radecs = np.delete(radecs, idx1[0], 0)

    xyz = []
    for r, d in radecs:
        xyz.append(hp.ang2vec(r, d, lonlat=True))

    npts, junk = radecs.shape
    if npts == 4:
        xyz = [xyz[0], xyz[1], xyz[3], xyz[2]]
        ipix = hp.query_polygon(nside, np.array(xyz))
    else:
        ipix = hp.query_polygon(nside, np.array(xyz))

    xyz = np.array(xyz)
    proj = hp.projector.MollweideProj(rot=rotation, coord=None)
    x, y = proj.vec2xy(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    xy = np.zeros(radecs.shape)
    xy[:, 0] = x
    xy[:, 1] = y
    path = matplotlib.path.Path(xy)
    patch = matplotlib.patches.PathPatch(
        path, alpha=alpha, facecolor=color, fill=True, zorder=3, edgecolor=edgecolor
    )

    return ipix, radecs, patch, area


def getSquarePixels(
    ra_pointing,
    dec_pointing,
    tileSide,
    nside,
    alpha=0.4,
    color="k",
    edgecolor="k",
    rotation=None,
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
        return [], [], [], []

    idx1 = np.where((radecs[:, 1] >= 87.0) | (radecs[:, 1] <= -87.0))[0]
    if len(idx1) > 0:
        radecs = np.delete(radecs, idx1[0], 0)

    xyz = []
    for r, d in radecs:
        xyz.append(hp.ang2vec(r, d, lonlat=True))

    npts, junk = radecs.shape
    if npts == 4:
        xyz = [xyz[0], xyz[1], xyz[3], xyz[2]]
        ipix = hp.query_polygon(nside, np.array(xyz))
    else:
        ipix = hp.query_polygon(nside, np.array(xyz))

    xyz = np.array(xyz)
    proj = hp.projector.MollweideProj(rot=rotation, coord=None)
    x, y = proj.vec2xy(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    xy = np.zeros(radecs.shape)
    xy[:, 0] = x
    xy[:, 1] = y
    path = matplotlib.path.Path(xy)
    patch = matplotlib.patches.PathPatch(
        path, alpha=alpha, facecolor=color, fill=True, zorder=3, edgecolor=edgecolor
    )

    return ipix, radecs, patch, area
