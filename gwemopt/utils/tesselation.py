import logging
from pathlib import Path

import healpy as hp
import numpy as np

logger = logging.getLogger(__name__)


def tesselation_spiral(
    fov_type: str, fov: float, scale: float = 0.80, save_path: Path | None = None
):
    if fov_type == "square":
        FOV = fov * fov * scale
    elif fov_type == "circle":
        FOV = np.pi * fov * fov * scale

    area_of_sphere = 4 * np.pi * (180 / np.pi) ** 2
    n = int(np.ceil(area_of_sphere / FOV))
    logger.debug("Using %d points to tile the sphere..." % n)

    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)

    points = np.zeros((n, 3))
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = z

    ra, dec = hp.pixelfunc.vec2ang(points, lonlat=True)
    if save_path:
        fid = open(save_path, "w")
        for ii in range(len(ra)):
            fid.write("%d %.5f %.5f\n" % (ii, ra[ii], dec[ii]))
        fid.close()
    return np.stack([np.arange(len(ra)), ra, dec], axis=-1)


def tesselation_packing(
    fov_type: str, fov: float, scale: float = 0.97, save_path: Path | None = None
):
    sphere_radius = 1.0
    if fov_type == "square":
        circle_radius = np.deg2rad(fov / 2.0) * scale
    elif fov_type == "circle":
        circle_radius = np.deg2rad(fov) * scale
    vertical_count = int((np.pi * sphere_radius) / (2 * circle_radius))

    phis = []
    thetas = []

    phi = -0.5 * np.pi
    phi_step = np.pi / vertical_count
    while phi < 0.5 * np.pi:
        horizontal_count = int(
            (2 * np.pi * np.cos(phi) * sphere_radius) / (2 * circle_radius)
        )
        if horizontal_count == 0:
            horizontal_count = 1
        theta = 0
        theta_step = 2 * np.pi / horizontal_count
        while theta < 2 * np.pi - 1e-8:
            phis.append(phi)
            thetas.append(theta)
            theta += theta_step
        phi += phi_step
    dec = np.array(np.rad2deg(phis))
    ra = np.array(np.rad2deg(thetas))

    if save_path:
        fid = open(save_path, "w")
        for ii in range(len(ra)):
            fid.write("%d %.5f %.5f\n" % (ii, ra[ii], dec[ii]))
        fid.close()
    return np.stack([np.arange(len(ra)), ra, dec], axis=-1)
