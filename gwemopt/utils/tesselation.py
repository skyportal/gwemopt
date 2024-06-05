import healpy as hp
import numpy as np


def tesselation_spiral(config_struct, scale=0.80):
    if config_struct["FOV_type"] == "square":
        FOV = config_struct["FOV"] * config_struct["FOV"] * scale
    elif config_struct["FOV_type"] == "circle":
        FOV = np.pi * config_struct["FOV"] * config_struct["FOV"] * scale

    area_of_sphere = 4 * np.pi * (180 / np.pi) ** 2
    n = int(np.ceil(area_of_sphere / FOV))
    print("Using %d points to tile the sphere..." % n)

    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)

    points = np.zeros((n, 3))
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = z

    ra, dec = hp.pixelfunc.vec2ang(points, lonlat=True)
    fid = open(config_struct["tesselationFile"], "w")
    for ii in range(len(ra)):
        fid.write("%d %.5f %.5f\n" % (ii, ra[ii], dec[ii]))
    fid.close()


def tesselation_packing(config_struct, scale=0.97):
    sphere_radius = 1.0
    if config_struct["FOV_type"] == "square":
        circle_radius = np.deg2rad(config_struct["FOV"] / 2.0) * scale
    elif config_struct["FOV_type"] == "circle":
        circle_radius = np.deg2rad(config_struct["FOV"]) * scale
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

    fid = open(config_struct["tesselationFile"], "w")
    for ii in range(len(ra)):
        fid.write("%d %.5f %.5f\n" % (ii, ra[ii], dec[ii]))
    fid.close()
