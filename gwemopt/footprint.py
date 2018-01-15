
import os
import numpy as np
import healpy as hp

def get_skymap(params):
   
    nside = params["nside"]

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra = np.rad2deg(phi)
    dec = np.rad2deg(0.5*np.pi - theta)

    footprint_ra, footprint_dec, footprint_radius = params["footprint_ra"], params["footprint_dec"], params["footprint_radius"]

    ra1, d1 = np.deg2rad(ra), np.deg2rad(dec)
    ra2, d2 = np.deg2rad(footprint_ra), np.deg2rad(footprint_dec)

    # Calculate angle between target and moon
    cosA = np.sin(d1)*np.sin(d2) + np.cos(d1)*np.cos(d2)*np.cos(ra1-ra2)
    angle = np.arccos(cosA)*(360/(2*np.pi))
    prob = np.exp(-np.power(angle, 2.) / (2 * np.power(footprint_radius, 2.)))
    prob = prob / np.sum(prob)

    fitsfile = os.path.join(params["outputDir"],'skymap.fits')
    hp.fitsfunc.write_map(fitsfile,prob)

    return fitsfile
