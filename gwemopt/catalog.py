
import os, sys

import numpy as np
import healpy as hp

from scipy.stats import norm
from scipy.special import gammaincinv

from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1

from astropy.cosmology import WMAP9 as cosmo
from astropy.table import Column
import astropy.units as u
import astropy.constants as c

def get_catalog(params, map_struct):

    cat, = Vizier.get_catalogs('J/ApJS/199/26/table3')

    completeness = 0.5
    alpha = -1.0
    MK_star = -23.55
    MK_max = MK_star + 2.5 * np.log10(gammaincinv(alpha + 2, completeness))

    z = (u.Quantity(cat['cz']) / c.c).to(u.dimensionless_unscaled)
    MK = cat['Ktmag'] - cosmo.distmod(z)
    keep = (z > 0) & (MK < MK_max)

    cat = cat[keep]
    z = z[keep]

    r = cosmo.luminosity_distance(z).to('Mpc').value
    theta = 0.5 * np.pi - cat['DEJ2000'].to('rad').value
    phi = cat['RAJ2000'].to('rad').value
    ipix = hp.ang2pix(map_struct["nside"], theta, phi)

    if "distnorm" in map_struct:
        dp_dV = map_struct["prob"][ipix] * map_struct["distnorm"][ipix] * norm(map_struct["distmu"][ipix], map_struct["distsigma"][ipix]).pdf(r) / map_struct["pixarea"]
        top50 = cat[np.flipud(np.argsort(dp_dV))][:50]
    else:
        top50 = cat[np.flipud(np.argsort(map_struct["prob"][ipix]))][:50]

    catalogfile = os.path.join(params["outputDir"],'catalog.dat')
    top50['RAJ2000', 'DEJ2000', 'Ktmag'].write(catalogfile, format='ascii')

    return top50

