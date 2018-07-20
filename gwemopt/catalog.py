
import os, sys, copy

import numpy as np
import healpy as hp

from scipy.stats import norm
from scipy.special import gammaincinv

from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1

from astropy.table import Table
from astropy.io import ascii
from astropy.cosmology import WMAP9 as cosmo
from astropy.table import Column
import astropy.units as u
import astropy.constants as c

def get_catalog(params, map_struct):

    if not os.path.isdir(params["catalogDir"]): os.makedirs(params["catalogDir"])
    catalogFile = os.path.join(params["catalogDir"],"%s.csv"%params["galaxy_catalog"])

    """AB Magnitude zero point."""
    MAB0 = -2.5 * np.log10(3631.e-23)
    pc_cm = 3.08568025e18
    const = 4. * np.pi * (10. * pc_cm)**2.

    if params["galaxy_catalog"] == "2MRS":
        if not os.path.isfile(catalogFile):
            cat, = Vizier.get_catalogs('J/ApJS/199/26/table3')
            ascii.write(cat,catalogFile,format='ecsv')
        else:
            cat = ascii.read(catalogFile,format='ecsv')

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

        mag = cat['Ktmag']
        L_nu =  const * 10.**((mag + MAB0)/(-2.5))
 
    elif params["galaxy_catalog"] == "GLADE":
        if not os.path.isfile(catalogFile):
            cat, = Vizier.get_catalogs('VII/275/glade1')
            ascii.write(cat,catalogFile,format='ecsv')
        else:
            cat = ascii.read(catalogFile,format='ecsv')
        r = cat["Dist"]

        mag = cat['Bmag']
        L_nu =  const * 10.**((mag + MAB0)/(-2.5))

    L_nu =  const * 10.**((mag + MAB0)/(-2.5))
    L_nu = np.log10(L_nu)
    L_nu = L_nu**params["catalog_n"]
    Slum = L_nu / np.sum(L_nu)
    mlim, M_KNmin, M_KNmax = 22, -17, -12
    L_KNmin, L_KNmax = const * 10.**((M_KNmin + MAB0)/(-2.5)), const * 10.**((M_KNmax + MAB0)/(-2.5))

    Llim = 4. * np.pi * (r * 1e6 * pc_cm)**2. * 10.**((mlim + MAB0)/(-2.5))
    Sdet = (L_KNmax-Llim)/(L_KNmax-L_KNmin)
    Sdet[Sdet<0.01] = 0.01
    Sdet[Sdet>1.0] = 1.0

    theta = 0.5 * np.pi - cat['DEJ2000'].to('rad').value
    phi = cat['RAJ2000'].to('rad').value
    ipix = hp.ang2pix(map_struct["nside"], cat['RAJ2000'].to('deg').value, cat['DEJ2000'].to('deg').value, lonlat=True)

    if "distnorm" in map_struct:
        Sloc = map_struct["prob"][ipix] * (map_struct["distnorm"][ipix] * norm(map_struct["distmu"][ipix], map_struct["distsigma"][ipix]).pdf(r))**params["powerlaw_dist_exp"] / map_struct["pixarea"]
        Sloc = copy.copy(map_struct["prob"][ipix])
    else:
        Sloc = copy.copy(map_struct["prob"][ipix])

    S = Sloc*Slum*Sdet
    prob = np.zeros(map_struct["prob"].shape)
    prob[ipix] = prob[ipix] + S
    prob = prob / np.sum(prob)

    map_struct['prob_catalog'] = prob
    if params["doUseCatalog"]:
        map_struct['prob'] = prob 

    idx = np.argsort(S)[::-1]
    top50 = cat[idx][:50]
    catalogfile = os.path.join(params["outputDir"],'catalog.csv')
    top50['RAJ2000', 'DEJ2000'].write(catalogfile, format='csv')

    return map_struct

