
import os, sys, copy
import h5py
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
    catalogFile = os.path.join(params["catalogDir"],"%s.hdf5"%params["galaxy_catalog"])

    """AB Magnitude zero point."""
    MAB0 = -2.5 * np.log10(3631.e-23)
    pc_cm = 3.08568025e18
    const = 4. * np.pi * (10. * pc_cm)**2.

    if params["galaxy_catalog"] == "2MRS":
        if not os.path.isfile(catalogFile):
            cat, = Vizier.get_catalogs('J/ApJS/199/26/table3')

            ras, decs = cat["RAJ2000"], cat["DEJ2000"]
            cz = cat["cz"]
            magk = cat["Ktmag"]
            z = (u.Quantity(cat['cz']) / c.c).to(u.dimensionless_unscaled)

            completeness = 0.5
            alpha = -1.0
            MK_star = -23.55
            MK_max = MK_star + 2.5 * np.log10(gammaincinv(alpha + 2, completeness))
            MK = magk - cosmo.distmod(z)
            idx = (z > 0) & (MK < MK_max)
        
            ra, dec = ra[idx], dec[idx]
            z = z[idx]
            magk = magk[idx]            

            with h5py.File(catalogFile, 'w') as f:
                f.create_dataset('ra', data=ras)
                f.create_dataset('dec', data=decs)
                f.create_dataset('z', data=z)
                f.create_dataset('magk', data=magk)

        else:
            with h5py.File(catalogFile, 'r') as f:
                ra, dec = f['ra'][:], f['dec'][:]
                z = f['z'][:]
                magk = f['magk'][:]

        r = cosmo.luminosity_distance(z).to('Mpc').value
        mag = magk * 1.0
 
    elif params["galaxy_catalog"] == "GLADE":
        if not os.path.isfile(catalogFile):
            cat, = Vizier.get_catalogs('VII/281/glade2')

            ra, dec = cat["RAJ2000"], cat["DEJ2000"]
            distmpc = cat["Dist"]
            magb, magk = cat["Bmag"], cat["Kmag"]

            idx = np.where(distmpc >= 0)[0]
            ra, dec = ra[idx], dec[idx]
            distmpc = distmpc[idx]
            magb, magk = magb[idx], magk[idx]

            with h5py.File(catalogFile, 'w') as f:
                f.create_dataset('ra', data=ra)
                f.create_dataset('dec', data=dec)
                f.create_dataset('distmpc', data=distmpc)
                f.create_dataset('magb', data=magb)
                f.create_dataset('magk', data=magk)
        else:
            with h5py.File(catalogFile, 'r') as f:
                ra, dec = f['ra'][:], f['dec'][:]
                distmpc = f['distmpc'][:]
                magb, magk = f['magb'][:], f['magk'][:]

        idx = np.where(~np.isnan(magb))[0]
        ra, dec, distmpc, magb, magk = ra[idx], dec[idx], distmpc[idx], magb[idx], magk[idx]

        r = distmpc * 1.0
        mag = magb * 1.0

    elif params["galaxy_catalog"] == "CLU":
        if not os.path.isfile(catalogFile):
            print("Please add %s."%catalogFile)
            exit(0)

        with h5py.File(catalogFile, 'r') as f:
            name = f['name'][:]
            ra, dec = f['ra'][:], f['dec'][:]
            sfr_fuv, mstar = f['sfr_fuv'][:], f['mstar'][:]
            distmpc, magb = f['distmpc'][:], f['magb'][:]
            a, b2a, pa = f['a'][:], f['b2a'][:], f['pa'][:]
            btc = f['btc'][:]

        idx = np.where(distmpc >= 0)[0]
        ra, dec = ra[idx], dec[idx]
        sfr_fuv, mstar = sfr_fuv[idx], mstar[idx]
        distmpc, magb = distmpc[idx], magb[idx]
        a, b2a, pa = a[idx], b2a[idx], pa[idx]
        btc = btc[idx]

        idx = np.where(~np.isnan(magb))[0]
        ra, dec = ra[idx], dec[idx]
        sfr_fuv, mstar = sfr_fuv[idx], mstar[idx]
        distmpc, magb = distmpc[idx], magb[idx]
        a, b2a, pa = a[idx], b2a[idx], pa[idx]
        btc = btc[idx]

        r = distmpc * 1.0
        mag = magb * 1.0

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

    n, cl, dist_exp = params["powerlaw_n"], params["powerlaw_cl"], params["powerlaw_dist_exp"]
    prob_scaled = copy.deepcopy(map_struct["prob"])
    prob_sorted = np.sort(prob_scaled)[::-1]
    prob_indexes = np.argsort(prob_scaled)[::-1]
    prob_cumsum = np.cumsum(prob_sorted)
    index = np.argmin(np.abs(prob_cumsum - cl)) + 1
    prob_scaled[prob_indexes[index:]] = 0.0
    prob_scaled = prob_scaled**n
#    prob_scaled = prob_scaled / np.nansum(prob_scaled)

    theta = 0.5 * np.pi - dec * 2 * np.pi /360.0
    phi = ra * 2 * np.pi /360.0
    ipix = hp.ang2pix(map_struct["nside"], ra, dec, lonlat=True).astype(int)
    if "distnorm" in map_struct:
        if map_struct["distnorm"] is not None:
            Sloc = prob_scaled[ipix] * (map_struct["distnorm"][ipix] * norm(map_struct["distmu"][ipix], map_struct["distsigma"][ipix]).pdf(r))**params["powerlaw_dist_exp"] / map_struct["pixarea"]
        else:
            Sloc = copy.copy(prob_scaled[ipix])
    else:
        Sloc = copy.copy(prob_scaled[ipix])

    S = Sloc*Slum*Sdet

    prob = np.zeros(map_struct["prob"].shape)
    prob[ipix] = prob[ipix] + S
    prob = prob / np.sum(prob)

    map_struct['prob_catalog'] = prob
    if params["doUseCatalog"]:
        map_struct['prob'] = prob 

    idx = np.where(~np.isnan(S))[0]
    ra, dec, Sloc, S = ra[idx], dec[idx], Sloc[idx], S[idx]

    Sthresh = np.max(S)*0.01
    idx = np.where(S >= Sthresh)[0]
    ra, dec, Sloc, S = ra[idx], dec[idx], Sloc[idx], S[idx] 

    idx = np.argsort(S)[::-1]
    ra, dec, Sloc, S = ra[idx], dec[idx], Sloc[idx], S[idx] 

    if len(ra) > 1000:
        print('Cutting catalog to top 1000 galaxies...')
        idx = np.arange(1000).astype(int)
        ra, dec, Sloc, S = ra[idx], dec[idx], Sloc[idx], S[idx]

    catalog_struct = {}
    catalog_struct["ra"] = ra
    catalog_struct["dec"] = dec
    catalog_struct["Sloc"] = Sloc
    catalog_struct["S"] = S

    if params["doPlots"]:
        catalogfile = os.path.join(params["outputDir"],'catalog.csv')
        fid = open(catalogfile,'w')
        cnt = 0
        for a, b, c, d in zip(ra, dec, Sloc, S):
            fid.write("%d %.5f %.5f %.5e %.5e\n"%(cnt,a,b,c,d))
            cnt = cnt + 1
        fid.close()

    return map_struct, catalog_struct

