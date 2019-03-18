#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import copy
import h5py
import numpy as np
import healpy as hp

from scipy.stats import norm
from scipy.special import gammaincinv

from astroquery.vizier import Vizier

from astropy.table import Table
from astropy.io import ascii
from astropy.cosmology import WMAP9 as cosmo
from astropy.table import Column
import astropy.units as u

# Unset row limits when querying Vizier
Vizier.ROW_LIMIT = -1


def get_catalog(params, map_struct):
    if not os.path.isdir(params["catalogDir"]):
        os.makedirs(params["catalogDir"])

    catalogFile = os.path.join(params["catalogDir"],
                               "%s.hdf5" % params["galaxy_catalog"])

    """AB Magnitude zero point."""
    MAB0 = -2.5 * np.log10(3631.e-23)
    pc_cm = 3.08568025e18
    const = 4. * np.pi * (10. * pc_cm)**2.

    if params["galaxy_catalog"] == "2MRS":
        if not os.path.isfile(catalogFile):
            import astropy.constants as c

            cat, = Vizier.get_catalogs('J/ApJS/199/26/table3')

            ra, dec = cat["RAJ2000"], cat["DEJ2000"]
            cz = cat["cz"]
            magk = cat["Ktmag"]

            z = (u.Quantity(cat['cz']) / c.c).to(u.dimensionless_unscaled)

            completeness = 0.5
            alpha = -1.0
            MK_star = -23.55
            MK_max = MK_star + 2.5 * np.log10(gammaincinv(alpha + 2,
                                                          completeness))
            MK = magk - cosmo.distmod(z)
            idx = (z > 0) & (MK < MK_max)

            ra, dec = ra[idx], dec[idx]
            z = z[idx]
            magk = magk[idx]

            distmpc = cosmo.luminosity_distance(z).to('Mpc').value

            with h5py.File(catalogFile, 'w') as f:
                f.create_dataset('ra', data=ra)
                f.create_dataset('dec', data=dec)
                f.create_dataset('z', data=z)
                f.create_dataset('magk', data=magk)
                f.create_dataset('distmpc', data=distmpc)

        else:
            with h5py.File(catalogFile, 'r') as f:
                ra, dec = f['ra'][:], f['dec'][:]
                z = f['z'][:]
                magk = f['magk'][:]
                distmpc = f['distmpc'][:]
        r = distmpc * 1.0
        mag = magk * 1.0

    elif params["galaxy_catalog"] == "GLADE":
        if not os.path.isfile(catalogFile):
            cat, = Vizier.get_catalogs('VII/281/glade2')

            ra, dec = cat["RAJ2000"], cat["DEJ2000"]
            distmpc, z = cat["Dist"], cat["z"]
            magb, magk = cat["Bmag"], cat["Kmag"]
            # Keep track of galaxy identifier
            GWGC, PGC, HyperLEDA = cat["GWGC"], cat["PGC"], cat["HyperLEDA"]
            _2MASS, SDSS = cat["_2MASS"], cat["SDSS-DR12"]

            idx = np.where(distmpc >= 0)[0]
            ra, dec = ra[idx], dec[idx]
            distmpc, z = distmpc[idx], z[idx]
            magb, magk = magb[idx], magk[idx]
            GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
            _2MASS, SDSS = _2MASS[idx], SDSS[idx]

            with h5py.File(catalogFile, 'w') as f:
                f.create_dataset('ra', data=ra)
                f.create_dataset('dec', data=dec)
                f.create_dataset('distmpc', data=distmpc)
                f.create_dataset('magb', data=magb)
                f.create_dataset('magk', data=magk)
                f.create_dataset('z', data=z)
                # Add galaxy identifier
                f.create_dataset('GWGC', data=GWGC)
                f.create_dataset('PGC', data=PGC)
                f.create_dataset('HyperLEDA', data=HyperLEDA)
                f.create_dataset('2MASS', data=_2MASS)
                f.create_dataset('SDSS', data=SDSS)

        else:
            with h5py.File(catalogFile, 'r') as f:
                ra, dec = f['ra'][:], f['dec'][:]
                distmpc, z = f['distmpc'][:], f['z'][:]
                magb, magk = f['magb'][:], f['magk'][:]
                GWGC, PGC, _2MASS = f['GWGC'][:], f['PGC'][:], f['2MASS'][:]
                HyperLEDA, SDSS = f['HyperLEDA'][:], f['SDSS'][:]
                # Convert bytestring to unicode
                GWGC = GWGC.astype('U')
                PGC = PGC.astype('U')
                HyperLEDA = HyperLEDA.astype('U')
                _2MASS = _2MASS.astype('U')
                SDSS = SDSS.astype('U')

        # Keep only galaxies with finite B mag when using it in the grade
        if params["galaxy_grade"] == "S":
            idx = np.where(~np.isnan(magb))[0]
            ra, dec, distmpc = ra[idx], dec[idx], distmpc[idx]
            magb, magk = magb[idx], magk[idx]
            GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
            _2MASS, SDSS = _2MASS[idx], SDSS[idx]

        r = distmpc * 1.0
        mag = magb * 1.0

    elif params["galaxy_catalog"] == "CLU":
        if not os.path.isfile(catalogFile):
            print("Please add %s." % catalogFile)
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

    L_nu = const * 10.**((mag + MAB0)/(-2.5))
    L_nu = np.log10(L_nu)
    L_nu = L_nu**params["catalog_n"]
    Slum = L_nu / np.sum(L_nu)

    mlim, M_KNmin, M_KNmax = 22, -17, -12
    L_KNmin = const * 10.**((M_KNmin + MAB0)/(-2.5))
    L_KNmax = const * 10.**((M_KNmax + MAB0)/(-2.5))

    Llim = 4. * np.pi * (r * 1e6 * pc_cm)**2. * 10.**((mlim + MAB0)/(-2.5))
    Sdet = (L_KNmax-Llim)/(L_KNmax-L_KNmin)
    Sdet[Sdet < 0.01] = 0.01
    Sdet[Sdet > 1.0] = 1.0

    n, cl = params["powerlaw_n"], params["powerlaw_cl"]
    dist_exp = params["powerlaw_dist_exp"]

    prob_scaled = copy.deepcopy(map_struct["prob"])
    prob_sorted = np.sort(prob_scaled)[::-1]
    prob_indexes = np.argsort(prob_scaled)[::-1]
    prob_cumsum = np.cumsum(prob_sorted)
    index = np.argmin(np.abs(prob_cumsum - cl)) + 1
    prob_scaled[prob_indexes[index:]] = 0.0
    prob_scaled = prob_scaled**n

    theta = 0.5 * np.pi - dec * 2 * np.pi / 360.0
    phi = ra * 2 * np.pi / 360.0
    ipix = hp.ang2pix(map_struct["nside"], ra, dec, lonlat=True)

    if "distnorm" in map_struct:
        if map_struct["distnorm"] is not None:
            Sloc = prob_scaled[ipix] * (map_struct["distnorm"][ipix] *
                                        norm(map_struct["distmu"][ipix],
                                        map_struct["distsigma"][ipix]).pdf(r))**params["powerlaw_dist_exp"] / map_struct["pixarea"]
        else:
            Sloc = copy.copy(prob_scaled[ipix])
    else:
        Sloc = copy.copy(prob_scaled[ipix])

    # Set nan values to zero
    Sloc[np.isnan(Sloc)] = 0

    S = Sloc*Slum*Sdet
    prob = np.zeros(map_struct["prob"].shape)
    if params["galaxy_grade"] == "Sloc":
        prob[ipix] = prob[ipix] + Sloc
        grade = Sloc
    elif params["galaxy_grade"] == "S":
        prob[ipix] = prob[ipix] + S
        grade = S

    prob = prob / np.sum(prob)

    map_struct['prob_catalog'] = prob
    if params["doUseCatalog"]:
        map_struct['prob'] = prob

    idx = np.where(~np.isnan(grade))[0]
    grade = grade[idx]
    ra, dec, Sloc, S = ra[idx], dec[idx], Sloc[idx], S[idx]
    distmpc, z = distmpc[idx], z[idx]
    if params["galaxy_catalog"] == "GLADE":
        GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
        _2MASS, SDSS = _2MASS[idx], SDSS[idx]

    Sthresh = np.max(grade)*0.01
    idx = np.where(grade >= Sthresh)[0]
    grade = grade[idx]
    ra, dec, Sloc, S = ra[idx], dec[idx], Sloc[idx], S[idx]
    distmpc, z = distmpc[idx], z[idx]
    if params["galaxy_catalog"] == "GLADE":
        GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
        _2MASS, SDSS = _2MASS[idx], SDSS[idx]

    idx = np.argsort(grade)[::-1]
    grade = grade[idx]
    ra, dec, Sloc, S = ra[idx], dec[idx], Sloc[idx], S[idx]
    distmpc, z = distmpc[idx], z[idx]
    if params["galaxy_catalog"] == "GLADE":
        GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
        _2MASS, SDSS = _2MASS[idx], SDSS[idx]

    if len(ra) > 1000:
        print('Cutting catalog to top 1000 galaxies...')
        idx = np.arange(1000).astype(int)
        ra, dec, Sloc, S = ra[idx], dec[idx], Sloc[idx], S[idx]
        distmpc, z = distmpc[idx], z[idx]
        if params["galaxy_catalog"] == "GLADE":
            GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
            _2MASS, SDSS = _2MASS[idx], SDSS[idx]

    catalog_struct = {}
    catalog_struct["ra"] = ra
    catalog_struct["dec"] = dec
    catalog_struct["Sloc"] = Sloc
    catalog_struct["S"] = S

    if params["writeCatalog"]:
        catalogfile = os.path.join(params["outputDir"], 'catalog.csv')
        fid = open(catalogfile, 'w')
        cnt = 1
        if params["galaxy_catalog"] == "GLADE":
            fid.write("id, RAJ2000, DEJ2000, Sloc, S, Dist, z, GWGC, PGC, HyperLEDA, 2MASS, SDSS\n")
            for a, b, c, d, e, f, g, h, i, j, k in zip(ra, dec, Sloc, S, distmpc, z, GWGC, PGC, HyperLEDA, _2MASS, SDSS):
                fid.write("%d, %.5f, %.5f, %.5e, %.5e, %.4f, %.4f, %s, %s, %s, %s, %s\n" % (cnt, a, b, c, d, e, f, g, h, i, j, k))
                cnt = cnt + 1
        else:
            fid.write("id, RAJ2000, DEJ2000, Sloc, S, Dist, z\n")
            for a, b, c, d in zip(ra, dec, Sloc, S, distmpc, z):
                fid.write("%d, %.5f, %.5f, %.5e, %.5e, %.4f, %.4f\n" % (cnt, a, b, c, d, e, f))
                cnt = cnt + 1

        fid.close()

    return map_struct, catalog_struct
