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
from ligo.skymap import distance
from astropy.table import Table
from astropy.io import ascii
from astropy.cosmology import WMAP9 as cosmo
from astropy.table import Column
import astropy.units as u
import scipy.stats
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
            raise ValueError("Please add %s." % catalogFile)

        cat = Table.read(catalogFile)
        name = cat['name']
        ra, dec = cat['ra'], cat['dec']
        sfr_fuv, mstar = cat['sfr_fuv'], cat['mstar']
        distmpc, magb = cat['distmpc'], cat['magb']
        a, b2a, pa = cat['a'], cat['b2a'], cat['pa']
        btc = cat['btc']

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

        z = -1*np.ones(distmpc.shape)
        r = distmpc * 1.0
        mag = magb * 1.0


    elif params["galaxy_catalog"] == "mangrove":
        catalogFile = os.path.join(params["catalogDir"],
                               "%s.hdf5" % params["galaxy_catalog"])
        
        if not os.path.isfile(catalogFile):
            print("mangrove catalog not found localy, start the automatic download")
            url = 'https://mangrove.lal.in2p3.fr/data/mangrove.hdf5'
            os.system("wget -O {}/mangrove.hdf5 {}".format(params["catalogDir"], url))
            
        cat = Table.read(catalogFile)
        
        ra, dec = cat["RA"], cat["dec"]
        distmpc, z = cat["dist"], cat["z"]
        magb, magk = cat["B_mag"], cat["K_mag"]
        magW1, magW2, magW3, magW4 = cat["w1mpro"], cat["w2mpro"], cat["w3mpro"], cat["w4mpro"]
        stellarmass = cat['stellarmass']
        # Keep track of galaxy identifier
        GWGC, PGC, HyperLEDA = cat["GWGC_name"], cat["PGC"], cat["HyperLEDA_name"]
        _2MASS, SDSS = cat["2MASS_name"], cat["SDSS-DR12_name"]
        #keep track of the AGN flag
        AGN_flag = cat["AGN_flag"]

        idx = np.where(distmpc >= 0)[0]
        ra, dec = ra[idx], dec[idx]
        distmpc, z = distmpc[idx], z[idx]
        magb, magk = magb[idx], magk[idx]
        magW1, magW2, magW3, magW4 = magW1[idx], magW2[idx], magW3[idx], magW4[idx]
        stellarmass = stellarmass[idx]
        GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
        _2MASS, SDSS = _2MASS[idx], SDSS[idx]
        AGN_flag = AGN_flag[idx]

        # Convert bytestring to unicode
        GWGC = GWGC.astype('U')
        PGC = PGC.astype('U')
        HyperLEDA = HyperLEDA.astype('U')
        _2MASS = _2MASS.astype('U')
        SDSS = SDSS.astype('U')

        r = distmpc * 1.0
        mag = magb * 1.0

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
            
            #creat an mask to cut at 3 sigma in distance
            mask = np.zeros(len(r))
            
            #calculate the moments from distmu, distsigma and distnorm
            mom_mean, mom_std, mom_norm = distance.parameters_to_moments(map_struct["distmu"],map_struct["distsigma"])

            condition_indexer = np.where( (r < (mom_mean[ipix] + (3*mom_std[ipix]))) & (r > (mom_mean[ipix] - (3*mom_std[ipix])) )) 
            mask[condition_indexer] = 1

            Sloc = prob_scaled[ipix] * (map_struct["distnorm"][ipix] *
                                        norm(map_struct["distmu"][ipix],
                                        map_struct["distsigma"][ipix]).pdf(r))**params["powerlaw_dist_exp"] / map_struct["pixarea"]
    
            #multiplie the Sloc by 1 or 0 according to the 3 sigma condistion
            Sloc = Sloc*mask
            idx = np.where(condition_indexer)[0]
        else:
            Sloc = copy.copy(prob_scaled[ipix])
            idx = np.arange(len(r)).astype(int)
    else:
        Sloc = copy.copy(prob_scaled[ipix])
        idx = np.arange(len(r)).astype(int)

    # this happens when we are using a tiny catalog...
    CompatibleGal = True
    if np.all(Sloc == 0.0):
        Sloc[:] = 1.0
        CompatibleGal = False

    #new version of the Slum calcul (from HOGWARTs)
    Lsun = 3.828e26
    Msun = 4.83
    Mknmin = -19
    Mknmax = -12
    Lblist = []
    
    for i in range(0, len(r)):
        Mb = mag[i] - 5 * np.log10((r[i] * 10 ** 6)) + 5
        #L = Lsun * 2.512 ** (Msun - Mb)
        Lb = Lsun * 2.512 ** (Msun - Mb)
        Lblist.append(Lb)
        
                           
    #set 0 when Sloc is 0 (keep compatible galaxies for normalization)
    Lblist = np.array(Lblist)
    
    Lblist[Sloc == 0] = 0

    Slum = Lblist / np.nansum(np.array(Lblist))
   
  
    """
    L_nu = const * 10.**((mag + MAB0)/(-2.5))
    L_nu = L_nu / np.nanmax(L_nu[idx])
    L_nu = L_nu**params["catalog_n"]
    L_nu[L_nu < 0.001] = 0.001
    L_nu[L_nu > 1.0] = 1.0
    Slum = L_nu / np.sum(L_nu)
    """


    mlim, M_KNmin, M_KNmax = 22, -17, -12
    L_KNmin = const * 10.**((M_KNmin + MAB0)/(-2.5))
    L_KNmax = const * 10.**((M_KNmax + MAB0)/(-2.5))

    Llim = 4. * np.pi * (r * 1e6 * pc_cm)**2. * 10.**((mlim + MAB0)/(-2.5))
    Sdet = (L_KNmax-Llim)/(L_KNmax-L_KNmin)
    Sdet[Sdet < 0.01] = 0.01
    Sdet[Sdet > 1.0] = 1.0

    # Set nan values to zero
    Sloc[np.isnan(Sloc)] = 0
    Slum[np.isnan(Slum)] = 0

    if params["galaxy_grade"] == "Smass":
        
        if params["galaxy_catalog"] != "mangrove":
            raise ValueError("You are trying to use the stellar mass information (Smass), please select the mangrove catalog for such use.")

        #set Smass
        Smass = np.array(stellarmass)
        
        #put null values to nan
        Smass[np.where(Smass == 0)] = np.nan

        #go back to linear scaling and not log
        Smass = 10**Smass

        # Keep only galaxies with finite stellarmass when using it in the grade
        #set nan values to 0
        Smass[~np.isfinite(Smass)] = 0  
 
        #set Smass
        Smass = Smass / np.sum(Smass)
        
	#alpha is defined only with non null mass galaxies, we set a mask for that
        ind_without_mass = np.where(Smass == 0)
        Sloc_temp = copy.deepcopy(Sloc)
        Sloc_temp[ind_without_mass] = 0

        #alpha_mass parameter is defined in such way that in mean Sloc count in as much as Sloc*alpha*Smass
        alpha_mass = (np.sum(Sloc_temp) / np.sum( Sloc_temp*Smass ) )
        print("You chose to use the grade using stellar mass, the parameters values are:")
        print("alpha_mass =", alpha_mass)

	#beta_mass is a parameter allowing to change the importance of Sloc according to Sloc*alpha*Smass
        #beta_mass should be fitted in the futur on real GW event for which we have the host galaxy
        #fixed to one at the moment
        beta_mass = 1
        print("beta_mass =", beta_mass)

        Smass = Sloc*(1+ (alpha_mass*beta_mass*Smass) )
        #Smass = Sloc*Smass
        
    S = Sloc*Slum*Sdet

    prob = np.zeros(map_struct["prob"].shape)
    if params["galaxy_grade"] == "Sloc":     
        for j in range(len(ipix)):
            prob[ipix[j]] += Sloc[j]
        grade = Sloc
    elif params["galaxy_grade"] == "S":
        for j in range(len(ipix)):
            prob[ipix[j]] += S[j]
        grade = S
    elif params["galaxy_grade"] == "Smass":
        for j in range(len(ipix)):
            prob[ipix[j]] += Smass[j]
        grade = Smass
 
    prob = prob / np.sum(prob)

    map_struct['prob_catalog'] = prob
    if params["doUseCatalog"]:
        map_struct['prob'] = prob

    Lblist = np.array(Lblist)

    idx = np.where(~np.isnan(grade))[0]
    grade = grade[idx]
    ra, dec, Sloc, S = ra[idx], dec[idx], Sloc[idx], S[idx]
    distmpc, z = distmpc[idx], z[idx]

    if params["galaxy_catalog"] == "GLADE":
        GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
        _2MASS, SDSS = _2MASS[idx], SDSS[idx]
        Lblist = Lblist[idx]
        magb = magb[idx]
 
    if params["galaxy_catalog"] == "mangrove":
        GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
        _2MASS, SDSS = _2MASS[idx], SDSS[idx]   
        stellarmass, magb = stellarmass[idx], magb[idx]
        AGN_flag = AGN_flag[idx]
        if params["galaxy_grade"] == "Smass":
            Smass = Smass[idx]

    """
    Sthresh = np.max(grade)*0.01
    idx = np.where(grade >= Sthresh)[0]
    grade = grade[idx]
    ra, dec, Sloc, S = ra[idx], dec[idx], Sloc[idx], S[idx]
    distmpc, z = distmpc[idx], z[idx]
    if params["galaxy_catalog"] == "GLADE":
        GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
        _2MASS, SDSS = _2MASS[idx], SDSS[idx]
    """
    
    idx = np.argsort(grade)[::-1]
    grade = grade[idx]

    ra, dec, Sloc, S = ra[idx], dec[idx], Sloc[idx], S[idx]
    distmpc, z = distmpc[idx], z[idx]
    if params["galaxy_catalog"] == "GLADE":
        GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
        _2MASS, SDSS = _2MASS[idx], SDSS[idx]
        Lblist = Lblist[idx]
        magb = magb[idx]

    if params["galaxy_catalog"] == "mangrove":
        GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
        _2MASS, SDSS = _2MASS[idx], SDSS[idx]
        stellarmass, magb = stellarmass[idx], magb[idx]
        AGN_flag = AGN_flag[idx]
        if params["galaxy_grade"] == "Smass":
            Smass = Smass[idx]
       
    # Keep only galaxies within 3sigma in distance
    if params["galaxy_catalog"] != "mangrove":
        mask = Sloc > 0
        ra, dec, Sloc, S = ra[mask], dec[mask], Sloc[mask], S[mask],
        distmpc, z = distmpc[mask], z[mask]
        if params["galaxy_catalog"] == "GLADE":
            GWGC, PGC, HyperLEDA = GWGC[mask], PGC[mask], HyperLEDA[mask]
            _2MASS, SDSS = _2MASS[mask], SDSS[mask]
            Lblist = Lblist[mask]

    if params["galaxy_catalog"] == "mangrove":
        # Keep only galaxies within 3sigma in distance
        mask = Sloc > 0
        ra, dec, Sloc, S = ra[mask], dec[mask], Sloc[mask], S[mask]
        distmpc, z = distmpc[mask], z[mask]
        GWGC, PGC, HyperLEDA = GWGC[mask], PGC[mask], HyperLEDA[mask]
        _2MASS, SDSS = _2MASS[mask], SDSS[mask]
        stellarmass, magb = stellarmass[mask], magb[mask]
        AGN_flag = AGN_flag[mask]

        if params["galaxy_grade"] == "Smass":
            Smass = Smass[mask]

        
    if len(ra) > 2000:
        print('Cutting catalog to top 2000 galaxies...')
        idx = np.arange(2000).astype(int)
        ra, dec, Sloc, S = ra[idx], dec[idx], Sloc[idx], S[idx]
        distmpc, z = distmpc[idx], z[idx]
        if params["galaxy_catalog"] == "GLADE":
            GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
            _2MASS, SDSS = _2MASS[idx], SDSS[idx]
            Lblist = Lblist[idx]
            magb = magb[idx]

        elif params["galaxy_catalog"] == "mangrove":
            GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
            _2MASS, SDSS = _2MASS[idx], SDSS[idx]
            stellarmass, magb = stellarmass[idx], magb[idx]
            AGN_flag = AGN_flag[idx]

            if params["galaxy_grade"] == "Smass":
                Smass = Smass[idx]

    # now normalize the distributions
    S = S / np.sum(S)
    Sloc = Sloc / np.sum(Sloc)

    if params["galaxy_grade"] == "Smass":
        Smass = Smass/np.sum(Smass)
    else:
        Smass = np.ones(Sloc.shape)
        Smass = Smass/np.sum(Smass)

    catalog_struct = {}
    catalog_struct["ra"] = ra
    catalog_struct["dec"] = dec
    catalog_struct["Sloc"] = Sloc
    catalog_struct["S"] = S
    catalog_struct["Smass"] = Smass

    catalog_struct["CompatibleGal"] = CompatibleGal


    if params["writeCatalog"]:
        catalogfile = os.path.join(params["outputDir"], 'catalog.csv')
        fid = open(catalogfile, 'w')
        cnt = 1
        if params["galaxy_catalog"] == "GLADE":

            if params["galaxy_grade"] == "S":
 
                fid.write("id, RAJ2000, DEJ2000, Sloc, S, Dist, z,GWGC, PGC, HyperLEDA, 2MASS, SDSS, BLum\n")
                for a, b, c, d, e, f, g, h, i, j, k, l in zip(ra, dec, Sloc, S, distmpc, z, GWGC, PGC, HyperLEDA, _2MASS, SDSS, Lblist):
                    fid.write("%d, %.5f, %.5f, %.5e, %.5e, %.4f, %.4f, %s, %s, %s, %s, %s, %.2E\n" % (cnt, a, b, c, d, e, f, g, h, i, j, k, l))
                    cnt = cnt + 1
                
            else:

                fid.write("id, RAJ2000, DEJ2000, Sloc, S, Dist, z, GWGC, PGC, HyperLEDA, 2MASS, SDSS\n")
                for a, b, c, d, e, f, g, h, i, j, k in zip(ra, dec, Sloc, S, distmpc, z, GWGC, PGC, HyperLEDA, _2MASS, SDSS):
                    fid.write("%d, %.5f, %.5f, %.5e, %.5e, %.4f, %.4f, %s, %s, %s, %s, %s\n" % (cnt, a, b, c, d, e, f, g, h, i, j, k))
                    cnt = cnt + 1

        elif params["galaxy_catalog"] == "mangrove":

            if params["galaxy_grade"] == "Smass":

                if params["AGN_flag"]:

                    fid.write("id, RAJ2000, DEJ2000, Smass, S, Sloc, Dist, z, GWGC, PGC, HyperLEDA, 2MASS, SDSS, B_mag, AGN_flag, stellarmass\n")
                    for a, b, c, d, e, f, g, h, i, j, k, l, m, n, o in zip(ra, dec, Smass, S, Sloc, distmpc, z, GWGC, PGC, HyperLEDA, _2MASS, SDSS, magb, AGN_flag,stellarmass):
                        fid.write("%d, %.5f, %.5f, %.5e, %.5e, %.5e, %.4f, %.4f, %s, %s, %s, %s, %s, %s, %s, %s\n" % (cnt, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o))
                        cnt = cnt + 1
                    
                else:

                    fid.write("id, RAJ2000, DEJ2000, Smass, S, Sloc, Dist, z, GWGC, PGC, HyperLEDA, 2MASS, SDSS, B_mag, stellarmass\n")
                    for a, b, c, d, e, f, g, h, i, j, k, l, m, n in zip(ra, dec, Smass, S, Sloc, distmpc, z, GWGC, PGC, HyperLEDA, _2MASS, SDSS, magb, stellarmass):
                        fid.write("%d, %.5f, %.5f, %.5e, %.5e, %.5e, %.4f, %.4f, %s, %s, %s, %s, %s, %s, %s\n" % (cnt, a, b, c, d, e, f, g, h, i, j, k, l, m, n))
                        cnt = cnt + 1


            else:

                if params["AGN_flag"]:

                    fid.write("id, RAJ2000, DEJ2000, Sloc, S, Dist, z, GWGC, PGC, HyperLEDA, 2MASS, SDSS, AGN_flag\n")
                    for a, b, c, d, e, f, g, h, i, j, k, l in zip(ra, dec, Sloc, S, distmpc, z, GWGC, PGC, HyperLEDA, _2MASS, SDSS, AGN_flag):
                        fid.write("%d, %.5f, %.5f, %.5e, %.5e, %.4f, %.4f, %s, %s, %s, %s, %s, %s\n" % (cnt, a, b, c, d, e, f, g, h, i, j, k, l))
                        cnt = cnt + 1

                else:

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

