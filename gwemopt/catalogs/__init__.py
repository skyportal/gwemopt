#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import os

import astropy.constants as c
import astropy.units as u
import h5py
import healpy as hp
import numpy as np
from astropy.cosmology import WMAP9 as cosmo
from astropy.table import Table
from astroquery.vizier import Vizier
from ligo.skymap import distance
from scipy.special import gammaincinv
from scipy.stats import norm

from gwemopt.catalogs.clu import CluCatalog
from gwemopt.catalogs.glade import GladeCatalog
from gwemopt.catalogs.mangrove import MangroveCatalog
from gwemopt.catalogs.twomrs import TwoMRSCatalog

# Unset row limits when querying Vizier
Vizier.ROW_LIMIT = -1


def get_catalog(params, map_struct, export_catalog: bool = True):
    params["catalogDir"].mkdir(parents=True, exist_ok=True)

    """AB Magnitude zero point."""
    MAB0 = -2.5 * np.log10(3631.0e-23)
    pc_cm = 3.08568025e18
    const = 4.0 * np.pi * (10.0 * pc_cm) ** 2.0

    if params["galaxy_catalog"] == "2MRS":
        cat = TwoMRSCatalog(catalog_dir=params["catalogDir"])

    elif params["galaxy_catalog"] == "GLADE":
        cat = GladeCatalog(catalog_dir=params["catalogDir"])

    elif params["galaxy_catalog"] == "CLU":
        cat = CluCatalog(catalog_dir=params["catalogDir"])

    elif params["galaxy_catalog"] == "mangrove":
        cat = MangroveCatalog(catalog_dir=params["catalogDir"])
    else:
        raise KeyError(f"Unknown galaxy catalog: {params['galaxy_catalog']}")

    cat_df = cat.get_catalog()

    if params["galaxy_catalog"] == "glade":
        # Keep only galaxies with finite B mag when using it in the grade
        if params["galaxy_grade"] == "S":
            mask = np.where(~np.isnan(cat_df["magb"]))[0]
            cat_df = cat_df.iloc[mask]

    n, cl = params["powerlaw_n"], params["powerlaw_cl"]

    prob_scaled = copy.deepcopy(map_struct["prob"])
    prob_sorted = np.sort(prob_scaled)[::-1]
    prob_indexes = np.argsort(prob_scaled)[::-1]
    prob_cumsum = np.cumsum(prob_sorted)
    index = np.argmin(np.abs(prob_cumsum - cl)) + 1
    prob_scaled[prob_indexes[index:]] = 0.0
    prob_scaled = prob_scaled**n

    ipix = hp.ang2pix(map_struct["nside"], cat_df["ra"], cat_df["dec"], lonlat=True)

    if "distnorm" in map_struct:
        if map_struct["distnorm"] is not None:
            # creat an mask to cut at 3 sigma in distance
            mask = np.zeros(len(cat_df["r"]))

            # calculate the moments from distmu, distsigma and distnorm
            mom_mean, mom_std, mom_norm = distance.parameters_to_moments(
                map_struct["distmu"], map_struct["distsigma"]
            )

            condition_indexer = np.where(
                (cat_df["r"] < (mom_mean[ipix] + (3 * mom_std[ipix])))
                & (cat_df["r"] > (mom_mean[ipix] - (3 * mom_std[ipix])))
            )
            mask[condition_indexer] = 1

            Sloc = (
                prob_scaled[ipix]
                * (
                    map_struct["distnorm"][ipix]
                    * norm(
                        map_struct["distmu"][ipix], map_struct["distsigma"][ipix]
                    ).pdf(cat_df["r"])
                )
                ** params["powerlaw_dist_exp"]
                / map_struct["pixarea"]
            )

            # multiplie the Sloc by 1 or 0 according to the 3 sigma condistion
            Sloc = Sloc * mask
        else:
            Sloc = copy.copy(prob_scaled[ipix])
    else:
        Sloc = copy.copy(prob_scaled[ipix])

    # this happens when we are using a tiny catalog...
    compatible_gal = True
    if np.all(Sloc == 0.0):
        Sloc[:] = 1.0
        compatible_gal = False

    # new version of the Slum calcul (from HOGWARTs)
    Lsun = 3.828e26
    Msun = 4.83
    Lblist = []

    for i, r_i in enumerate(cat_df["r"]):
        Mb = cat_df["mag"][i] - 5 * np.log10((r_i * 10**6)) + 5
        Lb = Lsun * 2.512 ** (Msun - Mb)
        Lblist.append(Lb)

    # set 0 when Sloc is 0 (keep compatible galaxies for normalization)
    Lblist = np.array(Lblist)

    Lblist[Sloc == 0] = 0

    Slum = Lblist / np.nansum(np.array(Lblist))

    mlim, M_KNmin, M_KNmax = 22, -17, -12
    L_KNmin = const * 10.0 ** ((M_KNmin + MAB0) / (-2.5))
    L_KNmax = const * 10.0 ** ((M_KNmax + MAB0) / (-2.5))

    Llim = (
        4.0
        * np.pi
        * (cat_df["r"] * 1e6 * pc_cm) ** 2.0
        * 10.0 ** ((mlim + MAB0) / (-2.5))
    )
    Sdet = (L_KNmax - Llim) / (L_KNmax - L_KNmin)
    Sdet[Sdet < 0.01] = 0.01
    Sdet[Sdet > 1.0] = 1.0

    # Set nan values to zero
    Sloc[np.isnan(Sloc)] = 0
    Slum[np.isnan(Slum)] = 0

    if params["galaxy_grade"] == "Smass":
        if params["galaxy_catalog"] != "mangrove":
            raise ValueError(
                "You are trying to use the stellar mass information (Smass), please select the mangrove catalog for such use."
            )

        # set Smass
        Smass = cat_df["stellarmass"].to_numpy()

        # put null values to nan
        Smass[np.where(Smass == 0)] = np.nan

        # go back to linear scaling and not log
        Smass = 10.0**Smass

        # Keep only galaxies with finite stellarmass when using it in the grade
        # set nan values to 0
        Smass[~np.isfinite(Smass)] = 0

        # set Smass
        Smass = Smass / np.sum(Smass)

        # alpha is defined only with non null mass galaxies, we set a mask for that
        ind_without_mass = np.where(Smass == 0)
        Sloc_temp = copy.deepcopy(Sloc)
        Sloc_temp[ind_without_mass] = 0

        # alpha_mass parameter is defined in such way that in mean Sloc count in as much as Sloc*alpha*Smass
        alpha_mass = np.sum(Sloc_temp) / np.sum(Sloc_temp * Smass)
        print(
            "You chose to use the grade using stellar mass, the parameters values are:"
        )
        print("alpha_mass =", alpha_mass)

        # beta_mass is a parameter allowing to change the importance of Sloc according to Sloc*alpha*Smass
        # beta_mass should be fitted in the futur on real GW event for which we have the host galaxy
        # fixed to one at the moment
        beta_mass = 1
        print("beta_mass =", beta_mass)

        Smass = Sloc * (1 + (alpha_mass * beta_mass * Smass))

    S = Sloc * Slum * Sdet

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
    else:
        raise ValueError(
            "You are trying to use a galaxy grade that is not implemented yet."
        )

    prob = prob / np.sum(prob)

    map_struct["prob_catalog"] = prob
    if params["doUseCatalog"]:
        map_struct["prob"] = prob

    Lblist = np.array(Lblist)

    cat_df["grade"] = grade
    cat_df["S"] = S
    cat_df["Sloc"] = Sloc
    if params["galaxy_grade"] != "Smass":
        cat_df["Smass"] = 1.0

    mask = np.where(~np.isnan(grade))[0]
    cat_df = cat_df.iloc[mask]

    cat_df.sort_values(by=["grade"], inplace=True, ascending=False, ignore_index=True)

    # idx = np.argsort(grade)[::-1]
    #
    # ra, dec, Sloc, S = ra[idx], dec[idx], Sloc[idx], S[idx]
    # distmpc, z = distmpc[idx], z[idx]
    # if params["galaxy_catalog"] == "GLADE":
    #     GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
    #     _2MASS, SDSS = _2MASS[idx], SDSS[idx]
    #     Lblist = Lblist[idx]
    #     magb = magb[idx]
    #
    # if params["galaxy_catalog"] == "mangrove":
    #     GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
    #     _2MASS, SDSS = _2MASS[idx], SDSS[idx]
    #     stellarmass, magb = stellarmass[idx], magb[idx]
    #     AGN_flag = AGN_flag[idx]
    #     if params["galaxy_grade"] == "Smass":
    #         Smass = Smass[idx]

    # Keep only galaxies within 3sigma in distance
    # if params["galaxy_catalog"] != "mangrove":
    #     mask = Sloc > 0
    #     cat_df = cat_df[mask]
    #     ra, dec, Sloc, S = (
    #         ra[mask],
    #         dec[mask],
    #         Sloc[mask],
    #         S[mask],
    #     )
    #     distmpc, z = distmpc[mask], z[mask]
    #     if params["galaxy_catalog"] == "GLADE":
    #         GWGC, PGC, HyperLEDA = GWGC[mask], PGC[mask], HyperLEDA[mask]
    #         _2MASS, SDSS = _2MASS[mask], SDSS[mask]
    #         Lblist = Lblist[mask]

    # if params["galaxy_catalog"] == "mangrove":
    #     # Keep only galaxies within 3sigma in distance
    #     mask = Sloc > 0
    #     ra, dec, Sloc, S = ra[mask], dec[mask], Sloc[mask], S[mask]
    #     distmpc, z = distmpc[mask], z[mask]
    #     GWGC, PGC, HyperLEDA = GWGC[mask], PGC[mask], HyperLEDA[mask]
    #     _2MASS, SDSS = _2MASS[mask], SDSS[mask]
    #     stellarmass, magb = stellarmass[mask], magb[mask]
    #     AGN_flag = AGN_flag[mask]
    #
    #     if params["galaxy_grade"] == "Smass":
    #         Smass = Smass[mask]

    if len(cat_df) > 2000:
        print("Cutting catalog to top 2000 galaxies...")
        cat_df = cat_df.iloc[:2000]
        # idx = np.arange(2000).astype(int)

        # ra, dec, Sloc, S = ra[idx], dec[idx], Sloc[idx], S[idx]
        # distmpc, z = distmpc[idx], z[idx]
        # if params["galaxy_catalog"] == "GLADE":
        #     GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
        #     _2MASS, SDSS = _2MASS[idx], SDSS[idx]
        #     Lblist = Lblist[idx]
        #     magb = magb[idx]
        #
        # elif params["galaxy_catalog"] == "mangrove":
        #     GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
        #     _2MASS, SDSS = _2MASS[idx], SDSS[idx]
        #     stellarmass, magb = stellarmass[idx], magb[idx]
        #     AGN_flag = AGN_flag[idx]
        #
        #     if params["galaxy_grade"] == "Smass":
        #         Smass = Smass[idx]

    # now normalize the distributions
    for key in ["S", "Sloc", "Smass"]:
        cat_df[key] = cat_df[key] / np.sum(cat_df[key])

    # catalog_struct = {}
    # catalog_struct["ra"] = ra
    # catalog_struct["dec"] = dec
    # catalog_struct["Sloc"] = Sloc
    # catalog_struct["S"] = S
    # catalog_struct["Smass"] = Smass
    #
    # catalog_struct["CompatibleGal"] = compatible_gal

    if export_catalog:
        output_path = params["outputDir"].joinpath(f"catalog_{cat.name}.csv")
        print(f"Saving catalog to {output_path}")
        cat_df.to_csv(output_path)

        # catalogfile = os.path.join()
        # fid = open(catalogfile, "w")
        # cnt = 1
        # if params["galaxy_catalog"] == "GLADE":
        #     if params["galaxy_grade"] == "S":
        #         fid.write(
        #             "id, RAJ2000, DEJ2000, Sloc, S, Dist, z,GWGC, PGC, HyperLEDA, 2MASS, SDSS, BLum\n"
        #         )
        #         for a, b, c, d, e, f, g, h, i, j, k, l in zip(
        #             ra,
        #             dec,
        #             Sloc,
        #             S,
        #             distmpc,
        #             z,
        #             GWGC,
        #             PGC,
        #             HyperLEDA,
        #             _2MASS,
        #             SDSS,
        #             Lblist,
        #         ):
        #             fid.write(
        #                 "%d, %.5f, %.5f, %.5e, %.5e, %.4f, %.4f, %s, %s, %s, %s, %s, %.2E\n"
        #                 % (cnt, a, b, c, d, e, f, g, h, i, j, k, l)
        #             )
        #             cnt = cnt + 1
        #
        #     else:
        #         fid.write(
        #             "id, RAJ2000, DEJ2000, Sloc, S, Dist, z, GWGC, PGC, HyperLEDA, 2MASS, SDSS\n"
        #         )
        #         for a, b, c, d, e, f, g, h, i, j, k in zip(
        #             ra, dec, Sloc, S, distmpc, z, GWGC, PGC, HyperLEDA, _2MASS, SDSS
        #         ):
        #             fid.write(
        #                 "%d, %.5f, %.5f, %.5e, %.5e, %.4f, %.4f, %s, %s, %s, %s, %s\n"
        #                 % (cnt, a, b, c, d, e, f, g, h, i, j, k)
        #             )
        #             cnt = cnt + 1
        #
        # elif params["galaxy_catalog"] == "mangrove":
        #     if params["galaxy_grade"] == "Smass":
        #         if params["AGN_flag"]:
        #             fid.write(
        #                 "id, RAJ2000, DEJ2000, Smass, S, Sloc, Dist, z, GWGC, PGC, HyperLEDA, 2MASS, SDSS, B_mag, AGN_flag, stellarmass\n"
        #             )
        #             for a, b, c, d, e, f, g, h, i, j, k, l, m, n, o in zip(
        #                 ra,
        #                 dec,
        #                 Smass,
        #                 S,
        #                 Sloc,
        #                 distmpc,
        #                 z,
        #                 GWGC,
        #                 PGC,
        #                 HyperLEDA,
        #                 _2MASS,
        #                 SDSS,
        #                 magb,
        #                 AGN_flag,
        #                 stellarmass,
        #             ):
        #                 fid.write(
        #                     "%d, %.5f, %.5f, %.5e, %.5e, %.5e, %.4f, %.4f, %s, %s, %s, %s, %s, %s, %s, %s\n"
        #                     % (cnt, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o)
        #                 )
        #                 cnt = cnt + 1
        #
        #         else:
        #             fid.write(
        #                 "id, RAJ2000, DEJ2000, Smass, S, Sloc, Dist, z, GWGC, PGC, HyperLEDA, 2MASS, SDSS, B_mag, stellarmass\n"
        #             )
        #             for a, b, c, d, e, f, g, h, i, j, k, l, m, n in zip(
        #                 ra,
        #                 dec,
        #                 Smass,
        #                 S,
        #                 Sloc,
        #                 distmpc,
        #                 z,
        #                 GWGC,
        #                 PGC,
        #                 HyperLEDA,
        #                 _2MASS,
        #                 SDSS,
        #                 magb,
        #                 stellarmass,
        #             ):
        #                 fid.write(
        #                     "%d, %.5f, %.5f, %.5e, %.5e, %.5e, %.4f, %.4f, %s, %s, %s, %s, %s, %s, %s\n"
        #                     % (cnt, a, b, c, d, e, f, g, h, i, j, k, l, m, n)
        #                 )
        #                 cnt = cnt + 1
        #
        #     else:
        #         if params["AGN_flag"]:
        #             fid.write(
        #                 "id, RAJ2000, DEJ2000, Sloc, S, Dist, z, GWGC, PGC, HyperLEDA, 2MASS, SDSS, AGN_flag\n"
        #             )
        #             for a, b, c, d, e, f, g, h, i, j, k, l in zip(
        #                 ra,
        #                 dec,
        #                 Sloc,
        #                 S,
        #                 distmpc,
        #                 z,
        #                 GWGC,
        #                 PGC,
        #                 HyperLEDA,
        #                 _2MASS,
        #                 SDSS,
        #                 AGN_flag,
        #             ):
        #                 fid.write(
        #                     "%d, %.5f, %.5f, %.5e, %.5e, %.4f, %.4f, %s, %s, %s, %s, %s, %s\n"
        #                     % (cnt, a, b, c, d, e, f, g, h, i, j, k, l)
        #                 )
        #                 cnt = cnt + 1
        #
        #         else:
        #             fid.write(
        #                 "id, RAJ2000, DEJ2000, Sloc, S, Dist, z, GWGC, PGC, HyperLEDA, 2MASS, SDSS\n"
        #             )
        #             for a, b, c, d, e, f, g, h, i, j, k in zip(
        #                 ra, dec, Sloc, S, distmpc, z, GWGC, PGC, HyperLEDA, _2MASS, SDSS
        #             ):
        #                 fid.write(
        #                     "%d, %.5f, %.5f, %.5e, %.5e, %.4f, %.4f, %s, %s, %s, %s, %s\n"
        #                     % (cnt, a, b, c, d, e, f, g, h, i, j, k)
        #                 )
        #                 cnt = cnt + 1
        #
        # else:
        #     fid.write("id, RAJ2000, DEJ2000, Sloc, S, Dist, z\n")
        #     for a, b, c, d in zip(ra, dec, Sloc, S, distmpc, z):
        #         fid.write(
        #             "%d, %.5f, %.5f, %.5e, %.5e, %.4f, %.4f\n" % (cnt, a, b, c, d, e, f)
        #         )
        #         cnt = cnt + 1
        #
        # fid.close()

    return map_struct, cat_df
