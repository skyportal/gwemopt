import copy

import healpy as hp
import numpy as np
from astroquery.vizier import Vizier
from ligo.skymap import distance
from scipy.stats import norm

from gwemopt.catalogs.clu import CluCatalog
from gwemopt.catalogs.glade import GladeCatalog
from gwemopt.catalogs.mangrove import MangroveCatalog
from gwemopt.catalogs.nedlvs import NEDCatalog
from gwemopt.catalogs.twomrs import TwoMRSCatalog

# Unset row limits when querying Vizier
Vizier.ROW_LIMIT = -1


def get_catalog(params, map_struct, export_catalog: bool = True):
    """
    Get the catalog of galaxies to be used in the optimization.
    """
    params["catalogDir"].mkdir(parents=True, exist_ok=True)

    """AB Magnitude zero point."""
    MAB0 = -2.5 * np.log10(3631.0e-23)
    pc_cm = 3.08568025e18
    const = 4.0 * np.pi * (10.0 * pc_cm) ** 2.0

    if params["catalog"] == "2MRS":
        cat = TwoMRSCatalog(catalog_dir=params["catalogDir"])
        default_mag_column = "magk"

    elif params["catalog"] == "GLADE":
        cat = GladeCatalog(catalog_dir=params["catalogDir"])
        default_mag_column = "magk"

    elif params["catalog"] == "CLU":
        cat = CluCatalog(catalog_dir=params["catalogDir"])
        default_mag_column = "magb"

    elif params["catalog"] == "mangrove":
        cat = MangroveCatalog(catalog_dir=params["catalogDir"])
        default_mag_column = "magb"

    elif params["catalog"] == "NED":
        cat = NEDCatalog(catalog_dir=params["catalogDir"])
        default_mag_column = "magk"
    else:
        raise KeyError(
            f"Unknown galaxy catalog: {params['galaxy_catalog']}. "
            f"Must be one of '2MRS', 'GLADE', 'CLU', 'mangrove', or 'NED'"
        )

    cat_df = cat.get_catalog()
    mag_column = params.get("catalog_mag_column", default_mag_column)

    if params["catalog"] == "glade":
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

    ipix = hp.ang2pix(
        map_struct["nside"],
        np.array(cat_df["ra"]),
        np.array(cat_df["dec"]),
        lonlat=True,
    )

    if "distnorm" in map_struct:
        if map_struct["distnorm"] is not None:
            # creat an mask to cut at 3 sigma in distance
            mask = np.zeros(len(cat_df["distmpc"]))

            # calculate the moments from distmu, distsigma and distnorm
            mom_mean, mom_std, mom_norm = distance.parameters_to_moments(
                map_struct["distmu"], map_struct["distsigma"]
            )

            condition_indexer = np.where(
                (cat_df["distmpc"] < (mom_mean[ipix] + (3 * mom_std[ipix])))
                & (cat_df["distmpc"] > (mom_mean[ipix] - (3 * mom_std[ipix])))
            )
            mask[condition_indexer] = 1

            s_loc = (
                prob_scaled[ipix]
                * (
                    map_struct["distnorm"][ipix]
                    * norm(
                        map_struct["distmu"][ipix], map_struct["distsigma"][ipix]
                    ).pdf(cat_df["distmpc"])
                )
                ** params["powerlaw_dist_exp"]
                / map_struct["pixarea"]
            )

            # multiplie the Sloc by 1 or 0 according to the 3 sigma condistion
            s_loc = s_loc * mask
        else:
            s_loc = copy.copy(prob_scaled[ipix])
    else:
        s_loc = copy.copy(prob_scaled[ipix])

    # this happens when we are using a tiny catalog...
    if np.all(s_loc == 0.0):
        s_loc[:] = 1.0

    # new version of the Slum calcul (from HOGWARTs)
    Lsun = 3.828e26
    Msun = 4.83
    Lblist = []

    for _, row in cat_df.iterrows():
        if row[mag_column] is not None:
            Mb = row[mag_column] - 5 * np.log10((row["distmpc"] * 10**6)) + 5
            Lb = Lsun * 2.512 ** (Msun - Mb)
            Lblist.append(Lb)
        else:
            Lblist.append(0)

    # set 0 when Sloc is 0 (keep compatible galaxies for normalization)
    Lblist = np.array(Lblist)

    Lblist[s_loc == 0] = 0

    Slum = Lblist / np.nansum(np.array(Lblist))

    mlim, M_KNmin, M_KNmax = 22, -17, -12
    L_KNmin = const * 10.0 ** ((M_KNmin + MAB0) / (-2.5))
    L_KNmax = const * 10.0 ** ((M_KNmax + MAB0) / (-2.5))

    Llim = (
        4.0
        * np.pi
        * (cat_df["distmpc"] * 1e6 * pc_cm) ** 2.0
        * 10.0 ** ((mlim + MAB0) / (-2.5))
    )
    sdet = (L_KNmax - Llim) / (L_KNmax - L_KNmin)
    sdet[sdet < 0.01] = 0.01
    sdet[sdet > 1.0] = 1.0

    # Set nan values to zero
    s_loc[np.isnan(s_loc)] = 0
    Slum[np.isnan(Slum)] = 0

    if params["galaxy_grade"] == "Smass":
        if params["galaxy_catalog"] != "mangrove":
            raise ValueError(
                "You are trying to use the stellar mass information (Smass), "
                "please select the mangrove catalog for such use."
            )

        # set Smass
        s_mass = cat_df["stellarmass"].to_numpy()

        # put null values to nan
        s_mass[np.where(s_mass == 0)] = np.nan

        # go back to linear scaling and not log
        s_mass = 10.0**s_mass

        # Keep only galaxies with finite stellarmass when using it in the grade
        # set nan values to 0
        s_mass[~np.isfinite(s_mass)] = 0

        # set Smass
        s_mass = s_mass / np.sum(s_mass)

        # alpha is defined only with non null mass galaxies, we set a mask for that
        ind_without_mass = np.where(s_mass == 0)
        Sloc_temp = copy.deepcopy(s_loc)
        Sloc_temp[ind_without_mass] = 0

        # alpha_mass parameter is defined in such way that in mean Sloc count
        # in as much as Sloc*alpha*Smass
        alpha_mass = np.sum(Sloc_temp) / np.sum(Sloc_temp * s_mass)
        print(
            "You chose to use the grade using stellar mass, the parameters values are:"
        )
        print("alpha_mass =", alpha_mass)

        # beta_mass is a parameter allowing to change the importance of Sloc
        # according to Sloc*alpha*Smass
        # beta_mass should be fitted in the futur on real GW event
        # for which we have the host galaxy
        # fixed to one at the moment
        beta_mass = 1
        print("beta_mass =", beta_mass)

        s_mass = s_loc * (1 + (alpha_mass * beta_mass * s_mass))

    s = np.array(s_loc * Slum * sdet)
    prob = np.zeros(map_struct["prob"].shape)
    if params["galaxy_grade"] == "Sloc":
        for j in range(len(ipix)):
            prob[ipix[j]] += s_loc[j]
        grade = s_loc
    elif params["galaxy_grade"] == "S":
        for j in range(len(ipix)):
            prob[ipix[j]] += s[j]
        grade = s
    elif params["galaxy_grade"] == "Smass":
        for j in range(len(ipix)):
            prob[ipix[j]] += s_mass[j]
        grade = s_mass
    else:
        raise ValueError(
            "You are trying to use a galaxy grade that is not implemented yet."
        )

    prob = prob / np.sum(prob)

    map_struct["prob_catalog"] = prob
    if params["doUseCatalog"]:
        map_struct["prob"] = prob

    cat_df["grade"] = grade
    cat_df["S"] = s
    cat_df["Sloc"] = s_loc
    if params["galaxy_grade"] != "Smass":
        cat_df["Smass"] = 1.0

    mask = np.where(~np.isnan(grade))[0]
    cat_df = cat_df.iloc[mask]

    cat_df.sort_values(by=["grade"], inplace=True, ascending=False, ignore_index=True)

    if len(cat_df) > params["galaxy_limit"]:
        print(f'Cutting catalog to top {params["galaxy_limit"]} galaxies...')
        cat_df = cat_df.iloc[: params["galaxy_limit"]]

    # now normalize the distributions
    for key in ["S", "Sloc", "Smass"]:
        cat_df[key] = cat_df[key] / np.sum(cat_df[key])

    if export_catalog:
        output_path = params["outputDir"].joinpath(f"catalog_{cat.name}.csv")
        print(f"Saving catalog to {output_path}")
        cat_df.to_csv(output_path)

    return map_struct, cat_df
