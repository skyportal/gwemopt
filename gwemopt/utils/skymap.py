import healpy as hp
import ligo.skymap.distance as ligodist
import numpy as np
from ligo.skymap.bayestar import rasterize
from ligo.skymap.io import read_sky_map

from gwemopt.utils.rotate import rotate_map
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy import stats
from astropy.io import fits
from astropy.time import Time


def read_skymap(params, do_3d: bool | None = None, map_struct=None):

    # Let's just figure out what's in the skymap first

    skymap_path = params["skymap"]
    is_3d = False
    t_obs = Time.now()

    with fits.open(skymap_path) as hdul:
        for x in hdul:
            if "DATE-OBS" in x.header:
                t_obs = Time(x.header["DATE-OBS"], format="isot")

            elif "EVENTMJD" in x.header:
                t_obs_mjd = x.header["EVENTMJD"]
                t_obs = Time(t_obs_mjd, format="mjd")

            if ("DISTMEAN" in x.header) | ("DISTSTD" in x.header):
                is_3d = True

    # Set GPS time from skymap, if not specified. Defaults to today
    params["eventtime"] = t_obs
    if params["gpstime"] is None:
        params["gpstime"] = t_obs.gps

    # "Do 3D" based on map, if not specified
    if do_3d is None:
        do_3d = is_3d

    header = []
    if map_struct is None:
        map_struct = {}

        if params["doDatabase"]:
            models = params["models"]
            localizations_all = models.Localization.query.all()
            localizations = models.Localization.query.filter_by(
                dateobs=params["dateobs"], localization_name=params["localization_name"]
            ).all()
            if localizations == None:
                raise ValueError("No localization with dateobs=%s" % params["dateobs"])
            else:
                prob_data = localizations[0].healpix
                prob_data = prob_data / np.sum(prob_data)
                map_struct["prob"] = prob_data

                distmu = localizations[0].distmu
                distsigma = localizations[0].distsigma
                distnorm = localizations[0].distnorm

                if distmu is None:
                    map_struct["distmu"] = None
                    map_struct["distsigma"] = None
                    map_struct["distnorm"] = None
                else:
                    map_struct["distmu"] = np.array(distmu)
                    map_struct["distsigma"] = np.array(distsigma)
                    map_struct["distnorm"] = np.array(distnorm)
                    do_3d = True
        else:
            filename = params["skymap"]

            if do_3d:
                try:
                    healpix_data, header = hp.read_map(
                        filename, field=(0, 1, 2, 3), verbose=False, h=True
                    )
                except:
                    table = read_sky_map(filename, moc=True, distances=True)
                    order = hp.nside2order(params["nside"])
                    t = rasterize(table, order)
                    result = t["PROB"], t["DISTMU"], t["DISTSIGMA"], t["DISTNORM"]
                    healpix_data = hp.reorder(result, "NESTED", "RING")

                distmu_data = healpix_data[1]
                distsigma_data = healpix_data[2]
                prob_data = healpix_data[0]
                norm_data = healpix_data[3]

                map_struct["distmu"] = distmu_data / params["DScale"]
                map_struct["distsigma"] = distsigma_data / params["DScale"]
                map_struct["prob"] = prob_data
                map_struct["distnorm"] = norm_data

            else:
                prob_data, header = hp.read_map(
                    filename, field=0, verbose=False, h=True
                )
                prob_data = prob_data / np.sum(prob_data)

                map_struct["prob"] = prob_data

    if params["doRotate"]:
        for key in map_struct.keys():
            map_struct[key] = rotate_map(
                map_struct[key], np.deg2rad(params["theta"]), np.deg2rad(params["phi"])
            )
        map_struct["prob"] = map_struct["prob"] / np.sum(map_struct["prob"])

    natural_nside = hp.pixelfunc.get_nside(map_struct["prob"])
    nside = params["nside"]

    print("natural_nside =", natural_nside)
    print("nside =", nside)

    if not do_3d:
        map_struct["prob"] = hp.ud_grade(map_struct["prob"], nside, power=-2)

    if do_3d:
        if natural_nside != nside:
            map_struct["prob"] = hp.pixelfunc.ud_grade(
                map_struct["prob"], nside, power=-2
            )
            map_struct["distmu"] = hp.pixelfunc.ud_grade(map_struct["distmu"], nside)
            map_struct["distsigma"] = hp.pixelfunc.ud_grade(
                map_struct["distsigma"], nside
            )
            map_struct["distnorm"] = hp.pixelfunc.ud_grade(
                map_struct["distnorm"], nside
            )

            map_struct["distmu"][map_struct["distmu"] < -1e30] = np.inf

        nside_down = 32

        distmu_down = hp.pixelfunc.ud_grade(map_struct["distmu"], nside_down)
        distsigma_down = hp.pixelfunc.ud_grade(map_struct["distsigma"], nside_down)
        distnorm_down = hp.pixelfunc.ud_grade(map_struct["distnorm"], nside_down)

        (
            map_struct["distmed"],
            map_struct["diststd"],
            mom_norm,
        ) = ligodist.parameters_to_moments(
            map_struct["distmu"], map_struct["distsigma"]
        )

        distmu_down[distmu_down < -1e30] = np.inf

        map_struct["distmed"] = hp.ud_grade(map_struct["distmed"], nside, power=-2)
        map_struct["diststd"] = hp.ud_grade(map_struct["diststd"], nside, power=-2)

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra = np.rad2deg(phi)
    dec = np.rad2deg(0.5 * np.pi - theta)

    map_struct["ra"] = ra
    map_struct["dec"] = dec

    if params["doRASlice"]:
        ra_low, ra_high = params["raslice"][0], params["raslice"][1]
        if ra_low <= ra_high:
            ipix = np.where(
                (ra_high * 360.0 / 24.0 < ra) | (ra_low * 360.0 / 24.0 > ra)
            )[0]
        else:
            ipix = np.where(
                (ra_high * 360.0 / 24.0 < ra) & (ra_low * 360.0 / 24.0 > ra)
            )[0]
        map_struct["prob"][ipix] = 0.0
        map_struct["prob"] = map_struct["prob"] / np.sum(map_struct["prob"])

    if params["doAvoidGalacticPlane"]:
        coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        ipix = np.where(np.abs(coords.galactic.b.deg) <= params["galactic_limit"])[0]
        map_struct["prob"][ipix] = 0.0
        map_struct["prob"] = map_struct["prob"] / np.sum(map_struct["prob"])

    sort_idx = np.argsort(map_struct["prob"])[::-1]
    csm = np.empty(len(map_struct["prob"]))
    csm[sort_idx] = np.cumsum(map_struct["prob"][sort_idx])

    map_struct["cumprob"] = csm
    map_struct["ipix_keep"] = np.where(csm <= params["iterativeOverlap"])[0]

    pixarea = hp.nside2pixarea(nside)
    pixarea_deg2 = hp.nside2pixarea(nside, degrees=True)

    map_struct["nside"] = nside
    map_struct["npix"] = npix
    map_struct["pixarea"] = pixarea
    map_struct["pixarea_deg2"] = pixarea_deg2

    for j in range(len(header)):
        if header[j][0] == "DATE":
            map_struct["trigtime"] = header[j][1]

    return params, map_struct


def samples_from_skymap(map_struct, is3D=False, Nsamples=100):
    prob_data_sorted = np.sort(map_struct["prob"])[::-1]
    prob_data_indexes = np.argsort(map_struct["prob"])[::-1]
    prob_data_cumsum = np.cumsum(prob_data_sorted)

    rand_values = np.random.rand(
        Nsamples,
    )

    ras = []
    decs = []
    dists = []

    if is3D:
        r = np.linspace(0, 2000)
        rand_values_dist = np.random.rand(
            Nsamples,
        )

    for ii in range(Nsamples):
        ipix = np.argmin(np.abs(prob_data_cumsum - rand_values[ii]))
        ra_inj = map_struct["ra"][prob_data_indexes][ipix]
        dec_inj = map_struct["dec"][prob_data_indexes][ipix]

        ras.append(ra_inj)
        decs.append(dec_inj)

        if is3D:
            dp_dr = (
                r**2
                * map_struct["distnorm"][prob_data_indexes][ipix]
                * stats.norm(
                    map_struct["distmu"][prob_data_indexes][ipix],
                    map_struct["distsigma"][prob_data_indexes][ipix],
                ).pdf(r)
            )
            dp_dr_norm = np.cumsum(dp_dr / np.sum(dp_dr))
            idx = np.argmin(np.abs(dp_dr_norm - rand_values_dist[ii]))
            dist_inj = r[idx]
            dists.append(dist_inj)
        else:
            dists.append(50.0)

    samples_struct = {}
    samples_struct["ra"] = np.array(ras)
    samples_struct["dec"] = np.array(decs)
    samples_struct["dist"] = np.array(dists)

    return samples_struct
