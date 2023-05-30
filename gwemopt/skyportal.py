import copy

import healpy as hp
import ligo.segments as segments
import numpy as np
from mocpy import MOC
from regions import CircleSkyRegion, PolygonSkyRegion, RectangleSkyRegion

import gwemopt.tiles
from gwemopt.tiles import (
    absmag_tiles_struct,
    angular_distance,
    get_rectangle,
    powerlaw_tiles_struct,
)


def create_galaxy_from_skyportal(params, map_struct, catalog_struct, regions=None):
    nside = params["nside"]

    tile_structs = {}
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]
        if type(regions[0]) == RectangleSkyRegion:
            config_struct["FOV_type"] = "square"
            config_struct["FOV"] = np.max(
                [regions[0].width.value, regions[0].height.value]
            )
        elif type(regions[0]) == CircleSkyRegion:
            config_struct["FOV_type"] = "circle"
            config_struct["FOV"] = regions[0].radius.value
        elif type(regions[0]) == PolygonSkyRegion:
            ra = np.array([regions[0].vertices.ra for reg in regions])
            dec = np.array([regions[0].vertices.dec for reg in regions])
            min_ra, max_ra = np.min(ra), np.max(ra)
            min_dec, max_dec = np.min(dec), np.max(dec)
            config_struct["FOV_type"] = "square"
            config_struct["FOV"] = np.max([max_ra - min_ra, max_dec - min_dec])

        # Combine in a single pointing, galaxies that are distant by
        # less than FoV * params['galaxies_FoV_sep']
        # Take galaxy with highest proba at the center of new pointing
        FoV = params["config"][telescope]["FOV"] * params["galaxies_FoV_sep"]
        if "FOV_center" in params["config"][telescope]:
            FoV_center = (
                params["config"][telescope]["FOV_center"] * params["galaxies_FoV_sep"]
            )
        else:
            FoV_center = params["config"][telescope]["FOV"] * params["galaxies_FoV_sep"]

        new_ra = []
        new_dec = []
        new_Sloc = []
        new_Smass = []
        new_S = []
        galaxies = []
        idxRem = np.arange(len(catalog_struct["ra"])).astype(int)

        while len(idxRem) > 0:
            ii = idxRem[0]
            ra, dec, Sloc, S, Smass = (
                catalog_struct["ra"][ii],
                catalog_struct["dec"][ii],
                catalog_struct["Sloc"][ii],
                catalog_struct["S"][ii],
                catalog_struct["Smass"][ii],
            )

            if config_struct["FOV_type"] == "square":
                decCorners = (dec - FoV, dec + FoV)
                # assume small enough to use average dec for corners
                raCorners = (
                    ra - FoV / np.cos(np.deg2rad(dec)),
                    ra + FoV / np.cos(np.deg2rad(dec)),
                )
                idx1 = np.where(
                    (catalog_struct["ra"][idxRem] >= raCorners[0])
                    & (catalog_struct["ra"][idxRem] <= raCorners[1])
                )[0]
                idx2 = np.where(
                    (catalog_struct["dec"][idxRem] >= decCorners[0])
                    & (catalog_struct["dec"][idxRem] <= decCorners[1])
                )[0]
                mask = np.intersect1d(idx1, idx2)

                if len(mask) > 1:
                    ra_center, dec_center = get_rectangle(
                        catalog_struct["ra"][idxRem][mask],
                        catalog_struct["dec"][idxRem][mask],
                        FoV / np.cos(np.deg2rad(dec)),
                        FoV,
                    )

                    decCorners = (dec_center - FoV / 2.0, dec_center + FoV / 2.0)
                    raCorners = (
                        ra_center - FoV / (2.0 * np.cos(np.deg2rad(dec))),
                        ra_center + FoV / (2.0 * np.cos(np.deg2rad(dec))),
                    )
                    idx1 = np.where(
                        (catalog_struct["ra"][idxRem] >= raCorners[0])
                        & (catalog_struct["ra"][idxRem] <= raCorners[1])
                    )[0]
                    idx2 = np.where(
                        (catalog_struct["dec"][idxRem] >= decCorners[0])
                        & (catalog_struct["dec"][idxRem] <= decCorners[1])
                    )[0]
                    mask2 = np.intersect1d(idx1, idx2)

                    decCorners = (
                        dec_center - FoV_center / 2.0,
                        dec_center + FoV_center / 2.0,
                    )
                    raCorners = (
                        ra_center - FoV_center / (2.0 * np.cos(np.deg2rad(dec))),
                        ra_center + FoV_center / (2.0 * np.cos(np.deg2rad(dec))),
                    )
                    idx1 = np.where(
                        (catalog_struct["ra"][idxRem] >= raCorners[0])
                        & (catalog_struct["ra"][idxRem] <= raCorners[1])
                    )[0]
                    idx2 = np.where(
                        (catalog_struct["dec"][idxRem] >= decCorners[0])
                        & (catalog_struct["dec"][idxRem] <= decCorners[1])
                    )[0]
                    mask3 = np.intersect1d(idx1, idx2)

                    # did the optimization help?
                    if (len(mask2) > 2) and (len(mask3) > 0):
                        mask = mask2
                else:
                    ra_center, dec_center = np.mean(
                        catalog_struct["ra"][idxRem][mask]
                    ), np.mean(catalog_struct["dec"][idxRem][mask])

            elif config_struct["FOV_type"] == "circle":
                dist = angular_distance(
                    ra, dec, catalog_struct["ra"][idxRem], catalog_struct["dec"][idxRem]
                )
                mask = np.where((2 * FoV) >= dist)[0]
                if len(mask) > 1:
                    ra_center, dec_center = get_rectangle(
                        catalog_struct["ra"][idxRem][mask],
                        catalog_struct["dec"][idxRem][mask],
                        (FoV / np.sqrt(2)) / np.cos(np.deg2rad(dec)),
                        FoV / np.sqrt(2),
                    )

                    dist = angular_distance(
                        ra_center,
                        dec_center,
                        catalog_struct["ra"][idxRem],
                        catalog_struct["dec"][idxRem],
                    )
                    mask2 = np.where(FoV >= dist)[0]
                    # did the optimization help?
                    if len(mask2) > 2:
                        mask = mask2
                else:
                    ra_center, dec_center = np.mean(
                        catalog_struct["ra"][idxRem][mask]
                    ), np.mean(catalog_struct["dec"][idxRem][mask])

            new_ra.append(ra_center)
            new_dec.append(dec_center)
            new_Sloc.append(np.sum(catalog_struct["Sloc"][idxRem][mask]))
            new_S.append(np.sum(catalog_struct["S"][idxRem][mask]))
            new_Smass.append(np.sum(catalog_struct["Smass"][idxRem][mask]))
            galaxies.append(idxRem[mask])

            idxRem = np.setdiff1d(idxRem, idxRem[mask])

        # redefine catalog_struct
        catalog_struct_new = {}
        catalog_struct_new["ra"] = new_ra
        catalog_struct_new["dec"] = new_dec
        catalog_struct_new["Sloc"] = new_Sloc
        catalog_struct_new["S"] = new_S
        catalog_struct_new["Smass"] = new_Smass
        catalog_struct_new["galaxies"] = galaxies

        moc_struct = {}
        cnt = 0
        for ra, dec, Sloc, S, Smass, galaxies in zip(
            catalog_struct_new["ra"],
            catalog_struct_new["dec"],
            catalog_struct_new["Sloc"],
            catalog_struct_new["S"],
            catalog_struct_new["Smass"],
            catalog_struct_new["galaxies"],
        ):
            moc_struct[int(cnt)] = gwemopt.moc.Fov2Moc(
                params, config_struct, telescope, ra, dec, nside
            )
            moc_struct[int(cnt)]["galaxies"] = galaxies
            cnt = cnt + 1

        if params["timeallocationType"] == "absmag":
            tile_struct = absmag_tiles_struct(
                params, config_struct, telescope, map_struct, moc_struct
            )
        elif params["timeallocationType"] == "powerlaw":
            tile_struct = powerlaw_tiles_struct(
                params,
                config_struct,
                telescope,
                map_struct,
                moc_struct,
                catalog_struct=catalog_struct,
            )

        tile_struct = gwemopt.segments.get_segments_tiles(
            params, config_struct, tile_struct
        )

        cnt = 0
        for ra, dec, Sloc, S, Smass, galaxies in zip(
            catalog_struct_new["ra"],
            catalog_struct_new["dec"],
            catalog_struct_new["Sloc"],
            catalog_struct_new["S"],
            catalog_struct_new["Smass"],
            catalog_struct_new["galaxies"],
        ):
            if params["galaxy_grade"] == "Sloc":
                tile_struct[cnt]["prob"] = Sloc
            elif params["galaxy_grade"] == "S":
                tile_struct[cnt]["prob"] = S
            elif params["galaxy_grade"] == "Smass":
                tile_struct[cnt]["prob"] = Smass

            tile_struct[cnt]["galaxies"] = galaxies
            if config_struct["FOV_type"] == "square":
                tile_struct[cnt]["area"] = params["config"][telescope]["FOV"] ** 2
            elif config_struct["FOV_type"] == "circle":
                tile_struct[cnt]["area"] = (
                    4 * np.pi * params["config"][telescope]["FOV"] ** 2
                )
            cnt = cnt + 1

        tile_structs[telescope] = tile_struct

    return tile_structs


def create_moc_from_skyportal(params, map_struct=None, field_ids=None):
    nside = params["nside"]
    npix = hp.nside2npix(nside)

    if params["doMinimalTiling"]:
        prob = map_struct["prob"]

        n, cl, dist_exp = (
            params["powerlaw_n"],
            params["powerlaw_cl"],
            params["powerlaw_dist_exp"],
        )
        prob_scaled = copy.deepcopy(prob)
        prob_sorted = np.sort(prob_scaled)[::-1]
        prob_indexes = np.argsort(prob_scaled)[::-1]
        prob_cumsum = np.cumsum(prob_sorted)
        index = np.argmin(np.abs(prob_cumsum - cl)) + 1
        prob_indexes = prob_indexes[: index + 1]

    if "doUsePrimary" in params:
        doUsePrimary = params["doUsePrimary"]
    else:
        doUsePrimary = False

    if "doUseSecondary" in params:
        doUseSecondary = params["doUseSecondary"]
    else:
        doUseSecondary = False

    npix = hp.nside2npix(nside)
    pixarea = hp.nside2pixarea(nside)

    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra = np.rad2deg(phi)
    dec = np.rad2deg(0.5 * np.pi - theta)

    moc_structs = {}
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]
        tesselation = config_struct["tesselation"]
        moc_struct = {}
        ipixs = []
        for ii, tess in enumerate(tesselation):
            if field_ids is not None:
                if tess.field_id not in field_ids[telescope]:
                    ipixs.append([])
                    continue
            ipixs.append(skyportal2FOV(tess, nside))
        for ii, tess in enumerate(tesselation):
            index = tess.field_id

            if (telescope == "ZTF") and doUsePrimary and (index > 880):
                continue
            if (telescope == "ZTF") and doUseSecondary and (index < 1000):
                continue

            ipix = ipixs[ii]
            if len(ipix) == 0:
                continue

            moc_struct[index] = {}
            moc_struct[index]["ra"] = tess.ra
            moc_struct[index]["dec"] = tess.dec

            moc_struct[index]["ipix"] = ipix
            moc_struct[index]["corners"] = [
                [np.min(ra[ipix]), np.min(dec[ipix])],
                [np.min(ra[ipix]), np.max(dec[ipix])],
                [np.max(ra[ipix]), np.max(dec[ipix])],
                [np.max(ra[ipix]), np.min(dec[ipix])],
            ]
            moc_struct[index]["patch"] = []
            moc_struct[index]["area"] = len(ipix) * pixarea

        if map_struct is not None:
            ipix_keep = map_struct["ipix_keep"]
        else:
            ipix_keep = []

        if params["doMinimalTiling"]:
            moc_struct_new = copy.copy(moc_struct)
            if params["tilesType"] == "galaxy":
                tile_probs = gwemopt.tiles.compute_tiles_map(
                    params, moc_struct_new, prob, func="center", ipix_keep=ipix_keep
                )
            else:
                tile_probs = gwemopt.tiles.compute_tiles_map(
                    params, moc_struct_new, prob, func="np.sum(x)", ipix_keep=ipix_keep
                )

            keys = moc_struct.keys()

            sort_idx = np.argsort(tile_probs)[::-1]
            csm = np.empty(len(tile_probs))
            csm[sort_idx] = np.cumsum(tile_probs[sort_idx])
            ipix_keep = np.where(csm <= cl)[0]

            probs = []
            moc_struct = {}
            cnt = 0
            for ii, key in enumerate(keys):
                if ii in ipix_keep:
                    moc_struct[key] = moc_struct_new[key]
                    cnt = cnt + 1

        moc_structs[telescope] = moc_struct

    return moc_structs


def skyportal2FOV(tess, nside):
    moc = moc_from_tiles([tile.healpix for tile in tess.tiles], 2**29)
    pix_id = moc.degrade_to_order(int(np.log2(nside))).flatten()
    if len(pix_id) > 0:
        ipix = hp.nest2ring(int(nside), pix_id.tolist())
    else:
        ipix = []

    return ipix


def moc_from_tiles(rangeSet, nside):
    depth = int(np.log2(nside))
    segmentlist = segments.segmentlist()
    for x in rangeSet:
        segment = segments.segment(x.lower, x.upper - 1)
        segmentlist = segmentlist + segments.segmentlist([segment])
    segmentlist.coalesce()

    MOCstr = f"{depth}/" + " ".join(map(lambda x: f"{x[0]}-{x[1]}", segmentlist))
    return MOC.from_string(MOCstr)
