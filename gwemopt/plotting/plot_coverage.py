import copy

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from matplotlib.pyplot import cm

from gwemopt.io import export_tiles_coverage_hist, export_tiles_coverage_int
from gwemopt.plotting.movie import make_movie
from gwemopt.plotting.style import CBAR_BOOL, UNIT, add_edges, add_sun_moon, cmap
from gwemopt.utils.geometry import angular_distance


def plot_mollweide_coverage(params, map_struct, coverage_struct):
    """
    Plot the coverage in Mollweide projection.
    """
    plot_name = params["outputDir"].joinpath("mollview_coverage.pdf")
    plt.figure()
    hp.mollview(map_struct["prob"], title="", unit=UNIT, cbar=CBAR_BOOL, cmap=cmap)
    hp.projplot(
        coverage_struct["data"][:, 0],
        coverage_struct["data"][:, 1],
        "wx",
        lonlat=True,
        coord="G",
    )
    add_edges()
    plt.savefig(plot_name, dpi=200)
    plt.close()


def plot_coverage(params, coverage_struct):
    """
    Plot the coverage
    """
    plot_name = params["outputDir"].joinpath("coverage.pdf")
    plt.figure(figsize=(10, 8))
    for ii in range(len(coverage_struct["ipix"])):
        data = coverage_struct["data"][ii, :]
        filt = coverage_struct["filters"][ii]

        if filt == "g":
            color = "g"
        elif filt == "r":
            color = "r"
        else:
            color = "k"

        plt.scatter(data[2], data[5], s=20, color=color)

    plt.xlabel("Time [MJD]")
    plt.ylabel("Tile Number")
    plt.savefig(plot_name, dpi=200)
    plt.close()


def plot_tiles_coverage(params, map_struct, coverage_struct, plot_sun_moon=False):
    """
    Plot the tiles coverage in Mollweide projection.
    """
    plot_name = params["outputDir"].joinpath("tiles_coverage.pdf")
    plt.figure()
    hp.mollview(map_struct["prob"], title="", unit=UNIT, cbar=CBAR_BOOL, cmap=cmap)
    add_edges()
    ax = plt.gca()
    for ii in range(len(coverage_struct["ipix"])):
        patch = coverage_struct["patch"][ii]

        if patch == []:
            continue

        if not type(patch) == list:
            patch = [patch]

        for p in patch:
            patch_cpy = copy.copy(p)
            patch_cpy.axes = None
            patch_cpy.figure = None
            patch_cpy.set_transform(ax.transData)
            hp.projaxes.HpxMollweideAxes.add_patch(ax, patch_cpy)

    if plot_sun_moon:
        add_sun_moon(params)

    plt.savefig(plot_name, dpi=200)
    plt.close()


def plot_tiles_coverage_hist(params, diffs):
    """
    Plot the histogram of the tiles coverage.
    """
    plot_name = params["outputDir"].joinpath("tiles_coverage_hist.pdf")
    plt.figure(figsize=(12, 8))
    bins = np.linspace(0.0, 24.0, 25)
    plt.hist(24.0 * np.array(diffs), bins=bins)
    plt.xlabel("Difference Between Observations [hours]")
    plt.ylabel("Number of Observations")
    plt.savefig(plot_name, dpi=200)
    plt.close("all")


def plot_tiles_coverage_int(
    params, map_struct, catalog_struct, coverage_struct, plot_sun_moon=False
):
    gpstime = params["gpstime"]
    event_mjd = Time(gpstime, format="gps", scale="utc").mjd

    colors = cm.rainbow(np.linspace(0, 1, len(params["telescopes"])))

    plot_name = params["outputDir"].joinpath("tiles_coverage_int.pdf")
    fig = plt.figure(figsize=(12, 8))

    gs = fig.add_gridspec(4, 1)
    ax1 = fig.add_subplot(gs[0:3, 0], projection="astro hours mollweide")
    ax2 = fig.add_subplot(gs[3, 0])
    ax3 = ax2.twinx()  # mirror them

    plt.axes(ax1)
    hp.mollview(
        map_struct["prob"], title="", unit=UNIT, cbar=CBAR_BOOL, cmap=cmap, hold=True
    )
    add_edges()
    ax = plt.gca()

    if params["tilesType"] == "galaxy":
        for telescope, color in zip(params["telescopes"], colors):
            idx = np.where(coverage_struct["telescope"] == telescope)[0]
            hp.projscatter(
                coverage_struct["data"][idx, 0],
                coverage_struct["data"][idx, 1],
                lonlat=True,
                s=10,
                color=color,
            )
    else:
        for ii in range(len(coverage_struct["ipix"])):
            patch = coverage_struct["patch"][ii]

            idx = params["telescopes"].index(coverage_struct["telescope"][ii])

            if patch == []:
                continue

            if not type(patch) == list:
                patch = [patch]

            for p in patch:
                patch_cpy = copy.copy(p)
                patch_cpy.axes = None
                patch_cpy.figure = None
                patch_cpy.set_transform(ax.transData)
                patch_cpy.set_facecolor(colors[idx])

                hp.projaxes.HpxMollweideAxes.add_patch(ax, patch_cpy)

    idxs = np.argsort(coverage_struct["data"][:, 2])
    plt.axes(ax2)
    for telescope, color in zip(params["telescopes"], colors):
        ipixs = np.empty((0, 2))
        cum_prob = 0.0

        tts, cum_probs, cum_areas = [], [], []
        if params["tilesType"] == "galaxy":
            cum_galaxies = []

        for jj, ii in enumerate(idxs):
            if np.mod(jj, 100) == 0:
                print("%s: %d/%d" % (telescope, jj, len(idxs)))

            data = coverage_struct["data"][ii, :]
            ipix = coverage_struct["ipix"][ii]
            if params["tilesType"] == "galaxy":
                galaxies = coverage_struct["galaxies"][ii]

            if not telescope == coverage_struct["telescope"][ii]:
                continue

            if params["tilesType"] == "galaxy":
                overlap = np.setdiff1d(galaxies, cum_galaxies)
                if len(overlap) > 0:
                    for galaxy in galaxies:
                        if galaxy in cum_galaxies:
                            continue
                        if catalog_struct is None:
                            continue
                        if params["galaxy_grade"] == "Sloc":
                            cum_prob = cum_prob + catalog_struct["Sloc"][galaxy]
                        elif params["galaxy_grade"] == "S":
                            cum_prob = cum_prob + catalog_struct["S"][galaxy]
                        elif params["galaxy_grade"] == "Smass":
                            cum_prob = cum_prob + catalog_struct["Smass"][galaxy]
                    cum_galaxies = np.append(cum_galaxies, galaxies)
                    cum_galaxies = np.unique(cum_galaxies).astype(int)
                cum_area = len(cum_galaxies)
            else:
                ipixs = np.append(ipixs, ipix)
                ipixs = np.unique(ipixs).astype(int)

                cum_prob = np.sum(map_struct["prob"][ipixs])
                cum_area = len(ipixs) * map_struct["pixarea_deg2"]

            cum_probs.append(cum_prob)
            cum_areas.append(cum_area)
            tts.append(data[2] - event_mjd)

        ax2.plot(tts, cum_probs, color=color, linestyle="-", label=telescope)
        ax3.plot(tts, cum_areas, color=color, linestyle="--")

        filename = params["outputDir"].joinpath(f"tiles_coverage_int_{telescope}.txt")
        export_tiles_coverage_int(filename, tts, cum_probs, cum_areas)

    ax2.set_xlabel("Time since event [days]")
    if params["tilesType"] == "galaxy":
        ax2.set_ylabel("Integrated Metric")
    else:
        ax2.set_ylabel("Integrated Probability")

    if params["tilesType"] == "galaxy":
        ax3.set_ylabel("Number of galaxies")
    else:
        ax3.set_ylabel("Sky area [sq. deg.]")

    ipixs = np.empty((0, 2))
    cum_prob = 0.0

    tts, cum_probs, cum_areas = [], [], []
    if params["tilesType"] == "galaxy":
        cum_galaxies = []

    for jj, ii in enumerate(idxs):
        data = coverage_struct["data"][ii, :]
        ipix = coverage_struct["ipix"][ii]
        if params["tilesType"] == "galaxy":
            galaxies = coverage_struct["galaxies"][ii]

        if params["tilesType"] == "galaxy":
            overlap = np.setdiff1d(galaxies, cum_galaxies)
            if len(overlap) > 0:
                for galaxy in galaxies:
                    if galaxy in cum_galaxies:
                        continue
                    if catalog_struct is None:
                        continue
                    if params["galaxy_grade"] == "Sloc":
                        cum_prob = cum_prob + catalog_struct["Sloc"][galaxy]
                    elif params["galaxy_grade"] == "S":
                        cum_prob = cum_prob + catalog_struct["S"][galaxy]
                    elif params["galaxy_grade"] == "Smass":
                        cum_prob = cum_prob + catalog_struct["Smass"][galaxy]
                cum_galaxies = np.append(cum_galaxies, galaxies)
                cum_galaxies = np.unique(cum_galaxies).astype(int)
            cum_area = len(cum_galaxies)
        else:
            ipixs = np.append(ipixs, ipix)
            ipixs = np.unique(ipixs).astype(int)

            cum_prob = np.sum(map_struct["prob"][ipixs])
            cum_area = len(ipixs) * map_struct["pixarea_deg2"]

        tts.append(data[2] - event_mjd)
        cum_probs.append(cum_prob)
        cum_areas.append(cum_area)

    ax2.plot(tts, cum_probs, color="k", linestyle="-", label="All")
    ax3.plot(tts, cum_areas, color="k", linestyle="--")

    if len(params["telescopes"]) > 3:
        ax2.legend(loc=1, ncol=3, fontsize=10)
        ax2.set_ylim([0, 1])
        ax3.set_ylim([0, 10000])
    elif "IRIS" in params["telescopes"]:
        ax2.set_ylim([0, 1.0])
        ax3.set_ylim([0, 10000])
        ax2.legend(loc=1)
    elif "ZTF" in params["telescopes"]:
        ax2.set_ylim([0, 1.0])
        ax3.set_ylim([0, 10000])
        ax2.legend(loc=1)
    elif "PS1" in params["telescopes"]:
        ax2.set_ylim([0, 1.0])
        ax3.set_ylim([0, 10000])
        ax2.legend(loc=1)
    else:
        ax2.legend(loc=1)

    if plot_sun_moon:
        add_sun_moon(params)

    plt.savefig(plot_name, dpi=200)
    plt.close()


def plot_coverage_scaled(params, map_struct, coverage_struct, plot_sun_moon, max_time):
    plot_name = params["outputDir"].joinpath("tiles_coverage_scaled.pdf")
    plt.figure()
    hp.mollview(map_struct["prob"], title="", unit=UNIT, cbar=CBAR_BOOL, cmap=cmap)
    add_edges()
    ax = plt.gca()
    for ii in range(len(coverage_struct["ipix"])):
        data = coverage_struct["data"][ii, :]
        patch = coverage_struct["patch"][ii]

        if patch == []:
            continue

        if not type(patch) == list:
            patch = [patch]

        for p in patch:
            patch_cpy = copy.copy(p)
            patch_cpy.axes = None
            patch_cpy.figure = None
            patch_cpy.set_transform(ax.transData)
            current_alpha = patch_cpy.get_alpha()

            if current_alpha > 0.0:
                alpha = data[4] / max_time
                if alpha > 1:
                    alpha = 1.0
                patch_cpy.set_alpha(alpha)
            hp.projaxes.HpxMollweideAxes.add_patch(ax, patch_cpy)

    if plot_sun_moon:
        add_sun_moon(params)

    plt.savefig(plot_name, dpi=200)
    plt.close()

    if params["doMovie"]:
        idx = np.isfinite(coverage_struct["data"][:, 2])
        mjd_min = np.min(coverage_struct["data"][idx, 2])
        mjd_max = np.max(coverage_struct["data"][idx, 2])
        mjd_N = 100

        mjds = np.linspace(mjd_min, mjd_max, num=mjd_N)
        moviedir = params["outputDir"].joinpath("movie")
        moviedir.mkdir(exist_ok=True, parents=True)

        for jj in range(len(mjds)):
            mjd = mjds[jj]
            plot_name = moviedir.joinpath(f"coverage-{jj:04d}.png")
            title = f"Coverage Map: {mjd:.2f}"

            plt.figure()
            hp.mollview(
                map_struct["prob"], title=title, unit=UNIT, cbar=CBAR_BOOL, cmap=cmap
            )
            add_edges()
            ax = plt.gca()

            idx = np.where(coverage_struct["data"][:, 2] <= mjd)[0]
            for ii in idx:
                patch = coverage_struct["patch"][ii]

                if patch == []:
                    continue

                if not type(patch) == list:
                    patch = [patch]

                for p in patch:
                    patch_cpy = copy.copy(p)
                    patch_cpy.axes = None
                    patch_cpy.figure = None
                    patch_cpy.set_transform(ax.transData)
                    hp.projaxes.HpxMollweideAxes.add_patch(ax, patch_cpy)

            plt.savefig(plot_name, dpi=200)
            plt.close()

        moviefiles = moviedir.joinpath("coverage-%04d.png")
        filename = params["outputDir"].joinpath("coverage.mpg")

        make_movie(moviefiles, filename)


def make_coverage_plots(
    params, map_struct, coverage_struct, catalog_struct=None, plot_sun_moon: bool = True
):
    plot_mollweide_coverage(params, map_struct, coverage_struct)

    idx = np.isfinite(coverage_struct["data"][:, 4])
    if not idx.size:
        return

    max_time = np.max(coverage_struct["data"][idx, 4])

    plot_coverage(params, coverage_struct)

    plot_tiles_coverage(
        params, map_struct, coverage_struct, plot_sun_moon=plot_sun_moon
    )

    diffs = []
    if params["tilesType"] == "galaxy":
        coverage_ras = coverage_struct["data"][:, 0]
        coverage_decs = coverage_struct["data"][:, 1]

        for ii in range(len(coverage_ras) - 1):
            current_ra, current_dec = coverage_ras[ii], coverage_decs[ii]

            dist = angular_distance(
                current_ra, current_dec, coverage_ras[ii + 1 :], coverage_decs[ii + 1 :]
            )
            idx = np.where(dist <= 1 / 3600.0)[0]
            if len(idx) > 0:
                jj = idx[0]
                diffs.append(
                    np.abs(
                        coverage_struct["data"][ii, 2] - coverage_struct["data"][jj, 2]
                    )
                )
    else:
        for ii in range(len(coverage_struct["ipix"])):
            ipix = coverage_struct["ipix"][ii]
            for jj in range(len(coverage_struct["ipix"])):
                if ii >= jj:
                    continue
                if coverage_struct["telescope"][ii] == coverage_struct["telescope"][jj]:
                    continue
                ipix2 = coverage_struct["ipix"][jj]
                overlap = np.intersect1d(ipix, ipix2)
                rat = np.array(
                    [
                        float(len(overlap)) / float(len(ipix)),
                        float(len(overlap)) / float(len(ipix2)),
                    ]
                )
                if np.any(rat > 0.5):
                    diffs.append(
                        np.abs(
                            coverage_struct["data"][ii, 2]
                            - coverage_struct["data"][jj, 2]
                        )
                    )

    export_tiles_coverage_hist(params, diffs)

    plot_tiles_coverage_hist(params, diffs)

    plot_tiles_coverage_int(
        params, map_struct, catalog_struct, coverage_struct, plot_sun_moon
    )

    plot_coverage_scaled(params, map_struct, coverage_struct, plot_sun_moon, max_time)
