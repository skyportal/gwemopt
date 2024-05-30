import copy

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from matplotlib.pyplot import cm
from tqdm import tqdm

from gwemopt.io import export_tiles_coverage_int
from gwemopt.plotting.movie import make_movie
from gwemopt.plotting.style import add_sun_moon
from gwemopt.utils.geometry import angular_distance


def plot_tiles_coverage(params, map_struct, coverage_struct, plot_sun_moon=False):
    """
    Plot the tiles coverage
    """
    plot_name = params["outputDir"].joinpath("tiles_coverage.pdf")

    hdu = map_struct["hdu"]
    columns = [col.name for col in hdu.columns]

    fig = plt.figure(figsize=(8, 6), dpi=100)
    args = {"projection": params["projection"]}
    if args["projection"] == "astro globe":
        args["center"] = map_struct["center"]
    ax = plt.axes([0.05, 0.05, 0.9, 0.9], **args)
    ax.imshow_hpx(hdu, field=columns.index("PROB"), cmap="cylon")
    ax.grid()

    for ii in range(len(coverage_struct["moc"])):
        moc = coverage_struct["moc"][ii]
        data = coverage_struct["data"][ii]

        moc.fill(
            ax=ax, wcs=ax.wcs, alpha=data[6], fill=True, color="black", linewidth=1
        )
        moc.border(ax=ax, wcs=ax.wcs, alpha=1, color="black")

    if plot_sun_moon:
        add_sun_moon(params, ax)

    plt.savefig(plot_name, dpi=200)
    plt.close()


def plot_tiles_coverage_int(
    params, map_struct, catalog_struct, coverage_struct, plot_sun_moon=False
):
    gpstime = params["gpstime"]
    event_mjd = Time(gpstime, format="gps", scale="utc").mjd

    colors = cm.rainbow(np.linspace(0, 1, len(params["telescopes"])))

    hdu = map_struct["hdu"]
    columns = [col.name for col in hdu.columns]

    plot_name = params["outputDir"].joinpath("tiles_coverage_int.pdf")
    fig = plt.figure(figsize=(12, 8))

    gs = fig.add_gridspec(4, 1)
    args = {"projection": params["projection"]}
    if args["projection"] == "astro globe":
        args["center"] = map_struct["center"]
    ax1 = fig.add_subplot(gs[0:3, 0], **args)
    ax2 = fig.add_subplot(gs[3, 0])
    ax3 = ax2.twinx()  # mirror them

    plt.axes(ax1)
    ax1.imshow_hpx(hdu, field=columns.index("PROB"), cmap="cylon")
    ax1.grid()
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
        for ii in range(len(coverage_struct["moc"])):
            moc = coverage_struct["moc"][ii]
            data = coverage_struct["data"][ii]
            moc.fill(
                ax=ax, wcs=ax.wcs, alpha=data[6], fill=True, color="black", linewidth=1
            )
            moc.border(ax=ax, wcs=ax.wcs, alpha=1, color="black")

    idxs = np.argsort(coverage_struct["data"][:, 2])
    plt.axes(ax2)
    for telescope, color in zip(params["telescopes"], colors):
        moc = None
        cum_prob = 0.0

        tts, cum_probs, cum_areas = [], [], []
        if params["tilesType"] == "galaxy":
            cum_galaxies = []

        for jj, ii in enumerate(idxs):
            if np.mod(jj, 100) == 0:
                print("%s: %d/%d" % (telescope, jj, len(idxs)))

            data = coverage_struct["data"][ii, :]
            m = coverage_struct["moc"][ii]
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
                if moc is None:
                    moc = m
                else:
                    moc = moc.union(m)

                cum_prob = moc.probability_in_multiordermap(map_struct["skymap"])
                cum_area = moc.sky_fraction * 360**2 / np.pi

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

    moc = None
    cum_prob = 0.0

    tts, cum_probs, cum_areas = [], [], []
    if params["tilesType"] == "galaxy":
        cum_galaxies = []

    for jj, ii in enumerate(idxs):
        data = coverage_struct["data"][ii, :]
        m = coverage_struct["moc"][ii]
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
            if moc is None:
                moc = m
            else:
                moc = moc.union(m)

            cum_prob = moc.probability_in_multiordermap(map_struct["skymap"])
            cum_area = moc.sky_fraction * 360**2 / np.pi

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
        add_sun_moon(params, ax)

    plt.savefig(plot_name, dpi=200)
    plt.close()


def plot_coverage_scaled(params, map_struct, coverage_struct, plot_sun_moon, max_time):
    plot_name = params["outputDir"].joinpath("tiles_coverage_scaled.pdf")

    hdu = map_struct["hdu"]
    columns = [col.name for col in hdu.columns]

    fig = plt.figure(figsize=(8, 6), dpi=100)
    args = {"projection": params["projection"]}
    if args["projection"] == "astro globe":
        args["center"] = map_struct["center"]
    ax = plt.axes([0.05, 0.05, 0.9, 0.9], **args)
    ax.imshow_hpx(hdu, field=columns.index("PROB"), cmap="cylon")
    ax.grid()

    for ii in range(len(coverage_struct["moc"])):
        moc = coverage_struct["moc"][ii]
        data = coverage_struct["data"][ii]

        alpha = data[4] / max_time
        if alpha > 1:
            alpha = 1.0

        moc.fill(ax=ax, wcs=ax.wcs, alpha=alpha, fill=True, color="black", linewidth=1)
        moc.border(ax=ax, wcs=ax.wcs, alpha=1, color="black")

    if plot_sun_moon:
        add_sun_moon(params, ax)

    plt.savefig(plot_name, dpi=200)
    plt.close()

    if params["doMovie"]:
        print("Creating movie from schedule...")

        idx = np.isfinite(coverage_struct["data"][:, 2])
        mjd_min = np.min(coverage_struct["data"][idx, 2])
        mjd_max = np.max(coverage_struct["data"][idx, 2])
        mjd_N = 100

        mjds = np.linspace(mjd_min, mjd_max, num=mjd_N)
        moviedir = params["outputDir"].joinpath("movie")
        moviedir.mkdir(exist_ok=True, parents=True)

        for jj in tqdm(range(len(mjds))):
            mjd = mjds[jj]
            plot_name = moviedir.joinpath(f"coverage-{jj:04d}.png")
            title = f"Coverage Map: {mjd:.2f}"

            fig = plt.figure(figsize=(8, 6), dpi=100)
            args = {"projection": params["projection"]}
            if args["projection"] == "astro globe":
                args["center"] = map_struct["center"]
            ax = plt.axes([0.05, 0.05, 0.9, 0.9], **args)
            ax.imshow_hpx(hdu, field=columns.index("PROB"), cmap="cylon")
            ax.grid()

            idx = np.where(coverage_struct["data"][:, 2] <= mjd)[0]
            for ii in idx:
                moc = coverage_struct["moc"][ii]
                data = coverage_struct["data"][ii]

                alpha = data[4] / max_time
                if alpha > 1:
                    alpha = 1.0

                moc.fill(
                    ax=ax,
                    wcs=ax.wcs,
                    alpha=alpha,
                    fill=True,
                    color="black",
                    linewidth=1,
                )

            plt.savefig(plot_name, dpi=200)
            plt.close()

        moviefiles = moviedir.joinpath("coverage-%04d.png")
        filename = params["outputDir"].joinpath("coverage.mpg")

        make_movie(moviefiles, filename)


def make_coverage_plots(
    params, map_struct, coverage_struct, catalog_struct=None, plot_sun_moon: bool = True
):

    idx = np.isfinite(coverage_struct["data"][:, 4])
    if not idx.size:
        return

    max_time = np.max(coverage_struct["data"][idx, 4])

    plot_tiles_coverage_int(
        params, map_struct, catalog_struct, coverage_struct, plot_sun_moon
    )

    plot_coverage_scaled(params, map_struct, coverage_struct, plot_sun_moon, max_time)
