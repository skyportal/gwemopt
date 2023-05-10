import copy
import os

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from gwemopt.plotting.coverage import make_coverage_plots
from gwemopt.plotting.efficiency import make_efficiency_plots
from gwemopt.plotting.movie import make_movie
from gwemopt.plotting.observability import plot_observability
from gwemopt.plotting.plot_skymap import plot_skymap
from gwemopt.plotting.schedule import make_schedule_plots
from gwemopt.plotting.style import add_edges, cmap
from gwemopt.plotting.tiles import make_tile_plots


def tauprob(params, tau, prob):
    plotName = os.path.join(params["outputDir"], "tau_prob.pdf")
    plt.figure()
    plt.plot(tau, prob)
    plt.grid()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Log of observing time $\tau$", fontsize=20)
    plt.ylabel(
        "Log of detection prob. given the target is at the observing field", fontsize=20
    )
    plt.savefig(plotName, dpi=200)
    plt.close()


def doMovie_supersched(params, coverage_structs, tile_structs, map_struct):
    unit = "Gravitational-wave probability"
    cbar = False

    idx = np.isfinite(coverage_structs["data"][:, 2])
    try:
        mjd_min = np.min(coverage_structs["data"][idx, 2])
    except:
        return
    mjd_max = np.max(coverage_structs["data"][idx, 2])
    mjd_N = 100
    Tobs = list(params["Tobs_all"])
    Tobs = np.linspace(Tobs[0], Tobs[1], params["Tobs_split"] + 1)
    mjds = np.linspace(mjd_min, mjd_max, num=mjd_N)

    parentdir = os.path.abspath(os.path.join(params["outputDir"], os.pardir))
    moviedir = os.path.join(parentdir, "movie")  # saves movie file in parent directory

    if not os.path.isdir(moviedir):
        os.mkdir(moviedir)

    # for jj in range(len(coverage_struct["ipix"])):
    #    mjd = coverage_struct["data"][jj,3]

    for jj in range(len(mjds)):
        mjd = mjds[jj]
        for i in range(
            len(Tobs)
        ):  # can change this to enumerate(Tobs) if Tobs is variable
            ii = jj + (100 * i)
            if not os.path.exists(
                os.path.join(moviedir, f"coverage-{ii:04d}.png")
            ):  # adds multiples of 100 for each round of Tobs
                plotName = os.path.join(moviedir, f"coverage-{ii:04d}.png")
                break

        title = "Coverage Map: %.2f" % mjd
        plt.figure()
        hp.mollview(map_struct["prob"], title=title, unit=unit, cbar=cbar, cmap=cmap)
        add_edges()
        ax = plt.gca()

        idx = np.where(coverage_structs["data"][:, 2] <= mjd)[0]

        # for ii in range(jj):
        for ii in idx:
            data = coverage_structs["data"][ii, :]
            filt = coverage_structs["filters"][ii]
            ipix = coverage_structs["ipix"][ii]
            patch = coverage_structs["patch"][ii]
            FOV = coverage_structs["FOV"][ii]

            if patch == []:
                continue

            # hp.visufunc.projplot(corners[:,0], corners[:,1], 'k', lonlat = True)
            patch_cpy = copy.copy(patch)
            patch_cpy.axes = None
            patch_cpy.figure = None
            patch_cpy.set_transform(ax.transData)

            # alpha = data[4]/max_time
            # if alpha > 1:
            #    alpha = 1.0
            # patch_cpy.set_alpha(alpha)
            hp.projaxes.HpxMollweideAxes.add_patch(ax, patch_cpy)
        # tiles.plot()

        if len(params["coverage_structs"]) == 1:
            continue
        else:
            i = len(params["coverage_structs"]) - 1  # finds out which round we are in

        while i > 0:  # goes through all previous rounds
            readfile = os.path.join(parentdir, f"{Tobs[i-1]:.2}_to_{Tobs[i]:.2}_Tobs")
            prevtelescopes = params["alltelescopes"][i - 1].split(",")
            prev_tile_structs = params["tile_structs"][f"tile_structs_{i-1}"]
            i -= 1
            for prevtelescope in prevtelescopes:
                if prevtelescope not in prev_tile_structs:
                    continue
                prev_tile_struct = prev_tile_structs[prevtelescope]
                schedfile = f"schedule_{prevtelescope}.dat"
                data_file = os.path.join(readfile, schedfile)

                with open(data_file, "r") as f:
                    for line in f:
                        data = list(line.split(" "))
                        field_id = int(data[0])
                        if int(data[8]) == 1:
                            patch = prev_tile_struct[field_id]["patch"]
                            if patch == []:
                                continue
                            patch_cpy = copy.copy(patch)
                            patch_cpy.axes = None
                            patch_cpy.figure = None
                            patch_cpy.set_transform(ax.transData)
                            patch_cpy.set_facecolor("white")
                            hp.projaxes.HpxMollweideAxes.add_patch(ax, patch_cpy)

        plt.savefig(plotName, dpi=200)
        plt.close()
