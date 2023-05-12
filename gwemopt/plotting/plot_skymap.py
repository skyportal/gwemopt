import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from gwemopt.plotting.style import CBAR_BOOL, UNIT, add_edges, cmap


def plot_skymap(params, map_struct):
    """
    Function to plot the skymap
    """

    plot_name = params["outputDir"].joinpath("prob.pdf")

    if np.percentile(map_struct["prob"], 99) > 0:
        hp.mollview(
            map_struct["prob"],
            title="",
            unit=UNIT,
            cbar=CBAR_BOOL,
            min=np.percentile(map_struct["prob"], 1),
            max=np.percentile(map_struct["prob"], 99),
            cmap=cmap,
        )
    else:
        hp.mollview(
            map_struct["prob"],
            title="",
            unit=UNIT,
            cbar=CBAR_BOOL,
            min=np.percentile(map_struct["prob"], 1),
            cmap=cmap,
        )

    add_edges()
    plt.savefig(plot_name, dpi=200)
    plt.close()

    if "distmu" in map_struct:
        fin = np.copy(map_struct["distmu"])
        fin[~np.isfinite(fin)] = np.nan
        plot_name = params["outputDir"].joinpath("dist.pdf")
        hp.mollview(
            map_struct["distmu"],
            unit="Distance [Mpc]",
            min=np.nanpercentile(fin, 10),
            max=np.nanpercentile(fin, 90),
        )
        add_edges()
        plt.savefig(plot_name, dpi=200)
        plt.close()

        fin = np.copy(map_struct["distmed"])
        fin[~np.isfinite(fin)] = np.nan
        plot_name = params["outputDir"].joinpath("dist_median.pdf")
        hp.mollview(
            map_struct["distmed"],
            unit="Distance [Mpc]",
            min=np.nanpercentile(fin, 10),
            max=np.nanpercentile(fin, 90),
        )
        add_edges()
        plt.savefig(plot_name, dpi=200)
        plt.close()
