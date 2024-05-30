import healpy as hp
import ligo.skymap.plot
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from ligo.skymap import moc
from matplotlib import pyplot as plt


def plot_skymap(params, map_struct, colnames=["PROB", "DISTMEAN", "DISTSTD"]):
    """
    Function to plot the skymap
    """

    plot_name = params["outputDir"].joinpath("prob.pdf")

    hdu = map_struct["hdu"]
    columns = [col.name for col in hdu.columns]

    for col in colnames:
        if col in columns:

            args = {"projection": params["projection"]}
            if args["projection"] == "astro globe":
                args["center"] = map_struct["center"]

            fig = plt.figure(figsize=(8, 6), dpi=100)
            ax = plt.axes(
                [0.05, 0.05, 0.9, 0.9],
                **args,
            )
            ax.imshow_hpx(hdu, field=columns.index(col), cmap="cylon")
            ax.grid()
            plot_name = params["outputDir"].joinpath(f"{col.lower()}.pdf")
            plt.savefig(plot_name, dpi=200)
            plt.close()
