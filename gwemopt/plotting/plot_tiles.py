import copy

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from gwemopt.plotting.style import CBAR_BOOL, UNIT, add_edges, add_sun_moon, cmap


def make_tile_plots(params, map_struct, tiles_structs, plot_sun_moon=True):
    """
    Function to plot the tiles
    """

    plot_name = params["outputDir"].joinpath("tiles.pdf")

    hdu = map_struct["hdu"]
    columns = [col.name for col in hdu.columns]

    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.axes([0.05, 0.05, 0.9, 0.9], projection="astro mollweide")
    ax.imshow_hpx(hdu, field=columns.index("PROB"), cmap="cylon")
    for telescope in tiles_structs:
        tiles_struct = tiles_structs[telescope]
        keys = list(tiles_struct.keys())
        probs = np.array([tiles_struct[key]["prob"] for key in keys])
        alphas = probs / np.max(probs)

        for ii, index in tqdm(enumerate(keys), total=len(keys)):
            moc = tiles_struct[index]["moc"]
            moc.fill(
                ax=ax,
                wcs=ax.wcs,
                alpha=alphas[ii],
                fill=True,
                color="black",
                linewidth=1,
            )
            moc.border(ax=ax, wcs=ax.wcs, alpha=1, color="black")

    if plot_sun_moon:
        add_sun_moon(params, ax)

    plt.savefig(plot_name, dpi=200)
    plt.close()
