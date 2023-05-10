import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from gwemopt.plotting.movie import make_movie
from gwemopt.plotting.style import CBAR_BOOL, UNIT, add_edges, cmap


def plot_observability(params, map_struct):
    """
    Function to plot the observability
    """
    observability_struct = map_struct["observability"]

    for telescope in observability_struct.keys():
        plot_name = params["outputDir"].joinpath(f"observability_{telescope}.pdf")
        hp.mollview(
            map_struct["prob"] * observability_struct[telescope]["observability"],
            title="",
            unit=UNIT,
            cbar=CBAR_BOOL,
            min=np.min(map_struct["prob"]),
            max=np.max(map_struct["prob"]),
            cmap=cmap,
        )
        add_edges()
        plt.show()
        plt.savefig(plot_name, dpi=200)
        plt.close("all")

    if params["doMovie"]:
        moviedir = params["outputDir"].joinpath("movie")
        moviedir.mkdir(parents=True, exist_ok=True)

        for telescope in observability_struct.keys():
            dts = list(observability_struct[telescope]["dts"].keys())
            dts = np.sort(dts)
            for ii, dt in enumerate(dts):
                plot_name = moviedir.joinpath(f"observability-{ii:04d}.png")
                title = f"Detectability Map: {dt:.2f} Days"
                hp.mollview(
                    map_struct["prob"] * observability_struct[telescope]["dts"][dt],
                    title=title,
                    cbar=CBAR_BOOL,
                    min=np.min(map_struct["prob"]),
                    max=np.max(map_struct["prob"]),
                    cmap=cmap,
                )
                add_edges()
                plt.show()
                plt.savefig(plot_name, dpi=200)
                plt.close("all")

            moviefiles = moviedir.joinpath("observability-%04d.png")
            filename = params["outputDir"].joinpath(f"observability_{telescope}.mpg")

            make_movie(moviefiles, filename)
