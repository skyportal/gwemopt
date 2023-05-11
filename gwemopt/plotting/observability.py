import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from gwemopt.plotting.movie import make_movie
from gwemopt.plotting.style import CBAR_BOOL, UNIT, add_edges, cmap


def plot_observability(params, map_struct):
    """
    Function to plot the observability
    """
    observability_struct = map_struct["observability"]

    for telescope in observability_struct.keys():
        plot_name = params["outputDir"].joinpath(f"observable_area_{telescope}.pdf")

        vals = map_struct["prob"] * observability_struct[telescope]["observability"]
        vals[~observability_struct[telescope]["observability"].astype(bool)] = np.nan

        hp.mollview(
            vals,
            title=f"Observable Area - {telescope} (integrated)",
            unit=UNIT,
            cbar=CBAR_BOOL,
            min=np.min(map_struct["prob"]),
            max=np.max(map_struct["prob"]),
            cmap=cmap,
        )
        add_edges()
        print(f"Saving to {plot_name}")
        plt.savefig(plot_name, dpi=200)
        plt.close()

    if params["doMovie"]:
        moviedir = params["outputDir"].joinpath("movie")
        moviedir.mkdir(parents=True, exist_ok=True)

        for telescope in observability_struct.keys():
            dts = list(observability_struct[telescope]["dts"].keys())
            dts = np.sort(dts)

            for ii, dt in tqdm(enumerate(dts), total=len(dts)):
                plot_name = moviedir.joinpath(f"observability-{ii:04d}.png")
                title = f"Observability Map: {dt:.2f} Days"

                vals = map_struct["prob"] * observability_struct[telescope]["dts"][dt]
                vals[~observability_struct[telescope]["dts"][dt].astype(bool)] = np.nan

                hp.mollview(
                    vals,
                    title=title,
                    cbar=CBAR_BOOL,
                    unit=UNIT,
                    min=np.min(map_struct["prob"]),
                    max=np.max(map_struct["prob"]),
                    cmap=cmap,
                )

                add_edges()
                plt.savefig(plot_name, dpi=200)
                plt.close()

            moviefiles = moviedir.joinpath("observability-%04d.png")
            filename = params["outputDir"].joinpath(
                f"observability_timelapse_{telescope}.mpg"
            )

            make_movie(moviefiles, filename)
