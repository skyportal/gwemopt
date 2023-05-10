import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from gwemopt.plotting.style import CBAR_BOOL, UNIT, add_edges


def make_efficiency_plots(params, map_struct, efficiency_structs):
    plot_name = params["outputDir"].joinpath("efficiency.pdf")
    plt.figure()
    for key in efficiency_structs:
        efficiency_struct = efficiency_structs[key]
        plt.plot(
            efficiency_struct["distances"],
            efficiency_struct["efficiency"],
            label=efficiency_struct["legend_label"],
        )
    plt.xlabel("Distance [Mpc]")
    plt.ylabel("Efficiency")
    plt.legend(loc="best")
    plt.ylim([0.01, 1])
    plt.savefig(plot_name, dpi=200)
    plt.close()

    plot_name = params["outputDir"].joinpath("injs.pdf")
    plt.figure()
    plt.plot(efficiency_struct["ra"], efficiency_struct["dec"], "kx")
    plt.xlabel("RA [Degrees]")
    plt.ylabel("Declination [Degrees]")
    plt.savefig(plot_name, dpi=200)
    plt.close()

    plot_name = params["outputDir"].joinpath("mollview_injs.pdf")
    plt.figure()
    hp.mollview(map_struct["prob"], unit=UNIT, cbar=CBAR_BOOL)
    hp.projplot(
        efficiency_struct["ra"], efficiency_struct["dec"], "wx", lonlat=True, coord="G"
    )
    add_edges()
    plt.savefig(plot_name, dpi=200)
    plt.close()

    if params["do_3d"]:
        for key in efficiency_structs:
            efficiency_struct = efficiency_structs[key]
            plot_name = params["outputDir"].joinpath(
                f'3Deff_distbins_{efficiency_struct["legend_label"]}.pdf',
            )
            plt.figure()

            injected_KNe, recovered_KNe = (
                efficiency_struct["dists_inj"]["tot"],
                efficiency_struct["dists_inj"]["recovered"],
            )
            plt.hist(
                injected_KNe,
                bins=10,
                range=(np.nanmin(injected_KNe), np.nanmax(injected_KNe)),
                color="r",
                label="Injected KNe",
                histtype="step",
                linewidth=2,
            )
            plt.hist(
                recovered_KNe,
                bins=10,
                range=(np.nanmin(injected_KNe), np.nanmax(injected_KNe)),
                label="Recovered KNe",
                color="g",
                histtype="step",
                linewidth=2,
            )

            plt.legend(
                loc="upper left", fancybox=True, edgecolor="k", prop={"size": 10}
            )
            plt.xlabel("Dist (Mpc)", fontsize=15)
            plt.savefig(plot_name, dpi=200)
            plt.close()
