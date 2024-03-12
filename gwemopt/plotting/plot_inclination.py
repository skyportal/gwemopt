import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def density_2d(cosine_iota, distance, map_struct):
    return (
        map_struct["prob_density_interp"](cosine_iota)
        * map_struct["dist_norm_interp"](cosine_iota)
        * distance**2
        * norm(
            loc=map_struct["dist_mu_interp"](cosine_iota),
            scale=map_struct["dist_sigma_interp"](cosine_iota),
        ).pdf(distance)
    )


def plot_inclination(params, map_struct):
    """
    Function to plot the inclination dependence
    """

    plot_name = params["outputDir"].joinpath("inclination.pdf")

    # we evaluate the probability density on a grid of iota and distance
    cosine_iota_dense = np.linspace(-1, 1, 1000)
    distance_dense = np.linspace(0, 250, 1000)

    COS, DIST = np.meshgrid(cosine_iota_dense, distance_dense)

    plt.figure()
    plt.xlabel("cos(Inclination")
    plt.ylabel("Distance [Mpc]")
    plt.contourf(COS, DIST, density_2d(COS, DIST, map_struct), cmap="Blues", levels=100)
    if params["true_location"]:
        plt.plot(
            np.cos(np.deg2rad(params["true_inclination"])),
            params["true_distance"],
            "x",
            color="black",
            markersize=10,
        )
    plt.savefig(plot_name, dpi=200)
    plt.close()

    plot_name = params["outputDir"].joinpath("inclination_marginal.pdf")

    plt.figure()
    plt.xlabel("Inclination")
    plt.ylabel("PDF")
    plt.plot(map_struct["iota_EM"], map_struct["prob_iota_EM"])
    if params["true_location"]:
        if params["true_inclination"] > 90:
            iota_EM = params["true_inclination"] - 90.0
        else:
            iota_EM = params["true_inclination"]
        plt.axvline(x=np.cos(np.deg2rad(iota_EM)), linestyle="--", color="black")

    plt.savefig(plot_name, dpi=200)
    plt.close()
