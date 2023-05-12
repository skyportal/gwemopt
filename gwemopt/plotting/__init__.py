import os

import matplotlib.pyplot as plt

from gwemopt.plotting.movie import make_movie
from gwemopt.plotting.observability import plot_observability
from gwemopt.plotting.plot_coverage import make_coverage_plots
from gwemopt.plotting.plot_efficiency import make_efficiency_plots
from gwemopt.plotting.plot_schedule import make_schedule_plots
from gwemopt.plotting.plot_skymap import plot_skymap
from gwemopt.plotting.plot_tiles import make_tile_plots
from gwemopt.plotting.style import add_edges, cmap


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
