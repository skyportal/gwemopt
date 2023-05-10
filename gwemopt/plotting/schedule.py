import matplotlib.pyplot as plt
import numpy as np


def make_schedule_plots(params, exposurelist, keys):
    plot_name = params["outputDir"].joinpath("scheduler.pdf")
    if params["scheduleType"].endswith("_slew"):
        e = []
        k = []
        start_time = exposurelist[0][0]
        for exposure, key in zip(exposurelist, keys):
            e.append((exposure[0] - start_time) * 24)
            k.append(key)
            e.append((exposure[1] - start_time) * 24)
            k.append(key)
        plt.figure()
        plt.grid()
        plt.xlabel("Time (h)")
        plt.ylabel("Tile Number")
        plt.plot(e, k, "b-")
    else:
        xs = []
        ys = []
        for ii, key in zip(np.arange(len(exposurelist)), keys):
            xs.append(ii)
            ys.append(key)
        plt.figure()
        plt.xlabel("Exposure Number")
        plt.ylabel("Tile Number")
        plt.plot(xs, ys, "kx")
    plt.savefig(plot_name, dpi=200)
    plt.close()
