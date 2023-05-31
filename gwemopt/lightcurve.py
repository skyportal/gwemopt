import numpy as np


def tophat(mag0=0.0, dmag=0.0):
    ntime, t_i, t_f = 30, 0.00, 15.25
    phase = np.linspace(t_i, t_f, ntime)  # epochs

    filters = ["u", "g", "r", "i", "z", "y", "J", "H", "K"]
    mags = {}
    mags["tophat"] = {}
    mags["tophat"]["name"] = "tophat"
    mags["tophat"]["legend_label"] = "Tophat: %.3f, %.3f" % (mag0, dmag)
    mags["tophat"]["t"] = phase

    for filt in filters:
        mag = mag0 + phase * dmag
        mags["tophat"][filt] = mag

    return mags


def read_files(files, tmin=-100.0, tmax=100.0):
    mags = {}
    for filename in files:
        name = filename.replace(".txt", "").replace(".dat", "").split("/")[-1]

        if "neutron_precursor" in name:
            legend_label = "Barnes et al. (2016)"
        elif "rpft" in name:
            legend_label = "Metzger et al. (2015)"
        elif "BHNS" in name:
            legend_label = "Kawaguchi et al. (2016)"
        elif "BNS" in name:
            legend_label = "Dietrich et al. (2016)"
        elif "k1" in name:
            legend_label = "Tanaka and Hotokezaka (2013)"
        else:
            legend_label = name

        mag_d = np.loadtxt(filename)

        bands = ["u", "g", "r", "i", "z", "y", "J", "H", "K"]

        mags[name] = {}
        mags[name]["t"] = mag_d[:, 0]
        indexes1 = np.where(mags[name]["t"] >= tmin)[0]
        indexes2 = np.where(mags[name]["t"] <= tmax)[0]
        indexes = np.intersect1d(indexes1, indexes2)

        mags[name]["t"] = mag_d[indexes, 0]

        for ii, filt in enumerate(bands):
            mags[name][filt] = mag_d[indexes, ii + 1]

        mags[name]["c"] = (mags[name]["g"] + mags[name]["r"]) / 2.0
        mags[name]["o"] = (mags[name]["r"] + mags[name]["i"]) / 2.0

        mags[name]["name"] = name
        mags[name]["legend_label"] = legend_label

    return mags
