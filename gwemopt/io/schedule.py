import numpy as np
import pandas as pd
from astropy.time import Time

from gwemopt.scheduler import computeSlewReadoutTime
from gwemopt.utils import angular_distance


def read_summary(summary_path):
    """
    Reads a summary file and returns a pandas dataframe

    :param summary_path: path to summary file
    :return: pandas dataframe
    """

    summary = pd.read_csv(
        summary_path,
        index_col=False,
        sep=",",
        names=[
            "time",
            "dt",
            "prob",
            "area",
            "mjds",
        ],
    )

    return summary


def read_schedule(schedule_path):
    """
    Reads a schedule file and returns a pandas dataframe

    :param schedule_path: path to schedule file
    :return: pandas dataframe
    """

    schedule = pd.read_csv(
        schedule_path,
        index_col=False,
        sep=" ",
        names=[
            "field",
            "ra",
            "dec",
            "tobs",
            "limmag",
            "texp",
            "prob",
            "airmass",
            "filter",
        ],
    )

    return schedule


def summary(params, map_struct, coverage_struct, catalog_struct=None):
    filts = list(set(coverage_struct["filters"]))
    for jj, telescope in enumerate(params["telescopes"]):
        schedulefile = params["outputDir"].joinpath(f"schedule_{telescope}.dat")
        schedulexmlfile = params["outputDir"].joinpath(f"schedule_{telescope}.xml")

        config_struct = params["config"][telescope]

        if (params["tilesType"] == "hierarchical") or (params["tilesType"] == "greedy"):
            fields = np.zeros((params["Ntiles"][jj], len(filts) + 2))
        else:
            fields = np.zeros((len(config_struct["tesselation"]), len(filts) + 2))

        totexp = 0

        with open(schedulefile, "w") as fid:
            for ii in range(len(coverage_struct["filters"])):
                if not telescope == coverage_struct["telescope"][ii]:
                    continue

                data = coverage_struct["data"][ii, :]
                filt = coverage_struct["filters"][ii]

                ra, dec = data[0], data[1]
                observ_time, mag, exposure_time, field_id, prob, airmass = (
                    data[2],
                    data[3],
                    data[4],
                    data[5],
                    data[6],
                    data[7],
                )

                if params["tilesType"] == "galaxy":
                    galaxies = coverage_struct["galaxies"][ii]
                    prob = np.sum(catalog_struct[params["galaxy_grade"]][galaxies])

                fid.write(
                    "%d %.5f %.5f %.5f %.5f %d %.5f %.5f %s \n"
                    % (
                        field_id,
                        ra,
                        dec,
                        observ_time,
                        mag,
                        exposure_time,
                        prob,
                        airmass,
                        filt,
                    )
                )

                dist = angular_distance(
                    data[0],
                    data[1],
                    config_struct["tesselation"][:, 1],
                    config_struct["tesselation"][:, 2],
                )
                idx1 = np.argmin(dist)
                idx2 = filts.index(filt)
                fields[idx1, 0] = config_struct["tesselation"][idx1, 0]
                fields[idx1, 1] = prob
                fields[idx1, idx2 + 2] = fields[idx1, idx2 + 2] + 1

                totexp = totexp + exposure_time

        idx = np.where(fields[:, 1] > 0)[0]
        fields = fields[idx, :]
        idx = np.argsort(fields[:, 1])[::-1]
        fields = fields[idx, :]

        fields_sum = np.sum(fields[:, 2:], axis=1)
        idx = np.where(fields_sum >= 2)[0]
        print("%d/%d fields were observed at least twice" % (len(idx), len(fields_sum)))
        print(f"Expected time spent on exposures: {totexp / 3600:.1f} hr.")
        slew_readout_time = computeSlewReadoutTime(config_struct, coverage_struct)
        print(f"Expected time spent on slewing and readout: {slew_readout_time:.0f} s.")

        coveragefile = params["outputDir"].joinpath(f"coverage_{telescope}.dat")
        with open(coveragefile, "w") as fid:
            for field in fields:
                fid.write("%d %.10f " % (field[0], field[1]))
                for ii in range(len(filts)):
                    fid.write("%d " % (field[2 + ii]))
                fid.write("\n")

    summaryfile = params["outputDir"].joinpath("summary.dat")
    cummoc = None

    with open(summaryfile, "w") as fid:
        gpstime = params["gpstime"]
        event_mjd = Time(gpstime, format="gps", scale="utc").mjd

        tts = np.array([1, 7, 60])
        for tt in tts:
            mjds_floor = []
            mjds = []
            cum_prob = 0.0
            cum_area = 0.0

            if params["tilesType"] == "galaxy":
                galaxies = np.empty((0, 2))

            for ii in range(len(coverage_struct["filters"])):
                data = coverage_struct["data"][ii, :]
                filt = coverage_struct["filters"][ii]
                moc = coverage_struct["moc"][ii]

                ra, dec = data[0], data[1]
                observ_time, mag, exposure_time, field_id, prob, airmass = (
                    data[2],
                    data[3],
                    data[4],
                    data[5],
                    data[6],
                    data[7],
                )

                if data[2] > event_mjd + tt:
                    continue

                if cummoc is None:
                    cummoc = moc
                else:
                    cummoc = cummoc + moc
                cum_prob = cummoc.probability_in_multiordermap(map_struct["skymap"])

                if params["tilesType"] == "galaxy":
                    galaxies = np.append(galaxies, coverage_struct["galaxies"][ii])
                    galaxies = np.unique(galaxies).astype(int)
                    cum_prob = np.sum(catalog_struct[params["galaxy_grade"]][galaxies])

                cum_area = cummoc.sky_fraction * 360**2 / np.pi
                mjds.append(data[2])
                mjds_floor.append(int(np.floor(data[2])))

            if len(mjds_floor) == 0:
                print("No images after %.1f days..." % tt)
                fid.write("%.1f,-1,-1,-1,-1\n" % (tt))
            else:
                mjds = np.unique(mjds)
                mjds_floor = np.unique(mjds_floor)

                print("After %.1f days..." % tt)
                print(
                    "Number of hours after first image: %.5f"
                    % (24 * (np.min(mjds) - event_mjd))
                )
                print("MJDs covered: %s" % (" ".join(str(x) for x in mjds_floor)))
                print("Cumultative probability: %.5f" % cum_prob)
                print("Cumultative area: %.5f degrees" % cum_area)

                fid.write(
                    "%.1f,%.5f,%.5f,%.5f,%s\n"
                    % (
                        tt,
                        24 * (np.min(mjds) - event_mjd),
                        cum_prob,
                        cum_area,
                        " ".join(str(x) for x in mjds_floor),
                    )
                )
