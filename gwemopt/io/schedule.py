import numpy as np
import pandas as pd
from astropy.time import Time
from VOEventLib.VOEvent import Field, Table, What
from VOEventLib.Vutil import stringVOEvent, utilityTable

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


def export_schedule_xml(xmlfile, map_struct, coverage_struct, config_struct):
    what = What()

    table = Table(name="data", Description=["The datas of GWAlert"])
    table.add_Field(
        Field(
            name=r"grid_id",
            ucd="",
            unit="",
            dataType="int",
            Description=["ID of the grid of fov"],
        )
    )
    table.add_Field(
        Field(
            name="field_id",
            ucd="",
            unit="",
            dataType="int",
            Description=["ID of the filed"],
        )
    )
    table.add_Field(
        Field(
            name=r"ra",
            ucd=r"pos.eq.ra ",
            unit="deg",
            dataType="float",
            Description=[
                "The right ascension at center of fov in equatorial coordinates"
            ],
        )
    )
    table.add_Field(
        Field(
            name="dec",
            ucd="pos.eq.dec ",
            unit="deg",
            dataType="float",
            Description=["The declination at center of fov in equatorial coordinates"],
        )
    )
    table.add_Field(
        Field(
            name="ra_width",
            ucd=" ",
            unit="deg",
            dataType="float",
            Description=["Width in RA of the fov"],
        )
    )
    table.add_Field(
        Field(
            name="dec_width",
            ucd="",
            unit="deg",
            dataType="float",
            Description=["Width in Dec of the fov"],
        )
    )
    table.add_Field(
        Field(
            name="prob_sum",
            ucd="",
            unit="None",
            dataType="float",
            Description=["The sum of all pixels in the fov"],
        )
    )
    table.add_Field(
        Field(
            name="observ_time",
            ucd="",
            unit="sec",
            dataType="float",
            Description=["Tile mid. observation time in MJD"],
        )
    )
    table.add_Field(
        Field(
            name="airmass",
            ucd="",
            unit="None",
            dataType="float",
            Description=["Airmass of tile at mid. observation time"],
        )
    )
    table.add_Field(
        Field(name="priority", ucd="", unit="", dataType="int", Description=[""])
    )
    table_field = utilityTable(table)
    table_field.blankTable(len(coverage_struct["ipix"]))

    for ii in range(len(coverage_struct["ipix"])):
        data = coverage_struct["data"][ii, :]
        ipix = coverage_struct["ipix"][ii]

        prob = np.sum(map_struct["prob"][ipix])

        ra, dec = data[0], data[1]
        observ_time, exposure_time, field_id, prob, airmass = (
            data[2],
            data[4],
            data[5],
            data[6],
            data[7],
        )

        table_field.setValue("grid_id", ii, 0)
        table_field.setValue("field_id", ii, field_id)
        table_field.setValue("ra", ii, ra)
        table_field.setValue("dec", ii, dec)
        table_field.setValue("ra_width", ii, config_struct["FOV"])
        table_field.setValue("dec_width", ii, config_struct["FOV"])
        table_field.setValue("observ_time", ii, observ_time)
        table_field.setValue("airmass", ii, airmass)
        table_field.setValue("prob_sum", ii, prob)
        table_field.setValue("priority", ii, ii)

    table = table_field.getTable()
    what.add_Table(table)
    xml = stringVOEvent(what)
    lines = xml.splitlines()
    linesrep = []
    for line in lines:
        linenew = (
            line.replace(">b'", ">")
            .replace("'</", "</")
            .replace("=b'", "=")
            .replace("'>", ">")
        )
        linesrep.append(linenew)
    xmlnew = "\n".join(linesrep)
    with open(xmlfile, "w") as fid:
        fid.write(xmlnew)


def summary(params, map_struct, coverage_struct, catalog_struct=None):
    idx50 = len(map_struct["cumprob"]) - np.argmin(np.abs(map_struct["cumprob"] - 0.50))
    idx90 = len(map_struct["cumprob"]) - np.argmin(np.abs(map_struct["cumprob"] - 0.90))

    mapfile = params["outputDir"].joinpath("map.dat")
    with open(mapfile, "w") as fid:
        fid.write(
            "%.5f %.5f\n"
            % (map_struct["pixarea_deg2"] * idx50, map_struct["pixarea_deg2"] * idx90)
        )

    filts = list(set(coverage_struct["filters"]))
    for jj, telescope in enumerate(params["telescopes"]):
        schedulefile = params["outputDir"].joinpath(f"schedule_{telescope}.dat")
        schedulexmlfile = params["outputDir"].joinpath(f"schedule_{telescope}.xml")

        config_struct = params["config"][telescope]

        export_schedule_xml(schedulexmlfile, map_struct, coverage_struct, config_struct)

        if (params["tilesType"] == "hierarchical") or (params["tilesType"] == "greedy"):
            fields = np.zeros((params["Ntiles"][jj], len(filts) + 2))
        else:
            fields = np.zeros((len(config_struct["tesselation"]), len(filts) + 2))

        totexp = 0

        with open(schedulefile, "w") as fid:
            for ii in range(len(coverage_struct["ipix"])):
                if not telescope == coverage_struct["telescope"][ii]:
                    continue

                data = coverage_struct["data"][ii, :]
                filt = coverage_struct["filters"][ii]
                ipix = coverage_struct["ipix"][ii]

                prob = np.sum(map_struct["prob"][ipix])

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
        print(
            "%d/%d fields were observed at least twice\n" % (len(idx), len(fields_sum))
        )

        print(
            "Integrated probability, All: %.5f, 2+: %.5f"
            % (np.sum(fields[:, 1]), np.sum(fields[idx, 1]))
        )

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
    with open(summaryfile, "w") as fid:
        gpstime = params["gpstime"]
        event_mjd = Time(gpstime, format="gps", scale="utc").mjd

        tts = np.array([1, 7, 60])
        for tt in tts:
            mjds_floor = []
            mjds = []
            ipixs = np.empty((0, 2))
            cum_prob = 0.0
            cum_area = 0.0

            if params["tilesType"] == "galaxy":
                galaxies = np.empty((0, 2))

            for ii in range(len(coverage_struct["ipix"])):
                data = coverage_struct["data"][ii, :]
                filt = coverage_struct["filters"][ii]
                ipix = coverage_struct["ipix"][ii]

                prob = np.sum(map_struct["prob"][ipix])

                if data[2] > event_mjd + tt:
                    continue

                ipixs = np.append(ipixs, ipix)
                ipixs = np.unique(ipixs).astype(int)
                cum_prob = np.sum(map_struct["prob"][ipixs])

                if params["tilesType"] == "galaxy":
                    galaxies = np.append(galaxies, coverage_struct["galaxies"][ii])
                    galaxies = np.unique(galaxies).astype(int)
                    cum_prob = np.sum(catalog_struct[params["galaxy_grade"]][galaxies])

                cum_area = len(ipixs) * map_struct["pixarea_deg2"]
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
