import os


def export_efficiency_data(params, efficiency_struct, lightcurve_struct):
    """
    Export the efficiency data to a file.

    :param params: The parameters of the run.
    :param efficiency_struct: The efficiency data.
    :param lightcurve_struct: The lightcurve data.
    :return: None
    """
    filename = params["outputDir"].joinpath(
        "efficiency_" + lightcurve_struct["name"] + ".txt"
    )

    append_write = ["w", "a"][filename.exists()]

    with open(filename, append_write) as f:
        if append_write == "w":
            f.write("Distance" + "\t" + "efficiency\n")
        for i in range(0, len(efficiency_struct["distances"])):
            dist = efficiency_struct["distances"][i]
            eff = efficiency_struct["efficiency"][i]
            f.write(str(dist) + "\t" + str(eff) + "\n")


def save_efficiency_metric(
    params, efficiency_filename, efficiency_metric, lightcurve_struct
):
    """
    Save the efficiency metric to a file.

    :param params: The parameters of the run.
    :param efficiency_filename: The name of the file to save the efficiency metric to.
    :param efficiency_metric: The efficiency metric.
    :param lightcurve_struct: The lightcurve data.
    :return: None
    """

    append_write = ["w", "a"][os.path.exists(efficiency_filename)]

    with open(efficiency_filename, append_write) as f:
        if append_write == "w":
            f.write(
                "tilesType\t"
                + "timeallocationType\t"
                + "scheduleType\t"
                + "Ntiles\t"
                + "efficiencyMetric\t"
                + "efficiencyMetric_err\t"
                + "injection\n"
            )
        f.write(
            params["tilesType"]
            + "\t"
            + params["timeallocationType"]
            + "\t"
            + params["scheduleType"]
            + "\t"
            + str(params["Ntiles"])
            + "\t"
            + str(efficiency_metric[0])
            + "\t"
            + str(efficiency_metric[1])
            + "\t"
            + lightcurve_struct["name"]
            + "\n"
        )
