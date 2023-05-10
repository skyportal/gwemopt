def export_tiles_coverage_hist(params, diffs):
    """
    Export the histogram of the tiles coverage.
    """
    filename = params["outputDir"].joinpath("tiles_coverage_hist.dat")
    with open(filename, "w") as fid:
        for ii in range(len(diffs)):
            fid.write("%.10f\n" % diffs[ii])


def export_tiles_coverage_int(filename, tts, cum_probs, cum_areas):
    with open(filename, "w") as fid:
        for ii in range(len(tts)):
            fid.write("%.10f %.10e %.10f\n" % (tts[ii], cum_probs[ii], cum_areas[ii]))

    print(
        f"Total Cumulative Probability: {100.*cum_probs[-1]:.1f}% \n"
        f"Total Cumulative Area: {cum_areas[-1]:.1f} sq. deg."
    )
