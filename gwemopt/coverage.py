
import numpy as np

def read_coverage(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    lines = lines[1:]
    lines = filter(None,lines)

    coverage_struct = {}
    coverage_struct["data"] = np.empty((0,4))
    coverage_struct["filters"] = []
    for line in lines:
        lineSplit = line.split(",")
        ra = float(lineSplit[2])
        dec = float(lineSplit[3])
        mjd = float(lineSplit[4])
        filt = lineSplit[6]
        mag = float(lineSplit[7])

        coverage_struct["data"] = np.append(coverage_struct["data"],np.array([[ra,dec,mjd,mag]]),axis=0)
        coverage_struct["filters"].append(filt)
    coverage_struct["filters"] = np.array(coverage_struct["filters"])

    return coverage_struct

def read_files(params):

    coverage_structs = []
    for telescope, dataFile in zip(params["telescopes"],params["dataFiles"]):
        coverage_struct = read_coverage(dataFile)
        
        coverage_struct["FOV"] = params["config"][telescope]["FOV"]*np.ones((len(coverage_struct["filters"]),))
        coverage_structs.append(coverage_struct)

    coverage_struct_combined = {}
    coverage_struct_combined["data"] = np.empty((0,4))
    coverage_struct_combined["filters"] = np.empty((0,1))
    coverage_struct_combined["FOV"] = np.empty((0,1))
    for coverage_struct in coverage_structs:
        coverage_struct_combined["data"] = np.append(coverage_struct_combined["data"],coverage_struct["data"],axis=0)
        coverage_struct_combined["filters"] = np.append(coverage_struct_combined["filters"],coverage_struct["filters"])
        coverage_struct_combined["FOV"] = np.append(coverage_struct_combined["FOV"],coverage_struct["FOV"])

    return coverage_struct_combined


