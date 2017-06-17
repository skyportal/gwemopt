
import os, sys
import numpy as np
import healpy as hp

def readParamsFromFile(file):
    """@read gwemopt params file

    @param file
        gwemopt params file
    """

    params = {}
    if os.path.isfile(file):
        with open(file,'r') as f:
            for line in f:
                line_without_return = line.split("\n")
                line_split = line_without_return[0].split(" ")
                line_split = filter(None, line_split)
                if line_split:
                    params[line_split[0]] = line_split[1]
    return params

def read_skymap(filename,is3D=False):

    map_struct = {}

    if is3D:
        healpix_data = hp.read_map(filename, field=(0,1,2,3))

        distmu_data = healpix_data[1]
        diststd_data = healpix_data[2]
        prob_data = healpix_data[0]
        norm_data = healpix_data[3]

        map_struct["distmu"] = distmu_data
        map_struct["diststd"] = diststd_data
        map_struct["prob"] = prob_data
        map_struct["norm"] = healpix_data[3]
    else:
        prob_data = hp.read_map(filename, field=0)
        prob_data = prob_data / np.sum(prob_data)

        map_struct["prob"] = prob_data

    return map_struct   
