#!/usr/bin/env python

import os
import datetime
import json
import voeventparse as vp
#from slackclient import SlackClient

import GRANDMA_FAshifts as fa
from astropy.io import fits, ascii

import numpy as np
import pytz
import copy
from astropy import table
from astropy import time

import lxml.objectify as objectify

import gwemopt.utils
import gwemopt.moc
import gwemopt.gracedb
import gwemopt.rankedTilesGenerator
import gwemopt.waw
import gwemopt.lightcurve
import gwemopt.coverage
import gwemopt.efficiency
import gwemopt.plotting
import gwemopt.tiles
import gwemopt.segments
import gwemopt.catalog

from VOEventLib.VOEvent import *
from VOEventLib.Vutil import *

sys.path.append("./utils_DB/")
from VOEparser import VOEparser
from send2database import populate_DB
from astropy.table import Table

path_config = 'config/'

force_Db_use = True

def Tel_dicf():
    Tel_dic = {
        "Name": "",
        "FOV": "",
        "magnitude": "",
        "exposuretime": "",
        "latitude": "",
        "longitude": "",
        "elevation": "",
        "FOV_coverage": "",
        "FOV_coverage_type": "",
        "FOV_type": "",
        "slew_rate": 0.0,
        "readout": "",
        "filt": "",
        "OS": "",
    }
    return Tel_dic


def GW_dicf():
    GW_dic = {
        "Packet_Type": "",
        "Pkt_Ser_Num": "",
        "AlertType": "",
        "Retraction": "",
        "HardwareInj": "",
        "EventPage": "",
        "FAR": 0.,
        "Group": "",
        "Pipeline": "",
        "HasNS": "",
        "HasRemnant": "",
        "BNS": "",
        "NSBH": "",
        "BBH": "",
        "Terrestrial": "",
        "location": "",
        "lum": "",
        "errlum": "",
        "90cr": "",
        "50cr": "",
    }
    return GW_dic


def GRB_dicf():
    GRB_dic = {
        "Packet_Type": "",
        "Pkt_Ser_Num": "",
        "Trigger_TJD": "",
        "Trigger_SOD": "",
        "grbid": "",
        "ratesnr": 0.,
        "ratets": 0.,
        "imagesnr": 0.,
        "lc": "",
        "hratio": 0.,
        "longshort": False,
        "probGRB": -1.0,
        "defGRB": True,
        "locsnr": 0.,
        "locdur": 0.,
        "locref": "",
        "obs": "",
        "snr": "",
        "descriptsnr": "",
        "dur": "",
        "inst": "",
        "location": "",
        "descriptdur": "",
    }
    return GRB_dic


def VO_dicf():
    Contentvo_dic = {
        "name": "",
        "role": "",
        "stream": "grandma.lal.in2p3.fr/GRANDMA_Alert",
        "streamid": "",
        "voivorn": "ivo://svom.bao.ac.cn/LV#SFC_GW_",
        "authorivorn": "GRANDMA_Alert",
        "shortName": "GRANDMA",
        "contactName": "Nicolas  Leroy",
        "contactPhone": "+33-1-64-46-83-73",
        "contactEmail": "leroy@lal.in2p3.fr",
        "description": "Selected by ",
        "vodescription": "VOEvent created in GRANDMA",
        "locationserver": "",
        "voschemaurl": "http://www.cacr.caltech.edu/~roy/VOEvent/VOEvent2-110220.xsd",
        "ba": "",
        "ivorn": "",
        "letup": "a",
        "trigid": None,
        "eventype": None,
        "eventstat": None,
        "inst": "",
        "trigdelay": 0.,
        "locpix": "",
        "trigtime": "",
        "ra": 0.,
        "dec": 0.,
        "error": 0.,
        "eventstatus": "",
        "voimportance": "",
        "location": "",
        "iter_statut": 0,
    }

    return Contentvo_dic

def define_output_config(filename, role):
    """
    Function to read configuration json file to setup the ouput directory architecture
    :param filename: name of the json file

    the requiered field in the json are output_dir, skymappath and vopath
    We could add check with json scheme !!!!
    """
    with open(filename) as f:
        output_config = json.load(f)

    if (role == 'test'):
        output_config["output_dir"] = output_config["output_dir"] + '/test/'

    return output_config




def trigtime(isotime):
    date_t = isotime.split("-")
    yr_t = int(date_t[0])
    mth_t = int(date_t[1])
    dy_t = int(date_t[2].split("T")[0])
    hr_t = int(date_t[2].split("T")[1].split(":")[0])
    mn_t = int(date_t[2].split("T")[1].split(":")[1])
    sd_t = int(float(date_t[2].split("T")[1].split(":")[2]))
    trigger_time_format = datetime.datetime(yr_t, mth_t, dy_t, hr_t, mn_t, sd_t, tzinfo=pytz.utc)
    return trigger_time_format


def delay_fct(isotime):
    """

    :param isotime:
    :return:
    """
    # Time_now
    time_now = datetime.datetime.utcnow()

    # Time_alert
    date_t = isotime.split("-")
    yr_t = int(date_t[0])
    mth_t = int(date_t[1])
    dy_t = int(date_t[2].split("T")[0])
    hr_t = int(date_t[2].split("T")[1].split(":")[0])
    mn_t = int(date_t[2].split("T")[1].split(":")[1])
    sd_t = int(float(date_t[2].split("T")[1].split(":")[2]))
    trigger_time_format = datetime.datetime(yr_t, mth_t, dy_t, hr_t, mn_t, sd_t)
    return time_now - trigger_time_format


def gbm_lc_name(GRB_LC):
    """

    :param GRB_LC:
    :return:
    """
    grb_name = str(GRB_LC).split("/")
    return grb_name[9]


def search_ba():
    """

    :return:
    """
    fa_duty = fa.FA_shift()
    return fa_duty

def init_observation_plan(VO_dic, skymappath, dirpath="", filename=""):
    tobs = None
    filt = ["r"]
    exposuretimes = [30]
    mindiff = 30.0 * 60.0

    params = {}
    params["config"] = {}

    params["powerlaw_cl"] = 0.9
    params["powerlaw_n"] = 1.0
    params["powerlaw_dist_exp"] = 1.0

    params["doPlots"] = False
    params["doMovie"] = False
    # params["doObservability"] = True
    params["doObservability"] = False
    params["do3D"] = True
    params["DScale"] = 1.0

    params["doFootprint"] = False
    params["footprint_ra"] = 30.0
    params["footprint_dec"] = 60.0
    params["footprint_radius"] = 10.0

    params["airmass"] = 2.5

    params["doCommitDatabase"] = False
    params["doRequestScheduler"] = False
    params["dateobs"] = False
    params["doEvent"] = False
    params["doSkymap"] = True
    params["doFootprint"] = False
    params["doDatabase"] = False
    params["doReferences"] = False
    params["doChipGaps"] = False
    params["doSplit"] = False
    params["doSchedule"] = False
    params["doMinimalTiling"] = True
    params["doIterativeTiling"] = True
    params["doMaxTiles"] = True
    params["iterativeOverlap"] = 0.2
    params["maximumOverlap"] = 0.2

    params["catalog_n"] = 1.0
    params["doUseCatalog"] = False
    params["catalogDir"] = "../catalogs"
    params["galaxy_catalog"] = "GLADE"
    params["doCatalog"] = False
    params["galaxy_grade"] = 'Sloc'
    params["writeCatalog"] = False
    params["doParallel"] = False
    params["Ncores"] = 2
    params["doAlternatingFilters"] = False
    params["galaxies_FoV_sep"] = 0.9
    params["doOverlappingScheduling"] = False
    params["doPerturbativeTiling"] = True

    params["doSingleExposure"] = True
    params["filters"] = filt
    params["exposuretimes"] = exposuretimes
    params["mindiff"] = mindiff
    params["skymap"] = skymappath + VO_dic["locpix"].split("/")[-1]

    # Get nside from skymap fits file
    # Assume to be in first header extension
    skymap_header = fits.getheader(params["skymap"], 1)
    params["nside"] = skymap_header['NSIDE']
    params["DISTMEAN"] = skymap_header['DISTMEAN']
    params["DISTSTD"] = skymap_header['DISTSTD']

    params["DISTMEAN"], params["DISTSTD"] = 100.0, 50.0

    # Use galaxies to compute the grade, both for tiling and galaxy targeting, only when dist_mean + dist_std < 300Mpc
    if params["DISTMEAN"]+params["DISTSTD"]<=300:
        params["doUseCatalog"] = True
        params["doCatalog"] = True
        params["writeCatalog"] = True

    params = gwemopt.utils.params_checker(params)

    return params

def Observation_plan_multiple(telescopes, VO_dic, trigger_id, params, map_struct_input, obs_mode):
    tobs = None
    map_struct = copy.deepcopy(map_struct_input)

    event_time = time.Time(VO_dic["trigtime"], scale='utc')

    # gwemoptpath return egg file and then is not working, need to be inside ToO directory
    gwemoptpath = os.path.dirname(gwemopt.__file__)
    config_directory = "../config"
    tiling_directory = "../tiling"

    for telescope in telescopes:
        config_file = "%s/%s.config" % (config_directory,telescope)
        params["config"][telescope] = \
            gwemopt.utils.readParamsFromFile(config_file)
        params["config"][telescope]["telescope"] = telescope
    
        if "tesselationFile" in params["config"][telescope]:
            params["config"][telescope]["tesselationFile"] = \
                os.path.join(config_directory,
                             params["config"][telescope]["tesselationFile"])
            tesselation_file = params["config"][telescope]["tesselationFile"]
            if not os.path.isfile(tesselation_file):
                if params["config"][telescope]["FOV_type"] == "circle":
                    gwemopt.tiles.tesselation_spiral(
                        params["config"][telescope])
                elif params["config"][telescope]["FOV_type"] == "square":
                    gwemopt.tiles.tesselation_packing(
                        params["config"][telescope])
    
            params["config"][telescope]["tesselation"] = \
                np.loadtxt(params["config"][telescope]["tesselationFile"],
                           usecols=(0, 1, 2), comments='%')
    
        if "referenceFile" in params["config"][telescope]:
            params["config"][telescope]["referenceFile"] = \
                os.path.join(config_directory,
                             params["config"][telescope]["referenceFile"])
            refs = table.unique(table.Table.read(
                params["config"][telescope]["referenceFile"],
                format='ascii', data_start=2, data_end=-1)['field', 'fid'])
            reference_images = \
                {group[0]['field']: group['fid'].astype(int).tolist()
                 for group in refs.group_by('field').groups}
            reference_images_map = {1: 'g', 2: 'r', 3: 'i'}
            for key in reference_images:
                reference_images[key] = [reference_images_map.get(n, n)
                                         for n in reference_images[key]]
            params["config"][telescope]["reference_images"] = reference_images

    params["gpstime"] = event_time.gps
    params["outputDir"] = "output/%.5f" % event_time.mjd
    params["tilingDir"] = tiling_directory
    params["event"] = ""
    params["telescopes"] = telescopes

    if obs_mode == 'Tiling': 
        params["tilesType"] = "moc"
        params["scheduleType"] = "greedy"
        params["timeallocationType"] = "powerlaw"
        # params["doCatalog"] = False
    elif obs_mode == 'Galaxy targeting':
        # params["tilesType"] = "hierarchical"
        params["tilesType"] = "galaxy"
        params["scheduleType"] = "greedy"
        params["timeallocationType"] = "powerlaw"
        obs_mode = 'Galaxy targeting'

        # params["Ntiles"] = 50
        #params["Ntiles"] = 20
        params["doCatalog"] = True

    if params["doEvent"]:
        params["skymap"], eventinfo = gwemopt.gracedb.get_event(params)
        params["gpstime"] = eventinfo["gpstime"]
        event_time = time.Time(params["gpstime"], format='gps', scale='utc')
        params["dateobs"] = event_time.iso
    elif params["doSkymap"]:
        event_time = time.Time(params["gpstime"], format='gps', scale='utc')
        params["dateobs"] = event_time.iso
    elif params["doFootprint"]:
        params["skymap"] = gwemopt.footprint.get_skymap(params)
        event_time = time.Time(params["gpstime"], format='gps', scale='utc')
        params["dateobs"] = event_time.iso
    elif params["doDatabase"]:
        event_time = time.Time(params["dateobs"], format='datetime',
                               scale='utc')
        params["gpstime"] = event_time.gps
        params["models"] = models
    else:
        print("""Need to enable --doEvent, --doFootprint,
              --doSkymap, or --doDatabase""")
        exit(0)

    if tobs is None:
        now_time = time.Time.now()
        timediff = now_time.gps - event_time.gps
        timediff_days = timediff / 86400.0
        # Start observation plan 1h after execution of this code
        Tstart_delay = 1 / 24
        # Check observability for the next 24h, starting 1h after
        # the execution of this code
        params["Tobs"] = np.array([timediff_days + Tstart_delay,
                                   timediff_days + Tstart_delay + 1])
    else:
        params["Tobs"] = tobs

    params = gwemopt.segments.get_telescope_segments(params)

    if not os.path.isdir(params["outputDir"]):
        print('make directory' + params["outputDir"])
        os.makedirs(params["outputDir"])

    # turn around after removing this earlier
    #NEED TO COMPUTE THEM
    VO_dic["50cr"] = "0.0"
    VO_dic["90cr"] = "0.0"

    # Initialise table
    tiles_table = None

    if params["doCatalog"]:
        map_struct, catalog_struct = gwemopt.catalog.get_catalog(params, map_struct)

    if params["tilesType"] == "moc":
        print("Generating MOC struct...")
        moc_structs = gwemopt.moc.create_moc(params, map_struct=map_struct)
        tile_structs = gwemopt.tiles.moc(params, map_struct, moc_structs)
    elif params["tilesType"] == "ranked":
        print("Generating ranked struct...")
        moc_structs = gwemopt.rankedTilesGenerator.create_ranked(params,
                                                                     map_struct)
        tile_structs = gwemopt.tiles.moc(params, map_struct, moc_structs)
    elif params["tilesType"] == "hierarchical":
        print("Generating hierarchical struct...")
        tile_structs = gwemopt.tiles.hierarchical(params, map_struct)
        for telescope in params["telescopes"]:
            params["config"][telescope]["tesselation"] = np.empty((0,3))
            tiles_struct = tile_structs[telescope]
            for index in tiles_struct.keys():
                ra, dec = tiles_struct[index]["ra"], tiles_struct[index]["dec"]
                params["config"][telescope]["tesselation"] = np.append(params["config"][telescope]["tesselation"],[[index,ra,dec]],axis=0)
    elif params["tilesType"] == "greedy":
        print("Generating greedy struct...")
        tile_structs = gwemopt.tiles.greedy(params, map_struct)
        for telescope in params["telescopes"]:
            params["config"][telescope]["tesselation"] = np.empty((0,3))
            tiles_struct = tile_structs[telescope]
            for index in tiles_struct.keys():
                ra, dec = tiles_struct[index]["ra"], tiles_struct[index]["dec"]
                params["config"][telescope]["tesselation"] = np.append(params["config"][telescope]["tesselation"],[[index,ra,dec]],axis=0)
    elif params["tilesType"] == "galaxy":
        print("Generating galaxy struct...")
        tile_structs = gwemopt.tiles.galaxy(params, map_struct, catalog_struct)
        for telescope in params["telescopes"]:
            params["config"][telescope]["tesselation"] = np.empty((0,3))
            tiles_struct = tile_structs[telescope]
            for index in tiles_struct.keys():
                ra, dec = tiles_struct[index]["ra"], tiles_struct[index]["dec"]
                params["config"][telescope]["tesselation"] = np.append(params["config"][telescope]["tesselation"],[[index,ra,dec]],axis=0)
    else:
        print("Need tilesType to be galaxy, moc, greedy, hierarchical, or ranked")
        exit(0)

    tile_structs, coverage_struct = gwemopt.coverage.timeallocation(params,
                                                                    map_struct,
                                                                    tile_structs)
    if params["doPlots"]:
        gwemopt.plotting.skymap(params, map_struct)
        gwemopt.plotting.tiles(params, map_struct, tile_structs)
        gwemopt.plotting.coverage(params, map_struct, coverage_struct)

    tiles_tables = {}
    for jj, telescope in enumerate(telescopes):

        config_struct = params["config"][telescope]
    
        # table_field = utilityTable(thistable)
        # table_field.blankTable(len(coverage_struct))
        field_id_vec = []
        ra_vec = []
        dec_vec = []
        grade_vec = []
        rank_id = []
    
        for ii in range(len(coverage_struct["ipix"])):
            data = coverage_struct["data"][ii, :]
            filt = coverage_struct["filters"][ii]
            ipix = coverage_struct["ipix"][ii]
            patch = coverage_struct["patch"][ii]
            FOV = coverage_struct["FOV"][ii]
            area = coverage_struct["area"][ii]

            if not telescope == coverage_struct["telescope"][ii]:
                continue
    
            prob = np.sum(map_struct["prob"][ipix])
            #prob = tile_structs[telescope][ii]["prob"]
    
            ra, dec = data[0], data[1]
            #exposure_time, field_id, prob = data[4], data[5], data[6]
            exposure_time, field_id = data[4], data[5]
            field_id_vec.append(int(field_id))
            ra_vec.append(np.round(ra, 4))
            dec_vec.append(np.round(dec, 4))
            grade_vec.append(np.round(prob, 4))
    
        # Store observation in database only if there are tiles
        if field_id_vec:
    
            field_id_vec = np.array(field_id_vec)
            ra_vec = np.array(ra_vec)
            dec_vec = np.array(dec_vec)
            grade_vec = np.array(grade_vec)
            # Sort by descing order of probability
            idx = np.argsort(grade_vec)[::-1]
            field_id_vec = field_id_vec[idx]
            ra_vec = ra_vec[idx]
            dec_vec = dec_vec[idx]
            grade_vec = grade_vec[idx]
    
            # Store observation plan with tiles for a given telescope in GRANDMA database
            # Formatting data
    
            # Create an array indicating descing order of probability
            if telescope in ["GWAC"]:
                for i in range(len(field_id_vec)):
                    rank_id.append(str(int(field_id_vec[i])).zfill(8))
            else:
                rank_id = np.arange(len(field_id_vec)) + 1
    
            # Get each tile corners in RA, DEC
            tiles_corners_str = []
            tiles_corners_list = []
            for tile_id in field_id_vec:
                unsorted_corners = tile_structs[telescope][tile_id]['corners']
                # print (unsorted_corners.shape[0])
    
                try:
                    if unsorted_corners.shape[0] == 1:
                        sorted_corners_str = "[[%.3f, %.3f]]" % (unsorted_corners[0][0], unsorted_corners[0][1])
                        sorted_corners_list = [[unsorted_corners[0][0], unsorted_corners[0][1]]]
    
                    elif unsorted_corners.shape[0] == 2:
                        sorted_corners_str = "[[%.3f, %.3f], [%.3f, %.3f]]" % (unsorted_corners[0][0], unsorted_corners[0][1], unsorted_corners[1][0], unsorted_corners[1][1])
                        sorted_corners_list = [[unsorted_corners[0][0], unsorted_corners[0][1]], [unsorted_corners[1][0], unsorted_corners[1][1]]]
    
                    elif unsorted_corners.shape[0] == 3:
                        sorted_corners_str = "[[%.3f, %.3f], [%.3f, %.3f], [%.3f, %.3f]]" % (unsorted_corners[0][0], unsorted_corners[0][1], unsorted_corners[1][0], unsorted_corners[1][1], unsorted_corners[2][0], unsorted_corners[2][1])
                        sorted_corners_list = [[unsorted_corners[0][0], unsorted_corners[0][1]], [unsorted_corners[1][0], unsorted_corners[1][1]], [unsorted_corners[2][0], unsorted_corners[2][1]]]
    
                    elif unsorted_corners.shape[0] == 4:
                        sorted_corners_str = "[[%.3f, %.3f], [%.3f, %.3f], [%.3f, %.3f], [%.3f, %.3f]]" % (unsorted_corners[0][0], unsorted_corners[0][1], unsorted_corners[1][0], unsorted_corners[1][1], unsorted_corners[3][0], unsorted_corners[3][1], unsorted_corners[2][0], unsorted_corners[2][1])
                        sorted_corners_list = [[unsorted_corners[0][0], unsorted_corners[0][1]], [unsorted_corners[1][0], unsorted_corners[1][1]], [unsorted_corners[3][0], unsorted_corners[3][1]], [unsorted_corners[2][0], unsorted_corners[2][1]]]
                    else:
                        sorted_corners_str = "[]"
                        sorted_corners_list = []   
                except:
                    sorted_corners_str = "[]"
                    sorted_corners_list = []
    
                tiles_corners_str.append(sorted_corners_str)
                tiles_corners_list.append(sorted_corners_list)
    
            # Limit number of tiles sent to each telescope
            if params["max_nb_tiles"][jj] > 0:
                max_nb_tiles = min(len(rank_id), params["max_nb_tiles"][jj])
            else:
                max_nb_tiles = len(rank_id)
            rank_id = rank_id[:max_nb_tiles]
            field_id_vec = field_id_vec[:max_nb_tiles]
            ra_vec = ra_vec[:max_nb_tiles]
            dec_vec = dec_vec[:max_nb_tiles]
            grade_vec = grade_vec[:max_nb_tiles]
            tiles_corners_str = tiles_corners_str[:max_nb_tiles]
            tiles_corners_list = tiles_corners_list[:max_nb_tiles]
    
            # Create astropy table containing observation plan and telescope name
            tiles_table = Table([list(rank_id), field_id_vec, ra_vec, dec_vec, grade_vec, tiles_corners_str, tiles_corners_list],
                                names=('rank_id', 'tile_id', 'RA', 'DEC', 'Prob', 'Corners', 'Corners_list'),
                                meta={'telescope_name': telescope,
                                      'trigger_id': trigger_id,
                                      'obs_mode': obs_mode,
                                      'FoV_telescope': config_struct['FOV'],
                                      'FoV_sep': params["galaxies_FoV_sep"],
                                      'doUseCatalog': params["doUseCatalog"],
                                      'galaxy_grade': params["galaxy_grade"]})
            tiles_tables[telescope] = tiles_table
        else:
            tiles_tables[telescope] = None

    galaxies_table=None

    if params["doCatalog"] == True:
        # Send galaxies to DB
        filename_gal = params["outputDir"] + '/catalog.csv'

        galaxies_table = ascii.read(filename_gal, format='csv')
        #if storeGal: send_ObsPlan_galaxies_to_DB(galaxies_table, trigger_id)

    #        return np.transpose(np.array([rank_id, ra_vec, dec_vec, grade_vec])), galaxies_table
    return tiles_tables, galaxies_table

def Observation_plan(telescope, VO_dic, trigger_id, params, map_struct_input):
    tobs = None
    map_struct = copy.deepcopy(map_struct_input)

    print(telescope)
    event_time = time.Time(VO_dic["trigtime"], scale='utc')

    # gwemoptpath return egg file and then is not working, need to be inside ToO directory
    gwemoptpath = os.path.dirname(gwemopt.__file__)
    config_directory = "../config"
    tiling_directory = "../tiling"

    config_file = "%s/%s.config" % (config_directory,telescope)
    params["config"][telescope] = \
        gwemopt.utils.readParamsFromFile(config_file)
    params["config"][telescope]["telescope"] = telescope

    if "tesselationFile" in params["config"][telescope]:
        params["config"][telescope]["tesselationFile"] = \
            os.path.join(config_directory,
                         params["config"][telescope]["tesselationFile"])
        tesselation_file = params["config"][telescope]["tesselationFile"]
        if not os.path.isfile(tesselation_file):
            if params["config"][telescope]["FOV_type"] == "circle":
                gwemopt.tiles.tesselation_spiral(
                    params["config"][telescope])
            elif params["config"][telescope]["FOV_type"] == "square":
                gwemopt.tiles.tesselation_packing(
                    params["config"][telescope])

        params["config"][telescope]["tesselation"] = \
            np.loadtxt(params["config"][telescope]["tesselationFile"],
                       usecols=(0, 1, 2), comments='%')

    if "referenceFile" in params["config"][telescope]:
        params["config"][telescope]["referenceFile"] = \
            os.path.join(config_directory,
                         params["config"][telescope]["referenceFile"])
        refs = table.unique(table.Table.read(
            params["config"][telescope]["referenceFile"],
            format='ascii', data_start=2, data_end=-1)['field', 'fid'])
        reference_images = \
            {group[0]['field']: group['fid'].astype(int).tolist()
             for group in refs.group_by('field').groups}
        reference_images_map = {1: 'g', 2: 'r', 3: 'i'}
        for key in reference_images:
            reference_images[key] = [reference_images_map.get(n, n)
                                     for n in reference_images[key]]
        params["config"][telescope]["reference_images"] = reference_images

    params["gpstime"] = event_time.gps
    params["outputDir"] = "output/%.5f" % event_time.mjd
    params["tilingDir"] = tiling_directory
    params["event"] = ""
    params["telescopes"] = [telescope]

    if telescope in ["GWAC", "TRE", "TCA", "TCH", "OAJ"]:
        params["tilesType"] = "moc"
        params["scheduleType"] = "greedy"
        params["timeallocationType"] = "powerlaw"
        obs_mode = 'Tiling'
        # params["doCatalog"] = False
    else:
        # params["tilesType"] = "hierarchical"
        params["tilesType"] = "galaxy"
        params["scheduleType"] = "greedy"
        params["timeallocationType"] = "powerlaw"
        obs_mode = 'Galaxy targeting'

        # params["Ntiles"] = 50
        #params["Ntiles"] = 20
        # params["doCatalog"] = True

    if params["doEvent"]:
        params["skymap"], eventinfo = gwemopt.gracedb.get_event(params)
        params["gpstime"] = eventinfo["gpstime"]
        event_time = time.Time(params["gpstime"], format='gps', scale='utc')
        params["dateobs"] = event_time.iso
    elif params["doSkymap"]:
        event_time = time.Time(params["gpstime"], format='gps', scale='utc')
        params["dateobs"] = event_time.iso
    elif params["doFootprint"]:
        params["skymap"] = gwemopt.footprint.get_skymap(params)
        event_time = time.Time(params["gpstime"], format='gps', scale='utc')
        params["dateobs"] = event_time.iso
    elif params["doDatabase"]:
        event_time = time.Time(params["dateobs"], format='datetime',
                               scale='utc')
        params["gpstime"] = event_time.gps
        params["models"] = models
    else:
        print("""Need to enable --doEvent, --doFootprint,
              --doSkymap, or --doDatabase""")
        exit(0)

    if tobs is None:
        now_time = time.Time.now()
        timediff = now_time.gps - event_time.gps
        timediff_days = timediff / 86400.0
        # Start observation plan 1h after execution of this code
        Tstart_delay = 1 / 24
        # Check observability for the next 24h, starting 1h after
        # the execution of this code
        params["Tobs"] = np.array([timediff_days + Tstart_delay,
                                   timediff_days + Tstart_delay + 1])
    else:
        params["Tobs"] = tobs

    params = gwemopt.segments.get_telescope_segments(params)

    if not os.path.isdir(params["outputDir"]):
        print('make directory' + params["outputDir"])
        os.makedirs(params["outputDir"])

    # turn around after removing this earlier
    #NEED TO COMPUTE THEM
    VO_dic["50cr"] = "0.0"
    VO_dic["90cr"] = "0.0"

    # Initialise table
    tiles_table = None

    if params["doCatalog"]:
        map_struct, catalog_struct = gwemopt.catalog.get_catalog(params, map_struct)

    if params["tilesType"] == "moc":
        print("Generating MOC struct...")
        moc_structs = gwemopt.moc.create_moc(params)
        tile_structs = gwemopt.tiles.moc(params, map_struct, moc_structs)
    elif params["tilesType"] == "ranked":
        print("Generating ranked struct...")
        moc_structs = gwemopt.rankedTilesGenerator.create_ranked(params,
                                                                     map_struct)
        tile_structs = gwemopt.tiles.moc(params, map_struct, moc_structs)
    elif params["tilesType"] == "hierarchical":
        print("Generating hierarchical struct...")
        tile_structs = gwemopt.tiles.hierarchical(params, map_struct)
    elif params["tilesType"] == "greedy":
        print("Generating greedy struct...")
        tile_structs = gwemopt.tiles.greedy(params, map_struct)
    elif params["tilesType"] == "galaxy":
        print("Generating galaxy struct...")
        tile_structs = gwemopt.tiles.galaxy(params, map_struct, catalog_struct)
    else:
        print("Need tilesType to be galaxy, moc, greedy, hierarchical, or ranked")
        exit(0)

    coverage_struct = gwemopt.coverage.timeallocation(params,
                                                      map_struct,
                                                      tile_structs)
    if params["doPlots"]:
        gwemopt.plotting.skymap(params, map_struct)
        gwemopt.plotting.tiles(params, map_struct, tile_structs)
        gwemopt.plotting.coverage(params, map_struct, coverage_struct)

    config_struct = params["config"][telescope]

    # table_field = utilityTable(thistable)
    # table_field.blankTable(len(coverage_struct))
    field_id_vec = []
    ra_vec = []
    dec_vec = []
    grade_vec = []
    rank_id = []

    for ii in range(len(coverage_struct["ipix"])):
        data = coverage_struct["data"][ii, :]
        filt = coverage_struct["filters"][ii]
        ipix = coverage_struct["ipix"][ii]
        patch = coverage_struct["patch"][ii]
        FOV = coverage_struct["FOV"][ii]
        area = coverage_struct["area"][ii]

        prob = np.sum(map_struct["prob"][ipix])
        #prob = tile_structs[telescope][ii]["prob"]

        ra, dec = data[0], data[1]
        #exposure_time, field_id, prob = data[4], data[5], data[6]
        exposure_time, field_id = data[4], data[5]

        field_id_vec.append(int(field_id))
        ra_vec.append(np.round(ra, 4))
        dec_vec.append(np.round(dec, 4))
        grade_vec.append(np.round(prob, 4))

    # Store observation in database only if there are tiles
    if field_id_vec:

        field_id_vec = np.array(field_id_vec)
        ra_vec = np.array(ra_vec)
        dec_vec = np.array(dec_vec)
        grade_vec = np.array(grade_vec)
        # Sort by descing order of probability
        idx = np.argsort(grade_vec)[::-1]
        field_id_vec = field_id_vec[idx]
        ra_vec = ra_vec[idx]
        dec_vec = dec_vec[idx]
        grade_vec = grade_vec[idx]

        # Store observation plan with tiles for a given telescope in GRANDMA database
        # Formatting data

        # Create an array indicating descing order of probability
        if telescope in ["GWAC"]:
            for i in range(len(field_id_vec)):
                rank_id.append(str(int(field_id_vec[i])).zfill(8))
        else:
            rank_id = np.arange(len(field_id_vec)) + 1

        # Get each tile corners in RA, DEC
        tiles_corners_str = []
        tiles_corners_list = []
        for tile_id in field_id_vec:
            unsorted_corners = tile_structs[telescope][tile_id]['corners']
            # print (unsorted_corners.shape[0])
            try:
                if unsorted_corners.shape[0] == 1:
                    sorted_corners_str = "[[%.3f, %.3f]]" % (unsorted_corners[0][0], unsorted_corners[0][1])
                    sorted_corners_list = [[unsorted_corners[0][0], unsorted_corners[0][1]]]

                elif unsorted_corners.shape[0] == 2:
                    sorted_corners_str = "[[%.3f, %.3f], [%.3f, %.3f]]" % (unsorted_corners[0][0], unsorted_corners[0][1], unsorted_corners[1][0], unsorted_corners[1][1])
                    sorted_corners_list = [[unsorted_corners[0][0], unsorted_corners[0][1]], [unsorted_corners[1][0], unsorted_corners[1][1]]]

                elif unsorted_corners.shape[0] == 3:
                    sorted_corners_str = "[[%.3f, %.3f], [%.3f, %.3f], [%.3f, %.3f]]" % (unsorted_corners[0][0], unsorted_corners[0][1], unsorted_corners[1][0], unsorted_corners[1][1], unsorted_corners[2][0], unsorted_corners[2][1])
                    sorted_corners_list = [[unsorted_corners[0][0], unsorted_corners[0][1]], [unsorted_corners[1][0], unsorted_corners[1][1]], [unsorted_corners[2][0], unsorted_corners[2][1]]]

                elif unsorted_corners.shape[0] == 4:
                    sorted_corners_str = "[[%.3f, %.3f], [%.3f, %.3f], [%.3f, %.3f], [%.3f, %.3f]]" % (unsorted_corners[0][0], unsorted_corners[0][1], unsorted_corners[1][0], unsorted_corners[1][1], unsorted_corners[3][0], unsorted_corners[3][1], unsorted_corners[2][0], unsorted_corners[2][1])
                    sorted_corners_list = [[unsorted_corners[0][0], unsorted_corners[0][1]], [unsorted_corners[1][0], unsorted_corners[1][1]], [unsorted_corners[3][0], unsorted_corners[3][1]], [unsorted_corners[2][0], unsorted_corners[2][1]]]

            except:
                sorted_corners_str = "[]"
                sorted_corners_list = []

            tiles_corners_str.append(sorted_corners_str)
            tiles_corners_list.append(sorted_corners_list)

        # Limit number of tiles sent to each telescope
        if params["max_nb_tiles"] > 0:
            max_nb_tiles = min(len(rank_id), params["max_nb_tiles"])
        else:
            max_nb_tiles = len(rank_id)
        rank_id = rank_id[:max_nb_tiles]
        field_id_vec = field_id_vec[:max_nb_tiles]
        ra_vec = ra_vec[:max_nb_tiles]
        dec_vec = dec_vec[:max_nb_tiles]
        grade_vec = grade_vec[:max_nb_tiles]
        tiles_corners_str = tiles_corners_str[:max_nb_tiles]
        tiles_corners_list = tiles_corners_list[:max_nb_tiles] 

        # Create astropy table containing observation plan and telescope name
        tiles_table = Table([list(rank_id), field_id_vec, ra_vec, dec_vec, grade_vec, tiles_corners_str, tiles_corners_list],
                            names=('rank_id', 'tile_id', 'RA', 'DEC', 'Prob', 'Corners', 'Corners_list'),
                            meta={'telescope_name': telescope,
                                  'trigger_id': trigger_id,
                                  'obs_mode': obs_mode,
                                  'FoV_telescope': config_struct['FOV'],
                                  'FoV_sep': params["galaxies_FoV_sep"],
                                  'doUseCatalog': params["doUseCatalog"],
                                  'galaxy_grade': params["galaxy_grade"]})


    galaxies_table=None

    if params["doCatalog"] == True:
        # Send galaxies to DB
        filename_gal = params["outputDir"] + '/catalog.csv'

        galaxies_table = ascii.read(filename_gal, format='csv')
        #if storeGal: send_ObsPlan_galaxies_to_DB(galaxies_table, trigger_id)

    #        return np.transpose(np.array([rank_id, ra_vec, dec_vec, grade_vec])), galaxies_table
    return tiles_table, galaxies_table


def swift_trigger(v, collab, output_dic, file_log_s):
    """

    :param v:
    :param collab:
    :param output_dic:
    :return:
    """
    print('Swift trigger, instrument ' + str(collab[2]))

    Swift_dic = GRB_dicf()
    Swift_vo = VO_dicf()

    Swift_vo["role"] = v.attrib['role']

    # Do not save test alerts, except when forced to True
    if force_Db_use:
        Db_use = force_Db_use
    else:
        if v.attrib['role'] != "test":
            Db_use = True
        else:
            Db_use = False


    instru = str(collab[2])
    Swift_vo["ba"] = fa.FA_shift()

    if instru == "BAT":

        Swift_dic["inst"] = instru

        top_level_params = vp.get_toplevel_params(v)
        trigger_id = top_level_params['TrigID']['value']
        Swift_vo["trigid"] = trigger_id

        rate_signif = top_level_params['Rate_Signif']['value']
        Swift_dic["ratesnr"] = float(rate_signif)

        image_signif = top_level_params['Image_Signif']['value']
        Swift_dic["imagesnr"] = float(image_signif)
        Swift_dic["snr"] = float(image_signif)
        Swift_dic["descriptsnr"] = "SNR calculated from the image"

        if float(image_signif) < 4.0:
            Swift_vo["voimportance"] = 3
        if ((float(image_signif) >= 6.0) & (float(image_signif) < 7.0)):
            Swift_vo["voimportance"] = 2
        if ((float(image_signif) > 7.0)):
            Swift_vo["voimportance"] = 1

        def_not_grb = v.find(".//Param[@name='Def_NOT_a_GRB']").attrib['value']
        Swift_dic["defGRB"] = def_not_grb

        Swift_vo["eventstatus"] = "initial"
        Swift_vo["eventype"] = "GRB"
        Swift_vo["inst"] = "Swift-BAT"
        Swift_vo["location"] = "Sky"

        isotime = v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Time.TimeInstant. \
            ISOTime.text
        delay = delay_fct(isotime)
        isotime_format = trigtime(isotime)
        delay_min = (delay.seconds) / 60.0
        Swift_vo["trigtime"] = isotime_format
        Swift_vo["trigdelay"] = delay_min

        right_ascension = str(v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords. \
                              Position2D.Value2.C1.text)
        declination = str(v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Position2D \
                          .Value2.C2.text)
        error2_radius = str(v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords. \
                            Position2D.Error2Radius.text)

        Swift_vo["ra"] = right_ascension
        Swift_vo["trigdelay"] = right_ascension
        Swift_vo["dec"] = declination
        Swift_vo["error"] = error2_radius
        Swift_dic["locref"] = "BAT onboard"
        message_obs = ""

        name_dic = "Swift" + trigger_id
        if (Swift_vo["role"] != "test"):
            update_output_config(output_dic,Swift_vo,"GRB")

            setup_cloud_event(output_dic)

            lalid = name_lalid(v, file_log_s, name_dic, Swift_vo["letup"], "_DB")
            name_dic = "Swift" + trigger_id
            dic_grb[name_dic] = Swift_dic
            dic_vo[name_dic] = Swift_vo

            create_GRANDMAvoevent(lalid, Swift_dic, Swift_vo, output_dic, send2DB=Db_use)
            file_log_s.write(lalid + " " + str(trigger_id) + "\n")

            for telescope in LISTE_TELESCOPE:
                Tel_dic = Tel_dicf()
                Tel_dic["Name"] = telescope
                message_obs = message_obs + " " + telescope
                Tel_dic["OS"] = ""
                if (Swift_vo["role"] != "test"):
                    name_dic = "Swift" + trigger_id
                    lalid = name_lalid(v, file_log_s, name_dic, Swift_vo["letup"], "_" + Tel_dic["Name"])
                    create_GRANDMAvoevent(lalid, Swift_dic, Swift_vo, output_dic)
                    file_log_s.write(lalid + " " + str(trigger_id) + "\n")

        text_mes = str("---------- \n") + str("BAT alert \n") + str("---------- \n") + "Trigger ID: " + \
                   trigger_id + "\n" + "Trigger Time: " + isotime + "\n" + "Delay since alert: " + \
                   str(delay) + "\n" + "\n" + str("---Follow-up Advocate---\n") + str("FA on duty: ") + \
                   str(fa.FA_shift()) + "\n" + "\n" + "\n" + str("---SPACE TRIGGER---\n") + \
                   str("Trigger Rate SNR: ") + str(rate_signif) + " " + str("Image_Signif: ") + image_signif + "\n" + \
                   str("\n") + str("---Position---\n") + \
                   "RA: " + str(round(float(right_ascension), 1)) + " " + "DEC: " + str(round(float(declination), 1)) + \
                   " " + str("Error2Radius: ") + \
                   str(round(float(error2_radius), 1)) + "\n" + message_obs + "\n" + str("\n") + str(
            "---------- \n")  # +("---SVOM FOLLOWUP---\n")+\
        # str(observability__xinglong)+" "+str(follow)+"\n"

    return text_mes


def GW_trigger_retracted(v, output_dic, file_log_s):
    # Get config params from json file

    GW_dic = GW_dicf()
    GW_vo = VO_dicf()

    GW_vo["role"] = v.attrib['role']

    GW_vo["ba"] = fa.FA_shift()

    GW_vo["eventype"] = "GW"

    toplevel_params = vp.get_toplevel_params(v)
    Pktser = toplevel_params['Pkt_Ser_Num']['value']
    GW_vo["iter_statut"] = str(int(Pktser) - 1)
    GW_vo["eventstatus"] = toplevel_params['AlertType']['value']
    trigger_id = toplevel_params['GraceID']['value']

    GW_vo["letup"] = letters[int(Pktser) - 1]

    GW_dic["Retraction"] = '1'  # toplevel_params['Retraction']['value']
    GW_vo["eventstatus"] = "Retractation"

    isotime = v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Time.TimeInstant.ISOTime.text
    isotime_format = trigtime(isotime)
    GW_vo["trigtime"] = isotime_format
    GW_vo["trigid"] = trigger_id
    GW_vo["location"] = "LIGO Virgo"
    GW_vo["eventype"] = "GW"
    GW_vo["voimportance"] = 1

    update_output_config(output_dic, GW_vo, "GW")

    setup_cloud_event(output_dic)

    name_dic = "GW" + trigger_id
    lalid = name_lalid(v, file_log_s, name_dic, GW_vo["letup"], "_DB")

    for telescope in LISTE_TELESCOPE:
        Tel_dic = Tel_dicf()
        Tel_dic["Name"] = telescope

        name_dic = "GW" + trigger_id
        lalid = name_lalid(v, file_log_s, name_dic, GW_vo["letup"], "_" + Tel_dic["Name"])
        filename_vo = create_GRANDMAvoevent(lalid, GW_dic, GW_vo, output_dic)
        send_voevent(path_config+'/broker.json',filename_vo)

    file_log_s.write(lalid + " " + str(trigger_id) + "\n")

    text = str("---------- \n") + str("GW alert \n") + str("---------- \n") + str("GW NAME : ") \
           + str(GW_vo["trigid"]) + (" ") + str("Trigger Time: ") + isotime + "\n" + \
           str("WARNING RETRACTATION") + str("\n") + str("---Follow-up Advocate--\n") + str(
        "Follow-up advocate on duty: ") + str(fa.FA_shift()) + "\n"
    return text


def GW_treatment_alert(v, output_dic, file_log_s):
    # Get config params from json file
    GW_dic = GW_dicf()
    GW_vo = VO_dicf()

    GW_vo["role"] = v.attrib['role']

    # Do not save test alerts, except when forced to True
    if force_Db_use:
        Db_use = force_Db_use
    else:
        if v.attrib['role'] != "test":
            Db_use = True
        else:
            Db_use = False

    GW_vo["ba"] = fa.FA_shift()

    GW_vo["eventype"] = "GW"

    toplevel_params = vp.get_toplevel_params(v)
    Pktser = toplevel_params['Pkt_Ser_Num']['value']
    GW_vo["iter_statut"] = str(int(Pktser) - 1)
    GW_vo["inst"] = toplevel_params['Instruments']['value']
    GW_vo["eventstatus"] = toplevel_params['AlertType']['value']
    trigger_id = toplevel_params['GraceID']['value']

    GW_vo["letup"] = letters[int(Pktser) - 1]

    GW_dic["Retraction"] = '0'
    GW_dic["HardwareInj"] = toplevel_params['HardwareInj']['value']
    GW_dic["EventPage"] = toplevel_params['EventPage']['value']
    GW_dic["FAR"] = toplevel_params['FAR']['value']
    GW_dic["Group"] = toplevel_params['Group']['value']
    GW_dic["Pipeline"] = toplevel_params['Pipeline']['value']
    GW_dic["Classification"] = vp.get_grouped_params(v)
    GW_dic["locref"] = "bayestar"
    grouped_params = vp.get_grouped_params(v)
    HasRemnant = float(v.find(".//Param[@name='HasRemnant']").attrib['value'])
    BNS = str(v.find(".//Param[@name='BNS']").attrib['value'])
    GW_dic["BNS"] = BNS
    NSBH = str(v.find(".//Param[@name='NSBH']").attrib['value'])
    GW_dic["NSBH"] = NSBH
    BBH = str(v.find(".//Param[@name='BBH']").attrib['value'])
    GW_dic["BBH"] = BBH
    Terrestrial = str(v.find(".//Param[@name='Terrestrial']").attrib['value'])
    GW_dic["Terrestrial"] = Terrestrial
    HasNS = str(v.find(".//Param[@name='HasNS']").attrib['value'])
    GW_dic["HasRemnant"] = str(HasRemnant)
    GW_dic["HasNS"] = str(HasNS)
    # print(len(GW_dic["inst"].split(",")))

    if HasRemnant > 0.9:
        GW_vo["voimportance"] = 1
    else:
        if (len(GW_vo["inst"].split(","))) > 2:
            GW_vo["voimportance"] = 2
        else:
            GW_vo["voimportance"] = 3

    isotime = v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Time.TimeInstant.ISOTime.text

    delay = delay_fct(isotime)
    isotime_format = trigtime(isotime)
    GW_vo["trigtime"] = isotime_format
    delay_min = (delay.seconds) / 60.0
    GW_vo["trigdelay"] = delay_min

    GW_vo["trigid"] = trigger_id

    GW_vo["locpix"] = str(v.find(".//Param[@name='skymap_fits']").attrib['value'])
    GW_vo["location"] = "LIGO Virgo"
    name_dic = "GW" + trigger_id

    message_obs = "NO SKYMAP AVAILABLE"
    if (GW_vo["locpix"] != "") and (GW_vo["eventstatus"] == "Preliminary"):
        GW_vo["eventstatus"] = "Initial"
    # we do not react on preliminary with hypothesis that there are no skymap associated
    update_output_config(output_dic, GW_vo, "GW")
    
    setup_cloud_event(output_dic)

    if (GW_vo["eventstatus"] != "Preliminary"):
        message_obs = "Observation plan sent to "
        skypath = output_dic["skymappath"] + str(GW_vo["locpix"].split("/")[-1])
        if not os.path.isfile(skypath):
            command = "curl " + " -o " + skypath + " -O " + GW_vo["locpix"]
            os.system(command)

        hdul = fits.open(skypath)
        lumin = np.round(hdul[1].header['DISTMEAN'], 3)
        errorlumin = np.round(hdul[1].header['DISTSTD'], 3)
        hdul.close()
        s50cr = 0.0
        s90cr = 0.0
        GW_dic["lum"] = str(lumin)
        GW_dic["errlum"] = str(errorlumin)

        # Load params dictionary for gwemopt
        params = init_observation_plan(GW_vo, output_dic["skymappath"])

        print("Loading skymap...")
        # Function to read maps
        map_struct = gwemopt.utils.read_skymap(params, is3D=params["do3D"])
        # Compute 50% and 90% CR skymap area
        idx50 = map_struct["cumprob"] <= 0.50
        cr50 = len(map_struct["cumprob"][idx50])
        idx90 = map_struct["cumprob"] <= 0.90
        cr90 = len(map_struct["cumprob"][idx90])
        GW_dic["50cr"] = "{:.2f}".format(map_struct["pixarea_deg2"] * cr50)
        GW_dic["90cr"] = "{:.2f}".format(map_struct["pixarea_deg2"] * cr90)

        lalid = name_lalid(v, file_log_s, name_dic, GW_vo["letup"], "_DB")
        create_GRANDMAvoevent(lalid, GW_dic, GW_vo, "", output_dic, send2DB=Db_use)
        file_log_s.write(lalid + " " + str(trigger_id) + "\n")

        if GW_vo["voimportance"] == 1:
            LISTE_TELESCOPE_TILING = ["OAJ", "TRE", 'TCH', 'TCA']
            max_nb_tiles_tiling = np.array([60, 50, 50, 50])
            #max_nb_tiles_tiling = -1 * np.ones(len(LISTE_TELESCOPE_TILING))
            LISTE_TELESCOPE_GALAXY = ["Makes-60","Lisnyky-AZT8","Zadko","TNT","UBAI-T60N","ShAO-T60","Abastunami-T70","UBAI-T60S","Abastunami-T48","IRIS"]
            max_nb_tiles_galaxy = np.array([50]*len(LISTE_TELESCOPE_GALAXY))
            #max_nb_tiles_galaxy = -1 * np.ones(len(LISTE_TELESCOPE_GALAXY))
        else:
            LISTE_TELESCOPE_TILING = ["TRE", "TCH", "TCA"]
            max_nb_tiles_tiling = np.array([50, 50, 50])
            #max_nb_tiles_tiling = -1 * np.ones(len(LISTE_TELESCOPE_TILING))
            LISTE_TELESCOPE_GALAXY = ["Makes-60","Lisnyky-AZT8","Zadko","TNT","UBAI-T60N","ShAO-T60","Abastunami-T70","UBAI-T60S","Abastunami-T48","IRIS"]
            max_nb_tiles_galaxy = np.array([50]*len(LISTE_TELESCOPE_GALAXY))
            #max_nb_tiles_galaxy = -1 * np.ones(len(LISTE_TELESCOPE_GALAXY))

        LISTE_TELESCOPE_TILING = []
        max_nb_tiles_tiling = np.array([])
        LISTE_TELESCOPE_GALAXY = ["Makes-60"]
        max_nb_tiles_galaxy = np.array([50]*len(LISTE_TELESCOPE_GALAXY))

        ### TILING ###
        params["max_nb_tiles"] = max_nb_tiles_tiling
        # Adapt percentage of golden tiles with the 90% skymap size. Arbitrary, needs to be optimised!!!
        if float(GW_dic["90cr"]) < 60:
            params["iterativeOverlap"] = 0.8
            params["doIterativeTiling"] = False
            params["doPerturbativeTiling"] = False
        else:
            params["iterativeOverlap"] = 0.2
            params["doIterativeTiling"] = True
            params["doPerturbativeTiling"] = True
        print (GW_dic["90cr"], GW_dic["50cr"])
        print ('ITERATIVE OVERLAP: ', params["iterativeOverlap"])
        #params["galaxy_grade"] = 'Sloc'

        aTables_tiling, galaxies_table = Observation_plan_multiple(LISTE_TELESCOPE_TILING, GW_vo, trigger_id, params, map_struct, 'Tiling')
        # Send data to DB and send xml files to telescopes through broker for tiling
        send_data(LISTE_TELESCOPE_TILING, params, aTables_tiling, galaxies_table, GW_vo, GW_dic, trigger_id, v, file_log_s, path_config, output_dic, message_obs, name_dic, Db_use=Db_use, gal2DB=False)

        ### Galaxy targeting ###
        #if the  mean distance(+error) of the skymap is less than 300Mpc we perform galaxy targeting
        if params["DISTMEAN"]+params["DISTSTD"]<=300:
            #params["galaxy_grade"] = 'Sloc'
            params["max_nb_tiles"] = max_nb_tiles_galaxy
            params["doPerturbativeTiling"] = True
            aTables_galaxy, galaxies_table = Observation_plan_multiple(LISTE_TELESCOPE_GALAXY, GW_vo, trigger_id, params, map_struct, 'Galaxy targeting')
            # Send data to DB and send xml files to telescopes through broker for galaxy targeting
            send_data(LISTE_TELESCOPE_GALAXY, params, aTables_galaxy, galaxies_table, GW_vo, GW_dic, trigger_id, v, file_log_s, path_config, output_dic, message_obs, name_dic, Db_use=Db_use, gal2DB=True)

    else:
        lalid = name_lalid(v, file_log_s, name_dic, GW_vo["letup"], "_DB")
        create_GRANDMAvoevent(lalid, GW_dic, GW_vo, "", output_dic, send2DB=Db_use)
        file_log_s.write(lalid + " " + str(trigger_id) + "\n")

    text = str("---------- \n") + str("GW alert \n") + str("---------- \n") + str("GW NAME : ") \
           + str(GW_vo["trigid"]) + (" ") + str("Trigger Time: ") + isotime + "\n" + \
           str("Instruments: ") + str(str(GW_vo["inst"])) + str("\n") \
           + str("EventPage: ") + str(str(GW_dic["EventPage"])) + str("\n") + str("Search: ") + str(
        str(GW_dic["Group"])) + str("\n") + str("HasRemnant: ") + str(HasRemnant) + str("\n") \
           + str("Delay since alert: ") + str(delay) + ("\n") + str("\n") + str("---Follow-up Advocate--\n") + str(
        "Follow-up advocate on duty: ") + str(fa.FA_shift()) + "\n" + message_obs + "\n"
    return text

def fermi_trigger_found(v, output_dic, file_log_s):
    """

    :param v:
    :param collab:
    :return:
    """

    Fermi_dic = GRB_dicf()
    Fermi_vo = VO_dicf()

    Fermi_vo["role"] = v.attrib['role']

    # Do not save test alerts, except when forced to True
    if force_Db_use:
        Db_use = force_Db_use
    else:
        if v.attrib['role'] != "test":
            Db_use = True
        else:
            Db_use = False

    instru = "GBM"
    Fermi_dic["inst"] = instru

    Fermi_vo["ba"] = fa.FA_shift()

    Fermi_vo["eventstatus"] = "Preliminary"
    Fermi_vo["eventype"] = "GRB"
    Fermi_vo["inst"] = "Fermi-GBM"
    Fermi_vo["location"] = "Sky"
    message_obs = "NO SKYMAP AVAILABLE"

    toplevel_params = vp.get_toplevel_params(v)
    trigger_id = toplevel_params['TrigID']['value']
    # Fermi_vo["trigid"]=trigger_id

    rate__signif = toplevel_params['Trig_Signif']['value']
    rate__dur = toplevel_params['Trig_Dur']['value']
    Fermi_dic["ratesnr"] = rate__signif
    Fermi_dic["ratets"] = rate__dur
    Fermi_dic["snr"] = Fermi_dic["ratesnr"]
    Fermi_dic["descriptsnr"] = "SNR calculated from the onboard trigger"
    Fermi_dic["dur"] = Fermi_dic["ratets"]
    Fermi_dic["descriptdur"] = "Time scale employed by the onboard algorithm"

    # print("rate__signif",rate__signif)
    if float(rate__signif) < 4.0:
        Fermi_vo["voimportance"] = 3
    if ((float(rate__signif) >= 6.0) & (float(rate__signif) < 7.0)):
        Fermi_vo["voimportance"] = 3
    if ((float(rate__signif) > 7.0)):
        Fermi_vo["voimportance"] = 2

    isotime = v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Time.TimeInstant.ISOTime \
        .text

    delay = delay_fct(isotime)
    isotime_format = trigtime(isotime)
    Fermi_vo["trigtime"] = isotime_format
    delay_min = (delay.seconds) / 60.0
    Fermi_vo["trigdelay"] = delay_min

    # grb_proba = str(v.Why.Inference.attrib["probability"])
    grb_lc = toplevel_params['LightCurve_URL']['value']
    energy_bandmin = toplevel_params['Lo_Chan_Energy']['value']
    energy_bandmax = toplevel_params['Hi_Chan_Energy']['value']

    name_grb = gbm_lc_name(grb_lc)
    Fermi_vo["trigid"] = name_grb
    ra = str(v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Position2D. \
             Value2.C1.text)
    dec = str(v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Position2D. \
              Value2.C2.text)
    error2_radius = str(v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords. \
                        Position2D.Error2Radius.text)
    Fermi_vo["ra"] = ra
    Fermi_vo["dec"] = dec
    Fermi_vo["error"] = error2_radius
    Fermi_dic["locref"] = "GBM onboard"

    update_output_config(output_dic, Fermi_vo, "GRB")

    setup_cloud_event(output_dic)

    if (Fermi_vo["role"] != "test"):
        name_dic = "GBM" + trigger_id
        lalid = name_lalid(v, file_log_s, name_dic, Fermi_vo["letup"], "_DB")
        create_GRANDMAvoevent(lalid, Fermi_dic, Fermi_vo, output_dic, send2DB=Db_use)
        dic_grb[name_dic] = Fermi_dic
        dic_vo[name_dic] = Fermi_vo

        file_log_s.write(lalid + " " + str(trigger_id) + "\n")

    text = str("---------- \n") + str("FERMI/GBM alert \n") + str("---------- \n") + str("GRB NAME : ") \
           + str(name_grb) + (" ") + str("Trigger ID: ") + trigger_id + ("\n") + str("Trigger Time: ") + isotime + \
           ("\n") + str("Delay since alert: ") + str(delay) + ("\n") + str("\n") + str(
        "---Follow-up Advocate--\n") + str \
               ("FA on duty: ") + str(fa.FA_shift()) + "\n" + \
           str("\n") + str("---SPACE TRIGGER---\n") + str("Trigger Rate SNR ") + str(rate__signif) + " " + str \
               ("Trigger dur ") + rate__dur + ("\n") + str("\n") + str("---GRB CARAC---\n") + str("LC path: ") + str \
               (grb_lc) + str("\n") + str("Selected Energy band (keV): ") + str(energy_bandmin) + "-" + \
           str(energy_bandmax) + "\n" + message_obs + "\n" + "\n"
    return text


def fermi_trigger_follow(v, output_dic, message_type, file_log_s):
    """

    :param v:
    :param collab:
    :param message_type:
    :return:
    """

    toplevel_params = vp.get_toplevel_params(v)
    trigger_id = toplevel_params['TrigID']['value']
    name_dic = "GBM" + trigger_id
    Fermi_dic = GRB_dicf()
    Fermi_vo = VO_dicf()

    Fermi_vo["role"] = v.attrib['role']

    # Do not save test alerts, except when forced to True
    if force_Db_use:
        Db_use = force_Db_use
    else:
        if v.attrib['role'] != "test":
            Db_use = True
        else:
            Db_use = False


    message_obs = "No healpix skymap available"

    # pletter=Fermi_vo["letup"]
    # indice_pletter=np.where(letters==pletter)[0]
    # remove the letters as we can use Pkt_Ser_Num
    Fermi_vo["letup"] = toplevel_params['Pkt_Ser_Num']['value']

    grb_identified = str(v.What.Group.Param[0].attrib['value'])
    # long_short = "unknown"
    # rate__signif = 0.0
    # rate__dur = 0.0
    # prob_GRB=-1.0
    # hard_ratio=0.0
    # not_grb="unknown"

    Fermi_vo["ba"] = fa.FA_shift()

    Fermi_vo["eventype"] = "GRB"
    Fermi_vo["inst"] = "Fermi-GBM"
    Fermi_dic["inst"] = "Fermi-GBM"

    grb_lc = toplevel_params['LightCurve_URL']['value']

    name_grb = gbm_lc_name(grb_lc)
    Fermi_dic["lc"] = grb_lc
    Fermi_vo["grbid"] = name_grb

    isotime = v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Time.TimeInstant. \
        ISOTime.text
    delay = delay_fct(isotime)
    delay_min = (delay.seconds) / 60.0
    ra = v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Position2D.Value2.C1 \
        .text
    dec = v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Position2D.Value2.C2 \
        .text
    error2_radius = v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Position2D. \
        Error2Radius.text
    vo_ivorncit = v.Citations.EventIVORN
    Fermi_vo["ivorn"] = vo_ivorncit
    Fermi_vo["trigtime"] = trigtime(isotime)
    Fermi_vo["trigdelay"] = delay_min
    Fermi_vo["ra"] = ra
    Fermi_vo["dec"] = dec
    Fermi_vo["error"] = error2_radius

    if message_type == "FINAL FERMI/GBM POSITION MESSAGE":
        Fermi_dic["locref"] = "GBM final update"
        long_short = str(v.What.Group.Param[4].attrib['value'])
        Fermi_dic["longshort"] = long_short
        not_grb = def_not_grb = v.find(".//Param[@name='Def_NOT_a_GRB']").attrib['value']
        Fermi_dic["defGRB"] = not_grb
        loc_png = toplevel_params['LocationMap_URL']['value']
        png_name = (loc_png.split("/")[-1]).split("_")
        healpix_name = png_name[0] + "_" + "healpix" + "_" + png_name[2] + "_" + png_name[3].split(".")[0] + ".fit"
        path_helpcolec = loc_png.split("/")
        path_healpix = path_helpcolec[0]
        Fermi_vo["eventstatus"] = "Initial"
        for h in np.arange(len(path_helpcolec) - 1):
            if h != 0:
                path_healpix = path_healpix + "/" + path_helpcolec[h]
        link_healpix = path_healpix + "/" + healpix_name
        Fermi_vo["locpix"] = link_healpix
        Fermi_vo["iter_statut"] = 0

        update_output_config(output_dic, Fermi_vo, "GRB")

        setup_cloud_event(output_dic)

        message_obs = "Observation plan sent to "
        if (Fermi_vo["role"] != "test"):
            name_dic = "GBM" + trigger_id
            skypath = output_dic["skymappath"] + "/" + str(Fermi_vo["locpix"].split("/")[-1])
            if not os.path.isfile(skypath):
                command = "wget " + Fermi_vo["locpix"] + " -P ./HEALPIX/" + name_dic + "/"
                os.system(command)
            lalid = name_lalid(v, file_log_s, name_dic, Fermi_vo["letup"], "_DB")
            create_GRANDMAvoevent(lalid, Fermi_dic, Fermi_vo, "", output_dic, send2DB=Db_use)
            file_log_s.write(lalid + " " + str(trigger_id) + "\n")

            # Load params dictionary for gwemopt
            params = init_observation_plan(Fermi_vo, output_dic["skymappath"])
            print("Loading skymap...")
            # Function to read maps
            map_struct = gwemopt.utils.read_skymap(params, is3D=params["do3D"])

            for telescope in LISTE_TELESCOPE:
                Tel_dic = Tel_dicf()
                Tel_dic["Name"] = telescope
                message_obs = message_obs + " " + telescope
                aTable,galaxies_table = Observation_plan(telescope, Fermi_vo, trigger_id, params, map_struct)
                if aTable is None:
                    Tel_dic["OS"] = np.array([[], [], [], []])
                else:
                    Tel_dic["OS"] = np.transpose(np.array([aTable['rank_id'], aTable['RA'], aTable['DEC'], aTable['Prob']]))

                if (Fermi_vo["role"] != "test"):
                    name_dic = "GBM" + trigger_id
                    lalid = name_lalid(v, file_log_s, name_dic, Fermi_vo["letup"], "_" + Tel_dic["Name"])
                    create_GRANDMAvoevent(lalid, Fermi_dic, Fermi_vo, Tel_dic, output_dic)
                    file_log_s.write(lalid + " " + str(trigger_id) + "\n")

    if message_type == "FLIGHT UPDATE FERMI/GBM POSITION MESSAGE":
        Fermi_dic["locref"] = "GBM flight update"
        rate__signif = toplevel_params['Data_Signif']['value']
        Fermi_dic["locsnr"] = rate__signif
        rate__dur = toplevel_params['Data_Timescale']['value']
        Fermi_dic["locdur"] = rate__dur
        prob_GRB = toplevel_params['Most_Likely_Prob']['value']
        Fermi_dic["snr"] = Fermi_dic["locsnr"]
        Fermi_dic["descriptsnr"] = "SNR calculated from the localization flight pipeline"
        Fermi_dic["dur"] = Fermi_dic["locdur"]
        Fermi_dic["descriptdur"] = "Time scale employed by the final localization flight algorithm"
        Fermi_dic["probGRB"] = prob_GRB
        hard_ratio = toplevel_params['Hardness_Ratio']['value']
        Fermi_dic["hratio"] = hard_ratio
        not_grb = def_not_grb = v.find(".//Param[@name='Def_NOT_a_GRB']").attrib['value']
        Fermi_dic["defGRB"] = not_grb
        Fermi_vo["eventstatus"] = "preliminary"
        Fermi_vo["iter_statut"] = Fermi_vo["iter_statut"] + 1

        update_output_config(output_dic, Fermi_vo, "GRB")

        setup_cloud_event(output_dic)

        if (Fermi_vo["role"] != "test"):
            lalid = name_lalid(v, file_log_s, name_dic, Fermi_vo["letup"], "_DB")
            create_GRANDMAvoevent(lalid, Fermi_dic, Fermi_vo, "", output_dic, send2DB=Db_use)
            file_log_s.write(lalid + " " + str(trigger_id) + "\n")

    if message_type == "GROUND UPDATE FERMI/GBM POSITION MESSAGE":
        Fermi_dic["locref"] = "GBM ground update"
        rate__signif = toplevel_params['Burst_Signif']['value']
        Fermi_dic["locsnr"] = rate__signif
        rate__dur = toplevel_params['Data_Integ']['value']
        Fermi_dic["locdur"] = rate__dur
        Fermi_dic["snr"] = Fermi_dic["locsnr"]
        Fermi_dic["descriptsnr"] = "SNR calculated from the localization ground pipeline"
        Fermi_dic["dur"] = Fermi_dic["locdur"]
        Fermi_dic["descriptdur"] = "Time scale employed by the localization ground algorithm"
        not_grb = def_not_grb = v.find(".//Param[@name='Def_NOT_a_GRB']").attrib['value']
        Fermi_dic["defGRB"] = not_grb
        loc_fit = toplevel_params['LocationMap_URL']['value']
        Fermi_vo["locpix"] = loc_fit
        Fermi_vo["eventstatus"] = "preliminary"
        Fermi_vo["iter_statut"] = Fermi_vo["iter_statut"] + 1

        update_output_config(output_dic, Fermi_vo, "GRB")

        setup_cloud_event(output_dic)

        if (Fermi_vo["role"] != "test"):
            lalid = name_lalid(v, file_log_s, name_dic, Fermi_vo["letup"], "_DB")
            create_GRANDMAvoevent(lalid, Fermi_dic, Fermi_vo, "", output_dic, send2DB=Db_use)
            file_log_s.write(lalid + " " + str(trigger_id) + "\n")

    # print(grb_identified)
    if grb_identified == "false":
        text = "\n" + message_type + str(" \n") + ("---------- \n") + str("ID : ") + str(name_grb) + (
            " ") + trigger_id + (" ") + isotime + ("\n") + str \
                   ("Delay since alert: ") + str(delay) + ("\n") + str("---Follow-up Advocate--\n") + str \
                   ("FA on duty: ") + str(fa.FA_shift()) + ("\n") + message_obs + ("\n") + ("\n")
    else:
        text = "\n" + str("UPDATE FERMI/GBM POSITION MESSAGE NO LONGER CLASSIFIED AS GRB\n") + str \
            ("---------- \n") + str("ID : ") + str(name_grb) + (" ") + trigger_id + (" ") + isotime + ("\n") + str \
                   ("Delay since alert: ") + str(delay) + ("\n") + str("---Follow-up Advocate--\n") + str \
                   ("FA on duty: ") + str(fa.FA_shift()) + ("\n") + str("\n") + ("\n")
    return text


# A unique name per event
def name_lalid(v, file_log_s, name_dic, letter, tel):
    name_dic = name_dic + tel
    lalid = ""
    toplevel_params = vp.get_toplevel_params(v)
    time_now = datetime.datetime.utcnow()
    logfile_lines = file_log_s.readlines()
    Trigger_id = []

    if time_now.month < 10:
        if time_now.day < 10:
            lalid = "GRANDMA" + str(time_now.year) + "0" + str(time_now.month) + "0" + str(
                time_now.day) + "_" + name_dic + "_" + letter
        else:
            lalid = "GRANDMA" + str(time_now.year) + "0" + str(time_now.month) + str(
                time_now.day) + "_" + name_dic + "_" + letter
    else:
        lalid = "GRANDMA" + str(time_now.year) + str(time_now.month) + str(time_now.day) + "_" + name_dic + "_" + letter
    return lalid


# def create_GRBvoevent(lalid,instruselect,vo_trigid,vo_grbid,vo_trigtime,vo_trigtimeformat,vo_delay,vo_instru,vo_ratesnr,vo_imagesnr,vo_ratets,vo_ratehardness,vo_lc,vo_pra,vo_pdec,vo_perror,vo_ba,vo_ra,vo_dec,vo_error,vo_system_coor,vo_observa_loc,vo_importance, vo_reference,vo_ivorn,vo_snrloc,vo_durloc,vo_ls,vo_probGRB,vo_notGRB):

def add_GWvoeventcontent(GW_dic, v):
    # Removed as it give not correct information, need to fill the Event_status field
    #    if (GW_dic["Retraction"]=="O"):
    #        GW_dic["Retraction"]="1"
    if (GW_dic["Retraction"] == "1"):
        GW_dic["Retraction"] = "0"
    retractation = vp.Param(name="Prob", value=GW_dic["Retraction"], dataType="int", ucd="meta.number")
    retractation.Description = "Probability that the event is real"
    v.What.append(retractation)

    # hwinj = vp.Param(name="HardwareInj",value=GW_dic["HardwareInj"], ucd="meta.number",dataType="int")
    # hwinj.Description="Indicates that this event is a hardware injection if 1, no if 0"
    # v.What.append(hwinj)

    eventpage = vp.Param(name="Quicklook_url", value=GW_dic["EventPage"], ucd="meta.ref.url", dataType="string")
    eventpage.Description = "Web page for evolving status of this GW candidate"
    v.What.append(eventpage)

    lum = vp.Param(name="Distance", value=GW_dic["lum"], ucd="meta.number", dataType="float")
    lum.Description = "Luminosity distance (Mpc)"
    v.What.append(lum)

    errlum = vp.Param(name="Err_distance", value=GW_dic["errlum"], ucd="meta.number", dataType="float")
    errlum.Description = "Std for the luminosity distance (Mpc)"
    v.What.append(errlum)

    s50cr = vp.Param(name="50cr_skymap", value=GW_dic["50cr"], ucd="meta.number", dataType="float")
    s50cr.Description = "Sky localization area (50 pourcent confident region)"
    v.What.append(s50cr)

    s90cr = vp.Param(name="90cr_skymap", value=GW_dic["90cr"], ucd="meta.number", dataType="float")
    s90cr.Description = "Sky localization area (90 pourcent confident region)"
    v.What.append(s90cr)

    FAR = vp.Param(name="FAR", value=GW_dic["FAR"], ucd="arith.rate;stat.falsealarm", unit="Hz")
    FAR.Description = "Web page for evolving status of this GW candidate"
    v.What.append(FAR)

    Group = vp.Param(name="Group", value=GW_dic["Group"], ucd="meta.code", dataType="string")
    Group.Description = "Data analysis working group"
    v.What.append(Group)

    Pipeline = vp.Param(name="Pipeline", value=GW_dic["Pipeline"], ucd="meta.code", dataType="string")
    Group.Description = "Low-latency data analysis pipeline"
    v.What.append(Pipeline)

    BNS = vp.Param(name="BNS", value=GW_dic["BNS"], dataType="float", ucd="stat.probability")
    BNS.Description = "Probability that the source is a binary neutron star merger"
    NSBH = vp.Param(name="NSBH", value=GW_dic["NSBH"], dataType="float", ucd="stat.probability")
    NSBH.Description = "Probability that the source is a neutron star - black hole merger"
    BBH = vp.Param(name="BBH", value=GW_dic["BBH"], dataType="float", ucd="stat.probability")
    BBH.Description = "Probability that the source is a binary black hole merger"
    Terrestrial = vp.Param(name="Terrestrial", value=GW_dic["Terrestrial"], dataType="float", ucd="stat.probability")
    Terrestrial.Description = "Probability that the source is terrestrial (i.e., a background noise fluctuation or a glitch)"
    group_class = vp.Group(params=[BNS, NSBH, BBH, Terrestrial], name="Classification")
    group_class.Description = "Source classification: binary neutron star (BNS), neutron star-black hole (NSBH), binary black hole (BBH), or terrestrial (noise)"
    v.What.append(group_class)

    HasNS = vp.Param(name="HasNS", value=GW_dic["HasNS"], dataType="float", ucd="stat.probability")
    HasNS.Description = "Probability that at least one object in the binary has a mass that is less than 3 solar masses"
    HasRemnant = vp.Param(name="HasRemnant", value=GW_dic["HasRemnant"], dataType="float", ucd="stat.probability")
    HasRemnant.Description = "Probability that a nonzero mass was ejected outside the central remnant object"
    group_prop = vp.Group(params=[HasNS, HasRemnant], name="Properties")
    group_prop.Description = "Qualitative properties of the source, conditioned on the assumption that the signal is an astrophysical compact binary merger"
    v.What.append(group_prop)

    # v.What.append(GW_dic["Classification"])


def add_GRBvoeventcontent(GRB_dic, v):
    # GRB Parameters

    # grbid = Param(name="GRBID",value=GRB_dic["grbid"], ucd="meta.id")
    # grbid.set_Description(['GRB ID'])
    # what.add_Param(grbid)

    # trigonlinerate_snr = Param(name="Rate_snr",value=GRB_dic["ratesnr"], unit="sigma", ucd="stat.snr")
    # trigonlinerate_snr.set_Description(['Significance from the GRB rate onboard trigger algorithm of '+GRB_dic["inst"]])
    # what.add_Param(trigonlinerate_snr)

    snr_grb = vp.Param(name="Snr", value=str(GRB_dic["snr"]), unit="sigma", ucd="stat.snr", dataType="float")
    snr_grb.Description = GRB_dic["descriptsnr"]
    v.What.append(snr_grb)

    dur_grb = vp.Param(name="Dur", value=str(GRB_dic["dur"]), unit="s", ucd="time.interval", dataType="float")
    dur_grb.Description = GRB_dic["descriptdur"]
    print(str(GRB_dic["dur"]))
    v.What.append(dur_grb)

    # trigonlinerate_snr = vp.Param(name="Rate_snr",value=str(GRB_dic["ratesnr"]), unit="sigma", ucd="stat.snr",dataType="float")
    # trigonlinerate_snr.Description="Significance from the GRB rate onboard trigger algorithm of "+GRB_dic["inst"]
    # v.What.append(trigonlinerate_snr)

    # trigonlinerate_ts = Param(name="Rate_ts",value=GRB_dic["ratets"], unit="s", ucd="time.interval")
    # trigonlinerate_ts.set_Description = 'Timescale used in the GRB onboard pipeline of '+GRB_dic["inst"]
    # what.add_Param(trigonlinerate_ts)

    # trigonlinerate_ts = vp.Param(name="Rate_ts",value=str(GRB_dic["ratets"]), unit="s", ucd="time.interval",dataType="float")
    # trigonlinerate_ts.Description = "Timescale used in the GRB onboard pipeline of "+GRB_dic["inst"]
    # v.What.append(trigonlinerate_ts)

    # trigonlinerate_snr = Param(name="Img_snr",value=GRB_dic["imagesnr"], unit="sigma", ucd="stat.snr")
    # trigonlinerate_snr.set_Description(['Significance from the GRB image onboard pipeline of '+GRB_dic["inst"]])
    # what.add_Param(trigonlinerate_snr)

    # trigonlinerate_snr = vp.Param(name="Img_snr",value=str(GRB_dic["imagesnr"]), unit="sigma", ucd="stat.snr",dataType="float")
    # trigonlinerate_snr.Description="Significance from the GRB image onboard pipeline of "+GRB_dic["inst"]
    # v.What.append(trigonlinerate_snr)

    # lc = Param(name="LightCurve_URL",value=GRB_dic["lc"],ucd="meta.ref.url")
    # lc.Description(['The GRB LC_URL file will not be created/available until ~15 min after the trigger. Instrument:'+GRB_dic["inst"]])
    # what.add_Param(lc)

    lc = vp.Param(name="Quicklook_url", value=GRB_dic["lc"], ucd="meta.ref.url", dataType="string")
    lc.Description = "The GRB LC_URL file will not be created/available until ~15 min after the trigger. Instrument:" + \
                     GRB_dic["inst"]
    v.What.append(lc)

    # trigonlinerate_hardratio = Param(name="Hardness_Ratio",value=GRB_dic["hratio"], ucd="arith.ratio")
    # trigonlinerate_hardratio.set_Description(['GRB flight Spectral characteristics of '+GRB_dic["locref"]])
    # what.add_Param(trigonlinerate_hardratio)

    trigonlinerate_hardratio = vp.Param(name="Hardness_ratio", value=str(GRB_dic["hratio"]), ucd="arith.ratio",
                                        dataType="float")
    trigonlinerate_hardratio.Description = "GRB flight Spectral characteristics of " + GRB_dic["locref"]
    v.What.append(trigonlinerate_hardratio)

    longshort = vp.Param(name="Long_short", value=str(GRB_dic["longshort"]), dataType="string")
    longshort.Description = "GRB long-short of " + GRB_dic["locref"]
    v.What.append(longshort)

    # probGRB=Param(name="Prob_GRB",value=GRB_dic["probGRB"])
    # probGRB.set_Description(['Probability to be a GRB defined by '+GRB_dic["locref"]])
    # what.add_Param(probGRB)

    probGRB = vp.Param(name="Prob", value=str(GRB_dic["probGRB"]), dataType="float", ucd="meta.number")
    probGRB.Description = "Probability to be a GRB defined by " + GRB_dic["locref"]
    v.What.append(probGRB)

    # defGRB=Param(name="Def_NOT_a_GRB",value=GRB_dic["defGRB"])
    # defGRB.set_Description(['Not a GRB '+GRB_dic["locref"]])
    # what.add_Param(defGRB)

    # defGRB=vp.Param(name="Def_not_a_GRB",value=str(GRB_dic["defGRB"]),dataType="string")
    # defGRB.Description="Not a GRB "+GRB_dic["locref"]
    # v.What.append(defGRB)

    # orpos = Param(name="Loc_ref",value=GRB_dic["locref"])
    # orpos.set_Description(['Localization determined by '+GRB_dic["locref"]])
    # what.add_Param(orpos)

    orpos = vp.Param(name="Loc_ref", value=GRB_dic["locref"], dataType="string", ucd="meta.ref.url")
    orpos.Description = "Localization determined by " + GRB_dic["locref"]
    v.What.append(orpos)

    # snrloc = Param(name="Loc_snr",value=GRB_dic["locsnr"])
    # snrloc.Description = 'Fight/Ground position snr to calculate the position of '+GRB_dic["locref"]
    # what.add_Param(snrloc)

    # snrloc = vp.Param(name="Loc_snr",value=str(GRB_dic["locsnr"]),unit="sigma", ucd="stat.snr",dataType="float")
    # snrloc.Description ="Fight/Ground position snr to calculate the position of "+GRB_dic["locref"]
    # v.What.append(snrloc)

    # durloc = Param(name="Loc_dur",value=GRB_dic["locdur"])
    # durloc.set_Description(['Fight/Ground timescale to calculate the position of '+GRB_dic["locref"]])
    # what.add_Param(durloc)

    # durloc = vp.Param(name="Loc_dur",value=str(GRB_dic["locdur"]), unit="s", ucd="time.interval",dataType="float")
    # durloc.Description="Fight/Ground timescale to calculate the position of "+GRB_dic["locref"]
    # v.What.append(durloc)


def create_GRANDMAvoevent(lalid, Trigger_dic, VO_dic, Tel_dic, output_dic, send2DB=False):
    """
    Create the VOEvent
    """

    vo_name = lalid + ".xml"

    lalid_bis = lalid.split("_")
    VO_dic["streamid"] = ""
    for h in np.arange(len(lalid_bis)):
        VO_dic["streamid"] = VO_dic["streamid"] + lalid_bis[h]

    v = vp.Voevent(stream=VO_dic["stream"], stream_id=VO_dic["streamid"], role=VO_dic["role"])

    vp.set_who(v, date=datetime.datetime.utcnow(), author_ivorn=VO_dic["authorivorn"])

    vp.set_author(v, contactName=VO_dic["contactName"])
    vp.set_author(v, shortName=VO_dic["shortName"])
    vp.set_author(v, contactPhone=VO_dic["contactPhone"])
    vp.set_author(v, contactEmail=VO_dic["contactEmail"])

    # Now create some Parameters for entry in the 'What' section.

    server = vp.Param(name="VOLocat", value=VO_dic["locationserver"])
    server.Description = 'VOevent stored'

    trigid = vp.Param(name="Event_ID", value=VO_dic["trigid"], ucd="meta.id", dataType="string")
    trigid.Description = "Trigger ID"
    v.What.append(trigid)

    # alertype = Param(name="Event_type", value=VO_dic["eventype"])
    # alertype.set_Description(["Type of the event"])
    # what.add_Param(alertype)

    alertype = vp.Param(name="Event_type", value=VO_dic["eventype"], ucd="meta.id", dataType="string")
    alertype.Description = "Type of the alert"
    v.What.append(alertype)

    alerstatus = vp.Param(name="Event_status", value=VO_dic["eventstatus"], ucd="meta.version", dataType="string")
    alerstatus.Description = "Event status (preliminary, initial, update, retractation)"
    alerstatus_iter = vp.Param(name="Revision", value=str(VO_dic["iter_statut"]), ucd="meta.number", dataType="int")
    alerstatus_iter.Description = "Revision Number"
    status_alerts = vp.Group(params=[alerstatus, alerstatus_iter], name="Status")
    status_alerts.Description = "Preliminary is set when there is not healpix skymap, then initial and then updates"
    v.What.append(status_alerts)

    triginstru = vp.Param(name="Event_inst", value=VO_dic["inst"], ucd="meta.code", dataType="string")
    triginstru.Description = "Instrument which originated the alert"
    v.What.append(triginstru)

    pixloc = vp.Param(name="Loc_url", value=str(VO_dic["locpix"]), ucd="meta.ref.url", dataType="string")
    # print("cc",VO_dic["locpix"])
    pixloc.Description = "The url location of healpix skymap"
    v.What.append(pixloc)

    fa = vp.Param(name="FA", value=VO_dic["ba"], dataType="string", ucd="meta.code")
    fa.Description = "GRANDMA follow-up advocate on duty at the time of the VO alert"
    v.What.append(fa)

    if VO_dic["eventype"] == "GRB":
        add_GRBvoeventcontent(Trigger_dic, v)

    if VO_dic["eventype"] == "GW":
        add_GWvoeventcontent(Trigger_dic, v)
        skymap_folder = output_dic["skymappath"]

    if Tel_dic != "":
        Name_tel = vp.Param(name="Name_tel", value=str(Tel_dic["Name"]), ucd="instr", dataType="string")
        Name_tel.Description = "Name of the telescope used for the observation strategy"
        FOV_tel = vp.Param(name="FOV", value=str(Tel_dic["FOV"]), ucd="instr.fov", dataType="float", unit="deg")
        FOV_tel.Description = "FOV of the telescope used for the observation strategy"
        FOV_coverage = vp.Param(name="FOV_coverage", value=str(Tel_dic["FOV_coverage"]), ucd="instr.fov",
                                dataType="string")
        FOV_coverage.Description = "Shape of the FOV for the telescope used for the observation strategy"
        magnitude = vp.Param(name="Mag_limit", value=str(Tel_dic["magnitude"]), ucd="phot.magr", dataType="float",
                             unit="mag")
        magnitude.Description = "Magnitude limit of the telescope used for the observation strategy"
        exposuretime = vp.Param(name="exposure", value=str(Tel_dic["exposuretime"]), ucd="obs.exposure",
                                dataType="float", unit="s")
        exposuretime.Description = "Exposure time of the telescope used for the observation strategy"
        slewrate = vp.Param(name="Slew_rate", value=str(Tel_dic["slew_rate"]), ucd="time.interval", dataType="float",
                            unit="s")
        slewrate.Description = "Slew rate of the telescope for the observation strategy"
        readout = vp.Param(name="Readout", value=str(Tel_dic["readout"]), ucd="time.interval", dataType="float",
                           unit="s")
        readout.Description = "Read out of the telescope used for the observation strategy"
        filt = vp.Param(name="Filters_tel", value=str(Tel_dic["filt"]), ucd="instr.filter", dataType="string")
        filt.Description = "Filters of the telescope used for the observation strategy"
        latitude = vp.Param(name="Latitude", value=str(Tel_dic["latitude"]), ucd="meta.number", dataType="float",
                            unit="deg")
        latitude.Description = "Latitude of the observatory"
        longitude = vp.Param(name="Longitude", value=str(Tel_dic["longitude"]), ucd="meta.number", dataType="float",
                             unit="deg")
        longitude.Description = "Longitude of the observatory"
        elevation = vp.Param(name="Elevation", value=str(Tel_dic["elevation"]), ucd="meta.number", dataType="float",
                             unit="m")
        elevation.Description = "Elevation of the observatory"
        config_obs = vp.Group(
            params=[Name_tel, FOV_tel, FOV_coverage, magnitude, exposuretime, slewrate, readout, filt, latitude,
                    longitude, elevation], name="Set_up_OS")
        config_obs.Description = "Set-up parameters for producing the observation strategy"
        v.What.append(config_obs)

        # OS_plan=vp.Param(name="Observation strategy",type="Table",value=Tel_dic["OS"])
        # OS_plan=vp.Param(name="Observation strategy")
        # OS_plan.Description="The list of tiles for "+str(Tel_dic["Name"])
        # OS_plan.Table=vp.Param(name="Table")
        if len(Tel_dic["OS"]) > 0 and len(Tel_dic["OS"][0]) > 0:
            obs_req = vp.Param(name="Obs_req", value="1", ucd="meta.number", dataType="int")
            obs_req.Description = "Set to 1 if observation are required, 0 to stop the observations"
            v.What.append(obs_req)
            Fields = objectify.Element("Table", name="Obs_plan")
            Fields.Description = "Tiles for the observation plan"
            grid_id = objectify.SubElement(Fields, "Field", name="Grid_id", ucd="", unit="", dataType="int")
            grid_id.Description = "ID of the grid of FOV"
            if Tel_dic["Name"] == "GWAC":
                field_id = objectify.SubElement(Fields, "Field", name="Field_id", ucd="", unit="", dataType="string")
                field_id.Description = "ID of the field of FOV"
            ra = objectify.SubElement(Fields, "Field", name="Ra", ucd="pos.eq.ra ", unit="deg", dataType="float")
            ra.Description = "The right ascension at center of fov in equatorial coordinates"
            dec = objectify.SubElement(Fields, "Field", name="Dec", ucd="pos.eq.ra ", unit="deg", dataType="float")
            dec.Description = "The declination at center of fov in equatorial coordinates"
            Os_grade = objectify.SubElement(Fields, "Field", name="Os_grade", ucd="meta.number", unit="None",
                                            dataType="float")
            Os_grade.Description = "Gives the importance of the tile/galaxy to observe"
            Data = objectify.SubElement(Fields, "Data")
            for i in np.arange(len(Tel_dic["OS"])):
                TR = objectify.SubElement(Data, "TR")
                for j in np.arange(len(Tel_dic["OS"][i])):
                    # objectify.SubElement(TR, "TD",value=str(Tel_dic["OS"][i][j]))
                    objectify.SubElement(TR, 'TD')
                    TR.TD[-1] = str(Tel_dic["OS"][i][j])
            v.What.append(Fields)
        else:
            obs_req = vp.Param(name="Obs_req", value="0", ucd="meta.number", dataType="int")
            obs_req.Description = "Set to 1 if observation are required, 0 to stop the observations"
            v.What.append(obs_req)

    vp.add_where_when(v, coords=vp.Position2D(ra=VO_dic["ra"], dec=VO_dic["dec"], err=VO_dic["error"], units='deg',
                                              system=vp.definitions.sky_coord_system.utc_fk5_geo),
                      obs_time=VO_dic["trigtime"], observatory_location=VO_dic["location"])

    if not VO_dic["voimportance"]:
        vp.add_why(v, importance=99)
    else:
        vp.add_why(v, importance=VO_dic["voimportance"])
    v.Why.Description = "Internal Ranking for the event (from 1 : most interesting to 3 for least interesting)"

    # Check everything is schema compliant:
    # vp.assert_valid_as_v2_0(v)
    file_voevent = output_dic["vopath"] + vo_name
    try:
        vp.assert_valid_as_v2_0(v)
    except Exception as e:
        print(e)

    with open(file_voevent, 'wb') as f:
        vp.dump(v, f)

    if send2DB:
        # Send information contained in VOEvent to database
        # First try with variable v
        try:
            send_VOE_alert_to_DB(v, skymap_folder,  str(VO_dic["locpix"].split("/")[-1]), proba=90, isfile=False)
        # If failed, load the xml file. Might only do this in future.
        except:
            send_VOE_alert_to_DB(file_voevent, skymap_folder,  str(VO_dic["locpix"].split("/")[-1]), proba=90, isfile=True)

    return file_voevent


def GW_trigger(v, output_dic, file_log_s):
    toplevel_params = vp.get_toplevel_params(v)
    # READ MESSAGE
    trigger_id = toplevel_params['GraceID']['value']
    AlertType = toplevel_params['AlertType']['value']
    testRetract = int(toplevel_params['Packet_Type']['value'])
    if (testRetract == 164):
        Retractation = 1
    else:
        Retractation = 0

    text_mes = ""

    if (Retractation == 1):
        message_type = "GW RETRACTION POSITION MESSAGE"
        text_mes = GW_trigger_retracted(v, output_dic, file_log_s)
        print(text_mes)
        return text_mes

    text_mes = GW_treatment_alert(v, output_dic, file_log_s)

    return text_mes


def fermi_trigger(v, output_dic, collab, file_log_s):
    """
    Function used to process Fermi GCN notices
    :param v: VO event in dicitonary format to process
    :param output_dic: dictionary with ouput files configuration
    :param collab: string proving info on the instrument
    :return:
    """
    text_mes=""

    print('Fermi trigger')
    instru = str(collab[2])
    if instru == "GBM":
        toplevel_params = vp.get_toplevel_params(v)
        # READ MESSAGE
        trigger_id = toplevel_params['TrigID']['value']
        message_descrip = str(v.What.Description).split()

        if "transient." and "found" in message_descrip:
            id_grb_message_init = v.attrib['ivorn']
            text_mes = fermi_trigger_found(v, output_dic, file_log_s)

        if "location" in message_descrip:
            id_grbmessage_follow = v.attrib['ivorn']
            id_follow_up_message = str(v.Citations.EventIVORN)
            messages = str(id_grbmessage_follow).split("#")[1].split("_")
            message_type = ""
            if "Flt" in messages:
                message_type = "FLIGHT UPDATE FERMI/GBM POSITION MESSAGE"
            if "Gnd" in messages:
                message_type = "GROUND UPDATE FERMI/GBM POSITION MESSAGE"
            if "Fin" in messages:
                message_type = "FINAL FERMI/GBM POSITION MESSAGE"
            # IDFOLLOWUP_message
            text_mes = fermi_trigger_follow(v, output_dic, message_type, file_log_s)

    return text_mes


def slack_message(slack_channel, text_mes):
    """

    :param slack_channel:
    :param text_mes:
    :return:
    """
    slack_token = os.environ["SLACK_API_TOKEN"]
    sc = SlackClient(slack_token)
    sc.api_call(
        "chat.postMessage",
        channel=slack_channel,
        text=text_mes
    )


def send_data(telescope_list, params, aTables, galaxies_table, GW_vo, GW_dic, trigger_id, v, file_log_s, path_config, output_dic, message_obs, name_dic, Db_use=False, gal2DB=False):
    """ Send observation plans to DB and send xml files to broker """

    for i_tel, telescope in enumerate(telescope_list):
        print (telescope) 
        if telescope not in ["GWAC", "TRE", "TCA", "TCH", "OAJ"]:
   
            if not params["doUseCatalog"]:
                print("Observation plan for {} is not computed as we don't use the galaxies catalog at this distance".format(telescope))
                continue
                
        Tel_dic = Tel_dicf()
        Tel_dic["Name"] = telescope
        message_obs = message_obs + " " + telescope

        # Store galaxies only once in DB. Need to change if we do not use galaxies for some telescopes
        if i_tel == 0 and params["doUseCatalog"] and gal2DB:
            storeGal = True
        else:
            storeGal = False

        aTable = aTables[telescope]            
           
        if aTable is None:
            Tel_dic["OS"] = np.array([[], [], [], []])
        elif telescope == 'F60':
            gal_id = []
            for gal in galaxies_table:
                if (gal['GWGC'] != '---') and (gal['GWGC']):
                    gal_name = gal['GWGC']
                    gal_catalog = 'GWGC'
                #elif (gal['PGC'] != '--') and (gal['PGC']):
                #    gal_name = gal['PGC']
                #    gal_catalog = 'PGC'
                elif (gal['HyperLEDA'] != '---') and (gal['HyperLEDA']):
                    gal_name = gal['HyperLEDA']
                    gal_catalog = 'HyperLEDA'
                elif (gal['2MASS'] != '---') and (gal['2MASS']):
                    gal_name = gal['2MASS']
                    gal_catalog = '2MASS'
                elif (gal['SDSS'] != '---') and (gal['SDSS']):
                    gal_name = gal['SDSS']
                    gal_catalog = 'SDSS-DR12'
                
                gal_id.append(gal_name)
                
            Tel_dic["OS"] = np.transpose(np.array([gal_id, galaxies_table['RAJ2000'], galaxies_table['DEJ2000'], galaxies_table['S']]))
        else:
            Tel_dic["OS"] = np.transpose(np.array([aTable['rank_id'], aTable['RA'], aTable['DEC'], aTable['Prob']]))
            if Db_use :
                print ('Sending observation plan and tiles to database.')
                send_ObsPlan_to_DB(aTable.meta, GW_vo["eventstatus"], GW_vo["iter_statut"])
                send_ObsPlan_tiles_to_DB(aTable, GW_vo["eventstatus"], GW_vo["iter_statut"], output_dic["skymappath"], str(GW_vo["locpix"].split("/")[-1]))
        if Db_use and galaxies_table is not None:
            if storeGal: 
                print ('Sending list of galaxies to database. This can take some time. (done only once per event)')
                send_ObsPlan_galaxies_to_DB(galaxies_table, trigger_id, GW_vo["eventstatus"], GW_vo["iter_statut"])
            if aTable is not None:
                print ('Linking each galaxy to a tile and store information in database.')
                send_link_galaxies_tiles_to_DB(aTable, galaxies_table, trigger_id, GW_vo["eventstatus"], GW_vo["iter_statut"])

        lalid = name_lalid(v, file_log_s, name_dic, GW_vo["letup"], "_" + Tel_dic["Name"])
        filename_vo = create_GRANDMAvoevent(lalid, GW_dic, GW_vo, Tel_dic, output_dic)

        #only send plan if this is the first one
        #I can see here a problem : if the first LVC notice does not have any skymap, we will not send the plan at all
        # may be need to retreive info if the alert has already been received and send from DB
        if GW_vo["letup"] == 'a':
            send_voevent(path_config + '/broker.json', filename_vo)

        file_log_s.write(lalid + " " + str(trigger_id) + "\n")




def send_VOE_alert_to_DB(VOE_alert, path_skymap, filename_skymap, proba=90, isfile=False):
    """ Store information of the alert's VOEvent in GRANDMA database """

    # VOE_file = "VOEVENTS/" + lalid + ".xml"

    # Instanciate the VOEparser class.
    # Set file = False if VOE passed as an argument
    VOE = VOEparser(VOE_alert, isfile=isfile)

    # Load the VOE event
    VOE.load_voevent()

    # Store parameters in a dictionary
    VOE_dict = VOE.get_parameters()

    # Instanciate the populate_DB class with the json config file
    db = populate_DB(path_config=path_config, filename_config="db_config.json")
    # Connect to DB
    db.connect_db()

    # Fill the events table in the database.
    # If there is a skymap, set the path, filename,
    # and desired cumulative probability to use (50 or 90 for instance)
    db.fill_events(VOE_dict, path_skymap=path_skymap, filename_skymap=filename_skymap, proba=proba)

    # Fill the event_properties table in the database
    db.fill_event_properties(VOE_dict)

    # Close connection wih database
    db.close_db()

def send_ObsPlan_to_DB(tiles_table_meta, status, revision):
    """ Store observation plan for a given telescope in GRANDMA database """

    # Instanciate the populate_DB class with the json config file
    db = populate_DB(path_config=path_config, filename_config="db_config.json")

    # Connect to DB
    db.connect_db()

    # Fill the tiles table in the database
    db.fill_obs_plan(tiles_table_meta, status, revision)

    # Close connection wih database
    db.close_db()



def send_ObsPlan_tiles_to_DB(tiles_table, status, revision, path_skymap, filename_skymap, proba=90):
    """ Store observation plan of tiles for a given telescope in GRANDMA database """

    # Instanciate the populate_DB class with the json config file
    db = populate_DB(path_config=path_config, filename_config="db_config.json")

    # Connect to DB
    db.connect_db()

    # Fill the tiles table in the database
    db.fill_tiles(tiles_table, status, revision, path_skymap=path_skymap, filename_skymap=filename_skymap, proba=proba)

    # Close connection wih database
    db.close_db()


def send_ObsPlan_galaxies_to_DB(galaxies_table, trigger_id, status, revision):
    """ Store observation plan for galaxies in GRANDMA database """

    # Instanciate the populate_DB class with the json config file
    db = populate_DB(path_config=path_config, filename_config="db_config.json")
    # Connect to DB
    db.connect_db()

    # Fill the galaxies table in the database
    db.fill_galaxies(galaxies_table, trigger_id, status, revision)

    # Close connection wih database
    db.close_db()

def send_link_galaxies_tiles_to_DB(tiles_table, galaxies_table, trigger_id, status, revision):
    """ Store link between galaxies and tiles in GRANDMA database"""

    # Instanciate the populate_DB class with the json config file
    db = populate_DB(path_config=path_config, filename_config="db_config.json")
    # Connect to DB
    db.connect_db()

    # Fill the galaxies table in the database
    db.fill_link_tiles_galaxies(tiles_table, galaxies_table, trigger_id, status, revision)

    # Close connection wih database
    db.close_db()


def update_output_config(outputdic, vo_dic, evttype):
    """

    :param outputdic:
    :param Vo_dict:
    :return:
    """
    outputdic["trigid"] = vo_dic["trigid"]
    outputdic["type_ser"] = vo_dic["eventstatus"] + "_" + str(vo_dic["iter_statut"])
    outputdic["evt_type"] = evttype

def setup_cloud_event(outputdic):
    """ Setup the structure of directories in the GRANDMA cloud
        Definition is based on issue """

    # create the main directory to store all the infos related to the event
    dir_evt_name = outputdic["output_dir"] + '/' + outputdic["evt_type"] + '/' + outputdic["trigid"]
    if not os.path.exists(dir_evt_name):
        os.makedirs(dir_evt_name)

    # create the subdir for the given message received
    dir_ser_name =  dir_evt_name + '/' + outputdic["type_ser"] + '/'
    if not os.path.exists(dir_ser_name):
        os.makedirs(dir_ser_name)

    # create subdir for obs request
    dir_obs_name = dir_ser_name + outputdic["Obsrequest"] + '/'
    if not os.path.exists(dir_obs_name):
        os.makedirs(dir_obs_name)

    # create subdir for OT
    dir_ot_name = dir_ser_name + outputdic["OT"] + '/'
    if not os.path.exists(dir_ot_name):
        os.makedirs(dir_ot_name)

    outputdic["skymappath"] = dir_ser_name
    outputdic["vopath"] = dir_obs_name

def send_voevent(broker_config,filename_vo):
    with open(broker_config) as f:
        broker_config = json.load(f)

    cmd = "%s --host=%s --port=%s -f " % (broker_config['path'],
                                          broker_config['host'],
                                          broker_config['port'])
    cmd = cmd + filename_vo
    os.system(cmd)


def online_processing(v,role_filter='observation'):
    #    v = vp.loads(playload)
    collab = ""
    text_mes = ""

    output_dic = define_output_config(path_config + 'outputdirgrandma.json', v.attrib['role'])
    #for GRANDMA the json need also to contain : Obsrequest, OT, evt_type, trigid, type_ser

    if(v.attrib['role'] != role_filter):
        return

    file_log_r = open(LOGFILE_receivedalerts, "a+")

    file_log_s = open(LOGFILE_sendingalerts, "a+")

    try:
        collab = str(v.How['Description'])
    except AttributeError:
        contact = str(v.Who.Author.contactName)
        if "LIGO" in contact.split():
            collab = "gravitational"

    # is it a test alert or a real trigger and send via slack
    # Get config params from json file
    with open(path_config + 'slack.json') as f:
        slack_config = json.load(f)

    toplevel_params = vp.get_toplevel_params(v)
    LVC_status = toplevel_params['AlertType']['value']
    trigger_id = toplevel_params['GraceID']['value']
    event_page = toplevel_params['EventPage']['value']

    text_m = str("---------- \n") + str("new GCN notice \n") + str("---------- \n") \
            + str("ID : ") + str(trigger_id) + str(" with status ") +  str(LVC_status) + "\n" \
            + str("EventPage: ") + str(event_page) + "\n" + str("Observation program launched ") + "\n"
    """
    if v.attrib['role'] == "test":
        slack_channel_alert = "#testalerts"
        cmd ="curl -X POST -H 'Content-type: application/json' --data '{\"text\":\" "+ text_m+"\"}' %s" % slack_config['url_test']
        os.system(cmd)
    if v.attrib['role'] == "observation":
        if ("Swift" in collab.split()) or ("Fermi" in collab.split()):
            cmd ="curl -X POST -H 'Content-type: application/json' --data '{\"text\":\" "+ text_m+"\"}' %s" % slack_config['url_swift']
            os.system(cmd)
        if "gravitational" in collab.split():
            cmd ="curl -X POST -H 'Content-type: application/json' --data '{\"text\":\" "+ text_m+"\"}' %s" % slack_config['url_GW']
            os.system(cmd)
    """

    # which instrument comes from the alert Swift or Fermi ?
    if "Swift" in collab.split():
        text_mes = swift_trigger(v, output_dic, collab, file_log_s)
    if "Fermi" in collab.split():
        text_mes = fermi_trigger(v, output_dic, collab, file_log_s)
    if "gravitational" in collab.split():
        text_mes = GW_trigger(v, output_dic, file_log_s)

    """
    if v.attrib['role'] == "test":
        slack_channel_alert = "#testalerts"
        cmd ="curl -X POST -H 'Content-type: application/json' --data '{\"text\":\" "+ text_mes+"\"}' %s" % slack_config['url_test']
        os.system(cmd)
    if v.attrib['role'] == "observation":
        if ("Swift" in collab.split()) or ("Fermi" in collab.split()):
            cmd ="curl -X POST -H 'Content-type: application/json' --data '{\"text\":\" "+ text_mes+"\"}' %s" % slack_config['url_swift']
            os.system(cmd)
        if "gravitational" in collab.split():
            cmd ="curl -X POST -H 'Content-type: application/json' --data '{\"text\":\" "+ text_mes+"\"}' %s" % slack_config['url_GW']
            os.system(cmd)
#    print(text_mes)
     """


letters = np.array(
    ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w",
     "x", "y", "z"])
LOGFILE_receivedalerts = "LOG_ALERTS_RECEIVED.txt"
LOGFILE_sendingalerts = "LOG_ALERTS_SENT.txt"
LISTE_TELESCOPE_TILING = []
LISTE_TELESCOPE_GALAXY = []
LISTE_TELESCOPE = LISTE_TELESCOPE_TILING + LISTE_TELESCOPE_GALAXY

dic_grb = {}
dic_vo = {}
