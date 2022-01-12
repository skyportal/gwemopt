#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Modified by S. Antier 2022"

import os
import numpy as np
import copy
from astropy.io import fits, ascii
from astropy import table
from astropy import time
# from astropy.table import Table
import gwemopt.utils
import gwemopt.moc
#import gwemopt.gracedb
import gwemopt.rankedTilesGenerator
import gwemopt.waw
import gwemopt.lightcurve
import gwemopt.coverage
import gwemopt.efficiency
import gwemopt.plotting
import gwemopt.tiles
import gwemopt.segments
import gwemopt.catalog


def Observation_plan_multiple(telescopes, eventtime, trigger_id, params, map_struct_input, obs_mode):
    tobs = None
    map_struct = copy.deepcopy(map_struct_input)

    event_time = time.Time(eventtime, scale='utc')


    config_directory = params["configDirectory"]

    for telescope in telescopes:
        config_file = "%s/%s.config" % (config_directory, telescope)
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
        params["doCatalog"] = True
        params["writeCatalog"] = True

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
    else:
        print("""Need to enable --doEvent, --doFootprint,
              --doSkymap, or --doDatabase""")
        exit(0)

    if tobs is None:
        now_time = time.Time.now()
        timediff = now_time.gps - event_time.gps
        timediff_days = timediff / 86400.0
        #  Start observation plan 1h after execution of this code
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

    # Initialise table
    tiles_table = None

    if params["doCatalog"]:
        print('get catalog')
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
            params["config"][telescope]["tesselation"] = np.empty((0, 3))
            tiles_struct = tile_structs[telescope]
            for index in tiles_struct.keys():
                ra, dec = tiles_struct[index]["ra"], tiles_struct[index]["dec"]
                params["config"][telescope]["tesselation"] = np.append(params["config"][telescope]["tesselation"],
                                                                       [[index, ra, dec]], axis=0)
    elif params["tilesType"] == "greedy":
        print("Generating greedy struct...")
        tile_structs = gwemopt.tiles.greedy(params, map_struct)
        for telescope in params["telescopes"]:
            params["config"][telescope]["tesselation"] = np.empty((0, 3))
            tiles_struct = tile_structs[telescope]
            for index in tiles_struct.keys():
                ra, dec = tiles_struct[index]["ra"], tiles_struct[index]["dec"]
                params["config"][telescope]["tesselation"] = np.append(params["config"][telescope]["tesselation"],
                                                                       [[index, ra, dec]], axis=0)
    elif params["tilesType"] == "galaxy":
        print("Generating galaxy struct...")
        tile_structs = gwemopt.tiles.galaxy(params, map_struct, catalog_struct)
        for telescope in params["telescopes"]:
            params["config"][telescope]["tesselation"] = np.empty((0, 3))
            tiles_struct = tile_structs[telescope]
            for index in tiles_struct.keys():
                ra, dec = tiles_struct[index]["ra"], tiles_struct[index]["dec"]
                params["config"][telescope]["tesselation"] = np.append(params["config"][telescope]["tesselation"],
                                                                       [[index, ra, dec]], axis=0)
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
            # prob = tile_structs[telescope][ii]["prob"]

            ra, dec = data[0], data[1]
            # exposure_time, field_id, prob = data[4], data[5], data[6]
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
            if telescope in ["GWAC", "F30", "CGFT"]:
                print("tel : " + telescope)
                tessfile = np.loadtxt(config_struct["tesselationFile"])
                for i in range(len(field_id_vec)):
                    # rank_id.append(str(int(params["config"][telescope]["tesselation"][field_id_vec[i]][0])).zfill(8))
                    # print("field RA Dec "+str(int(params["config"][telescope]["tesselation"][field_id_vec[i]][0]))+" "+str(ra_vec[i])+" "+str(dec_vec[i]))
                    # need to find indices myslef
                    rows = np.where(tessfile[:, 1] == ra_vec[i])
                    #                    r2 = np.where(tessfile[rows,2]==dec_vec[i])
                    #                    f_id = tessfile[rows[r2[0][0]][0]][0]
                    r2 = np.where(tessfile[rows, 2][0] == dec_vec[i])
                    f_id = tessfile[rows[0][r2[0][0]]][0]
                    rank_id.append(str(int(f_id)).zfill(8))
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
                        sorted_corners_str = "[[%.3f, %.3f], [%.3f, %.3f]]" % (
                        unsorted_corners[0][0], unsorted_corners[0][1], unsorted_corners[1][0], unsorted_corners[1][1])
                        sorted_corners_list = [[unsorted_corners[0][0], unsorted_corners[0][1]],
                                               [unsorted_corners[1][0], unsorted_corners[1][1]]]

                    elif unsorted_corners.shape[0] == 3:
                        sorted_corners_str = "[[%.3f, %.3f], [%.3f, %.3f], [%.3f, %.3f]]" % (
                        unsorted_corners[0][0], unsorted_corners[0][1], unsorted_corners[1][0], unsorted_corners[1][1],
                        unsorted_corners[2][0], unsorted_corners[2][1])
                        sorted_corners_list = [[unsorted_corners[0][0], unsorted_corners[0][1]],
                                               [unsorted_corners[1][0], unsorted_corners[1][1]],
                                               [unsorted_corners[2][0], unsorted_corners[2][1]]]

                    elif unsorted_corners.shape[0] == 4:
                        sorted_corners_str = "[[%.3f, %.3f], [%.3f, %.3f], [%.3f, %.3f], [%.3f, %.3f]]" % (
                        unsorted_corners[0][0], unsorted_corners[0][1], unsorted_corners[1][0], unsorted_corners[1][1],
                        unsorted_corners[3][0], unsorted_corners[3][1], unsorted_corners[2][0], unsorted_corners[2][1])
                        sorted_corners_list = [[unsorted_corners[0][0], unsorted_corners[0][1]],
                                               [unsorted_corners[1][0], unsorted_corners[1][1]],
                                               [unsorted_corners[3][0], unsorted_corners[3][1]],
                                               [unsorted_corners[2][0], unsorted_corners[2][1]]]
                    else:
                        sorted_corners_str = "[]"
                        sorted_corners_list = []
                except:
                    sorted_corners_str = "[]"
                    sorted_corners_list = []

                tiles_corners_str.append(sorted_corners_str)
                tiles_corners_list.append(sorted_corners_list)

            #  Limit number of tiles sent to each telescope
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

            tiles_table = table.Table(
                [list(rank_id), field_id_vec, ra_vec, dec_vec, grade_vec, tiles_corners_str, tiles_corners_list],
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

    galaxies_table = None

    if params["doCatalog"] == True:
        # Store gaalxies in an astropy table
        filename_gal = params["outputDir"] + '/catalog.csv'

        galaxies_table = ascii.read(filename_gal, format='csv')

    return tiles_tables, galaxies_table
