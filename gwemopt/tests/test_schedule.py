import datetime
import os
import glob
import copy

from astropy import table
from astropy import time
import ephem
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
import healpy as hp
from ligo import segments
import numpy as np

def params_struct(skymap, gpstime, tobs=None, filt=['r'],
                  exposuretimes=[60.0],
                  mindiff=30.0*60.0, probability=0.9, tele='ZTF',
                  schedule_type='greedy',
                  doReferences=True,
                  filterScheduleType='block'):

    gwemoptpath = os.path.dirname(gwemopt.__file__)
    config_directory = os.path.join(gwemoptpath, '../config')
    tiling_directory = os.path.join(gwemoptpath, '../tiling')
    catalog_directory = os.path.join(gwemoptpath, '../catalog')

    params = {}
    params["config"] = {}
    config_files = glob.glob("%s/*.config" % config_directory)
    for config_file in config_files:
        telescope = config_file.split("/")[-1].replace(".config", "")
        if not telescope == tele: continue
        params["config"][telescope] =\
            gwemopt.utils.readParamsFromFile(config_file)
        params["config"][telescope]["telescope"] = telescope
        if "tesselationFile" in params["config"][telescope]:
            params["config"][telescope]["tesselationFile"] =\
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

            params["config"][telescope]["tesselation"] =\
                np.loadtxt(params["config"][telescope]["tesselationFile"],
                           usecols=(0, 1, 2), comments='%')

        if "referenceFile" in params["config"][telescope]:
            params["config"][telescope]["referenceFile"] =\
                os.path.join(config_directory,
                             params["config"][telescope]["referenceFile"])
            refs = table.unique(table.Table.read(
                params["config"][telescope]["referenceFile"],
                format='ascii', data_start=2, data_end=-1)['field', 'fid'])
            reference_images =\
                {group[0]['field']: group['fid'].astype(int).tolist()
                 for group in refs.group_by('field').groups}
            reference_images_map = {1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'J'}
            for key in reference_images:
                reference_images[key] = [reference_images_map.get(n, n)
                                         for n in reference_images[key]]
            params["config"][telescope]["reference_images"] = reference_images

        observer = ephem.Observer()
        observer.lat = str(params["config"][telescope]["latitude"])
        observer.lon = str(params["config"][telescope]["longitude"])
        observer.horizon = str(-12.0)
        observer.elevation = params["config"][telescope]["elevation"]
        params["config"][telescope]["observer"] = observer

    params["skymap"] = skymap
    params["gpstime"] = gpstime
    params["outputDir"] = "./" 
    params["tilingDir"] = tiling_directory
    params["event"] = ""
    params["telescopes"] = [tele]
    if tele == "ZTF":
        params["tilesType"] = "ranked"
    elif tele in ["KPED", "GROWTH-India"]:
        params["tilesType"] = "galaxy"
        params["catalogDir"] = catalog_directory
        params["galaxy_catalog"] = "GLADE"
        params["galaxy_grade"] = "S"
        params["writeCatalog"] = False
        params["catalog_n"] = 1.0
        params["powerlaw_dist_exp"] = 1.0
    elif tele in ['TRE']:
        params["tilesType"] = "moc"
    elif tele in ['TNT']:
        params["tilesType"] = "galaxy"
        params["catalogDir"] = catalog_directory
        params["galaxy_catalog"] = "GLADE"
        params["galaxy_grade"] = "Sloc"
        params["writeCatalog"] = False
        params["catalog_n"] = 1.0
        params["powerlaw_dist_exp"] = 1.0
    else:
        params["tilesType"] = "moc"
    params["scheduleType"] = schedule_type
    params["timeallocationType"] = "powerlaw"
    params["nside"] = 256
    params["powerlaw_cl"] = probability
    params["powerlaw_n"] = 1.0
    params["powerlaw_dist_exp"] = 0.0

    params["galaxies_FoV_sep"] = 0.0

    params["doPlots"] = False
    params["doMovie"] = False
    params["doObservability"] = True
    params["do3D"] = False

    params["doFootprint"] = False
    params["footprint_ra"] = 30.0
    params["footprint_dec"] = 60.0
    params["footprint_radius"] = 10.0

    params["airmass"] = 2.5

    params["doCommitDatabase"] = True
    params["doRequestScheduler"] = False
    params["doEvent"] = False
    params["doSkymap"] = True
    params["doFootprint"] = False
    params["doDatabase"] = False
    params["doReferences"] = doReferences
    params["doChipGaps"] = False
    params["doSplit"] = False
    params["doParallel"] = False
    params["doUseCatalog"] = False
    params["doIterativeTiling"] = False
    params["doMinimalTiling"] = False

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
        raise ValueError('Need to enable --doEvent, --doFootprint, '
                         '--doSkymap, or --doDatabase')

    if tobs is None:
        now_time = time.Time.now()
        timediff = now_time.gps - event_time.gps
        timediff_days = timediff / 86400.0
        params["Tobs"] = np.array([timediff_days, timediff_days+1])
    else:
        params["Tobs"] = tobs

    params["doSingleExposure"] = True
    if filterScheduleType == "block":
        params["doAlternatingFilters"] = True
    else:
        params["doAlternatingFilters"] = False
    params["filters"] = filt
    params["exposuretimes"] = exposuretimes
    params["mindiff"] = mindiff

    params = gwemopt.segments.get_telescope_segments(params)
    params['map_struct'] = None

    if params["doPlots"]:
        if not os.path.isdir(params["outputDir"]):
            os.makedirs(params["outputDir"])

    return params


def gen_structs(params):

    print('Loading skymap')
    # Function to read maps
    map_struct = gwemopt.utils.read_skymap(params, is3D=params["do3D"],
                                           map_struct=params['map_struct'])

    if params["tilesType"] == "galaxy":
        print("Generating catalog...")
        map_struct, catalog_struct =\
            gwemopt.catalog.get_catalog(params, map_struct)

    if params["tilesType"] == "moc":
        print('Generating MOC struct')
        moc_structs = gwemopt.moc.create_moc(params)
        tile_structs = gwemopt.tiles.moc(params, map_struct, moc_structs)
    elif params["tilesType"] == "ranked":
        print('Generating ranked struct')
        moc_structs = gwemopt.rankedTilesGenerator.create_ranked(params,
                                                                 map_struct)
        tile_structs = gwemopt.tiles.moc(params, map_struct, moc_structs)
    elif params["tilesType"] == "hierarchical":
        print('Generating hierarchical struct')
        tile_structs = gwemopt.tiles.hierarchical(params, map_struct)
    elif params["tilesType"] == "greedy":
        print('Generating greedy struct')
        tile_structs = gwemopt.tiles.greedy(params, map_struct)
    elif params["tilesType"] == "galaxy":
        print("Generating galaxy struct...")
        tile_structs = gwemopt.tiles.galaxy(params, map_struct, catalog_struct)
    else:
        raise ValueError(
            'Need tilesType to be moc, greedy, hierarchical, galaxy or ranked')

    coverage_struct = gwemopt.coverage.timeallocation(params,
                                                      map_struct,
                                                      tile_structs)

    if params["doPlots"]:
        gwemopt.plotting.skymap(params, map_struct)
        gwemopt.plotting.tiles(params, map_struct, tile_structs)
        gwemopt.plotting.coverage(params, map_struct, coverage_struct)

    return map_struct, tile_structs, coverage_struct


def test_scheduler():
    # Read test GCN

    gwemoptpath = os.path.dirname(gwemopt.__file__)
    testpath = config_directory = os.path.join(gwemoptpath, 'tests')
    skymap = os.path.join(testpath, 'data/MS190227n_bayestar.fits.gz')
    gpstime = 1235311089.400738

    teles=['ZTF','KPED', 'TRE', 'TNT']
    doReferences_list = [True,False, False, False]
    for tele,doReferences in zip(teles,doReferences_list): 
        params = params_struct(skymap, gpstime, tele=tele,
                               doReferences=doReferences)
        map_struct, tile_structs, coverage_struct = gen_structs(params)
