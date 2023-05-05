import os
import glob

from astropy import table
from astropy import time
import ephem
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
import numpy as np
from pathlib import Path
from gwemopt.read_output import read_schedule
import pandas as pd
import tempfile
from gwemopt.utils import readParamsFromFile, read_skymap, params_checker
from gwemopt.paths import test_skymap, DEFAULT_CONFIG_DIR, DEFAULT_TILING_DIR, REFS_DIR, TESSELATION_DIR

np.random.seed(42)

test_dir = Path(__file__).parent.absolute()
test_data_dir = test_dir.joinpath("data")
expected_results_dir = test_data_dir.joinpath("expected_results")
gwemopt_root_dir = test_dir.parent.parent


def params_struct(skymap, gpstime, filt=['r'],
                  exposuretimes=[60.0],
                  mindiff=30.0*60.0, probability=0.9, tele='ZTF',
                  schedule_type='greedy',
                  doReferences=True,
                  filterScheduleType='block'):

    config_directory = DEFAULT_CONFIG_DIR
    tiling_directory = DEFAULT_TILING_DIR
    catalog_directory = gwemopt_root_dir.joinpath("catalog")

    params = dict()
    params["config"] = {}
    config_files = glob.glob("%s/*.config" % config_directory)
    for config_file in config_files:
        telescope = config_file.split("/")[-1].replace(".config", "")
        if not telescope == tele: continue
        params["config"][telescope] =\
            readParamsFromFile(config_file)
        params["config"][telescope]["telescope"] = telescope
        if "tesselationFile" in params["config"][telescope]:
            params["config"][telescope]["tesselationFile"] = TESSELATION_DIR.joinpath(params["config"][telescope]["tesselationFile"])
            tesselation_file = params["config"][telescope]["tesselationFile"]
            if not os.path.isfile(tesselation_file):
                if params["config"][telescope]["FOV_type"] == "circle":
                    gwemopt.tiles.tesselation_spiral(
                        params["config"][telescope])
                elif params["config"][telescope]["FOV_type"] == "square":
                    gwemopt.tiles.tesselation_packing(
                        params["config"][telescope])

            params["config"][telescope]["tesselation"] =\
                np.loadtxt(tesselation_file, usecols=(0, 1, 2), comments='%')

        if "referenceFile" in params["config"][telescope]:
            params["config"][telescope]["referenceFile"] =\
                REFS_DIR.joinpath(params["config"][telescope]["referenceFile"])
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
    params["tilingDir"] = tiling_directory
    params["event"] = ""
    params["telescopes"] = [tele]
    if tele in ["KPED", "GROWTH-India"]:
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
    params["doOverlappingScheduling"] = False

    params = params_checker(params)

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

    params["Tobs"] = np.array([0., 1.])

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
<<<<<<< HEAD
    params, map_struct = read_skymap(params, is3D=params["do3D"], map_struct=params['map_struct'])
=======
    map_struct = read_skymap(params, is3D=params["do3D"], map_struct=params['map_struct'])
>>>>>>> 7a07e6b (Update test)

    catalog_struct = None

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

    return map_struct, tile_structs, coverage_struct, catalog_struct


def test_scheduler():
    """
    Test scheduler

    :return: None
    """
    skymap = test_skymap
    gpstime = 1235311089.400738

    telescope_list = [
        ('ZTF', True),
        ('KPED', False),
        # ('TRE', False),
        # ('TNT', False),
        # ("WINTER", False)
    ]

    for telescope, do_references in telescope_list:

        params = params_struct(skymap, gpstime, tele=telescope,
                               doReferences=do_references)

        with tempfile.TemporaryDirectory() as temp_dir:
            # To regenerate the test data, uncomment the following lines
            # temp_dir = Path(__file__).parent.absolute().joinpath("temp")
            # temp_dir.mkdir(exist_ok=True)

            params["outputDir"] = temp_dir

            map_struct, tile_structs, coverage_struct, catalog_struct = gen_structs(params)

            tile_structs, coverage_struct = gwemopt.coverage.timeallocation(
                params, map_struct, tile_structs
            )

            gwemopt.plotting.skymap(params, map_struct)
            gwemopt.plotting.tiles(params, map_struct, tile_structs)
            gwemopt.plotting.coverage(params, map_struct, coverage_struct)
            gwemopt.scheduler.summary(
                params, map_struct, coverage_struct, catalog_struct=catalog_struct
            )

            new_schedule = read_schedule(
                Path(temp_dir).joinpath(f"schedule_{telescope}.dat")
            )
            expected_schedule = read_schedule(
                expected_results_dir.joinpath(f"schedule_{telescope}.dat")
            )

            pd.testing.assert_frame_equal(
                new_schedule.reset_index(drop=True),
                expected_schedule.reset_index(drop=True)
            )
