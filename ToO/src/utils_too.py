"""
Set of functions common to different too

Author S. Antier 2022
"""

import os
import datetime
import voeventparse as vp
from ligo.skymap.tool.ligo_skymap_plot_observability import main
from astropy.utils.data import download_file


def init_gwemopt_observation_plan(config_gwemopt_dir="../configs"):
    """
    Function to initialize dictionary needed by gwemopt

    :param config_gwemopt_dir: directory where to find the configuration
    files need by gwemopt
    :return: dictionary with the list of parameters needed for gwemopt to run
    """
    
    #missing dateobs, tobs, exposuretimes, probability, tele, scedule_type, doUsePrimary, doNalanceExposure, filterScheduleType,doCompletedObservations=False, cobs=NOne, doPlannedObservations=False, max_nb_tiles=1000, raslice=[0, 24]

    
    #same https://github.com/growth-astro/growth-too-marshal/blob/main/growth/too/tasks/tiles.py
    #skymap none
    #gps time -1
    #event ""
    
    #    if schedule_strategy == "catalog":
    #params["tilesType"] = "moc"
    #params["scheduleType"] = schedule_type
    #params["timeallocationType"] = "powerlaw"

				#params["nside"] = 512
				
				
				#params["doUsePrimary"] = doUsePrimary



    filt = ["r"]
    exposuretimes = [30]
    
    mindiff = 30.0 * 60.0

    params = {
        "config": {},
        "powerlaw_cl": 0.9,
        "powerlaw_n": 1.0,
        "powerlaw_dist_exp": 1.0,
        "doPlots": True,
        "doMovie": False,
        "doObservability": True,
        "do3D": False,
        "DScale": 1.0,
        "doFootprint": False,
        "footprint_ra": 30.0,
        "footprint_dec": 60.0,
        "footprint_radius": 10.0,
        "airmass": 2.5,
        "doRASlice": False,
    #not present in ToO Marshall code
        "doRotate": False,
        "AGN_flag": False,
        "doOrderByObservability": False,
        "doTreasureMap": False,
        "doUpdateScheduler": False,
        "doBlocks": False,
    #Eventually, needs to be tested
        "doSuperSched": False,
        "doBalanceExposure": False,
        "doMovie_supersched": False,
    #checked
        "doCommitDatabase": False,
        "doRequestScheduler": False,
    #dateobs 
        "dateobs": False,
        "doEvent": False,
    #True if you use fits file
        "doSkymap": False,
    #checked True for Fermi
        "doDatabase": False,
    #True if references images
        "doReferences": False,
        "doChipGaps": False,
        "doSplit": False,
        "doSchedule": False,
   #schedule_strategy='tiling'    
        "doMinimalTiling": True,
   #network telescope according to the list
        "doIterativeTiling": True,
   #False in https://github.com/growth-astro/growth-too-marshal/blob/main/growth/too/tasks/tiles.py, do MaxTime and max_nb_times set up
        "doMaxTiles": True,
        "max_nb_times": 50.,
        "iterativeOverlap": 0.2,
        "maximumOverlap": 0.2,
        "catalog_n": 1.0,
        "doUseCatalog": False,
#CLU ?
        "catalogDir": config_gwemopt_dir + "/catalogs/",
        "tilingDir": config_gwemopt_dir + "/tiling/",
#still used ? 
        "configDirectory": config_gwemopt_dir + "/config/",

        "galaxy_catalog": "CLU",
        "doCatalog": False,
   #checked
        "galaxy_grade": "S", #'Smass'for Mangrove
			 #same https://github.com/growth-astro/growth-too-marshal/blob/main/growth/too/tasks/tiles.py
        "writeCatalog": False,
			 #same https://github.com/growth-astro/growth-too-marshal/blob/main/growth/too/tasks/tiles.py but none Ncores
        "doParallel": False, 
        "Ncores": 2,
    #True if bloc
        "doAlternatingFilters": False,

        "galaxies_FoV_sep": 0.9, 			 #1.0 for GROWTH
        "doOverlappingScheduling": False,
        "doPerturbativeTiling": True,
			 #same https://github.com/growth-astro/growth-too-marshal/blob/main/growth/too/tasks/tiles.py but none Ncores
        "doSingleExposure": True,
        "filters": filt,
        "exposuretimes": exposuretimes,
        "mindiff": mindiff,
        "Moon_check": False #TBC
    }

    return params


def init_voevent(conf_vo, what_com):
    """
    Create common structure for all VO events

    :param conf_vo: dictionary with common info, needed for header
    :param what_com: dictionary with common info for all type of events
    :return: vo event object
    """

    # initiate VO event object through voeventparse
    voevent = vp.Voevent(stream=conf_vo["stream"],
                         stream_id=conf_vo["streamid"],
                         role=conf_vo["role"])

    # include who part, required by VO protocol
    vp.set_who(voevent, date=datetime.datetime.utcnow(), author_ivorn=conf_vo["authorivorn"])

    # info of the origin, required by VO protocol
    vp.set_author(voevent, contactName=conf_vo["contactName"])
    vp.set_author(voevent, shortName=conf_vo["Experiment"])
    vp.set_author(voevent, contactPhone=conf_vo["contactPhone"])
    vp.set_author(voevent, contactEmail=conf_vo["contactEmail"])

    # fill infos from the original event, add them to What section
    # first type of the alert (GW, neutrinos, ...)
    alertype = vp.Param(name="Event_type",
                        value=what_com["event_type"], ucd="meta.id", dataType="string")
    alertype.Description = "Type of the alert"
    voevent.What.append(alertype)
    
    # include event name provided by the external observatory or GRANDMA
    eventid = vp.Param(name="Event_ID",
                      value=what_com["name_id"], ucd="meta.id", dataType="string")
    eventid.Description = "Name event, given by external observatory"
    voevent.What.append(eventid)


    # include trigger number provided by the external observatory
    trigid = vp.Param(name="Trigger_ID",
                      value=what_com["trigger_id"], ucd="meta.id", dataType="string")
    trigid.Description = "Trigger ID, given by external observatory"
    voevent.What.append(trigid)
    

    # Provide the status of the alert, given by external observatory
    alertstatus = vp.Param(name="Event_status",
                           value=what_com["event_status"],
                           ucd="meta.version", dataType="string")
    alertstatus.Description = "Event status (preliminary, initial, update, retraction)"
    voevent.What.append(alertstatus)

    # Serial number of the revision
    alertstatus_iter = vp.Param(name="Pkt_ser_num",
                                value=str(what_com["pkt_ser_num"]), ucd="meta.number",
                                dataType="int")
    alertstatus_iter.Description = \
        "Packet serial number since beginning, increase by 1 for every revision received"
    voevent.What.append(alertstatus_iter)

    if what_com["event_status"] not in ["Retraction", "retraction"]:
        # Instrument(s) involved in the alert
        triginstru = vp.Param(name="Instruments",
                              value=what_com["inst"], ucd="meta.code", dataType="string")
        triginstru.Description = "Instruments which originated of the alert"
        voevent.What.append(triginstru)
   	
   	
   	# include longshort provided by the external observatory
    eventid = vp.Param(name="LongShort",
                      value=str(what_com["longshort"]), ucd="meta.id", dataType="string")
    eventid.Description = "Long-short classification, given by external observatory"
    voevent.What.append(eventid)

   	# include hardness ratio classification provided by the external observatory
    hratio = vp.Param(name="Hratio",
                      value=str(what_com["hratio"]), ucd="meta.number", dataType="float")
    hratio.Description = "Hardness ratio classification, given by external observatory"
    voevent.What.append(hratio)

   	# include sun_distance classification provided by the external observatory
    sun_distance = vp.Param(name="Sun_distance",
                      value=str(what_com["sun_distance"]), ucd="meta.number", dataType="float")
    sun_distance.Description = "Sun distance from space, given by external observatory"
    voevent.What.append(sun_distance)

   	# include moon_distance classification provided by the external observatory
    moon_distance = vp.Param(name="Moon_distance",
                      value=str(what_com["moon_distance"]), ucd="meta.number", dataType="float")
    moon_distance.Description = "Moon distance from space, given by external observatory"
    voevent.What.append(moon_distance)    

   	# include moon_illumination classification provided by the external observatory
    moon_illum = vp.Param(name="Moon_illum",
                      value=str(what_com["moon_illum"]), ucd="meta.number", dataType="float")
    moon_illum.Description = "Moon illumination from space, given by external observatory"
    voevent.What.append(moon_illum)        
    
    
    return voevent
    

def voevent_name(name_dic):
    """
    Function to create VO file name

    :param name_dic: dictionary with needed info uniformized for all type of alert
    :return: file name
    """

    file_name = name_dic["experiment"] + "_" + name_dic["tel_name"] \
                + "_"+name_dic["event_type"] + name_dic["trigger_id"] \
                + "_" + name_dic["event_status"] + "_" \
                + str(name_dic["pkt_ser_num"]) + ".xml"

    return file_name


def uniformize_galnames(galaxies_table):
    """
    Create an uniform way to report galaxy name
    Set priority for name coming first from GWGC
    then HyperLEDA, 2MASS and SDSS

    :param galaxies_table: astropy table with galaxies selected by observation plan
    :return: list of uniform galaxy name
    """

    gal_id = []
    gal_name = ""
    for gal in galaxies_table:
        if (gal['GWGC'] != '---') and (gal['GWGC'] != '--') \
                and (gal['GWGC']) and (gal['GWGC'] != 'None'):

            gal_name = gal['GWGC']

        elif (gal['HyperLEDA'] != '---') and (gal['HyperLEDA'] != '--') \
                and (gal['HyperLEDA']) and (gal['HyperLEDA'] != 'None'):

            gal_name = gal['HyperLEDA']

        elif (gal['2MASS'] != '---') and (gal['2MASS'] != '--') \
                and (gal['2MASS']) and (gal['2MASS'] != 'None'):

            gal_name = gal['2MASS']

        elif (gal['SDSS'] != '---') and (gal['SDSS'] != '--') \
                and (gal['SDSS']) and (gal['SDSS'] != 'None'):

            gal_name = gal['SDSS']

        gal_id.append(gal_name)

    return gal_id


def update_voivorn(conf_dic):
    """
    Update the ivorn part before creating VO event
    :param conf_dic: dictionary with basic info for VO creation
    :return: update dictionary
    """

    conf_dic["stream"] = conf_dic["stream_address"] + "/" + conf_dic["Experiment"]+"_Alert"
    conf_dic["authorivorn"] = conf_dic["Experiment"] + "_Alert"
    conf_dic["vodescription"] = "VOEvent created in " + conf_dic["Experiment"]

    return conf_dic


def define_streamid_vo(telescope, pkt_ser_num, experiment):
    """
    Define the stream id needed for VO event
    it will be based on the experiment name
    time of the day
    id of the alert
    serial number of the on-going alert

    :param telescope: telescope to which VO is for
    :param pkt_ser_num: serial number for the on-going alert
    :param experiment: on-going experiment
    :return: string containing streamid
    """

    # get time of the day
    time_of_day = datetime.datetime.now()

    # build stream id as ExperimentTimeTelescopeSerial
    stream_id = experiment + str(time_of_day.year) + '{:02d}'.format(time_of_day.month) + \
                '{:02d}'.format(time_of_day.day) + telescope + str(pkt_ser_num)

    return stream_id


def send_voevent(broker_config, filename_vo):
    """
    Function so send vo events to a given broker defined in configuration file
    we plan to use comet

    :param broker_config: dictionary with broker infos
    :param filename_vo: list of files to be sent
    :return:
    """

    cmd_send = "%s --host=%s --port=%s -f " % (broker_config['broker_path'],
                                               broker_config['out_host'],
                                               broker_config['out_port'])

    for file_out in filename_vo:
        cmd = cmd_send + file_out
        os.system(cmd)


