#!/usr/bin/env python
"""
Module for postprod of neutrinos alert
"""

import os
import sys
import io
import datetime
#import config_lowlatency as cf
#from slackclient import SlackClient
import mma_schedule as mma
import voeventparse as vp
#SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__)) + "/ToO_field_match_v1.1/"
#sys.path.append(os.path.abspath(SCRIPT_PATH))
# CHANGE IMPORT GWAC ORDER TO WAIT PATH
#import GWAC_ToO_observable_calculator as GTOC

import numpy as np
import pytz
import copy

import glob
import ephem

from astropy import table
from astropy import time



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

sys.path.append("./VOEventLib/")
from VOEventLib.VOEvent import * 
from VOEventLib.Vutil  import *

ROLET

def GRB_dicf():
  GRB_dic = {
		  "trigtime" : "",

    "Packet_Type":"",
    "Pkt_Ser_Num":"",
    "Trigger_TJD":"",
    "Trigger_SOD":"",
    
    "inst" : "",
		  "trigid" : "",
    "trigdelay" : 0.,
    "grbid" : "",
		  "ratesnr" : 0.,
    "ratets" : 0.,
    "imagesnr" : 0.,
    "lc" : "",
    "hratio" : 0.,
    "longshort" : False,
    "probGRB" : -1.0,
    "defGRB" : True,
    "ra" : 0.,
    "dec" : 0.,
    "error" : 0.,
    "locsnr" : 0.,
    "locdur" : 0.,
    "locref" : "",
    "obs" : "",
    "locpix":"",
	 }
  return GRB_dic

def VO_dicf():
  Contentvo_dic = {
		  "name" : "",
    "role" : "",
		  "stream" : "",
		  "streamid" : "",
    "voivorn" : "ivo://svom.bao.ac.cn/LV#SFC_GW_",
		  "authorivorn" : "GRANDMA_Alert",
		  "shortName" : "GRANDMA (via VO-GCN)",
		  "contactName": "Sarah Antier",
    "contactPhone": "+33-1-64-46-83-73",
    "contactEmail": "antier@lal.in2p3.fr",
    "description": "Selected by ",
    "vodescription": "VOEvent between CSC and FSC",
    "locationserver": "",
    "voschemaurl":"http://www.cacr.caltech.edu/~roy/VOEvent/VOEvent2-110220.xsd",
    "ba":"",
    "gwacsci":"",
    "ivorn":"",
    "letup":"a",
    "obs": None,
	 }

  return Contentvo_dic
   

def trigtime(isotime):
   date_t = isotime.split("-")
   yr_t = int(date_t[0])
   mth_t = int(date_t[1])
   dy_t = int(date_t[2].split("T")[0])
   hr_t = int(date_t[2].split("T")[1].split(":")[0])
   mn_t = int(date_t[2].split("T")[1].split(":")[1])
   sd_t = int(float(date_t[2].split("T")[1].split(":")[2]))
   trigger_time_format = datetime.datetime(yr_t, mth_t, dy_t, hr_t, mn_t, sd_t,tzinfo=pytz.utc)
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
    #gwac_mma = mma.mma_on_duty(cf.GRID_BCK_DIR + "/MMA_duty.txt")
    #gwac_duty = mma.GWAC_onduty(cf.GRID_BCK_DIR+"/GWAC_duty.txt")
    gwac_mma = mma.mma_on_duty("MMA_duty.txt")
    gwac_duty = mma.GWAC_onduty("GWAC_duty.txt")
    return gwac_mma, gwac_duty


def Observation_plan(teles,obsinstru,trigtime,urlhelpix):

    tobs = None
    filt = ["r"]
    exposuretimes = [30]
    mindiff = 30.0*60.0

    skymap=urlhelpix.split("/")[-1]
    skymappath = "./HELPIX/%s"%skymap
 
    if not os.path.isfile(skymappath):
        command="wget "+urlhelpix+" -P ./HELPIX/"
        os.system(command)

    event_time = time.Time(trigtime,scale='utc')

    gwemoptpath = os.path.dirname(gwemopt.__file__)
    config_directory = "../config"
    tiling_directory = "../tiling"

    params = {}
    params["config"] = {}
    config_files = glob.glob("%s/*.config" % config_directory)
    for config_file in config_files:
        telescope = config_file.split("/")[-1].replace(".config", "")
        params["config"][telescope] =\
            gwemopt.utils.readParamsFromFile(config_file)
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
            reference_images_map = {1: 'g', 2: 'r', 3: 'i'}
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

    params["skymap"] = skymappath
    params["gpstime"] = event_time.gps
    params["outputDir"] = "output/%s" % event_time
    params["tilingDir"] = tiling_directory
    params["event"] = ""
    params["telescopes"] = [teles]
    params["tilesType"] = "moc"
    params["scheduleType"] = "greedy"
    params["timeallocationType"] = "powerlaw"
    params["nside"] = 256
    params["powerlaw_cl"] = 0.9
    params["powerlaw_n"] = 1.0
    params["powerlaw_dist_exp"] = 0.0

    params["doPlots"] = False
    params["doMovie"] = False
    params["doObservability"] = True
    params["do3D"] = False

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
        params["Tobs"] = np.array([timediff_days, timediff_days+1])
    else:
        params["Tobs"] = tobs

    params["doSingleExposure"] = True
    params["filters"] = filt
    params["exposuretimes"] = exposuretimes
    params["mindiff"] = mindiff

    params = gwemopt.segments.get_telescope_segments(params)

    if params["doPlots"]:
        if not os.path.isdir(params["outputDir"]):
            os.makedirs(params["outputDir"])

    print("Loading skymap...")
    # Function to read maps
    map_struct = gwemopt.utils.read_skymap(params, is3D=params["do3D"])

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
    else:
        print("Need tilesType to be moc, greedy, hierarchical, or ranked")
        exit(0)

    coverage_struct = gwemopt.coverage.timeallocation(params,
                                                      map_struct,
                                                      tile_structs)

    if params["doPlots"]:
        gwemopt.plotting.skymap(params, map_struct)
        gwemopt.plotting.tiles(params, map_struct, tile_structs)
        gwemopt.plotting.coverage(params, map_struct, coverage_struct)

    config_struct = params["config"][teles]

    what = What()

    thistable = Table(name="data", Description=["The datas of "+str(obsinstru)])
    thistable.add_Field(Field(name=r"grid_id", ucd="", unit="", dataType="int", \
                    Description=["ID of the grid of fov"]))
    thistable.add_Field(Field(name="field_id", ucd="", unit="", dataType="int",\
                    Description=["ID of the filed"]))
    thistable.add_Field(
        Field(
            name=r"ra", ucd=r"pos.eq.ra ", unit="deg", dataType="float",
            Description=["The right ascension at center of fov in equatorial coordinates"]
            )
        )
    thistable.add_Field(
        Field(
            name="dec", ucd="pos.eq.dec ", unit="deg", dataType="float",
            Description=["The declination at center of fov in equatorial coordinates"]
            )
        )
    thistable.add_Field(
        Field(
            name="ra_width", ucd=" ", unit="deg", dataType="float",
            Description=["Width in RA of the fov"]
            )
        )
    thistable.add_Field(
        Field(
            name="dec_width", ucd="", unit="deg", dataType="float",
            Description=["Width in Dec of the fov"]
            )
        )
    thistable.add_Field(
        Field(
            name="prob_sum", ucd="", unit="None", dataType="float",
            Description=["The sum of all pixels in the fov"]
            )
        )
    thistable.add_Field(Field(name="priority", ucd="", unit="", dataType="int", Description=[""]))
    table_field = utilityTable(thistable)
    table_field.blankTable(len(coverage_struct))

    for ii in range(len(coverage_struct["ipix"])):
        data = coverage_struct["data"][ii,:]
        filt = coverage_struct["filters"][ii]
        ipix = coverage_struct["ipix"][ii]
        patch = coverage_struct["patch"][ii]
        FOV = coverage_struct["FOV"][ii]
        area = coverage_struct["area"][ii]

        prob = np.sum(map_struct["prob"][ipix])

        ra, dec = data[0], data[1]
        exposure_time, field_id, prob = data[4], data[5], data[6]

        table_field.setValue("grid_id", ii, 0)
        table_field.setValue("field_id", ii, field_id)
        table_field.setValue("ra", ii, ra)
        table_field.setValue("dec", ii, dec)
        table_field.setValue("ra_width", ii, config_struct["FOV"])
        table_field.setValue("dec_width", ii, config_struct["FOV"])
        table_field.setValue("prob_sum", ii, prob)
        table_field.setValue("priority", ii, ii)

    thistable = table_field.getTable()
    #what.add_Table(thistable)
    #xml = stringVOEvent(what)

    return thistable

def swift_trigger(v, collab, text_mes,file_log_s,role):
    """

    :param v:
    :param collab:
    :param text_mes:
    :return:
    """

    #DIC not filled yet

    #"grbid" : "",
    #"ratets" : "",
    #"lc" : "",
    #"hratio" : "",
    #"longshort" : "",
    #"probGRB" : "",
    #"locsnr" : "",
    #"locdur" : "",
    #"obs" : "",

    Swift_dic=GRB_dicf()
    Swift_vo=VO_dicf()

    instru = str(collab[2])
    gwac_mma, gwac_duty = search_ba()
    Swift_vo["ba"]=gwac_mma
    Swift_vo["gwacsci"]=gwac_duty
    

    if instru == "BAT":

        Swift_dic["inst"]=instru

        top_level_params = vp.get_toplevel_params(v)
        trigger_id = top_level_params['TrigID']['value']
        Swift_dic["trigid"]=trigger_id
    
        rate_signif = top_level_params['Rate_Signif']['value']
        Swift_dic["ratesnr"]=float(rate_signif)

        image_signif = top_level_params['Image_Signif']['value']
        Swift_dic["imagesnr"]=float(rate_signif)

        def_not_grb =  v.find(".//Param[@name='Def_NOT_a_GRB']").attrib['value']
        Swift_dic["defGRB"]=def_not_grb 


        isotime = v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Time.TimeInstant.\
            ISOTime.text
        delay = delay_fct(isotime)
        isotime_format=trigtime(isotime)
        delay_min=(delay.seconds)/60.0
        Swift_dic["trigtime"]=isotime
        Swift_dic["trigdelay"]=delay_min


        right_ascension = str(v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.\
                              Position2D.Value2.C1.text)
        declination = str(v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Position2D\
                          .Value2.C2.text)
        error2_radius = str(v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.\
                            Position2D.Error2Radius.text)
   
        Swift_dic["ra"]=right_ascension
        Swift_dic["trigdelay"]=right_ascension
        Swift_dic["dec"]=declination
        Swift_dic["error"]=error2_radius
        Swift_dic["locref"]="BAT onboard"
        
        name_dic="Swift"+trigger_id
        if ((role!="test")):
                lalid=name_lalid(v,file_log_s,name_dic,Swift_vo["letup"],"gwac")
                name_dic="Swift"+trigger_id
                dic_grb[name_dic]=Swift_dic
                dic_vo[name_dic]=Swift_vo

                create_GRBvoevent(lalid,Swift_dic,Swift_vo)
                file_log_s.write(lalid +" "+str(trigger_id)+"\n")
        text_mes = str("---------- \n")+str("BAT alert \n")+str("---------- \n")+\
            str("Trigger ID: ")\
            +trigger_id+("\n")+str("Trigger Time: ")+isotime+("\n")+str("Delay since alert: ")+\
            str(delay)+("\n")+str("\n")+str("---BAs---\n")+str("BA on duty: ")+str(gwac_mma)+"\n"+\
            ""+str("GWAC scientist on duty: ")+str(gwac_duty)+str("\n")+("\n")+str("\n")+\
            str("---SPACE TRIGGER---\n")+str("Trigger Rate SNR: ")+str(rate_signif)+" "+\
            str("Image_Signif: ")+image_signif+("\n")+str("\n")+str("---Position---\n")+\
            "RA: "+str(round(float(right_ascension),1))+" "+"DEC: "+str(round(float(declination),1))+" "+str("Error2Radius: ")+\
            str(round(float(error2_radius),1))+"\n"+str("\n")+str("---------- \n")#+("---SVOM FOLLOWUP---\n")+\
            #str(observability__xinglong)+" "+str(follow)+"\n"
         
    return text_mes


def fermi_trigger_found(v, collab,role,file_log_s):
    """

    :param v:
    :param collab:
    :return:
    """

    Fermi_dic=GRB_dicf()
    Fermi_vo=VO_dicf()

    instru="GBM"
    Fermi_dic["inst"]=instru


    gwac_mma, gwac_duty = search_ba()
    Fermi_vo["ba"]=gwac_mma
    Fermi_vo["gwacsci"]=gwac_duty


    toplevel_params = vp.get_toplevel_params(v)
    trigger_id = toplevel_params['TrigID']['value']
    Fermi_dic["trigid"]=trigger_id

    rate__signif = toplevel_params['Trig_Signif']['value']
    rate__dur = toplevel_params['Trig_Dur']['value']
    Fermi_dic["ratesnr"]=rate__signif
    Fermi_dic["ratets"]=rate__dur



    isotime = v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Time.TimeInstant.ISOTime\
        .text
    Fermi_dic["trigtime"]=isotime

    delay = delay_fct(isotime)
    isotime_format=trigtime(isotime)
    delay_min=(delay.seconds)/60.0
    Fermi_dic["trigdelay"]=delay_min

    # grb_proba = str(v.Why.Inference.attrib["probability"])
    grb_lc = toplevel_params['LightCurve_URL']['value']
    energy_bandmin = toplevel_params['Lo_Chan_Energy']['value']
    energy_bandmax = toplevel_params['Hi_Chan_Energy']['value']

    name_grb = gbm_lc_name(grb_lc)
    Fermi_dic["grbid"]=name_grb
    ra = str(v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Position2D.\
    Value2.C1.text)
    dec = str(v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Position2D.\
    Value2.C2.text)
    error2_radius = str(v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.\
    Position2D.Error2Radius.text)
    Fermi_dic["ra"]=ra
    Fermi_dic["dec"]=dec
    Fermi_dic["error"]=error2_radius
    Fermi_dic["locref"]="GBM onboard"
  

    if ((role!="test")):
        name_dic="GBM"+trigger_id
        lalid=name_lalid(v,file_log_s,name_dic,Fermi_vo["letup"],"gwac")
        create_GRBvoevent(lalid,Fermi_dic, Fermi_vo)
        dic_grb[name_dic]=Fermi_dic
        dic_vo[name_dic]=Fermi_vo


        file_log_s.write(lalid +" "+str(trigger_id)+"\n")

    text = str("---------- \n")+str("FERMI/GBM alert \n")+str("---------- \n")+str("GRB NAME : ")\
        +str(name_grb)+(" ")+str("Trigger ID: ")+trigger_id+("\n")+str("Trigger Time: ")+isotime+\
        ("\n")+str("Delay since alert: ")+str(delay)+("\n")+str("\n")+str("---BAs--\n")+str\
        ("BA on duty: ")+str(gwac_mma)+"\n"+str("GWAC scientist on duty: ")+str(gwac_duty)+"\n"+\
        str("\n")+str("---SPACE TRIGGER---\n")+str("Trigger Rate SNR ")+str(rate__signif)+" "+str\
        ("Trigger dur ")+rate__dur+("\n")+str("\n")+str("---GRB CARAC---\n")+str("LC path: ")+str\
        (grb_lc)+str("\n")+str("Selected Energy band (keV): ")+str(energy_bandmin)+"-"+\
        str(energy_bandmax)+("\n")+str("\n")
    return text


def fermi_trigger_follow(v, collab, message_type,file_log_s,role):
    """

    :param v:
    :param collab:
    :param message_type:
    :return:
    """
    
    toplevel_params = vp.get_toplevel_params(v)
    trigger_id = toplevel_params['TrigID']['value']
    name_dic="GBM"+trigger_id
    Fermi_dic=dic_grb.get(name_dic)
    Fermi_vo=dic_vo.get(name_dic)
 
    pletter=Fermi_vo["letup"]
    indice_pletter=np.where(letters==pletter)[0]
    Fermi_vo["letup"]=letters[indice_pletter+1][0]

    grb_identified = str(v.What.Group.Param[0].attrib['value'])
    #long_short = "unknown"
    #rate__signif = 0.0
    #rate__dur = 0.0
    #prob_GRB=-1.0
    #hard_ratio=0.0
    #not_grb="unknown"


    if message_type == "FINAL FERMI/GBM POSITION MESSAGE":
        Fermi_dic["locref"]="GBM final update"
        long_short = str(v.What.Group.Param[4].attrib['value'])
        Fermi_dic["longshort"]=long_short
        not_grb = def_not_grb =  v.find(".//Param[@name='Def_NOT_a_GRB']").attrib['value']
        Fermi_dic["defGRB"]=not_grb
        loc_png=toplevel_params['LocationMap_URL']['value']
        png_name=(loc_png.split("/")[-1]).split("_")
        healpix_name=png_name[0]+"_"+"healpix"+"_"+png_name[2]+"_"+png_name[3].split(".")[0]+".fit"
        path_helpcolec=loc_png.split("/")
        path_healpix=path_helpcolec[0]
        for h in np.arange(len(path_helpcolec)-1):
         if h!=0:
            path_healpix=path_healpix+"/"+path_helpcolec[h]
        link_healpix=path_healpix+"/"+healpix_name
        Fermi_dic["locpix"]=link_healpix
        Fermi_vo["obs"]=Observation_plan("GWAC",Fermi_dic["inst"],Fermi_dic["trigtime"],Fermi_dic["locpix"])

    grb_lc = toplevel_params['LightCurve_URL']['value']
    
    
    name_grb = gbm_lc_name(grb_lc)
    Fermi_dic["lc"]=grb_lc 
    Fermi_dic["grbid"]=name_grb 
  
    if message_type == "FLIGHT UPDATE FERMI/GBM POSITION MESSAGE":
        Fermi_dic["locref"]="GBM flight update"
        rate__signif = toplevel_params['Data_Signif']['value']
        Fermi_dic["locsnr"]=rate__signif
        rate__dur = toplevel_params['Data_Timescale']['value']
        Fermi_dic["locdur"]=rate__dur
        prob_GRB= toplevel_params['Most_Likely_Prob']['value']
        Fermi_dic["probGRB"]=prob_GRB
        hard_ratio=toplevel_params['Hardness_Ratio']['value']
        Fermi_dic["hratio"]=hard_ratio
        not_grb = def_not_grb =  v.find(".//Param[@name='Def_NOT_a_GRB']").attrib['value']
        Fermi_dic["defGRB"]=not_grb 

    if message_type == "GROUND UPDATE FERMI/GBM POSITION MESSAGE":
        Fermi_dic["locref"]="GBM ground update"
        rate__signif = toplevel_params['Burst_Signif']['value']
        Fermi_dic["locsnr"]=rate__signif
        rate__dur = toplevel_params['Data_Integ']['value']
        Fermi_dic["locdur"]=rate__dur
        not_grb = def_not_grb =  v.find(".//Param[@name='Def_NOT_a_GRB']").attrib['value']
        Fermi_dic["defGRB"]=not_grb
        loc_fit=toplevel_params['LocationMap_URL']['value']
        Fermi_dic["locpix"]=loc_fit

    isotime = v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Time.TimeInstant.\
        ISOTime.text
    delay = delay_fct(isotime)
    delay_min=(delay.seconds)/60.0
    ra = v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Position2D.Value2.C1\
             .text
    dec = v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Position2D.Value2.C2\
              .text
    error2_radius = v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Position2D.\
                        Error2Radius.text
    vo_ivorncit=v.Citations.EventIVORN
    Fermi_vo["ivorn"]=vo_ivorncit
    Fermi_dic["trigtime"]=isotime
    Fermi_dic["trigdelay"]=delay_min
    Fermi_dic["ra"]=ra
    Fermi_dic["dec"]=dec
    Fermi_dic["error"]=error2_radius

    if  ((role!="test")):
     lalid=name_lalid(v,file_log_s,name_dic,Fermi_vo["letup"],"gwac")
     create_GRBvoevent(lalid,Fermi_dic, Fermi_vo)

     

    if grb_identified == "false":
        text = "\n"+message_type+str(" \n")+str("---------- \n")+str("ID : ")+str(name_grb)
    else:
        text = "\n"+str("UPDATE FERMI/GBM POSITION MESSAGE NO LONGER CLASSIFIED AS GRB\n")+str\
            ("---------- \n")+str("ID : ")+str(name_grb)+(" ")+trigger_id+(" ")+isotime+("\n")+str\
            ("Delay since alert: ")+str(delay)+("\n")+str("\n")
    return text



#A unique name per event
def name_lalid(v,file_log_s,name_dic,letter,tel):
    name_dic=name_dic+"_"+tel
    lalid="" 
    toplevel_params = vp.get_toplevel_params(v)
    time_now=datetime.datetime.utcnow()
    logfile_lines = file_log_s.readlines() 
    Trigger_id=[]

    if time_now.month < 10:
            if time_now.day < 10:
               lalid="GRANDMA"+str(time_now.year)+"0"+str(time_now.month)+"0"+str(time_now.day)+"_"+name_dic+"_"+letter
            else:
               lalid="GRANDMA"+str(time_now.year)+"0"+str(time_now.month)+str(time_now.day)+"_"+name_dic+"_"+letter
    else:
        lalid="GRANDMA"+str(time_now.year)+str(time_now.month)+str(time_now.day)+"_"+name_dic+"_"+letter
    return lalid

#def create_GRBvoevent(lalid,instruselect,vo_trigid,vo_grbid,vo_trigtime,vo_trigtimeformat,vo_delay,vo_instru,vo_ratesnr,vo_imagesnr,vo_ratets,vo_ratehardness,vo_lc,vo_pra,vo_pdec,vo_perror,vo_ba,vo_ra,vo_dec,vo_error,vo_system_coor,vo_observa_loc,vo_importance, vo_reference,vo_ivorn,vo_snrloc,vo_durloc,vo_ls,vo_probGRB,vo_notGRB):

def create_GRBvoevent(lalid,GRB_dic,VO_dic):
    """
    Create the VOEvent
    """

    vo_name=lalid+".xml"
    #vo_stream="TBD"
    #vo_streamid="TBD"
    #role=vp.definitions.roles.test

    #vo_authorivorn="GRANDMA_Alert_"#+str(datetime.datetime.utcnow().isoformat())
    #vo_shortName="GRANDMA (via VO-GCN)"
    #vo_contactName="Sarah Antier"
    #vo_contactPhone="+33-1-64-46-83-73"
    #vo_contactEmail="antier@lal.in2p3.fr"



    #Why section
    #vo_description="Selected by "+instruselect
    #vo_system_coor=vp.definitions.sky_coord_system.utc_fk5_geo
    #vo_observa_loc=vp.definitions.observatory_location.geosurface
    
    #vo_location_naocserver="http://svom.bao.ac.cn/grandma/"+lalid


    # Set the basic packet ID and Author details
    vo_event = VOEvent.VOEvent(version="2.0")

    ############ VOEvent header ############################

    #vo_event.set_ivorn("ivo:/lal.org/GRANDMA_'%s'" % GRB_dic["trigid"] + "_'%s'" % datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
#    vo_event.set_ivorn("ivo:/lal.org/GRANDMA"+"_"+GRB_dic["trigid"]+"_"+datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    vo_event.set_role("test")
    #vo_event.set_Description(VO_dic['vodescription'])

    ############ Who ############################
    who = Who()
    a = Author()
    #a.add_contactName(VO_dic["contactName"])
    #a.add_contactName(VO_dic["contactName"])
    a.add_contactName("cc")
    #a.add_contactEmail(VO_dic["contactEmail"])
    who.set_Author(a)
    vo_event.set_Who(who)


    #v = vp.Voevent(stream=VO_dic["stream"],stream_id=VO_dic["streamid"], role=vp.definitions.roles.test)

    #vp.set_who(v, date=datetime.datetime.utcnow(),author_ivorn=VO_dic["authorivorn"])

    #vp.set_author(v, shortName=VO_dic["shortName"],contactName=VO_dic["contactName"],contactPhone=VO_dic["contactPhone"],contactEmail=VO_dic["contactEmail"])

    # Now create some Parameters for entry in the 'What' section.

    #naocserver = vp.Param(name="VOLocat",value=VO_dic["locationserver"])
    #naocserver.Description = 'VOevent stored'

    #v.What.append(naocserver)

    ############ What ############################
    what = What()

    # params related to the event. None are in Groups.

    
    trigtime = Param(name="TrigTime",value=GRB_dic["trigtime"])
    trigtime.set_Description(['Time of the astrophysical event'])
    what.add_Param(trigtime)

    #v.What.append(trigtime)
 
    #triginstru = vp.Param(name="Instruments", value=GRB_dic["inst"])
    #triginstru.Description = "Instrument which observed the event"
    #v.What.append(triginstru)

    triginstru = Param(name="Instruments", value=GRB_dic["inst"])
    triginstru.set_Description(["Instrument which observed the event"])
    what.add_Param(triginstru)

    #trigid = vp.Param(name="TrigID",value=GRB_dic["trigid"])
    #trigid.Description = 'Trigger ID'
    #v.What.append(trigid)

    trigid = Param(name="TrigID",value=GRB_dic["trigid"],ucd="meta.id")
    trigid.set_Description(['Trigger ID'])
    what.add_Param(trigid)

 
    #trigdelay = vp.Param(name="Trig_delay",value=GRB_dic["trigdelay"],unit="min", ucd="time.duration")
    #trigdelay.Description = 'Delay since the GRB prompt event (min)'
    #v.What.append(trigdelay)

    trigdelay = Param(name="Trig_delay",value=GRB_dic["trigdelay"],unit="min", ucd="time.duration")
    trigdelay.set_Description(['Delay since the GRB prompt event (min)'])
    what.add_Param(trigdelay)

    #grbid = vp.Param(name="GRBID",value=GRB_dic["grbid"], ucd="meta.id")
    #grbid.Description = 'GRB ID'
    #v.What.append(grbid)

    grbid = Param(name="GRBID",value=GRB_dic["grbid"], ucd="meta.id")
    grbid.set_Description(['GRB ID'])
    what.add_Param(grbid)


    #trigonlinerate_snr = vp.Param(name="Rate_SNR",value=GRB_dic["ratesnr"], unit="sigma", ucd="stat.snr")
    #trigonlinerate_snr.Description = 'Significance from the rate onboard trigger algorithm of '+GRB_dic["inst"]
    #v.What.append(trigonlinerate_snr)

    trigonlinerate_snr = Param(name="Rate_SNR",value=GRB_dic["ratesnr"], unit="sigma", ucd="stat.snr")
    trigonlinerate_snr.set_Description(['Significance from the rate onboard trigger algorithm of '+GRB_dic["inst"]])
    what.add_Param(trigonlinerate_snr)

   
    #trigonlinerate_ts = vp.Param(name="Rate_Timescale",value=GRB_dic["ratets"], unit="s")
    #trigonlinerate_ts.Description = 'Timescale from the onboard trigger algorithm of '+GRB_dic["inst"]
    #v.What.append(trigonlinerate_ts)

    trigonlinerate_ts = Param(name="Rate_Timescale",value=GRB_dic["ratets"], unit="s")
    trigonlinerate_ts.set_Description = 'Timescale from the onboard trigger algorithm of '+GRB_dic["inst"]
    what.add_Param(trigonlinerate_ts)
    
    #trigonlinerate_snr = vp.Param(name="Image_SNR",value=GRB_dic["imagesnr"], unit="sigma", ucd="stat.snr")
    #trigonlinerate_snr.Description = 'Significance from the image onboard trigger algorithm of '+GRB_dic["inst"]
    #v.What.append(trigonlinerate_snr)

    trigonlinerate_snr = Param(name="Image_SNR",value=GRB_dic["imagesnr"], unit="sigma", ucd="stat.snr")
    trigonlinerate_snr.set_Description(['Significance from the image onboard trigger algorithm of '+GRB_dic["inst"]])
    what.add_Param(trigonlinerate_snr)


    #lc = vp.Param(name="LightCurve",value=GRB_dic["lc"])
    #lc.Description = 'The LC_URL file will not be created/available until ~15 min after the trigger. Instrument:'+GRB_dic["inst"]
    #v.What.append(lc)

    lc = Param(name="LightCurve",value=GRB_dic["lc"],ucd="meta.ref.url")
    lc.set_Description(['The LC_URL file will not be created/available until ~15 min after the trigger. Instrument:'+GRB_dic["inst"]])
    what.add_Param(lc)
#


    #trigonlinerate_hardratio = vp.Param(name="Hardness_Ratio",value=GRB_dic["hratio"])
    #trigonlinerate_hardratio.Description = 'Fight Spectral characteristics of '+GRB_dic["locref"]
    #v.What.append(trigonlinerate_hardratio)

    trigonlinerate_hardratio = Param(name="Hardness_Ratio",value=GRB_dic["hratio"])
    trigonlinerate_hardratio.set_Description(['Fight Spectral characteristics of '+GRB_dic["locref"]])
    what.add_Param(trigonlinerate_hardratio)


    #longshort = vp.Param(name="Long_Short",value=GRB_dic["longshort"])
    #longshort.Description = 'Final descision of '+GRB_dic["locref"]
    #v.What.append(longshort)

    longshort = Param(name="Long_Short",value=GRB_dic["longshort"])
    longshort.set_Description(['Final descision of '+GRB_dic["locref"]])
    what.add_Param(longshort)
    

    #probGRB=vp.Param(name="Prob_GRB",value=GRB_dic["probGRB"])
    #probGRB.Description = 'Probability to be a GRB defined by '+GRB_dic["locref"]
    #v.What.append(probGRB)

    probGRB=Param(name="Prob_GRB",value=GRB_dic["probGRB"])
    probGRB.set_Description(['Probability to be a GRB defined by '+GRB_dic["locref"]])
    what.add_Param(probGRB)


    #defGRB=vp.Param(name="Defined_GRB",value=GRB_dic["defGRB"])
    #defGRB.Description = 'Not a GRB '+GRB_dic["locref"]
    #v.What.append(defGRB)

    defGRB=Param(name="Defined_GRB",value=GRB_dic["defGRB"])
    defGRB.set_Description(['Not a GRB '+GRB_dic["locref"]])
    what.add_Param(defGRB)


    #orpos = vp.Param(name="Loc_ref",value=GRB_dic["locref"])
    #orpos.Description = 'Localization determined by '+GRB_dic["locref"]
    #v.What.append(orpos)
 
    orpos = Param(name="Loc_ref",value=GRB_dic["locref"])
    orpos.set_Description(['Localization determined by '+GRB_dic["locref"]])
    what.add_Param(orpos)

    pra = Param(name="RA",value=GRB_dic["ra"], unit="deg")
    #pra.Description = 'Localization of the transient determined by '+GRB_dic[""]
    what.add_Param(pra)

    
    #pdec = vp.Param(name="DEC",value=GRB_dic["dec"], unit="deg")
    #pdec.Description = 'Localization of the transient determined by '+GRB_dic["locref"]
    #v.What.append(pdec)

    pdec = Param(name="DEC",value=GRB_dic["dec"], unit="deg")
    #pdec.Description = 'Localization of the transient determined by '+GRB_dic["locref"]
    what.add_Param(pdec)


    #perror = vp.Param(name="ERROR",value=GRB_dic["error"], unit="deg")
    #perror.Description = 'Localization of the transient determined by '+GRB_dic["locref"]
    #v.What.append(perror)

    perror = Param(name="ERROR",value=GRB_dic["error"], unit="deg")
    #perror.Description = 'Localization of the transient determined by '+GRB_dic["locref"]
    what.add_Param(perror)

    #snrloc = vp.Param(name="Loc_SNR",value=GRB_dic["locsnr"], unit="sigma", ucd="stat.snr")
    #snrloc.Description = 'Fight/Ground position SNR '+GRB_dic["locref"]
    #v.What.append(snrloc)

    snrloc = Param(name="Loc_SNR",value=GRB_dic["locsnr"], unit="sigma", ucd="stat.snr")
    #snrloc.Description = 'Fight/Ground position SNR '+GRB_dic["locref"]
    what.add_Param(snrloc)    


    #durloc = vp.Param(name="Loc_dur",value=GRB_dic["locdur"])
    #durloc.Description = 'Fight/Ground position duration SNR '+GRB_dic["locref"]
    #v.What.append(durloc)

    durloc = Param(name="Loc_dur",value=GRB_dic["locdur"])
    durloc.set_Description(['Fight/Ground position duration SNR '+GRB_dic["locref"]])
    what.add_Param(durloc)



    #pixloc = vp.Param(name="Loc_url",value=GRB_dic["locpix"])
    #v.What.append(pixloc)

    pixloc = Param(name="Loc_url",value=GRB_dic["locpix"])
    what.add_Param(pixloc)


    #obsoth = vp.Param(name="Others",value=GRB_dic["obs"])
    #obsoth.Description = 'Other observatories'
    #v.What.append(obsoth)

    obsoth = Param(name="Others",value=GRB_dic["obs"])
    obsoth.set_Description (['Other observatories'])
    what.add_Param(obsoth)

    #too_r = vp.Param(name="ToO",value=vo_perror)
    #too_r.Description = 'Name of the telescopes to send a request in the network'
  
    #v.What.append(too_r)

    #ba = vp.Param(name="BA",value=VO_dic["ba"])
    #ba.Description = 'GRANDMA burst advocate on duty'
    #v.What.append(ba)

    ba = Param(name="BA",value=VO_dic["ba"])
    ba.set_Description(['GRANDMA burst advocate on duty'])
    what.add_Param(ba)

    if VO_dic["obs"]!=None:
     tableobs=VO_dic["obs"]
     what.add_Table(tableobs)
     #print(tableobs)


    vo_event.set_What(what)


    # When Where parameters
    wwd = {
    'observatory':  GRB_dic["inst"],
    'coord_system':'UTC-FK5-GEO',
    'time': GRB_dic["trigtime"],
    'timeError': None,
    'longitude': 0.0,
    'latitude': 0.0,
    'positionalError': 0.0
    }
    ww = makeWhereWhen(wwd)
    if ww: vo_event.set_WhereWhen(ww)

    # Now we set the sky location of our event:
    #vp.add_where_when(v,coords=vp.Position2D(ra=vo_ra, dec=vo_dec, err=vo_error,units='deg',system=vo_system_coor),obs_time=vo_trigtimeformat,observatory_location=vo_observa_loc)
   
    #vp.add_where_when(v,coords=vp.Position2D(ra=vo_ra, dec=vo_dec, err=vo_error,units='deg',system=vo_system_coor),obs_time=vo_trigtime,
                  #observatory_location=vo_observa_loc)

    ## You would normally describe or reference your telescope / instrument here
    #vp.add_how(v, descriptions=vo_description,references=vp.Reference(vo_reference))

    #vp.add_why(v, importance=vo_importance)

    # We can also cite earlier VOEvents:
    #vp.add_citations(v,
    #             vp.EventIvorn(
    #                 ivorn=VO_dic["ivorn"],
    #                 cite_type=vp.definitions.cite_types.followup))

    # Check everything is schema compliant:
    #vp.assert_valid_as_v2_0(v) 
    file_voevent="./VOEVENTS/"+vo_name
    xml = stringVOEvent(vo_event,VO_dic["voschemaurl"])
    #print(xml)

    vo_event_xml = open(file_voevent, 'w')
    vo_event.set_ivorn("ivo:/lal.org/GRANDMA"+"_"+GRB_dic["trigid"]+"_"+datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    vo_event_xml.write(xml)
    vo_event_xml.close()


   # output_filename = vo_name
    #with open("./VOEVENTS/"+output_filename, 'wb') as f:
        #vp.dump(v, f)
    #if VO_dic["obs"]!=None:
      #stop

   

def fermi_trigger(v, collab, text_mes,file_log_s,role):
    """

    :param v:
    :param collab:
    :param text_mes:
    :return:
    """

    instru = str(collab[2])
    if instru == "GBM":
        toplevel_params = vp.get_toplevel_params(v)
        # READ MESSAGE
        trigger_id = toplevel_params['TrigID']['value']
        message_descrip = str(v.What.Description).split()
      
        if "transient." and "found" in message_descrip:
            id_grb_message_init = v.attrib['ivorn']
            text_mes= fermi_trigger_found(v, collab,role,file_log_s)
                
                

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
            text_mes = fermi_trigger_follow(v, collab, message_type,file_log_s,role)

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


def treatment_alert(filename):
    """

    :param filename:
    :return:
    """
    text_mes = ""
    slack_channel_alert = ""

    #logfile = "/home/debian/lowlatency/grbs/LOG_ALERTS.txt"
    logfile_r=LOGFILE_receivedalerts
    file_log_r = open(LOGFILE_receivedalerts, "a+")

    file_log_s=open(LOGFILE_sendingalerts, "a+")
    
    #extract last alert received and sent
    rlast_alertid=""
    logfile_lines = file_log_r.readlines() 
    if len(logfile_lines)>0:
      rlast_alertid = logfile_lines [len(logfile_lines) -1]



    with open(filename, 'rb') as file_opened:
        v = vp.load(file_opened)
        rnew_alertid=str(v.attrib['ivorn'])
        #Check if we already treat the alert
        if rlast_alertid==rnew_alertid:
            return 0

        #else we will treat the alert
        file_log_r.write(rnew_alertid +""+str(filename)+ "\n")
        role = v.attrib['role']
        collab = str(v.How['Description'])


        #which instrument comes from the alert Swift or Fermi ?

        if "Swift" in collab.split():
            text_mes = swift_trigger(v, collab.split(), text_mes,file_log_s,role)
        if "Fermi" in collab.split():
            text_mes = fermi_trigger(v, collab.split(), text_mes,file_log_s,role)

        #is it a test alert or a real trigger and send via slack
        if role == "test":
            slack_channel_alert = "#testgwmes"
        if role == "observation":
            slack_channel_alert = "#lalalerts"
        print(text_mes)

        #slack_message(slack_channel_alert, text_mes)


letters=np.array(["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"])
LOGFILE_receivedalerts="LOG_ALERTS_RECEIVED.txt"
LOGFILE_sendingalerts="LOG_ALERTS_SENT.txt"
file_LOG = open(LOGFILE_receivedalerts, "a+") 
#LISTE_TELESCOPE=["Zadko","TAROT-Calern","TAROT-Chili","TAROT-Reunion","2.16m","GWACs","F60","TNT","F30","2.4m GMG","CGFT","CFHT","KAIT"]
LISTE_TELESCOPE=["GWACs","F60"]
dic_grb={}
dic_vo={}

#path="/home/leroy/temp/"
path="./EXAMPLE/"
fo = open(path+"READ_xml.txt","r")
lines = fo.readlines()
for line in lines[0:100]:
 filename=path+line.split("\n")[0]
 treatment_alert(filename)
