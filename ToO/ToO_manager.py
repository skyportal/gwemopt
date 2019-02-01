#!/usr/bin/env python


import os
import sys
import io
import datetime
#import config_lowlatency as cf
#from slackclient import SlackClient
#import mma_schedule as mma
import voeventparse as vp
#SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__)) + "/ToO_field_match_v1.1/"
#sys.path.append(os.path.abspath(SCRIPT_PATH))
# CHANGE IMPORT GWAC ORDER TO WAIT PATH
#import GWAC_ToO_observable_calculator as GTOC
import GRANDMA_FAshifts as fa

import numpy as np
import pytz
import copy

import glob
import ephem

from astropy import table
from astropy import time

import lxml.objectify as objectify
from lxml import etree
import pandas as pd



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

def Tel_dicf():
  Tel_dic = {

    "Name":"",
    "FOV":"",
    "magnitude":"",
    "exposuretime":"",
    "latitude":"",
    "longitude":"",
    "elevation":"",
    "FOV_coverage":"",
    "FOV_coverage_type":"",
    "FOV_type":"",
    "slew_rate":0.0,
    "readout":"",
    "filt":"",
    "OS":"",
	 }
  return Tel_dic


def GW_dicf():
  GW_dic = {

    "Packet_Type":"",
    "Pkt_Ser_Num":"",
    "AlertType":"",
    "Retraction":"",
#    "Instruments" : "",
    "HardwareInj" : "",
    "EventPage":"",
#		  "GraceID" : 0.,
    "FAR":0.,
    "Group" : "",
    "Pipeline":"",
    "HasNS":"",
    "HasRemnant":"",
    "BNS":"",
    "NSBH":"",
    "BBH":"",
    "Terrestrial":"",
    "location":"",
#    "Skymaplink" : :"",
	 }
  return GW_dic


def GRB_dicf():
  GRB_dic = {

    "Packet_Type":"",
    "Pkt_Ser_Num":"",
    "Trigger_TJD":"",
    "Trigger_SOD":"",
    
    
    "grbid" : "",
		  "ratesnr" : 0.,
    "ratets" : 0.,
    "imagesnr" : 0.,
    "lc" : "",
    "hratio" : 0.,
    "longshort" : False,
    "probGRB" : -1.0,
    "defGRB" : True,
    "locsnr" : 0.,
    "locdur" : 0.,
    "locref" : "",
    "obs" : "",
	 }
  return GRB_dic

def VO_dicf():
  Contentvo_dic = {
		  "name" : "",
    "role" : "",
		  "stream" : "grandma.lal.in2p3.fr/GRANDMA_Alert",
		  "streamid" : "",
    "voivorn" : "ivo://svom.bao.ac.cn/LV#SFC_GW_",
		  "authorivorn" : "GRANDMA_Alert",
		  "shortName" : "GRANDMA",
		  "contactName": "Nicolas  Leroy",
    "contactPhone": "+33-1-64-46-83-73",
    "contactEmail": "leroy@lal.in2p3.fr",
    "description": "Selected by ",
    "vodescription": "VOEvent between CSC and FSC",
    "locationserver": "",
    "voschemaurl":"http://www.cacr.caltech.edu/~roy/VOEvent/VOEvent2-110220.xsd",
    "ba":"",
    "ivorn":"",
    "letup":"a",
    "trigid":None,
    "eventype":None,
    "eventstat":None,
    "inst" : "",
    "trigdelay" : 0.,
    "locpix":"",
    "trigtime":"",
    "ra" : 0.,
    "dec" : 0.,
    "error" : 0.,
    "evenstatus" : "",
    "voimportance":"",
    "iter_statut":0,
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
    fa_duty=fa.FA_shift()
    return fa_duty


def Observation_plan(teles_target,obsinstru,trigtime,urlhelpix,VO_dic):

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
    params["telescopes"] = [teles_target]
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

    config_struct = params["config"][teles_target]



    #table_field = utilityTable(thistable)
    #table_field.blankTable(len(coverage_struct))

    field_id_vec=[]
    ra_vec=[]
    dec_vec=[]
    grade_vec=[]

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
 
        field_id_vec.append(field_id)
        ra_vec.append(ra)
        dec_vec.append(dec)
        grade_vec.append(prob)



    return np.transpose(np.array([np.array(field_id_vec),np.array(ra_vec),np.array(dec_vec),np.array(grade_vec)]))

def swift_trigger(v, collab, text_mes,file_log_s,role):
    """

    :param v:
    :param collab:
    :param text_mes:
    :return:
    """

    Swift_dic=GRB_dicf()
    Swift_vo=VO_dicf()

    instru = str(collab[2])
    Swift_vo["ba"]=fa.FA_shift()
    

    if instru == "BAT":

        Swift_dic["inst"]=instru

        top_level_params = vp.get_toplevel_params(v)
        trigger_id = top_level_params['TrigID']['value']
        Swift_vo["trigid"]=trigger_id
    
        rate_signif = top_level_params['Rate_Signif']['value']
        Swift_dic["ratesnr"]=float(rate_signif)

        image_signif = top_level_params['Image_Signif']['value']
        Swift_dic["imagesnr"]=float(image_signif)

        if float(image_signif) < 4.0:
           Swift_vo["voimportance"]=3
        if ((float(image_signif)>=6.0) &  (float(image_signif)<7.0)):
           Swift_vo["voimportance"]=2
        if ((float(image_signif)>7.0)):
           Swift_vo["voimportance"]=1



        def_not_grb =  v.find(".//Param[@name='Def_NOT_a_GRB']").attrib['value']
        Swift_dic["defGRB"]=def_not_grb 

        Swift_vo["evenstatus"]="initial"
        Swift_vo["eventype"]="GRB"
        Swift_vo["inst"]="Swift-BAT"
        Swift_vo["location"]="Sky"

        isotime = v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Time.TimeInstant.\
            ISOTime.text
        delay = delay_fct(isotime)
        isotime_format=trigtime(isotime)
        delay_min=(delay.seconds)/60.0
        Swift_vo["trigtime"]=isotime_format
        Swift_vo["trigdelay"]=delay_min


        right_ascension = str(v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.\
                              Position2D.Value2.C1.text)
        declination = str(v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Position2D\
                          .Value2.C2.text)
        error2_radius = str(v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.\
                            Position2D.Error2Radius.text)
   
        Swift_vo["ra"]=right_ascension
        Swift_vo["trigdelay"]=right_ascension
        Swift_vo["dec"]=declination
        Swift_vo["error"]=error2_radius
        Swift_dic["locref"]="BAT onboard"
        
        name_dic="Swift"+trigger_id
        if ((role!="test")):
                lalid=name_lalid(v,file_log_s,name_dic,Swift_vo["letup"],"")
                name_dic="Swift"+trigger_id
                dic_grb[name_dic]=Swift_dic
                dic_vo[name_dic]=Swift_vo

                create_GRANDMAvoevent(lalid,Swift_dic,Swift_vo,"")
                file_log_s.write(lalid +" "+str(trigger_id)+"\n")
        text_mes = str("---------- \n")+str("BAT alert \n")+str("---------- \n")+\
            str("Trigger ID: ")\
            +trigger_id+("\n")+str("Trigger Time: ")+isotime+("\n")+str("Delay since alert: ")+\
            str(delay)+("\n")+str("\n")+str("---Follow-up Advocate---\n")+str("FA on duty: ")+str(fa.FA_shift())+str("\n")+("\n")+str("\n")+\
            str("---SPACE TRIGGER---\n")+str("Trigger Rate SNR: ")+str(rate_signif)+" "+\
            str("Image_Signif: ")+image_signif+("\n")+str("\n")+str("---Position---\n")+\
            "RA: "+str(round(float(right_ascension),1))+" "+"DEC: "+str(round(float(declination),1))+" "+str("Error2Radius: ")+\
            str(round(float(error2_radius),1))+"\n"+str("\n")+str("---------- \n")#+("---SVOM FOLLOWUP---\n")+\
            #str(observability__xinglong)+" "+str(follow)+"\n"
         
    return text_mes


def GW_trigger_retracted(v, collab,role,file_log_s):

    GW_dic=GW_dicf()
    GW_vo=VO_dicf()

    
    
    GW_vo["ba"]=fa.FA_shift()


    GW_vo["eventype"]="GW"


    toplevel_params = vp.get_toplevel_params(v)
    Pktser=toplevel_params['Pkt_Ser_Num']['value']
    GW_vo["iter_statut"]=str(int(Pktser)-1)
    GW_vo["evenstatus"]=toplevel_params['AlertType']['value']
    trigger_id = toplevel_params['GraceID']['value']
 
    GW_vo["letup"]=letters[int(Pktser)-1]
    

    GW_dic["Retraction"]=toplevel_params['Retraction']['value']

    isotime = v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Time.TimeInstant.ISOTime.text
    isotime_format=trigtime(isotime)
    GW_vo["trigtime"]=isotime_format    
    GW_vo["trigid"]=trigger_id
    GW_vo["location"]="LIGO Virgo"
    GW_vo["eventype"]="GW"
    GW_vo["voimportance"]=1
 
    if ((role=="test")):
         name_dic="GW"+trigger_id
         lalid=name_lalid(v,file_log_s,name_dic,GW_vo["letup"],"_DB")
         create_GRANDMAvoevent(lalid,GW_dic, GW_vo,"")      
  
    for telescope in LISTE_TELESCOPE:
         Tel_dic=Tel_dicf()
         Tel_dic["Name"]=telescope 
         
         #Tel_dic["OS"]=Observation_plan(telescope,GW_vo["inst"],GW_vo["trigtime"],GW_vo["locpix"],Tel_dic)
         if ((role=="test")):
             name_dic="GW"+trigger_id
             lalid=name_lalid(v,file_log_s,name_dic,GW_vo["letup"],"_"+Tel_dic["Name"])
             create_GRANDMAvoevent(lalid,GW_dic, GW_vo,Tel_dic)      


    file_log_s.write(lalid +" "+str(trigger_id)+"\n")


    text = str("---------- \n")+str("GW alert \n")+str("---------- \n")+str("GW NAME : ")\
        +str(GW_vo["trigid"])+(" ")+str("Trigger Time: ")+isotime+"\n"+\
        str("WARNING RETRACTATION")+str("\n")+str("---Follow-up Advocate--\n")+str("Follow-up advocate on duty: ")+str(fa.FA_shift())+"\n"
    return text
   

def GW_treatment_alert(v, collab,role,file_log_s):




    GW_dic=GW_dicf()
    GW_vo=VO_dicf()

    
    
    GW_vo["ba"]=fa.FA_shift()


    GW_vo["eventype"]="GW"


    toplevel_params = vp.get_toplevel_params(v)
    Pktser=toplevel_params['Pkt_Ser_Num']['value']
    GW_vo["iter_statut"]=str(int(Pktser)-1)
    GW_vo["inst"]=toplevel_params['Instruments']['value']
    GW_vo["evenstatus"]=toplevel_params['AlertType']['value']
    trigger_id = toplevel_params['GraceID']['value']
 
    GW_vo["letup"]=letters[int(Pktser)-1]
    

    GW_dic["Retraction"]=toplevel_params['Retraction']['value']
    GW_dic["HardwareInj"]=toplevel_params['HardwareInj']['value']
    GW_dic["EventPage"]=toplevel_params['EventPage']['value']
    GW_dic["FAR"]=toplevel_params['FAR']['value']
    GW_dic["Group"]=toplevel_params['Group']['value']
    GW_dic["Pipeline"]=toplevel_params['Pipeline']['value']
    GW_dic["Classification"]=vp.get_grouped_params(v)
    GW_dic["locref"]="bayestar"
    grouped_params=vp.get_grouped_params(v)
    HasRemnant=float(v.find(".//Param[@name='HasRemnant']").attrib['value'])
    BNS=str(v.find(".//Param[@name='BNS']").attrib['value'])
    GW_dic["BNS"]=BNS
    NSBH=str(v.find(".//Param[@name='NSBH']").attrib['value'])
    GW_dic["NSBH"]=NSBH
    BBH=str(v.find(".//Param[@name='BBH']").attrib['value'])
    GW_dic["BBH"]=BBH
    Terrestrial=str(v.find(".//Param[@name='Terrestrial']").attrib['value'])
    GW_dic["Terrestrial"]=Terrestrial
    HasNS=str(v.find(".//Param[@name='HasNS']").attrib['value'])
    GW_dic["HasRemnant"]=str(HasRemnant)
    GW_dic["HasNS"]=str(HasNS)
    #print(len(GW_dic["inst"].split(",")))
 
    if HasRemnant > 0.9:
      GW_vo["voimportance"]=1
      if (len(GW_vo["inst"].split(","))) > 2:
        GW_vo["voimportance"]=2
      else:
        GW_vo["voimportance"]=3

    isotime = v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Time.TimeInstant.ISOTime.text
    

    delay = delay_fct(isotime)
    isotime_format=trigtime(isotime)
    GW_vo["trigtime"]=isotime_format
    delay_min=(delay.seconds)/60.0
    GW_vo["trigdelay"]=delay_min

    
    GW_vo["trigid"]=trigger_id
    
    GW_vo["locpix"]=str(v.find(".//Param[@name='skymap_fits']").attrib['value'])
    GW_vo["location"]="LIGO Virgo"

    message_obs="NO SKYMAP AVAILABLE"
    if (GW_vo["locpix"]!="") and (GW_vo["evenstatus"]=="Preliminary"):
     GW_vo["evenstatus"]="Initial"
    if (GW_vo["evenstatus"]!="Preliminary"):
     Observation_plan_tel=[] 
     message_obs="Observation plan sent to "   
     if ((role=="test")):
         name_dic="GW"+trigger_id
         lalid=name_lalid(v,file_log_s,name_dic,GW_vo["letup"],"_DB")
         create_GRANDMAvoevent(lalid,GW_dic, GW_vo,"")      
  
     for telescope in LISTE_TELESCOPE:
         Tel_dic=Tel_dicf()
         Tel_dic["Name"]=telescope 
         message_obs=message_obs+" "+telescope
         #print(telescope)
         
         Tel_dic["OS"]=Observation_plan(telescope,GW_vo["inst"],GW_vo["trigtime"],GW_vo["locpix"],Tel_dic)
         if ((role=="test")):
             name_dic="GW"+trigger_id
             lalid=name_lalid(v,file_log_s,name_dic,GW_vo["letup"],"_"+Tel_dic["Name"])
             create_GRANDMAvoevent(lalid,GW_dic, GW_vo,Tel_dic)      

    else:

     if ((role=="test")):
         name_dic="GW"+trigger_id
         lalid=name_lalid(v,file_log_s,name_dic,GW_vo["letup"],"_DB")
         create_GRANDMAvoevent(lalid,GW_dic, GW_vo,"")



    file_log_s.write(lalid +" "+str(trigger_id)+"\n")


    text = str("---------- \n")+str("GW alert \n")+str("---------- \n")+str("GW NAME : ")\
        +str(GW_vo["trigid"])+(" ")+str("Trigger Time: ")+isotime+"\n"+\
        str("Instruments: ")+str(str(GW_vo["inst"]))+str("\n")\
        +str("EventPage: ")+str(str(GW_dic["EventPage"]))+str("\n")+str("Search: ")+str(str(GW_dic["Group"]))+str("\n")+str("HasRemnant: ")+str(HasRemnant)+str("\n")\
        +str("Delay since alert: ")+str(delay)+("\n")+str("\n")+str("---Follow-up Advocate--\n")+str("Follow-up advocate on duty: ")+str(fa.FA_shift())+"\n"+message_obs+"\n"
    return text


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


    Fermi_vo["fa"]=fa.FA_shift()

    Fermi_vo["evenstatus"]="preliminary"
    Fermi_vo["eventype"]="GRB"
    Fermi_vo["inst"]="Fermi-GBM"
    Fermi_vo["location"]="Sky"


    toplevel_params = vp.get_toplevel_params(v)
    trigger_id = toplevel_params['TrigID']['value']
    #Fermi_vo["trigid"]=trigger_id

    rate__signif = toplevel_params['Trig_Signif']['value']
    rate__dur = toplevel_params['Trig_Dur']['value']
    Fermi_dic["ratesnr"]=rate__signif
    Fermi_dic["ratets"]=rate__dur

    #print("rate__signif",rate__signif)
    if float(rate__signif) < 4.0:
     Fermi_vo["voimportance"]=3
    if ((float(rate__signif)>=6.0) &  (float(rate__signif)<7.0)):
     Fermi_vo["voimportance"]=3
    if ((float(rate__signif)>7.0)):
     Fermi_vo["voimportance"]=2


    isotime = v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Time.TimeInstant.ISOTime\
        .text
    

    delay = delay_fct(isotime)
    isotime_format=trigtime(isotime)
    Fermi_vo["trigtime"]=isotime_format
    delay_min=(delay.seconds)/60.0
    Fermi_vo["trigdelay"]=delay_min

    # grb_proba = str(v.Why.Inference.attrib["probability"])
    grb_lc = toplevel_params['LightCurve_URL']['value']
    energy_bandmin = toplevel_params['Lo_Chan_Energy']['value']
    energy_bandmax = toplevel_params['Hi_Chan_Energy']['value']

    name_grb = gbm_lc_name(grb_lc)
    Fermi_vo["trigid"]=name_grb
    ra = str(v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Position2D.\
    Value2.C1.text)
    dec = str(v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Position2D.\
    Value2.C2.text)
    error2_radius = str(v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.\
    Position2D.Error2Radius.text)
    Fermi_vo["ra"]=ra
    Fermi_vo["dec"]=dec
    Fermi_vo["error"]=error2_radius
    Fermi_dic["locref"]="GBM onboard"
  

    if ((role!="test")):
        name_dic="GBM"+trigger_id
        lalid=name_lalid(v,file_log_s,name_dic,Fermi_vo["letup"],"")
        create_GRANDMAvoevent(lalid,Fermi_dic, Fermi_vo,"")
        dic_grb[name_dic]=Fermi_dic
        dic_vo[name_dic]=Fermi_vo


        file_log_s.write(lalid +" "+str(trigger_id)+"\n")

    text = str("---------- \n")+str("FERMI/GBM alert \n")+str("---------- \n")+str("GRB NAME : ")\
        +str(name_grb)+(" ")+str("Trigger ID: ")+trigger_id+("\n")+str("Trigger Time: ")+isotime+\
        ("\n")+str("Delay since alert: ")+str(delay)+("\n")+str("\n")+str("---Follow-up Advocate--\n")+str\
        ("FA on duty: ")+str(fa.FA_shift())+"\n"+\
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
 
    
    Fermi_vo["eventype"]="GRB"
    Fermi_vo["inst"]="Fermi-GBM"


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
        Fermi_vo["evenstatus"]="initial"
        for h in np.arange(len(path_helpcolec)-1):
         if h!=0:
            path_healpix=path_healpix+"/"+path_helpcolec[h]
        link_healpix=path_healpix+"/"+healpix_name
        Fermi_dic["locpix"]=link_healpix
        Fermi_vo["iter_statut"]=0
       
        Observation_plan_tel=[]        
        for telescope in LISTE_TELESCOPE:
            #Observation_plan_tel.append(Observation_plan(telescope,Fermi_vo["inst"],Fermi_vo["trigtime"],Fermi_vo["locpix"])
            a_revoir=1
        Fermi_vo["obs_stra"]=Observation_plan_tel

    grb_lc = toplevel_params['LightCurve_URL']['value']
    
    
    name_grb = gbm_lc_name(grb_lc)
    Fermi_dic["lc"]=grb_lc 
    Fermi_vo["grbid"]=name_grb 
  
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
        Fermi_vo["evenstatus"]="preliminary"
        Fermi_vo["iter_statut"]=Fermi_vo["iter_statut"]+1

    if message_type == "GROUND UPDATE FERMI/GBM POSITION MESSAGE":
        Fermi_dic["locref"]="GBM ground update"
        rate__signif = toplevel_params['Burst_Signif']['value']
        Fermi_dic["locsnr"]=rate__signif
        rate__dur = toplevel_params['Data_Integ']['value']
        Fermi_dic["locdur"]=rate__dur
        not_grb = def_not_grb =  v.find(".//Param[@name='Def_NOT_a_GRB']").attrib['value']
        Fermi_dic["defGRB"]=not_grb
        loc_fit=toplevel_params['LocationMap_URL']['value']
        Fermi_vo["locpix"]=loc_fit
        Fermi_vo["evenstatus"]="preliminary"
        Fermi_vo["iter_statut"]=Fermi_vo["iter_statut"]+1

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
    Fermi_vo["trigtime"]=trigtime(isotime)
    Fermi_vo["trigdelay"]=delay_min
    Fermi_vo["ra"]=ra
    Fermi_vo["dec"]=dec
    Fermi_vo["error"]=error2_radius

    if  ((role!="test")):
     lalid=name_lalid(v,file_log_s,name_dic,Fermi_vo["letup"],"")
     create_GRANDMAvoevent(lalid,Fermi_dic, Fermi_vo,"")

     

    if grb_identified == "false":
        text = "\n"+message_type+str(" \n")+str("---------- \n")+str("ID : ")+str(name_grb)
    else:
        text = "\n"+str("UPDATE FERMI/GBM POSITION MESSAGE NO LONGER CLASSIFIED AS GRB\n")+str\
            ("---------- \n")+str("ID : ")+str(name_grb)+(" ")+trigger_id+(" ")+isotime+("\n")+str\
            ("Delay since alert: ")+str(delay)+("\n")+str("\n")
    return text



#A unique name per event
def name_lalid(v,file_log_s,name_dic,letter,tel):
    name_dic=name_dic+tel
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

def add_GWvoeventcontent(GW_dic,v):


    retractation = vp.Param(name="Retraction",value=GW_dic["Retraction"],dataType="int",ucd="meta.number")
    retractation.Description="Set to 1 if the event is retracted."
    v.What.append(retractation)

    hwinj = vp.Param(name="HardwareInj",value=GW_dic["HardwareInj"], ucd="meta.number",dataType="int")
    hwinj.Description="Indicates that this event is a hardware injection if 1, no if 0"
    v.What.append(hwinj)

    eventpage=vp.Param(name="Event_page",value=GW_dic["EventPage"], ucd="meta.ref.url",dataType="string")
    eventpage.Description="Web page for evolving status of this GW candidate"
    v.What.append(eventpage)

    FAR=vp.Param(name="FAR",value=GW_dic["FAR"], ucd="arith.rate;stat.falsealarm",unit="Hz")
    FAR.Description="Web page for evolving status of this GW candidate"
    v.What.append(FAR)

    Group=vp.Param(name="Group",value=GW_dic["Group"], ucd="meta.code",dataType="string")
    Group.Description="Data analysis working group"
    v.What.append(Group)

    Pipeline=vp.Param(name="Pipeline",value=GW_dic["Pipeline"], ucd="meta.code",dataType="string")
    Group.Description="Low-latency data analysis pipeline"
    v.What.append(Pipeline)

    BNS = vp.Param(name="BNS", value=GW_dic["BNS"], dataType="float", ucd="stat.probability")
    BNS.Description="Probability that the source is a binary neutron star merger"
    NSBH = vp.Param(name="NSBH", value=GW_dic["NSBH"], dataType="float", ucd="stat.probability")
    NSBH.Description = "Probability that the source is a neutron star - black hole merger"
    BBH = vp.Param(name="BBH", value=GW_dic["BBH"], dataType="float", ucd="stat.probability")
    BBH.Description = "Probability that the source is a binary black hole merger"
    Terrestrial = vp.Param(name="Terrestrial", value=GW_dic["Terrestrial"], dataType="float", ucd="stat.probability")
    Terrestrial.Description = "Probability that the source is terrestrial (i.e., a background noise fluctuation or a glitch)"  
    group_class=vp.Group(params=[BNS, NSBH, BBH, Terrestrial], name="Classification")
    group_class.Description="Source classification: binary neutron star (BNS), neutron star-black hole (NSBH), binary black hole (BBH), or terrestrial (noise)"
    v.What.append(group_class)

    HasNS = vp.Param(name="HasNS", value=GW_dic["HasNS"], dataType="float", ucd="stat.probability")
    HasNS.Description = "Probability that at least one object in the binary has a mass that is less than 3 solar masses"
    HasRemnant = vp.Param(name="HasRemnant", value=GW_dic["HasRemnant"], dataType="float", ucd="stat.probability")
    HasRemnant.Description = "Probability that a nonzero mass was ejected outside the central remnant object"   
    group_prop=vp.Group(params=[HasNS, HasRemnant], name="Properties")
    group_prop.Description="Qualitative properties of the source, conditioned on the assumption that the signal is an astrophysical compact binary merger"
    v.What.append(group_prop)

    #v.What.append(GW_dic["Classification"])


def add_GRBvoeventcontent(GRB_dic,v):
   #GRB Parameters

    #grbid = Param(name="GRBID",value=GRB_dic["grbid"], ucd="meta.id")
    #grbid.set_Description(['GRB ID'])
    #what.add_Param(grbid)

    #trigonlinerate_snr = Param(name="Rate_snr",value=GRB_dic["ratesnr"], unit="sigma", ucd="stat.snr")
    #trigonlinerate_snr.set_Description(['Significance from the GRB rate onboard trigger algorithm of '+GRB_dic["inst"]])
    #what.add_Param(trigonlinerate_snr)

    trigonlinerate_snr = vp.Param(name="Rate_snr",value=GRB_dic["ratesnr"], unit="sigma", ucd="stat.snr",dataType="floay")
    trigonlinerate_snr.Description="Significance from the GRB rate onboard trigger algorithm of "+GRB_dic["inst"]
    v.What.append(trigonlinerate_snr)

   
    #trigonlinerate_ts = Param(name="Rate_ts",value=GRB_dic["ratets"], unit="s", ucd="time.interval")
    #trigonlinerate_ts.set_Description = 'Timescale used in the GRB onboard pipeline of '+GRB_dic["inst"]
    #what.add_Param(trigonlinerate_ts)

    trigonlinerate_ts = vp.Param(name="Rate_ts",value=GRB_dic["ratets"], unit="s", ucd="time.interval",dataType="float")
    trigonlinerate_ts.Description = "Timescale used in the GRB onboard pipeline of "+GRB_dic["inst"]
    v.What.append(trigonlinerate_ts)
    
    
    #trigonlinerate_snr = Param(name="Img_snr",value=GRB_dic["imagesnr"], unit="sigma", ucd="stat.snr")
    #trigonlinerate_snr.set_Description(['Significance from the GRB image onboard pipeline of '+GRB_dic["inst"]])
    #what.add_Param(trigonlinerate_snr)

    trigonlinerate_snr = vp.Param(name="Img_snr",value=str(GRB_dic["imagesnr"]), unit="sigma", ucd="stat.snr",dataType="float")
    trigonlinerate_snr.Description="Significance from the GRB image onboard pipeline of "+GRB_dic["inst"]
    v.What.append(trigonlinerate_snr)

    #lc = Param(name="LightCurve_URL",value=GRB_dic["lc"],ucd="meta.ref.url")
    #lc.Description(['The GRB LC_URL file will not be created/available until ~15 min after the trigger. Instrument:'+GRB_dic["inst"]])
    #what.add_Param(lc)
  
    lc = vp.Param(name="LightCurve_URL",value=GRB_dic["lc"],ucd="meta.ref.url",dataType="string")
    lc.Description="The GRB LC_URL file will not be created/available until ~15 min after the trigger. Instrument:"+GRB_dic["inst"]
    v.What.append(lc)


    #trigonlinerate_hardratio = Param(name="Hardness_Ratio",value=GRB_dic["hratio"], ucd="arith.ratio")
    #trigonlinerate_hardratio.set_Description(['GRB flight Spectral characteristics of '+GRB_dic["locref"]])
    #what.add_Param(trigonlinerate_hardratio)

    trigonlinerate_hardratio = vp.Param(name="Hardness_Ratio",value=str(GRB_dic["hratio"]), ucd="arith.ratio",dataType="float")
    trigonlinerate_hardratio.Description="GRB flight Spectral characteristics of "+GRB_dic["locref"]
    v.What.append(trigonlinerate_hardratio)


    longshort = vp.Param(name="Long_short",value=str(GRB_dic["longshort"]),dataType="bool")
    longshort.Description="GRB long-short of "+GRB_dic["locref"]
    v.What.append(longshort)
   
    #probGRB=Param(name="Prob_GRB",value=GRB_dic["probGRB"])
    #probGRB.set_Description(['Probability to be a GRB defined by '+GRB_dic["locref"]])
    #what.add_Param(probGRB)

    probGRB=vp.Param(name="Prob_GRB",value=str(GRB_dic["probGRB"]),dataType="float",ucd="meta.number")
    probGRB.Description="Probability to be a GRB defined by "+GRB_dic["locref"]
    v.What.append(probGRB)


    #defGRB=Param(name="Def_NOT_a_GRB",value=GRB_dic["defGRB"])
    #defGRB.set_Description(['Not a GRB '+GRB_dic["locref"]])
    #what.add_Param(defGRB)
 
    defGRB=vp.Param(name="Def_NOT_a_GRB",value=str(GRB_dic["defGRB"]),dataType="bool")
    defGRB.Description="Not a GRB "+GRB_dic["locref"]
    v.What.append(defGRB)


    #orpos = Param(name="Loc_ref",value=GRB_dic["locref"])
    #orpos.set_Description(['Localization determined by '+GRB_dic["locref"]])
    #what.add_Param(orpos)

    orpos = vp.Param(name="Loc_ref",value=GRB_dic["locref"],dataType="string",ucd="meta.ref.url")
    orpos.Description="Localization determined by "+GRB_dic["locref"]
    v.What.append(orpos)

    #snrloc = Param(name="Loc_snr",value=GRB_dic["locsnr"])
    #snrloc.Description = 'Fight/Ground position snr to calculate the position of '+GRB_dic["locref"]
    #what.add_Param(snrloc)   


    snrloc = vp.Param(name="Loc_snr",value=str(GRB_dic["locsnr"]),unit="sigma", ucd="stat.snr",dataType="float")
    snrloc.Description ="Fight/Ground position snr to calculate the position of "+GRB_dic["locref"]
    v.What.append(snrloc) 

    #durloc = Param(name="Loc_dur",value=GRB_dic["locdur"])
    #durloc.set_Description(['Fight/Ground timescale to calculate the position of '+GRB_dic["locref"]])
    #what.add_Param(durloc)

    durloc = vp.Param(name="Loc_dur",value=str(GRB_dic["locdur"]), unit="s", ucd="time.interval",dataType="float")
    durloc.Description="Fight/Ground timescale to calculate the position of "+GRB_dic["locref"]
    v.What.append(durloc)



   

def create_GRANDMAvoevent(lalid,Trigger_dic,VO_dic,Tel_dic):
    """
    Create the VOEvent
    """

    vo_name=lalid+".xml"

    role=vp.definitions.roles.test

    lalid_bis=lalid.split("_")
    VO_dic["streamid"]=""
    for h in np.arange(len(lalid_bis)):
     VO_dic["streamid"]=VO_dic["streamid"]+lalid_bis[h]

 
    v = vp.Voevent(stream=VO_dic["stream"],stream_id=VO_dic["streamid"], role=vp.definitions.roles.test)

    vp.set_who(v, date=datetime.datetime.utcnow(),author_ivorn=VO_dic["authorivorn"])

    vp.set_author(v, contactName=VO_dic["contactName"])
    vp.set_author(v, shortName=VO_dic["shortName"])
    vp.set_author(v,contactPhone=VO_dic["contactPhone"])
    vp.set_author(v,contactEmail=VO_dic["contactEmail"])

    # Now create some Parameters for entry in the 'What' section.

    server = vp.Param(name="VOLocat",value=VO_dic["locationserver"])
    server.Description = 'VOevent stored'


    trigid = vp.Param(name="Event_ID",value=VO_dic["trigid"],ucd="meta.id",dataType="string")
    trigid.Description = "Trigger ID"
    v.What.append(trigid)
    

    #alertype = Param(name="Event_type", value=VO_dic["eventype"])
    #alertype.set_Description(["Type of the event"])
    #what.add_Param(alertype)

    alertype = vp.Param(name="Event_type", value=VO_dic["eventype"],ucd="meta.id",dataType="string")
    alertype.Description = "Type of the alert"
    v.What.append(alertype)

    alerstatus = vp.Param(name="Event_status", value=VO_dic["evenstatus"],ucd="meta.version",dataType="string")
    alerstatus.Description="Event status (preliminary, initial, update, retractation)"
    alerstatus_iter = vp.Param(name="Iteration", value=str(VO_dic["iter_statut"]),ucd="meta.number",dataType="int")
    alerstatus_iter.Description = "Iteration Number"
    status_alerts=vp.Group(params=[alerstatus, alerstatus_iter], name="Status")
    status_alerts.Description="Preliminary is set when there is not healpix skymap, then initial and then updates"
    v.What.append(status_alerts)


    triginstru = vp.Param(name="Event_inst", value=VO_dic["inst"],ucd="meta.code",dataType="string")
    triginstru.Description="Instrument which originated the alert"
    v.What.append(triginstru)

    pixloc = vp.Param(name="Loc_url",value=str(VO_dic["locpix"]),ucd="meta.ref.url",dataType="string")
    #print("cc",VO_dic["locpix"])
    pixloc.Description="The url location of healpix skymap"
    v.What.append(pixloc)


    fa = vp.Param(name="FA",value=VO_dic["ba"],dataType="string",ucd="meta.code")
    fa.Description="GRANDMA follow-up advocate on duty at the time of the VO alert"
    v.What.append(fa)

    if VO_dic["eventype"]=="GRB":
      add_GRBvoeventcontent(Trigger_dic,v)

    if VO_dic["eventype"]=="GW":
      add_GWvoeventcontent(Trigger_dic,v)

    if Tel_dic!="":
      Name_tel = vp.Param(name="Name_tel", value=str(Tel_dic["Name"]),ucd="meta.id",dataType="string")
      Name_tel.Description="Name of the telescope used for the observation strategy"
      FOV_tel = vp.Param(name="FOV", value=str(Tel_dic["FOV"]),ucd="meta.number",dataType="float",unit="deg")
      FOV_tel.Description = "FOV of the telescope used for the observation strategy"
      FOV_coverage = vp.Param(name="FOV_coverage", value=str(Tel_dic["FOV_coverage"]),ucd="meta.number",dataType="string")
      FOV_coverage.Description = "Shape of the FOV for the telescope used for the observation strategy"   
      magnitude = vp.Param(name="Magnitude", value=str(Tel_dic["magnitude"]),ucd="meta.number",dataType="float",unit="mag")
      magnitude.Description = "Magnitude limit of the telescope used for the observation strategy"
      exposuretime = vp.Param(name="exposuretime", value=str(Tel_dic["exposuretime"]),ucd="time.interval",dataType="float",unit="s")
      exposuretime.Description = "Exposure time of the telescope used for the observation strategy"
      slewrate = vp.Param(name="Slew_rate", value=str(Tel_dic["slew_rate"]),ucd="time.interval",dataType="float",unit="s")
      slewrate.Description = "Slew rate of the telescope for the observation strategy"
      readout = vp.Param(name="Readout", value=str(Tel_dic["readout"]),ucd="time.interval",dataType="float",unit="s")
      readout.Description = "Read out of the telescope used for the observation strategy"
      filt = vp.Param(name="Filters", value=str(Tel_dic["filt"]),ucd="meta.ref",dataType="string")
      filt.Description = "Filters of the telescope used for the observation strategy"
      latitude = vp.Param(name="Latitude", value=str(Tel_dic["latitude"]),ucd="meta.number",dataType="float",unit="deg")
      latitude.Description = "Latitude of the observatory"      
      longitude = vp.Param(name="Latitude", value=str(Tel_dic["longitude"]),ucd="meta.number",dataType="float",unit="deg")
      longitude.Description = "Longitude of the observatory"  
      elevation = vp.Param(name="Elevation", value=str(Tel_dic["elevation"]),ucd="meta.number",dataType="float",unit="m")
      elevation.Description = "Elevation of the observatory"  
      config_obs=vp.Group(params=[Name_tel, FOV_tel,FOV_coverage,magnitude, exposuretime, slewrate, readout, filt, latitude, longitude, elevation],name="Set_up_OS")
      config_obs.Description="Set-up parameters for producing the observation strategy"
      v.What.append(config_obs)

      #OS_plan=vp.Param(name="Observation strategy",type="Table",value=Tel_dic["OS"])
      #OS_plan=vp.Param(name="Observation strategy")
      #OS_plan.Description="The list of tiles for "+str(Tel_dic["Name"])
      #OS_plan.Table=vp.Param(name="Table")
      if Tel_dic["OS"]!="":
       obs_req=vp.Param(name="Obs_req", value="1",ucd="meta.number",dataType="int")
       obs_req.Description="Set to 1 if observation are required, 0 to stop the observations"
       v.What.append(obs_req)
       Fields = objectify.Element("Table",name="Obs_plan")
       Fields.Description="Tiles for the observation plan"
       grid_id = objectify.SubElement(Fields, "Field", name="Grid_id",ucd="", unit="", dataType="int")
       grid_id.Description="ID of the grid of FOV"
       ra = objectify.SubElement(Fields, "Field", name="Ra", ucd="pos.eq.ra ", unit="deg", dataType="float") 
       ra.Description="The right ascension at center of fov in equatorial coordinates"
       dec = objectify.SubElement(Fields, "Field", name="Dec", ucd="pos.eq.ra ", unit="deg", dataType="float")      
       dec.Description="The declination at center of fov in equatorial coordinates"
       Os_grade = objectify.SubElement(Fields, "Field", name="Os_grade", ucd="meta.number", unit="None", dataType="float")      
       Os_grade.Description="Gives the importance of the tile/galaxy to observe"
       Data = objectify.SubElement(Fields, "Data") 
       for i in np.arange(len(Tel_dic["OS"])):
          TR = objectify.SubElement(Data, "TR")
          for j in np.arange(len(Tel_dic["OS"][i])):
            #objectify.SubElement(TR, "TD",value=str(Tel_dic["OS"][i][j]))
             objectify.SubElement(TR, 'TD')
             TR.TD[-1]=str(Tel_dic["OS"][i][j])
       v.What.append(Fields)
      else:
        obs_req=vp.Param(name="Obs_req", value="0",ucd="meta.number",dataType="int")
        obs_req.Description="Set to 1 if observation are required, 0 to stop the observations"
        v.What.append(obs_req)

    
    vp.add_where_when(v,coords=vp.Position2D(ra=VO_dic["ra"], dec=VO_dic["dec"], err=VO_dic["error"], units='deg',system=vp.definitions.sky_coord_system.utc_fk5_geo),obs_time=VO_dic["trigtime"],observatory_location=VO_dic["location"])

    vp.add_why(v,importance=VO_dic["voimportance"])
    v.Why.Description = "Internal Ranking for the event (from 1 : most interesting to 3 for least interesting)"

  
    # Check everything is schema compliant:
    #vp.assert_valid_as_v2_0(v) 
    file_voevent="./VOEVENTS/"+vo_name
    try:
       vp.assert_valid_as_v2_0(v)
    except Exception as e:
       print(e)



    with open(file_voevent, 'wb') as f:
             vp.dump(v, f)

    #xml = stringVOEvent(vo_event,VO_dic["voschemaurl"])
    #print(xml)

    #vo_event_xml = open(file_voevent, 'w')
    #vo_event.set_ivorn("ivo:/lal.org/GRANDMA"+"_"+VO_dic["trigid"]+"_"+datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    #vo_event_xml.write(xml)
    #vo_event_xml.close()


   # output_filename = vo_name
    #with open("./VOEVENTS/"+output_filename, 'wb') as f:
        #vp.dump(v, f)
    #if VO_dic["obs"]!=None:
      #stop

def GW_trigger(v, collab, text_mes,file_log_s,role):



   toplevel_params = vp.get_toplevel_params(v)
   # READ MESSAGE
   trigger_id = toplevel_params['GraceID']['value']
   AlertType=toplevel_params['AlertType']['value']
   Retractation=int(toplevel_params['Retraction']['value'])
   text_mes=""

   print(Retractation)
   if (AlertType=="Preliminary") and (Retractation==0):
       text_mes= GW_treatment_alert(v, collab,role,file_log_s)
           
   if (AlertType=="Initial" or AlertType=="Update"):
       if (Retractation==0):
          message_type = "GW UPDATE POSITION MESSAGE"
          text_mes= GW_treatment_alert(v, collab,role,file_log_s)
 
   if (Retractation==1):
       message_type = "GW RETRACTION POSITION MESSAGE"
       text_mes= GW_trigger_retracted(v, collab,role,file_log_s)
       print(text_mes)

   return text_mes
 

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
        collab=""
        try:
         collab = str(v.How['Description'])
        except AttributeError:
         contact=str(v.Who.Author.contactName)
         if "LIGO" in contact.split():
           collab="gravitational"

          
           

        #which instrument comes from the alert Swift or Fermi ?

        if "Swift" in collab.split():
            text_mes = swift_trigger(v, collab.split(), text_mes,file_log_s,role)
        if "Fermi" in collab.split():
            text_mes = fermi_trigger(v, collab.split(), text_mes,file_log_s,role)
        if "gravitational" in collab.split():
            text_mes = GW_trigger(v, collab.split(), text_mes,file_log_s,role)

        #is it a test alert or a real trigger and send via slack
        if role == "test":
            slack_channel_alert = "#testalerts"
        if role == "observation":
            if ("Swift" in collab.split()) or ("Fermi" in collab.split()):
              slack_channel_alert = "#grbalerts"
            if "gravitational" in collab.split():
              slack_channel_alert = "#gwalerts"
        print(text_mes)

        #slack_message(slack_channel_alert, text_mes)


letters=np.array(["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"])
LOGFILE_receivedalerts="LOG_ALERTS_RECEIVED.txt"
LOGFILE_sendingalerts="LOG_ALERTS_SENT.txt"
file_LOG = open(LOGFILE_receivedalerts, "a+") 
#LISTE_TELESCOPE=["Zadko","TAROT-Calern","TAROT-Chili","TAROT-Reunion","2.16m","GWACs","F60","TNT","F30","2.4m GMG","CGFT","CFHT","KAIT"]
LISTE_TELESCOPE=["GWAC"]
dic_grb={}
dic_vo={}

path="./EXAMPLE/"
fo = open(path+"READ_xml.txt","r")
lines = fo.readlines()
for line in lines[0:100]:
 filename=path+line.split("\n")[0]
 treatment_alert(filename)
