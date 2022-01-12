"""
Set of functions used in ToO processing in case of GRB alerts
- decoding of VO event
- perform selection
- prepare observation plans associated to the alert
- prepare output files needed by the different instruments

Authors S. Antier 2022
"""

import gcn
import os
import sys
import datetime
import json
import jsonschema
import dateutil.parser
import numpy as np
import voeventparse as vp
import lxml.objectify as objectify
from astropy.io import fits
import gwemopt.utils
import ToO_manager as too
import output_conf as outg
import score
import utils_too as utils_too
import astropy.time
import pytz
from datetime import datetime
import humanize
import healpy as hp
from ligo.skymap.bayestar import rasterize
from ligo.skymap.postprocess import find_greedy_credible_levels


sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '')))


def grb_trigger(voevent, output_dic, conf_dic):
    """
    Function to make GRB VO event analysis
    - decode the VO event and check the skymap
    - perform selection
    - perform observation plan
    - prepare output files for the different instruments

    :param voevent: xml object already parsed
    :param output_dic: dictionary with output architecture
    :param conf_dic: dictionary for server config and selection infos
    :return:
    """

    # initiate and fill with common part the grb dictionary object
    grb_dic = decode_common_voevent(voevent,output_dic)
    

    # setup output architecture
    outg.update_output_config(output_dic, grb_dic, "GRB")

    outg.setup_outputdir_event(output_dic)
    if grb_dic["Packet_Type"] == 110:
    				outg.update_scoreGRB(output_dic,grb_dic,"date_firstalert")

    if grb_dic["Packet_Type"] == 61:
    				outg.update_scoreGRB(output_dic,grb_dic,"date_firstalert")

    if (grb_dic["Packet_Type"] == 97) or (grb_dic["Packet_Type"] == 61):
    				#update GRANDMA json for the score        
        outg.update_scoreGRB(output_dic, grb_dic,"sun_distance")
        outg.update_scoreGRB(output_dic, grb_dic,"moon_illum")
        outg.update_scoreGRB(output_dic, grb_dic,"moon_distance")  
        outg.update_scoreGRB(output_dic, grb_dic,"snr")
        outg.update_scoreGRB(output_dic, grb_dic,"name")

    if grb_dic["Packet_Type"] == 67:
    				#update GRANDMA json for the score
    				grb_dic["name"] =outg.load_scoreGRB(output_dic,"name")
    				grb_dic["snr"] =outg.load_scoreGRB(output_dic,"snr")
    				grb_dic["date_firstalert"]=outg.load_scoreGRB(output_dic,"date_firstalert")
    				outg.update_scoreGRB(output_dic, grb_dic,"sun_distance")
    				outg.update_scoreGRB(output_dic, grb_dic,"moon_illum")
    				outg.update_scoreGRB(output_dic, grb_dic,"moon_distance") 
    				outg.update_scoreGRB(output_dic, grb_dic,"defGRB")   

            				
    if grb_dic["Packet_Type"] == 111:
    				#update GRANDMA json for the score        
        outg.update_scoreGRB(output_dic, grb_dic,"defGRB")
        outg.update_scoreGRB(output_dic, grb_dic,"sun_distance")
        outg.update_scoreGRB(output_dic, grb_dic,"moon_illum")
        outg.update_scoreGRB(output_dic, grb_dic,"moon_distance")  
        outg.update_scoreGRB(output_dic, grb_dic,"hratio")
        outg.update_scoreGRB(output_dic, grb_dic,"snr")

    if grb_dic["Packet_Type"] == 112:
    				#update GRANDMA json for the score        
        #update GRANDMA json for the score
        grb_dic["hratio"] =outg.load_scoreGRB(output_dic,"hratio")
        grb_dic["date_firstalert"]=outg.load_scoreGRB(output_dic,"date_firstalert")
        outg.update_scoreGRB(output_dic, grb_dic,"defGRB")
        outg.update_scoreGRB(output_dic, grb_dic,"sun_distance")
        outg.update_scoreGRB(output_dic, grb_dic,"moon_illum")
        outg.update_scoreGRB(output_dic, grb_dic,"moon_distance")  
        outg.update_scoreGRB(output_dic, grb_dic,"longshort")
        outg.update_scoreGRB(output_dic, grb_dic,"snr")

    if grb_dic["Packet_Type"] == 115:
    				#update GRANDMA json for the score   
    				grb_dic["hratio"] =outg.load_scoreGRB(output_dic,"hratio")
    				grb_dic["longshort"] =outg.load_scoreGRB(output_dic,"longshort")
    				grb_dic["date_firstalert"]=outg.load_scoreGRB(output_dic,"date_firstalert")     
    				outg.update_scoreGRB(output_dic, grb_dic,"defGRB")
    				outg.update_scoreGRB(output_dic, grb_dic,"sun_distance")
    				outg.update_scoreGRB(output_dic, grb_dic,"moon_illum")
    				outg.update_scoreGRB(output_dic, grb_dic,"moon_distance")
    				outg.update_scoreGRB(output_dic, grb_dic,"snr")


    # now download the skymap (and initiate gwemopt)
    # retrieve the configuration for gwemopt
    params = ""
    map_struct = ""
    if grb_dic["teles"] == "FERMI":
    				if (grb_dic["Packet_Type"] == 112 or grb_dic["Packet_Type"] == 115):
        				params, map_struct = update_skymap(grb_dic, output_dic, conf_dic)

    # need to compute the delay in hours
    # remove tz=datetime.timezone.utc) from previous version
    diff_time = (datetime.utcnow() - grb_dic["dateobs"])
    delaymin = diff_time.total_seconds() / 60.0
    delayhours = diff_time.total_seconds() / 3600.0
    delaydays = diff_time.total_seconds() / (3600.0*24.0)
    grb_dic["delaymin"] = delaymin
    grb_dic["delayhours"] = delayhours
    grb_dic["delaydays"] = delaydays
    grb_dic["delayhum"] = humanize.precisedelta(diff_time)

    # perform selection, this will return a score
    score_event = 0.0
    if grb_dic["teles"] == "SWIFT":
    				score_event = score.swift_score(grb_dic)

    # Needs to be checked
    if grb_dic["teles"] == "FERMI":
    				if (grb_dic["Packet_Type"] == 115) or (grb_dic["Packet_Type"] == 112):
        				score_event = score.fermi_score(grb_dic)
    grb_dic["GRANDMAscore"] = score_event

    # compute observation plans for all telescopes on ground
    # defined in the config file in not above delay
    # if delay < 0, by pass the test
    # then update DB nevertheless ?

    if delayhours < conf_dic["ground_delay"] or conf_dic["ground_delay"] < 0:
        print('Start ground observation plan')
        voevent_list = compute_ground(
            grb_dic, conf_dic, output_dic["vopath"], params, map_struct)
        utils_too.send_voevent(conf_dic, voevent_list)
    return "completed", grb_dic


def decode_common_voevent(voevent,outputdic):
    """
    Function to decode common part of VO event
    :param voevent: xml object already parsed
    :return: dictionary filled with info
    """

    # first init dictionary to store information
    grb_dic = init_grb_dic()

    # Fill the dictionary with the xml content
    # first retreive the parameters

    # fill the info on the sequence
    grb_dic["Packet_Type"] = gcn.get_packet(voevent)

    message_type = ""

    # SWIFT
    if grb_dic["Packet_Type"] == 82:
        message_type = "TEST SWIFT POSITION MESSAGE"
        grb_dic["teles"] = "SWIFT"
        grb_dic["inst"] = "BAT"
        grb_dic["grbid"] = gcn.get_trigname(voevent)
        grb_dic["ratesnr"] = gcn.get_swiftratesnr(voevent)
        grb_dic["imagesnr"] = gcn.get_swiftimagesnr(voevent)
        grb_dic["snr"] = grb_dic["imagesnr"]
        grb_dic["descriptsnr"] = "SNR calculated from the image"
        grb_dic["locref"] = "BAT onboard"
        grb_dic["dateobs"] = gcn.get_swiftdateobs(voevent)
        grb_dic["defGRB"] = gcn.get_defGRB(voevent)
        grb_dic["AlertType"] = "Test"
        grb_dic["Pkt_Ser_Num"] = gcn.get_packetsernum(voevent)
        grb_dic["ra"], grb_dic["dec"], grb_dic["error"], grb_dic["ra_formatted"], grb_dic["dec_formatted"] = gcn.get_swiftgbmlocalization(
            voevent)
    #Quick look only 90% of the burst
    if grb_dic["Packet_Type"] == 97:
        message_type = "FLIGHT SWIFT/BAT ALERT MESSAGE"
        grb_dic["teles"] = "SWIFT"
        grb_dic["inst"] = "BAT"
        grb_dic["name"] = gcn.get_swiftname(voevent)
        grb_dic["grbid"] = gcn.get_trigname(voevent)
        grb_dic["ratesnr"] = gcn.get_swiftratesnr(voevent)
        grb_dic["snr"] = grb_dic["ratesnr"]
        grb_dic["descriptsnr"] = "SNR calculated from the count rate"
        grb_dic["locref"] = "BAT onboard"
        grb_dic["dateobs"] = gcn.get_swiftdateobs(voevent)
        grb_dic["AlertType"] = "Preliminary"
        grb_dic["Pkt_Ser_Num"] = gcn.get_packetsernum(voevent)
        grb_dic["ra"], grb_dic["dec"], grb_dic["error"], grb_dic["ra_formatted"], grb_dic["dec_formatted"] = gcn.get_swiftgbmlocalization(
            voevent)
        grb_dic["sun_distance"]=gcn.get_gbmsundistance(voevent)
        grb_dic["moon_distance"]=gcn.get_gbmmoondistance(voevent)       
        grb_dic["moon_illum"]=gcn.get_gbmmoonillum(voevent)
        grb_dic["date_firstalert"]=str(astropy.time.Time(datetime.utcnow(), format="datetime").isot) 
        
    if grb_dic["Packet_Type"] == 61:
        message_type = "FLIGHT SWIFT/BAT POSITION MESSAGE"
        grb_dic["teles"] = "SWIFT"
        grb_dic["inst"] = "BAT"
        grb_dic["grbid"] = gcn.get_trigname(voevent)
        grb_dic["ratesnr"] = gcn.get_swiftratesnr(voevent)
        grb_dic["imagesnr"] = gcn.get_swiftimagesnr(voevent)
        grb_dic["name"] = gcn.get_swiftname(voevent)
        grb_dic["snr"] = grb_dic["imagesnr"]
        grb_dic["AlertType"] = "Initial"
        grb_dic["descriptsnr"] = "SNR calculated from the image"
        grb_dic["locref"] = "BAT onboard"
        grb_dic["defGRB"] = gcn.get_defGRB(voevent)
        grb_dic["dateobs"] = gcn.get_swiftdateobs(voevent)
        grb_dic["Pkt_Ser_Num"] = gcn.get_packetsernum(voevent)
        grb_dic["ra"], grb_dic["dec"], grb_dic["error"], grb_dic["ra_formatted"], grb_dic["dec_formatted"] = gcn.get_swiftgbmlocalization(
            voevent)
        grb_dic["date_firstalert"]=str(astropy.time.Time(datetime.utcnow(), format="datetime").isot)
        grb_dic["sun_distance"]=gcn.get_gbmsundistance(voevent)
        grb_dic["moon_distance"]=gcn.get_gbmmoondistance(voevent)       
        grb_dic["moon_illum"]=gcn.get_gbmmoonillum(voevent)
               
    if grb_dic["Packet_Type"] == 67:
        message_type = "FLIGHT SWIFT/XRT POSITION MESSAGE"
        grb_dic["teles"] = "SWIFT"
        grb_dic["inst"] = "XRT"
        grb_dic["dateobs"] = gcn.get_swiftdateobs(voevent)
        grb_dic["defGRB"] = gcn.get_defGRB(voevent)
        grb_dic["AlertType"] = "Update"
        grb_dic["Pkt_Ser_Num"] = gcn.get_packetsernum(voevent)
        grb_dic["grbid"] = gcn.get_trigname(voevent)
        grb_dic["ra"], grb_dic["dec"], grb_dic["error"], grb_dic["ra_formatted"], grb_dic["dec_formatted"] = gcn.get_swiftgbmlocalization(
            voevent)
        grb_dic["sun_distance"]=gcn.get_gbmsundistance(voevent)
        grb_dic["moon_distance"]=gcn.get_gbmmoondistance(voevent)       
        grb_dic["moon_illum"]=gcn.get_gbmmoonillum(voevent)
    # GBM
    if grb_dic["Packet_Type"] == 119:
        message_type = "TEST FERMI/GBM POSITION MESSAGE"
        grb_dic["teles"] = "FERMI"
        grb_dic["inst"] = "GBM"
        grb_dic["AlertType"] = "Test"
        grb_dic["grbid"] = gcn.get_trigname(voevent)
        grb_dic["snr"] = gcn.get_gbmratesnr(voevent)
        grb_dic["descriptsnr"] = "SNR calculated from the count rate"
        grb_dic["locref"] = "GBM onboard"
        grb_dic["dateobs"] = gcn.get_fermidateobs(voevent)
        grb_dic["defGRB"] = gcn.get_defGRB(voevent)
        grb_dic["Pkt_Ser_Num"] = gcn.get_packetsernum(voevent)
        grb_dic["ra"], grb_dic["dec"], grb_dic["error"], grb_dic["ra_formatted"], grb_dic["dec_formatted"] = gcn.get_swiftgbmlocalization(
            voevent)
        grb_dic["name"]=get_gbmname(voevent)
        grb_dic["date_firstalert"]=str(astropy.time.Time(datetime.utcnow(), format="datetime").isot)
    				
    if grb_dic["Packet_Type"] == 110:
        message_type = "FLIGHT FERMI/GBM ALERT MESSAGE"
        grb_dic["teles"] = "FERMI"
        grb_dic["inst"] = "GBM"
        grb_dic["AlertType"] = "Preliminary"
        grb_dic["grbid"] = gcn.get_trigname(voevent)
        grb_dic["descriptsnr"] = "SNR calculated from the count rate"
        grb_dic["locref"] = "GBM onboard"
        grb_dic["dateobs"] = gcn.get_fermidateobs(voevent)
        grb_dic["Pkt_Ser_Num"] = gcn.get_packetsernum(voevent)
        grb_dic["ra"], grb_dic["dec"], grb_dic["error"], grb_dic["ra_formatted"], grb_dic["dec_formatted"] = gcn.get_swiftgbmlocalization(
            voevent)
        grb_dic["name"]=gcn.get_gbmname(voevent)
        grb_dic["date_firstalert"]=str(astropy.time.Time(datetime.utcnow(), format="datetime").isot)
    				

    
    if grb_dic["Packet_Type"] == 111:
        message_type = "FLIGHT UPDATE FERMI/GBM POSITION MESSAGE"
        grb_dic["teles"] = "FERMI"
        grb_dic["inst"] = "GBM"
        grb_dic["AlertType"] = "Initial"
        grb_dic["grbid"] = gcn.get_trigname(voevent)
        grb_dic["descriptsnr"] = "SNR calculated from the count rate"
        grb_dic["locref"] = "GBM onboard"
        grb_dic["dateobs"] = gcn.get_fermidateobs(voevent)
        grb_dic["defGRB"] = gcn.get_defGRB(voevent)
        grb_dic["Pkt_Ser_Num"] = gcn.get_packetsernum(voevent)
        grb_dic["ra"], grb_dic["dec"], grb_dic["error"], grb_dic["ra_formatted"], grb_dic["dec_formatted"] = gcn.get_swiftgbmlocalization(
            voevent)
        grb_dic["sun_distance"]=gcn.get_gbmsundistance(voevent)
        grb_dic["moon_distance"]=gcn.get_gbmmoondistance(voevent)       
        grb_dic["moon_illum"]=gcn.get_gbmmoonillum(voevent)    
        grb_dic["defGRB"] = gcn.get_defGRB(voevent)  
        grb_dic["hratio"] = gcn.get_gbmhratio(voevent) 
        grb_dic["name"]=gcn.get_gbmname(voevent)
        
								
								    
    if grb_dic["Packet_Type"] == 112:
        message_type = "GROUND UPDATE FERMI/GBM POSITION MESSAGE"
        grb_dic["teles"] = "FERMI"
        grb_dic["inst"] = "GBM"
        grb_dic["AlertType"] = "Update"
        grb_dic["grbid"] = gcn.get_trigname(voevent)
        grb_dic["snr"] = gcn.get_gbmratesnr(voevent)
        grb_dic["descriptsnr"] = "SNR calculated from the count rate"
        grb_dic["locref"] = "GBM ground"
        grb_dic["dateobs"] = gcn.get_fermidateobs(voevent)
        grb_dic["defGRB"] = gcn.get_defGRB(voevent)
        grb_dic["Pkt_Ser_Num"] = gcn.get_packetsernum(voevent)
        grb_dic["ra"], grb_dic["dec"], grb_dic["error"], grb_dic["ra_formatted"], grb_dic["dec_formatted"] = gcn.get_swiftgbmlocalization(
            voevent)
        grb_dic["skymap"] = gcn.get_gbmskymapini(voevent)
        grb_dic["quicklook"] = gcn.get_gbmquicklook(voevent)
        #grb_dic["name"] = "GRB"+(grb_dic["skymap"]['localization_name'].split("_bn")[1]).split(".fit")[0]
        grb_dic["name"]=gcn.get_gbmname(voevent)
        grb_dic["longshort"]=gcn.get_longshort(voevent)
        grb_dic["sun_distance"]=gcn.get_gbmsundistance(voevent)
        grb_dic["moon_distance"]=gcn.get_gbmmoondistance(voevent)       
        grb_dic["moon_illum"]=gcn.get_gbmmoonillum(voevent)
								

								 
    if grb_dic["Packet_Type"] == 115:
        message_type = "FINAL FERMI/GBM POSITION MESSAGE"
        grb_dic["teles"] = "FERMI"
        grb_dic["inst"] = "GBM"
        grb_dic["AlertType"] = "Update"
        grb_dic["grbid"] = gcn.get_trigname(voevent)
        grb_dic["snr"] = gcn.get_gbmratesnr(voevent)
        grb_dic["descriptsnr"] = "SNR calculated from the count rate"
        grb_dic["locref"] = "GBM ground"
        grb_dic["dateobs"] = gcn.get_fermidateobs(voevent)
        grb_dic["defGRB"] = gcn.get_defGRB(voevent)
        grb_dic["Pkt_Ser_Num"] = gcn.get_packetsernum(voevent)
        grb_dic["skymap"] = gcn.get_gbmskymapupd(voevent)
        grb_dic["name"] = gcn.get_gbmname(voevent)#"GRB"+(grb_dic["skymap"]['localization_name'].split("_bn")[1]).split(".fit")[0]
        grb_dic["ra"], grb_dic["dec"], grb_dic["error"], grb_dic["ra_formatted"], grb_dic["dec_formatted"] = gcn.get_swiftgbmlocalization(
            voevent)
        grb_dic["quicklook"] = gcn.get_gbmquicklook(voevent)
        grb_dic["sun_distance"]=gcn.get_gbmsundistance(voevent)
        grb_dic["moon_distance"]=gcn.get_gbmmoondistance(voevent)       
        grb_dic["moon_illum"]=gcn.get_gbmmoonillum(voevent)
        

    return grb_dic


def init_grb_dic():
    """
    Function to initiate a dictionary object to store info
    registered in the VO event
    :return: unfilled dictionary
    """

    grb_dic = {
        "Packet_Type": "",
        "Pkt_Ser_Num": "",
        "dateobs": "",
        "grbid": "",
        "ratesnr": 0.,
        "ratets": 0.,
        "imagesnr": 0.,
        "date_firstalert":"",
        "lc": "",
        "hratio": 0.,
        "longshort": False,
        "probGRB": -1.0,
        "defGRB": True,
        "locsnr": 0.,
        "locdur": 0.,
        "locref": "",
        "snr": 0.,
        "descriptsnr": "",
        "dur": "",
        "inst": "",
        "teles": "",
        "location": "",
        "descriptdur": "",
        "AlertType": "",  # Preliminary,Initial,Update,Retraction
        "ra": 0.0,
        "dec": 0.0,
        "ra_formatted": 0.0,
        "dec_formatted": 0.0,
        "error": -1.0,
								"GRANDMAscore": 0.0,
        "delaymin": -1.0,
        "delayhours": -1.0,
        "delaydays": -1.0,
        "delayhum": -1.0,
        "skymap": -1.0,
        "quicklook": "",
        "sun_distance":0.0,
        "moon_illum":0.0,
        "moon_distance":0.0,
        "name":""
    }
    return grb_dic


def update_skymap(grb_dic, output_dic, conf_dic):
    """
    Function to download skymap and fill the missing info in grb_dic
    This will also initiate gwemopt

    :param grb_dic: dictionary with GRB related infos
    :param output_dic: dictionary with output architecture
    :param conf_dic: dictionary with info for selection
    :return: parms, updated param dictionary after loading skymap
    :return: map_struct, internal gwemopt structure filled when loading skymap
    avoid to reload it later
    """

    # initiate gwemopt dictionary configuration
    # need to use absolute path
    dir_path = os.path.dirname(os.path.abspath(__file__)) + "/"
    

    grb_dic["skymap"]["nside"]=conf_dic["nside_flat"]
    params = utils_too.init_gwemopt_observation_plan(
        dir_path + conf_dic["config_gwemopt"])

    # include in it full path to sky map fits file in the dictionary
    params["skymap"] = grb_dic["skymap"]
            
    # include nside from the sky map file

    params["nside"] = grb_dic["skymap"]["nside"]

    # update it with GRB specific part
    params = update_gwemoptconfig(grb_dic, conf_dic, params)

    # close the fits file
    # hdul.close()

    print("Loading skymap...")
    # read map to compute error regions
    map_struct = gwemopt.utils.read_skymap(
        params, is3D=params["do3D"], map_struct=params['map_struct'])

    # error regions
    i = np.flipud(np.argsort(params['map_struct']['prob']))
    credible_levels = find_greedy_credible_levels(
        params['map_struct']['prob'][i])
    cr50 = np.round(np.sum(credible_levels <= 0.5) *
                    hp.nside2pixarea(grb_dic["skymap"]["nside"], degrees=True), 1)
    cr90 = np.round(np.sum(credible_levels <= 0.9) *
                    hp.nside2pixarea(grb_dic["skymap"]["nside"], degrees=True), 1)

    # sorted_credible_levels = np.cumsum(grb_dic["skymap"]["probdensity"][i])
    # credible_levels = np.empty_like(sorted_credible_levels)

    idx50 = map_struct["cumprob"] < 0.50
    cr50 = len(map_struct["cumprob"][idx50])
    idx90 = map_struct["cumprob"] < 0.90
    cr90 = len(map_struct["cumprob"][idx90])
    grb_dic["50cr"] = cr50
    grb_dic["90cr"] = cr90

    return params, map_struct


def update_gwemoptconfig(grb_dic, conf_dic, params):
    """
    Update parameters for GRB alert on gwemopt

    :param gw_dic:
    :param conf_dic:
    :param params: dictionary to be used to start gwemopt and that
    will be updated/completed
    :return: updated params dictionary
    """
			

    # For GRB, do false
    if grb_dic["teles"] == "FERMI":
        params["do3D"] = False
        # parsing skymap
        params["doDatabase"] = True
        params["dateobs"] = grb_dic["dateobs"]
        order = hp.nside2order(grb_dic["skymap"]["nside"])
        t = rasterize(grb_dic["skymap"]["skymap"], order)
        result = t['PROB']
        flat = hp.reorder(result, 'NESTED', 'RING')
        params['map_struct'] = {}
        params['map_struct']['prob'] = flat
    if params["do3D"]:
        params["DISTMEAN"] = grb_dic["skymap"]["distmu"]
        params["DISTSTD"] = grb_dic["skymap"]["distsigma"]

        # Use galaxies to compute the grade, both for tiling and galaxy
        # targeting, only when dist_mean + dist_std < 400Mpc
        if params["DISTMEAN"]+params["DISTSTD"] <= conf_dic["Dist_cut"]:
            params["doUseCatalog"] = True
            params["doCatalog"] = True
            params["writeCatalog"] = True

    return params


def compute_ground(grb_dic, conf_dic, output_dir, params, map_struct):
    """
    Compute observation plan for the different telescopes
    :param grb_dic: dictionary with VO event information
    :param conf_dic: dictionary with configuration info for running
    the process plan (list telescopes, ...)
    :param output_dir: output directory where to save the VO event with
    observation plan
    :param params: dictionary with paraemters needed to start gwemopt
    :param map_struct: structure filled by gwemopt when loading skymap
    :return: list of VO files created
    """

    # initialize list of vo file
    vo_tosend = []
    if (grb_dic["teles"] == "SWIFT") or grb_dic["Packet_Type"] == 110 or grb_dic["Packet_Type"] == 119 or grb_dic["Packet_Type"] == 111:
        for tel_name in enumerate(conf_dic["Tels_followup"]):
            common_dic = create_dictionary(
                grb_dic, conf_dic["Experiment"], tel_name[1], 'Follow-up')
            voevent = create_grb_voevent(grb_dic, conf_dic, common_dic, "")
            file_voevent = utils_too.voevent_name(common_dic)
            # dump voevent
            with open(output_dir + "/" + file_voevent, 'wb') as fileo:
                vp.dump(voevent, fileo)
            vo_tosend.append(output_dir + "/" + file_voevent)

    if grb_dic["Packet_Type"] == 112 or grb_dic["Packet_Type"] == 115:
        # retrieve number of tiles we will use for the large field of view
        # telescope ie where will not target specific galaxy only
        params["max_nb_tiles"] = np.array(conf_dic["Tels_tiling_tilesnb"])

        # Adapt percentage of golden tiles with the 90% skymap size.
        # Arbitrary, needs to be optimised!!!
        if float(grb_dic["90cr"]) < 60:
            params["iterativeOverlap"] = 0.8
            params["doIterativeTiling"] = False
            params["doPerturbativeTiling"] = False
        else:
            params["iterativeOverlap"] = 0.2
        # move the two next lines to True if you want to optimize on the
            # network
            params["doIterativeTiling"] = False
            params["doPerturbativeTiling"] = False

        # Perform observation plan based on gwemopt package here using tiling for large FOV telescopes
        # output will be 2 tables : tiling and and list of associated galaxies
        atables_tiling = too.Observation_plan_multiple(
            conf_dic["Tels_tiling"], grb_dic["dateobs"], grb_dic["grbid"], params, map_struct, 'Tiling')

        # still need to prepare output files
        # loop on the different telescopes to create voevent associated to
        # observation plans
        for tel_name in enumerate(conf_dic["Tels_tiling"]):
            common_dic = create_dictionary(
                grb_dic, conf_dic["Experiment"], tel_name[1], 'Tiling')
            voevent = create_grb_voevent(
                grb_dic, conf_dic, common_dic, atables_tiling[0][tel_name[1]])
            # need to create correct file name and save it
            file_voevent = utils_too.voevent_name(common_dic)

        # dump voevent
        with open(output_dir + "/" + file_voevent, 'wb') as fileo:
            vp.dump(voevent, fileo)

        vo_tosend.append(output_dir + "/" + file_voevent)
    return vo_tosend



#missing galaxies option
"""

								# Perform observation plan based on gwemopt package here using galaxies for small FOV telescopes
								# output will be 2 tables : tiling (empty here) and list of associated galaxies
								params["galaxy_grade"] = 'S'
								params["max_nb_tiles"] = conf_dic["nb_galaxies"]
								atables_galaxy = too.Observation_plan_multiple(
												conf_dic["Tels_galaxy"], gw_dic["Time"], gw_dic["GraceID"], params, map_struct,
												'Galaxy targeting')

								# now save VO event for telescopes with galaxies
								for tel_name in enumerate(conf_dic["Tels_galaxy"]):
												# in case of CWB trigger the tables will be of type None
												# but we would like to create empty VO event file nevertheless
												# so here we are forcing the obs mode argument to be hard coded
												common_dic = create_dictionary(gw_dic, conf_dic["Experiment"], tel_name[1],
																                           'Galaxy')
												voevent = create_gw_voevent(
												    gw_dic, conf_dic, common_dic, atables_galaxy[1])
												# need to create correct file name and save it
												file_voevent = utils_too.voevent_name(common_dic)

												# dump voevent
												with open(output_dir + "/" + file_voevent, 'wb') as fileo:
																vp.dump(voevent, fileo)

												vo_tosend.append(output_dir + "/" + file_voevent)
"""


def basic_grb_voevent(grb_dic, conf_vo, what_com):
    """
    Create basis for VO event for a given telescope in the network

    :param gw_dic: dictionary with GRB infos
    :param conf_vo: dictionary to be used to fill header of VO event
    :param what_com: dictionary with common infos needed for VO creation
    :return: voevent object
    """

    # initialize stream id for
    conf_vo["streamid"] = \
        utils_too.define_streamid_vo(what_com['tel_name'], grb_dic["Pkt_Ser_Num"],
                                     conf_vo["Experiment"])

    # initialize the VO event object with basic structure
    voevent = utils_too.init_voevent(conf_vo, what_com)
    

    return voevent


def create_grb_voevent(grb_dic, conf_vo, what_com, atable):
	"""
	Create VO event with observation plan for a given telescope in the network

	:param gw_dic: dictionary with GRB infos
	:param conf_vo: dictionary to be used to fill header of VO event
	:param what_com: dictionary with common infos needed for VO creation
	:param atable: astropy table with observation plan and meta data
	:return: voevent object
	"""

	# get telescope name and role, will be used several time
	tel_name = what_com['tel_name']
	obs_mode = what_com['obs_mode']
	voevent = basic_grb_voevent(grb_dic, conf_vo, what_com)
	
	#GBM ground or final position
	if grb_dic["Packet_Type"] == 112 or grb_dic["Packet_Type"] == 115:
		pixloc = vp.Param(name="Loc_url", value=str(grb_dic["skymap"]["localization_name"]), ucd="meta.ref.url", dataType="string")
		pixloc.Description = "URL to retrieve location of the healpix skymap"
		voevent.What.append(pixloc)
		if tel_name != "":
			name_tel = vp.Param(name="Name_tel", value=str(tel_name),ucd="instr", dataType="string")
			name_tel.Description = "Name of the telescope used for the observation strategy"
			voevent.What.append(name_tel)
			if atable and atable[0]:


				# now prepare observation plan or galaxies list
				fields = objectify.Element("Table", name="Obs_plan")
				fields.Description = "Tiles for the observation plan"

				# first field is the identificaiton of the field
				# it will be different between tiling and galaxy targetting
				if obs_mode == "Tiling":
					field_id = objectify.SubElement(fields, "Field",
																																					name="Field_id", ucd="",
																																					unit="", dataType="string")
					field_id.Description = "ID of the field of FOV"
					tel_obsplan = np.transpose(np.array([atable['rank_id'],
																																										atable['RA'], atable['DEC'],
																																										atable['Prob']]))
				else:
					field_id = objectify.SubElement(fields, "Field", name="Gal_id",
																																															ucd="", unit="", dataType="string")
					field_id.Description = "ID of the galaxy"

					# For galaxie we will uniformize how we name them based on catalogs
					gal_id = utils_too.uniformize_galnames(atable)
					tel_obsplan = np.transpose(np.array([gal_id, atable['RAJ2000'], atable['DEJ2000'], atable['S']]))

					# right_asc = objectify.SubElement(fields, "Field", name="RA", ucd="pos.eq.ra ", unit="deg", dataType="float")

				right_asc = objectify.SubElement(
								fields, "Field", name="RA", ucd="pos.eq.ra ", unit="deg", dataType="float")
				right_asc.Description = "The right ascension at center of fov in equatorial coordinates"
				dec = objectify.SubElement(fields, "Field",
																															name="Dec", ucd="pos.eq.ra ", unit="deg", dataType="float")
				dec.Description = "The declination at center of fov in equatorial coordinates"
				os_grade = objectify.SubElement(fields, "Field",
																																				name="Os_score", ucd="meta.number", unit="None",
																																				dataType="float")
				os_grade.Description = "Gives the importance of the tile/galaxy to observe"
				data = objectify.SubElement(fields, "Data")

				# loop on the observation plan
				# put a limit to 500 fields otherwise get an error when sending a VO event

				for i in np.arange(min(500, len(tel_obsplan))):
								table_row = objectify.SubElement(data, "TR")
								for j in np.arange(len(tel_obsplan[i])):
												# objectify.SubElement(TR, "TD",value=str(Tel_dic["OS"][i][j]))
												objectify.SubElement(table_row, 'TD')
												table_row.TD[-1] = str(tel_obsplan[i][j])
			
				voevent.What.append(fields)
	grb_dic["dateobs"] = grb_dic["dateobs"].replace(tzinfo=pytz.utc)
	vp.add_where_when(voevent,
																			coords=vp.Position2D(ra=grb_dic["ra"], dec=grb_dic["dec"],
																																								err=grb_dic["error"], units='deg',
																																								system=vp.definitions.sky_coord_system.utc_fk5_geo),
																			obs_time=grb_dic["dateobs"], observatory_location=grb_dic["inst"])

	# Check everything is schema compliant:
	vp.assert_valid_as_v2_0(voevent)

	return voevent

def create_dictionary(grb_dic, experiment, tel, role):
    """
    Create standard dictionary to fill info for VO event and name the VO files
    :param grb_dic: dictionary with GRB information
    :param experiment: keyword for the experiment (defined in configuration file)
    :param tel: telescope name
    :param role: on-going role (test or observation), taken from VO input
    :return: dictionary with common infos
    """

    new_dic = {
        "event_type": "GRB",
        "trigger_id": grb_dic["grbid"],
        "name_id":grb_dic["name"],
        "event_status": grb_dic["AlertType"],
        "pkt_ser_num": grb_dic["Pkt_Ser_Num"],
        "inst": grb_dic["inst"],
        "teles": grb_dic["teles"],
        "tel_name": tel,
        "experiment": experiment,
        "GRANDMA score":grb_dic["GRANDMAscore"],
        "longshort":str(grb_dic["longshort"]),
        "hratio":float(grb_dic["hratio"]),
        "sun_distance":grb_dic["sun_distance"],
        "moon_distance":grb_dic["moon_distance"],
        "moon_illum":grb_dic["moon_illum"],
        "obs_mode": role
    }

    return new_dic
