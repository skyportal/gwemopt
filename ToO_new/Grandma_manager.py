#!/usr/bin/env python

import os
import json
import voeventparse as vp
import datetime
import numpy as np

#import GRANDMA_FAshifts as fa
from VOEventLib.Vutil import *
import utils_VO as vo

sys.path.append("./utils_DB/")
from astropy.table import Table
from send2database import *


import gwemopt.utils

from utils_VO import Tel_dicf

import grandma_outputconf as outg
import GW_ToO as gw
import Fermi_ToO as fermi
import Swift_ToO as swift
import ToO_manager as too

#to be use for test offline as the soft need to download update on timing (IERS-A table)
#from astropy.utils.iers import conf
#conf.auto_max_age = None

path_config = 'config/'
slack_output = True

#def search_ba():
#    """
#
#    :return:
#    """
#    fa_duty = fa.FA_shift()
#    return fa_duty


def GW_trigger(v, output_dic, Db_use, fa=""):
    toplevel_params = vp.get_toplevel_params(v)
    # READ MESSAGE
    trigger_id = toplevel_params['GraceID']['value']
    Pktser = str(int(toplevel_params['Pkt_Ser_Num']['value']) - 1)
    testRetract = int(toplevel_params['Packet_Type']['value'])

    if (testRetract == 164):
        Retraction = 1
    else:
        Retraction = 0


    if (Retraction == 1):
        message_type = "GW RETRACTION POSITION MESSAGE"
        name_dic = "GW" + trigger_id
        lalid = gw.name_lalid(name_dic, vo.letters[int(Pktser) - 1], "_DB")

        GW_vo,GW_dic = gw.GW_trigger_retracted(v, output_dic, lalid, path_config)

        # NOT SURE WHETHER THIS IS USEFUL
        if Db_use:
            # Send information contained in VOEvent to database
            send_VOE_alert_to_DB(v, isfile=False, path_skymap="%s" % "",
                                 filename_skymap="", proba=90)

        for telescope in LISTE_TELESCOPE:
            Tel_dic = Tel_dicf()
            Tel_dic["Name"] = telescope

            name_dic = "GW" + trigger_id
            lalid = gw.name_lalid(name_dic, GW_vo["letup"], "_" + Tel_dic["Name"])
            filename_vo = vo.create_GRANDMAvoevent(lalid, GW_dic, GW_vo, Tel_dic, output_dic)
            outg.send_voevent(path_config + '/broker.json', filename_vo)

        text_mes = str("---------- \n") + str("GW alert \n") + str("---------- \n") + str("GW NAME : ") \
               + str(GW_vo["trigid"]) + (" ") + str("Trigger Time: ") + GW_vo["trigtime"].strftime("%d. %B %Y %I:%M%p") + "\n" + \
               str("WARNING RETRACTION") + str("\n") + str("---Follow-up Advocate--\n") + str(
            "Follow-up advocate on duty: ") + str(fa) + "\n"

        return text_mes

    AlertType = toplevel_params['AlertType']['value']

    Lockey=v.find(".//Param[@name='skymap_fits']")
    if Lockey != None:
        locpix = str(v.find(".//Param[@name='skymap_fits']").attrib['value'])
    else:
        locpix=""

    if (locpix != "") and (AlertType == "Preliminary"):
        AlertType = "Initial"

    vo_dic={"trigid":trigger_id,"eventstatus":AlertType,"iter_statut":Pktser}

    outg.update_output_config(output_dic, vo_dic,"GW")
    outg.setup_cloud_event(output_dic)

    GW_vo, GW_dic = gw.init_GW_treatment_alert(v, path_config)

    #need to add test to check the nature of the trigger
    gw.cbc_GW_event(v,GW_vo,GW_dic)

    skypath = gw.update_skymap(GW_vo, GW_dic,output_dic)
    skymap_folder = output_dic["skymappath"]

    name_dic = "GW" + GW_vo["trigid"]
    lalid = gw.name_lalid(name_dic, GW_vo["letup"], "_DB")

    file_voevent = vo.create_GRANDMAvoevent(lalid, GW_dic, GW_vo, "", output_dic)
    if Db_use:
        # Send information contained in VOEvent to database
        try:
            send_VOE_alert_to_DB(v, isfile=False, path_skymap="%s" % skymap_folder, filename_skymap=skypath.split('/')[-1], proba=90)
            #  If failed, load the xml file. Might only do this in future.
        except:
            send_VOE_alert_to_DB(file_voevent, isfile=True, path_skymap="%s" % skymap_folder,
                                 filename_skymap=skypath.split('/')[-1], proba=90)

    #gw.GW_treatment_alert(GW_vo, output_dic)
    grandma_GW_treatment(v, GW_vo, GW_dic, name_dic, output_dic, skypath, Db_use)

    text_mes = str("---------- \n") + str("GW alert \n") + str("---------- \n") + str("GW NAME : ") \
           + str(GW_vo["trigid"]) + (" ") + str("Trigger Time: ") + GW_vo["trigtime"].strftime("%d. %B %Y %I:%M%p") + "\n" + \
           str("Instruments: ") + str(str(GW_vo["inst"])) + str("\n") \
           + str("EventPage: ") + str(str(GW_dic["EventPage"])) + str("\n") + str("Search: ") + str(
        str(GW_dic["Group"])) + str("\n") + str("HasRemnant: ") + str(GW_dic["HasRemnant"]) + str("\n") \
           + str("Delay since alert (minutes): ") + str(GW_vo["trigdelay"]) + ("\n") + str("\n") + str("---Follow-up Advocate--\n") + str(
        "Follow-up advocate on duty: ") + str(fa) + "\n" "\n"

    return text_mes

def grandma_GW_treatment(v, GW_vo, GW_dic, name_dic, output_dic, skypath, Db_use):
        #  Load params dictionary for gwemopt
        params = too.init_observation_plan(skypath)
        print("Loading skymap...")
        # Function to read maps
        map_struct = gwemopt.utils.read_skymap(params, is3D=params["do3D"])
        idx50 = map_struct["cumprob"] < 0.50
        cr50 = len(map_struct["cumprob"][idx50])
        idx90 = map_struct["cumprob"] < 0.90
        cr90 = len(map_struct["cumprob"][idx90])
        GW_dic["50cr"] = "{:.2f}".format(map_struct["pixarea_deg2"] * cr50)
        GW_dic["90cr"] = "{:.2f}".format(map_struct["pixarea_deg2"] * cr90)

        if GW_vo["voimportance"] == 1:
            LISTE_TELESCOPE_TILING = ["OAJ", "TRE", 'TCH', 'TCA']
            max_nb_tiles_tiling = np.array([20, 20, 20, 20])
            # max_nb_tiles_tiling = -1 * np.ones(len(LISTE_TELESCOPE_TILING))
            LISTE_TELESCOPE_GALAXY = ["Makes-60", "Lisnyky-AZT8", "Zadko", "TNT", "UBAI-T60N", "ShAO-T60", "Abastunami-T70", "UBAI-T60S", "Abastunami-T48", "IRIS"]
            max_nb_tiles_galaxy = np.array([50] * len(LISTE_TELESCOPE_GALAXY))
            # max_nb_tiles_galaxy = -1 * np.ones(len(LISTE_TELESCOPE_GALAXY))
        else:
            LISTE_TELESCOPE_TILING = ["TRE", "TCH", "TCA"]
            max_nb_tiles_tiling = np.array([20, 20, 20])
            # max_nb_tiles_tiling = -1 * np.ones(len(LISTE_TELESCOPE_TILING))
            LISTE_TELESCOPE_GALAXY = ["Makes-60", "Lisnyky-AZT8", "Zadko", "TNT", "UBAI-T60N", "ShAO-T60", "Abastunami-T70", "UBAI-T60S", "Abastunami-T48", "IRIS"]
            max_nb_tiles_galaxy = np.array([50] * len(LISTE_TELESCOPE_GALAXY))
            # max_nb_tiles_galaxy = -1 * np.ones(len(LISTE_TELESCOPE_GALAXY))

        message_obs = ""

        ### TILING ###
        params["max_nb_tiles"] = max_nb_tiles_tiling
        #  Adapt percentage of golden tiles with the 90% skymap size. Arbitrary, needs to be optimised!!!
        if float(GW_dic["90cr"]) < 60:
            params["iterativeOverlap"] = 0.8
            params["doIterativeTiling"] = False
            params["doPerturbativeTiling"] = False
        else:
            params["iterativeOverlap"] = 0.2
            params["doIterativeTiling"] = True
            params["doPerturbativeTiling"] = True
        print ('CR90: %.2f deg2   /  CR50: %.2f deg2' % (float(GW_dic["90cr"]), float(GW_dic["50cr"])))
        print ('ITERATIVE OVERLAP FRACTION: ', params["iterativeOverlap"])
        # params["galaxy_grade"] = 'Sloc'

        aTables_tiling, galaxies_table = too.Observation_plan_multiple(LISTE_TELESCOPE_TILING, GW_vo, GW_vo["trigid"], params, map_struct, 'Tiling')
        #  Send data to DB and send xml files to telescopes through broker for tiling
        send_data(LISTE_TELESCOPE_TILING, params, aTables_tiling, galaxies_table, GW_vo, GW_dic, GW_vo["trigid"], v, path_config, output_dic, message_obs, name_dic, Db_use=Db_use, gal2DB=False)

        ### Galaxy targeting ###
        # if the  mean distance(+error) of the skymap is less than 300Mpc we perform galaxy targeting
        if params["DISTMEAN"] + params["DISTSTD"] <= params['distance_useGal']:
            # params["galaxy_grade"] = 'Sloc'
            params["max_nb_tiles"] = max_nb_tiles_galaxy
            aTables_galaxy, galaxies_table = too.Observation_plan_multiple(LISTE_TELESCOPE_GALAXY, GW_vo,
                                                                           GW_vo["trigid"],
                                                                           params, map_struct, 'Galaxy targeting')
            #  Send data to DB and send xml files to telescopes through broker for galaxy targeting
            send_data(LISTE_TELESCOPE_GALAXY, params, aTables_galaxy, galaxies_table, GW_vo, GW_dic, GW_vo["trigid"], v, path_config, output_dic, message_obs, name_dic, Db_use=Db_use, gal2DB=True)

        return


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

        AlertType = toplevel_params['AlertType']['value']
        if "transient." and "found" in message_descrip:
            AlertType = "Preliminary"
        else:
            AlertType = "Initial"

        Pktser = str(int(toplevel_params['Pkt_Ser_Num']['value']) - 1)

        vo_dic = {"trigid": trigger_id, "eventstatus": AlertType, "iter_statut": Pktser}

        outg.update_output_config(output_dic, vo_dic, "GRB")
        outg.setup_cloud_event(output_dic)

        if "transient." and "found" in message_descrip:
            id_grb_message_init = v.attrib['ivorn']
            text_mes = fermi.fermi_trigger_found(v, output_dic, file_log_s)

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
            text_mes = fermi.fermi_trigger_follow(v, output_dic, message_type, file_log_s)

    return text_mes


def slack_message(text_mes, role, collab, slack_config = None):
    """

    :param slack_config:
    :param text_mes:
    :return:
    """
    if slack_config == None:
        with open(path_config + 'slack.json') as f:
            slack_config = json.load(f)

    if role == "test":
        cmd ="curl -X POST -H 'Content-type: application/json' --data '{\"text\":\" "+ text_mes+"\"}' %s" % slack_config['url_test']
        os.system(cmd)
    if role == "observation":
        if ("Swift" in collab.split()) or ("Fermi" in collab.split()):
            cmd ="curl -X POST -H 'Content-type: application/json' --data '{\"text\":\" "+ text_mes+"\"}' %s" % slack_config['url_swift']
            os.system(cmd)
        if "gravitational" in collab.split():
            cmd ="curl -X POST -H 'Content-type: application/json' --data '{\"text\":\" "+ text_mes+"\"}' %s" % slack_config['url_GW']
            os.system(cmd)



def online_processing(v, role_filter):
    #    v = vp.loads(playload)
    collab = ""
    text_mes = ""

    force_Db_use = False

    output_dic = outg.define_output_config(path_config + 'outputdirgrandma.json', v.attrib['role'])
    #for GRANDMA the json need also to contain : Obsrequest, OT, evt_type, trigid, type_ser

    if (v.attrib['role'] != role_filter):
        return

    if force_Db_use:
        Db_use = force_Db_use
    else:
        if v.attrib['role'] != "test":
            Db_use = True
        else:
            Db_use = False

#    Db_use = False
#    global slack_output
#    slack_output = False

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

    if slack_output:
        slack_message(text_m, v.attrib['role'], collab, slack_config)


    # which instrument comes from the alert Swift or Fermi ?
    if "Swift" in collab.split():
        text_mes = swift.swift_trigger(v, output_dic, collab)
    if "Fermi" in collab.split():
        text_mes = fermi_trigger(v, output_dic, collab)
    if "gravitational" in collab.split():
        text_mes = GW_trigger(v, output_dic, Db_use)

    #put message in Slack
    if slack_output:
        slack_message(text_mes, v.attrib['role'], collab, slack_config)

#    print(text_mes)



#LISTE_TELESCOPE = ["TRE", "TNT","OAJ","Abastunami-T70"]
#LISTE_TELESCOPE = ["OAJ","TRE","Abastunami-T70"]
LISTE_TELESCOPE = ["TRE","TCA","TCH","OAJ","Abastunami-T70","Zadko","TNT","Lisnyky-AZT8","ShAO-T60", "IRIS"]
LISTE_TELESCOPE = ["GWAC", "Zadko"]

