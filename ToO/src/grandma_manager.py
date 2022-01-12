#!/usr/bin/env python

"""
Basic function to start ToO_MM processing


S. Antier and N. Leroy 2022

"""


import os
import json
import logging
import sys
import output_conf as outg
# import gw_too as gw function to be added
import grb_too as grb
import utils_too
import astropy.time
import requests

#compatible with skyportal gcn.py
import gcn

# configuration path,
PATH_CONFIG = os.path.dirname(os.path.abspath(__file__)) + '/../inputs/'


def slack_message(grb_dic, slack_config,conf_dic):
	"""
	:param grb_dic:
	:param slack_config:
	:return:
	"""

	# information to send in addition follow-up advocate, observability, vo events sent to XX
	#contain only informations for GRB
	
	
	#Messages GBM
	
	#3rd slack message messages
	
	
	#SWIFT message
	if grb_dic["Packet_Type"] == 61:
		alert_text = """
		  *GRB candidate:* From {:s}-{:s} \n -{:s} renamed as {} \n -*{}* alert
		  """.format(grb_dic["teles"],grb_dic["inst"],grb_dic["grbid"],grb_dic["name"],grb_dic["AlertType"])
	  
		time_text = """
	  *Time*:\n -Trigger Time (T0): {} UTC\n -Time since T0: {:s}
	  """.format(str(astropy.time.Time(grb_dic["dateobs"], format="datetime").isot), grb_dic["delayhum"])
		 
		measurements_text = """
		 *Measurement:*\n- BAT GRB product: <https://gcn.gsfc.nasa.gov/notices_s/{}/BA/|link> \n- SPER: <https://https://www.swift.ac.uk/sper/{}/|link> \n- Burst Analyser: <https://www.swift.ac.uk/burst_analyser/{}/|link>
		  """.format(grb_dic["grbid"], grb_dic["grbid"], grb_dic["grbid"])
		  
		radec_text = """
		   *RA/Dec:*\n- [hours, deg]: {} {}\n- [deg, deg, err]: {:.5f} {:+.5f} {:+.5f}
		   """.format(grb_dic["ra_formatted"], grb_dic["dec_formatted"], float(grb_dic["ra"].text), float(grb_dic["dec"].text), float(grb_dic["error"].text))
		
		grandma_text = """
							*GRANDMA:*\n Received and treated by GRANDMA, Long vs short unknown \n-GRANDMA score: {:.0f}\n
							""".format(grb_dic["GRANDMAscore"])

		blocks = [
					{
						   "type": "section",
						   "fields": [
						       {
						           "type": "mrkdwn",
						           "text": alert_text
						       },
						   ]
					},
					{
						   "type": "section",
						   "fields": [
						       {
						           "type": "mrkdwn",
						           "text": time_text
						       },
						       
						       {
						           "type": "mrkdwn",
						           "text": radec_text
						       },
						       
						       {
						           "type": "mrkdwn",
						           "text": measurements_text
						       },
						       
						       {
						           "type": "mrkdwn",
						           "text": grandma_text
						       },
						   ]
					},
		]
		
		
	if grb_dic["Packet_Type"] == 67:
		alert_text = """
		  *GRB candidate:* From {:s}-{:s} \n -{:s} renamed as {} \n -*{}* alert
		  """.format(grb_dic["teles"],grb_dic["inst"],grb_dic["grbid"],grb_dic["name"],grb_dic["AlertType"])
	  
		time_text = """
	  *Time*:\n -Trigger Time (T0): {} UTC\n -Time since T0: {:s}
	  """.format(str(astropy.time.Time(grb_dic["dateobs"], format="datetime").isot), grb_dic["delayhum"])
		 
		measurements_text = """
		 *Measurement:*\n- BAT GRB product: <https://gcn.gsfc.nasa.gov/notices_s/{}/BA/|link> \n- SPER: <https://https://www.swift.ac.uk/sper/{}/|link> \n- Burst Analyser: <https://www.swift.ac.uk/burst_analyser/{}/|link>
		  """.format(grb_dic["grbid"], grb_dic["grbid"], grb_dic["grbid"])
		  
		radec_text = """
		   *RA/Dec:*\n- [hours, deg]: {} {}\n- [deg, deg, err]: {:.5f} {:+.5f} {:+.5f}
		   """.format(grb_dic["ra_formatted"], grb_dic["dec_formatted"], float(grb_dic["ra"].text), float(grb_dic["dec"].text), float(grb_dic["error"].text))
		
		grandma_text = """
							*GRANDMA:*\n Received and treated by GRANDMA, Long vs short unknown \n-GRANDMA score: {:.0f}\n
							""".format(grb_dic["GRANDMAscore"])

		blocks = [
					{
						   "type": "section",
						   "fields": [
						       {
						           "type": "mrkdwn",
						           "text": alert_text
						       },
						   ]
					},
					{
						   "type": "section",
						   "fields": [
						       {
						           "type": "mrkdwn",
						           "text": time_text
						       },
						       
						       {
						           "type": "mrkdwn",
						           "text": radec_text
						       },
						       
						       {
						           "type": "mrkdwn",
						           "text": measurements_text
						       },
						       
						       {
						           "type": "mrkdwn",
						           "text": grandma_text
						       },
						   ]
					},
		]
	
	#GBM message
	if grb_dic["Packet_Type"] == 110:
			alert_text = """
		  *GRB candidate:* From {:s}-{:s} \n -{:s}\n -*{}* alert
		  """.format(grb_dic["teles"],grb_dic["inst"],grb_dic["name"],grb_dic["AlertType"])

			grandma_text="""*GRANDMA*: Received and treated by GRANDMA"""
			
			time_text = """
	  *Time*:\n -Trigger Time (T0): {} UTC\n -Time since T0: {:s}
	  """.format(str(astropy.time.Time(grb_dic["dateobs"], format="datetime").isot), grb_dic["delayhum"])

			blocks = [
						{
							   "type": "section",
							   "fields": [
							       {
							           "type": "mrkdwn",
							           "text": alert_text
							       },
							   ]
						},
						{
							   "type": "section",
							   "fields": [
							   
							   				{
							           "type": "mrkdwn",
							           "text": time_text
							       },
							       {
							           "type": "mrkdwn",
							           "text": grandma_text
							       },
							   ]
						},
			]
			
	if grb_dic["Packet_Type"] == 115:
			alert_text = """
		  *GRB candidate:* From {:s}-{:s} \n -{:s}\n -*{}* alert
		  """.format(grb_dic["teles"],grb_dic["inst"],grb_dic["name"],grb_dic["AlertType"])
			
		
			

			measurements_text = """
		  *Measurement:*\n-Class:{}  \n -Hardness:{}  \n -GBM quicklook: <{}|link>
		  """.format(str(grb_dic["longshort"]),str(grb_dic["hratio"]),grb_dic["quicklook"])

			time_text = """
	  *Time*:\n -Trigger Time (T0): {} UTC\n -Time since T0: {:s}
	  """.format(str(astropy.time.Time(grb_dic["dateobs"], format="datetime").isot), grb_dic["delayhum"])
			
			grandma_text="""*GRANDMA*: Alert sent to {} \n-GRANDMA score: {:.0f}\n""".format(conf_dic["Tels_tiling"],grb_dic["GRANDMAscore"])

			blocks = [
						{
							   "type": "section",
							   "fields": [
							       {
							           "type": "mrkdwn",
							           "text": alert_text
							       },
							   ]
						},
						{
							   "type": "section",
							   "fields": [
							       
							       {
							           "type": "mrkdwn",
							           "text": time_text
							       },							   
							       {
							           "type": "mrkdwn",
							           "text": measurements_text
							       },								   				
							       {
							           "type": "mrkdwn",
							           "text": grandma_text
							       },
							   ]
						},
			]
	
	

	error_message = """
	{} is not defined as env variable
	if an alert has passed the filter,
	the message has not been sent to Slack
	"""
	
	
	if grb_dic["Packet_Type"] == 115 or grb_dic["Packet_Type"] == 110 or grb_dic["Packet_Type"] == 61 or grb_dic["Packet_Type"] == 67:
				requests.post(
								        slack_config["url_testGRANDMA"],
								        json={
								            'blocks': blocks,
								            'username': 'ToO-GRANDMA'
								        },
								        headers={'Content-Type': 'application/json'},
							    )

def online_processing(voevent, server_config_path, role_filter):
    """

    :param voevent: VO event bin object
    :param server_config_path: json path to server configuration
    :param role_filter: allow to react on a specific role
    :return:
    """

    collab = ""

    dir_path = os.path.dirname(os.path.abspath(__file__))
    # dir_path = "/home/antier/GRANDMA/too-mm-master"

    # do we need to have it as argument of the function?
    # could be interesting for reprocessing
    with open(server_config_path) as filename:
        server_config = json.load(filename)

    # now create structure for output file from configuration


    output_dic = outg.define_output_config(server_config["output_dir"], voevent.attrib['role'])



    # find selection definition
    with open(dir_path + '/' + server_config["selection"]) as filename:
        conf_dic = json.load(filename)

    conf_dic.update(server_config)
    conf_dic = utils_too.update_voivorn(conf_dic)

    if voevent.attrib['role'] != role_filter:
        print ("sd voevent.attrib DIFF role_filter RETURN")
        # returnvoevent

    conf_dic["role"] = voevent.attrib['role']

    try:
        collab = str(gcn.get_howdes(voevent))
    except AttributeError:
        contact = str(gcn.get_contact(voevent))
        if "LIGO" in contact.split():
            collab = "gravitational"

    # is it a test alert or a real trigger and send via slack
    # Get config params from json file
    # NEED to check of we keep it or put it in server_config

    with open(PATH_CONFIG + 'slack.json') as filename:
        slack_config = json.load(filename)

    text_m = ""
    event_dic=""
       
    if "Fermi" or "Swift" in collab.split():
    				print (" ++++++++++== = fermi/swift.grb_trigger  ++++++++++++++= ")
    				text_m, event_dic = grb.grb_trigger(voevent, output_dic, conf_dic)

    if "gravitational" in collab.split():
        print (" ++++++++++== = gw.gw_trigger  ++++++++++++++= ")
        text_m, event_dic = gw.gw_trigger(voevent, output_dic, conf_dic)

    # put message in Slack
    if slack_config["slack_output"]:
    				slack_message(event_dic, slack_config,conf_dic)
