"""
Functions to create and manage the output architecure
where files will be saved (skymap, voevent, json) ...

Author N. Leroy 2020
Modified S. Antier 2022
"""

import json
import os


def create_basic_conf():
    """
    Create default dictionary with the needed elements
    :return: dictionary with elements to create and store
    output files
    """

    output_conf = {
        "output_dir": "",
        "skymappath": "",
        "vopath": "",
        "evt_type": "",
        "type_ser": "",
        "trigid": ""
    }

    return output_conf


def define_output_config(dirname, role):


	"""
	Function to create configuration dictionary to setup the ouput directory architecture
	:param role: define the type of the alert, used to create subdirectory if needed
	:param dirname: top directory where to stored output file

	the required field in the json are output_dir, skymappath and vopath
	We could add check with json scheme !!!!
	"""

	# create empty dictionary that will be used to create output architecture
	output_config = create_basic_conf()

	# top directory is defined by input argument
	output_config["output_dir"] = dirname

	# if we face test example create a new subtree
	if role == 'test':
		   output_config["output_dir"] = output_config["output_dir"] + '/test/'

	return output_config


def update_output_config(outputdic, gcn_dic, evttype):
    """
    Modify the output configuration before creating the directory architecture
    :param outputdic: dictionnary with the output architecture
    :param gw_dic:
    :param evttype: type of the event to store in different places (GW, neutrinos, ...)
    :return:
    """

    # Be careful !!!! this will not work for other type of alerts !!!!
    # need to think on this later !!!!!!
    outputdic["evt_type"] = evttype
    outputdic["type_ser"] = gcn_dic["AlertType"] + "_" + str(gcn_dic["Pkt_Ser_Num"])
    if evttype=="GW":
						 outputdic["trigid"] = gcn_dic["GraceID"]
    if (evttype=="GRB") and gcn_dic["teles"]=="FERMI"		:
						 outputdic["trigid"] = gcn_dic["name"]					 

    if (evttype=="GRB") and gcn_dic["teles"]=="SWIFT"		:
						 outputdic["trigid"] = gcn_dic["grbid"]		

def set_up_scoreGRB(outputdic):

	#create json that contains info for GRANDMA score
	score_json = {
	"hratio": 0.0,
	"score": 0.0,
	"snr": 0.0,
	"date_firstalert":"",
	"sun_distance":0.0,
	"moon_illum":0.0,
	"moon_distance":0.0,
	"longshort":False,
	"defGRB":False,
	"name":""
	}
	json_name = outputdic["output_dir"]  + outputdic["evt_type"] +"/"  + outputdic["trigid"] +"/"+ outputdic["trigid"]+ "_GRANDMAscore.json"
	with open(json_name, "w") as write_file:
		json.dump(score_json, write_file)

def update_scoreGRB(outputdic, grb_dic,param):
			json_name = outputdic["output_dir"] + outputdic["evt_type"] +"/"  + outputdic["trigid"] +"/"+ outputdic["trigid"]+ "_GRANDMAscore.json"
			jsonFile = open(json_name, "r") # Open the JSON file for reading
			data = json.load(jsonFile) # Read the JSON into the buffer
			jsonFile.close() # Close the JSON file

			## Working with buffered content
			data[param] = grb_dic[param]

			## Save our changes to JSON file
			with open(json_name, "w") as write_file:
				json.dump(data, write_file)
			
def load_scoreGRB(outputdic,param):
			json_name = outputdic["output_dir"] + outputdic["evt_type"] + '/'  + outputdic["trigid"] +"/"+ outputdic["trigid"]+ "_GRANDMAscore.json"
			jsonFile = open(json_name, "r") # Open the JSON file for reading
			data = json.load(jsonFile) # Read the JSON into the buffer
			jsonFile.close() # Close the JSON file

			return data[param]
			

def setup_outputdir_event(outputdic):
    """
    Setup the structure of directories
    :param outputdic: dictionnary with the output architecture
    """

    # create the main directory to store all the infos related to the event
    dir_evt_name = outputdic["output_dir"] + '/' + outputdic["evt_type"] + '/' + outputdic["trigid"]
    #dir_evt_name = outputdic["output_dir"] + '/' + outputdic["evt_type"] + '/' + outputdic["trigid"]
    json_name = outputdic["output_dir"]  + outputdic["evt_type"] +"/"  + outputdic["trigid"] +"/"+ outputdic["trigid"]+ "_GRANDMAscore.json"
    if not os.path.exists(dir_evt_name):
        os.makedirs(dir_evt_name)
    if not os.path.exists(json_name):
								set_up_scoreGRB(outputdic)
								
    # create the subdir for the given message received
    dir_ser_name = dir_evt_name + '/' + outputdic["type_ser"] + '/'
    if not os.path.exists(dir_ser_name):
        os.makedirs(dir_ser_name)

    outputdic["skymappath"] = dir_ser_name
    outputdic["vopath"] = dir_ser_name


def send_voevent(broker_config, filename_vo):
    """
    Function used to send VOEvent using comet software
    :param broker_config: json file with broker structure
    :param filename_vo: VO event filename to be sent with comet
    :return:
    """

    # open the json config file
    with open(broker_config) as filein:
        broker_config = json.load(filein)

    # prepare the command with configuration available in json
    # previsouly loaded
    cmd = "%s --host=%s --port=%s -f " % (broker_config['path'],
                                          broker_config['host'],
                                          broker_config['port'])
    # and add file absolute path
    cmd = cmd + filename_vo

    # send file
    os.system(cmd)
