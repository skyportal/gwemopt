"""
Set of functions used in ToO processing in case of GW alerts
- decoding of VO event
- perform selection
- prepare observation plans associated to the alert
- prepare output files needed by the different instruments

Authors N. Leroy 2020
"""

import os
import datetime
import json
import jsonschema
import dateutil.parser
import numpy as np
import voeventparse as vp
import lxml.objectify as objectify
from astropy.io import fits
import gwemopt.utils
import gwemopt.ToO_manager as too
import output_conf as outg
import score
import utils_too


def gw_trigger(voevent, output_dic, conf_dic):
    """
    Function to make GW VO event analysis
    - decode the VO event and check the skymap
    - perform selection
    - perform observation plan
    - prepare output files for the different instruments

    :param voevent: xml object already parsed
    :param output_dic: dictionary with output architecture
    :param conf_dic: dictionary for server config and selection infos
    :return:
    """

    # initiate and fill with common part the gw dictionary object
    gw_dic = decode_common_voevent(voevent)

    # setup output architecture
    outg.update_output_config(output_dic, gw_dic, "GW")
    outg.setup_outputdir_event(output_dic)

    # check if this is a retraction
    # given by Packet_Type (hard coded in GCN production)
    # need also to update DB and stop all observations
    # or computing plan
    if gw_dic["Packet_Type"] == 164:
        voevent_list = gw_retraction(gw_dic, conf_dic, output_dic)
        utils_too.send_voevent(conf_dic, voevent_list)
        return "Retracted"

    # check if it is a CBC type event
    if gw_dic["Group"] in ["CBC"]:
        gw_dic = decode_cbc_voevent(voevent, gw_dic)

    # now download the skymap (and initiate gwemopt)
    # retrieve the configuration for gwemopt
    params, map_struct = update_skymap(gw_dic, output_dic, conf_dic)

    # need to compute the delay in hours
    diff_time = (datetime.datetime.now(tz=datetime.timezone.utc) - gw_dic["Time"])
    delay = diff_time.total_seconds() / 3600

    # perform selection, this will return a score
    score_event = score.gw_score(gw_dic, conf_dic)

    # compute observation plans for all telescopes on ground
    # defined in the config file in not above delay
    # if delay < 0, by pass the test
    # then update DB nevertheless ?
    if delay < conf_dic["ground_delay"] or conf_dic["ground_delay"] < 0:
        print('Start ground observation plan')
        voevent_list = compute_ground(gw_dic, conf_dic, output_dic["vopath"], params, map_struct)
        utils_too.send_voevent(conf_dic, voevent_list)

    return "completed"


def decode_common_voevent(voevent):
    """
    Function to decode common part of VO event
    :param voevent: xml object already parsed
    :return: dictionary filled with info
    """

    # first init dictionary to store information
    gw_dic = init_gw_dic()

    # Fill the dictionary with the xml content
    # first retreive the parameters
    toplevel_params = vp.get_toplevel_params(voevent)

    # fill the info on the sequence
    gw_dic["Packet_Type"] = int(toplevel_params['Packet_Type']['value'])
    gw_dic["Pkt_Ser_Num"] = int(toplevel_params['Pkt_Ser_Num']['value'])
    gw_dic["AlertType"] = toplevel_params['AlertType']['value']

    # fill the info on the event
    gw_dic["GraceID"] = toplevel_params['GraceID']['value']
    gw_dic["HardwareInj"] = toplevel_params['HardwareInj']['value']
    gw_dic["EventPage"] = toplevel_params['EventPage']['value']

    # fill remaining info in case of a non retracted event
    if gw_dic["Packet_Type"] != 164:
        gw_dic["Instruments"] = toplevel_params['Instruments']['value']
        gw_dic["FAR"] = float(toplevel_params['FAR']['value'])
        gw_dic["Group"] = toplevel_params['Group']['value']
        gw_dic["Pipeline"] = toplevel_params['Pipeline']['value']
        # fill skymap info
        gw_dic["Skymap"] = str(voevent.
                               find(".//Param[@name='skymap_fits']").
                               attrib['value'])

    # fill time info
    isotime = voevent.WhereWhen.ObsDataLocation.\
        ObservationLocation.AstroCoords.Time.TimeInstant.ISOTime.text

    # parse the iso time (string format) to a datetime object
    # add Z to the end of the ISO time string to specify UTC time zone
    gw_dic["Time"] = dateutil.parser.parse(isotime+"Z")

    return gw_dic


def init_gw_dic():
    """
    Function to initiate a dictionary object to store info
    registered in the VO event
    :return: unfilled dictionary
    """

    gw_dic = {
        "Packet_Type": 0,
        "Pkt_Ser_Num": 0,
        "GraceID": "",
        "AlertType": "",
        "HardwareInj": "",
        "EventPage": "",
        "Instruments": "",
        "FAR": 0.,
        "Group": "",
        "Pipeline": "",
        "Skymap": "",
        "HasNS": 0,
        "HasRemnant": 0,
        "BNS": 0,
        "NSBH": 0,
        "BBH": 0,
        "MassGap": 0,
        "Terrestrial": 0,
        "Time": "",
        "lum": 0,
        "errlum": 0,
        "90cr": 0,
        "50cr": 0
    }
    return gw_dic


def gw_retraction(gw_dic, conf_dic, output_dic):
    """
    Function to deal with retraction alert
    :param gw_dic: dictionary with GW related infos
    :param conf_dic: dictionary with configuration infos
    :param output_dic: dictionary with output architecture
    :return: list of VO event files to be send
    """

    # we need to stop all processing of still on-going
    # send update to DB -> this will also stop ToO process
    # on space segment; to be checked in interface
    # they can do it also on their side

    # stop processing
    # to be done

    # send infos to DB
    # to be done

    tels_ground = conf_dic["Tels_tiling"] + conf_dic["Tels_galaxy"]
    print(tels_ground)

    # initialize list of vo file
    vo_tosend = []

    # prepare output files
    for tel in tels_ground:
        print('Retraction done for candidate ' + gw_dic['GraceID'] + 'and tel ' + tel)
        common_dic = create_dictionary(gw_dic, conf_dic["Experiment"], tel, "")
        # utils_too.prepare_voe_retraction(tel, voevent, output_dic)
        voevent = basic_gw_voevent(gw_dic, conf_dic, common_dic)
        # need to create correct file name and save it
        file_voevent = utils_too.voevent_name(common_dic)

        # dump voevent
        with open(output_dic["vopath"] + "/" + file_voevent, 'wb') as fileo:
            vp.dump(voevent, fileo)

        vo_tosend.append(output_dic["vopath"] + "/" + file_voevent)

    return vo_tosend


def decode_cbc_voevent(voevent, gw_dic):
    """
    Funciton to decode the specific part of CBC VO event
    :param voevent: xml object already parsed
    :param gw_dic: dictionary with GW related infos
    :return: dictionary filled with cbc info
    """

    # info on possible EM counterpart
    gw_dic["HasRemnant"] = \
        float(voevent.find(".//Param[@name='HasRemnant']").attrib['value'])
    gw_dic["HasNS"] = \
        float(voevent.find(".//Param[@name='HasNS']").attrib['value'])

    # info on classification
    gw_dic["BNS"] = \
        float(voevent.find(".//Param[@name='BNS']").attrib['value'])
    gw_dic["NSBH"] = \
        float(voevent.find(".//Param[@name='NSBH']").attrib['value'])
    gw_dic["BBH"] = \
        float(voevent.find(".//Param[@name='BBH']").attrib['value'])
    gw_dic["Terrestrial"] = \
        float(voevent.find(".//Param[@name='Terrestrial']").attrib['value'])
    gw_dic["MassGap"] = \
        float(voevent.find(".//Param[@name='MassGap']").attrib['value'])

    return gw_dic


def update_skymap(gw_dic, output_dic, conf_dic):
    """
    Function to download skymap and fill the missing info in gw_dic
    This will also initiate gwemopt

    :param gw_dic: dictionary with GW related infos
    :param output_dic: dictionary with output architecture
    :param conf_dic: dictionary with info for selection
    :return: parms, updated param dictionary after loading skymap
    :return: map_struct, internal gwemopt structure filled when loading skymap
    avoid to reload it later
    """

    # download the sky map from LKV database (gracedb)
    # if does not exists already
    skypath = output_dic["skymappath"] + "/" \
              + str(gw_dic["Skymap"].split("/")[-1])

    if not os.path.isfile(skypath):
        command = "curl " + " -o " + skypath + " -O " + gw_dic["Skymap"]
        os.system(command)

    # open skymap file to get the header
    hdul = fits.open(skypath)

    if gw_dic["Pipeline"] not in ["CWB"]:
        # retrieve the distance and associated error
        lumin = np.round(hdul[1].header['DISTMEAN'], 3)
        errorlumin = np.round(hdul[1].header['DISTSTD'], 3)
    else:
        # if not CBC like event, put default values
        lumin = -1
        errorlumin = -1

    # include distance info in gw dictionary
    gw_dic["lum"] = lumin
    gw_dic["errlum"] = errorlumin

    # initiate gwemopt dictionary configuration
    # need to use absolute path
    dir_path = os.path.dirname(os.path.abspath(__file__)) + "/"
    params = utils_too.init_gwemopt_observation_plan(dir_path + conf_dic["config_gwemopt"])
    # include in it full path to sky map fits file in the dictionary
    params["skymap"] = skypath
    # include nside from the sky map file
    params["nside"] = hdul[1].header['NSIDE']
    # update it with GW specific part
    params = update_gwemoptconfig(gw_dic, conf_dic, params)

    # close the fits file
    hdul.close()

    print("Loading skymap...")
    # read map to compute error regions
    map_struct = gwemopt.utils.read_skymap(params, is3D=params["do3D"])
    idx50 = map_struct["cumprob"] < 0.50
    cr50 = len(map_struct["cumprob"][idx50])
    idx90 = map_struct["cumprob"] < 0.90
    cr90 = len(map_struct["cumprob"][idx90])
    gw_dic["50cr"] = "{:.2f}".format(map_struct["pixarea_deg2"] * cr50)
    gw_dic["90cr"] = "{:.2f}".format(map_struct["pixarea_deg2"] * cr90)

    return params, map_struct


def update_gwemoptconfig(gw_dic, conf_dic, params):
    """
    Update parameters for GW alert on gwemopt

    :param gw_dic:
    :param conf_dic:
    :param params: dictionary to be used to start gwemopt and that
    will be updated/completed
    :return: updated params dictionary
    """

    # if event is of CBC type we will have distace estimation
    # and then use 3D  galaxies info if possible

    if gw_dic["Pipeline"] not in ["CWB"]:
        params["do3D"] = True
    else:
        params["do3D"] = False

    if params["do3D"]:
        params["DISTMEAN"] = gw_dic['lum']
        params["DISTSTD"] = gw_dic['errlum']

        # Use galaxies to compute the grade, both for tiling and galaxy
        # targeting, only when dist_mean + dist_std < 400Mpc
        if params["DISTMEAN"]+params["DISTSTD"] <= conf_dic["Dist_cut"]:
            params["doUseCatalog"] = True
            params["doCatalog"] = True
            params["writeCatalog"] = True

    return params


def compute_ground(gw_dic, conf_dic, output_dir, params, map_struct):
    """
    Compute observation plan for the different telescopes
    :param gw_dic: dictionary with VO event information
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

    # retrieve number of tiles we will use for the large field of view
    # telescope ie where will not target specific galaxy only
    params["max_nb_tiles"] = np.array(conf_dic["Tels_tiling_tilesnb"])

    # Adapt percentage of golden tiles with the 90% skymap size.
    # Arbitrary, needs to be optimised!!!
    if float(gw_dic["90cr"]) < 60:
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
        conf_dic["Tels_tiling"], gw_dic["Time"], gw_dic["GraceID"], params, map_struct, 'Tiling')

    # still need to prepare output files
    # loop on the different telescopes to create voevent associated to
    # observation plans
    for tel_name in enumerate(conf_dic["Tels_tiling"]):
        common_dic = create_dictionary(gw_dic, conf_dic["Experiment"], tel_name[1], 'Tiling')
        voevent = create_gw_voevent(gw_dic, conf_dic, common_dic, atables_tiling[0][tel_name[1]])
        # need to create correct file name and save it
        file_voevent = utils_too.voevent_name(common_dic)

        # dump voevent
        with open(output_dir + "/" + file_voevent, 'wb') as fileo:
            vp.dump(voevent, fileo)

        vo_tosend.append(output_dir + "/" + file_voevent)

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
        voevent = create_gw_voevent(gw_dic, conf_dic, common_dic, atables_galaxy[1])
        # need to create correct file name and save it
        file_voevent = utils_too.voevent_name(common_dic)

        # dump voevent
        with open(output_dir + "/" + file_voevent, 'wb') as fileo:
            vp.dump(voevent, fileo)

        vo_tosend.append(output_dir + "/" + file_voevent)

    return vo_tosend


def basic_gw_voevent(gw_dic, conf_vo, what_com):
    """
    Create basis for VO event for a given telescope in the network

    :param gw_dic: dictionary with GW infos
    :param conf_vo: dictionary to be used to fill header of VO event
    :param what_com: dictionary with common infos needed for VO creation
    :return: voevent object
    """

    # initialize stream id for
    conf_vo["streamid"] = \
        utils_too.define_streamid_vo(what_com['tel_name'], gw_dic["Pkt_Ser_Num"],
                                     conf_vo["Experiment"])

    # initialize the VO event object with basic structure
    voevent = utils_too.init_voevent(conf_vo, what_com)

    return voevent


def create_gw_voevent(gw_dic, conf_vo, what_com, atable):
    """
    Create VO event with observation plan for a given telescope in the network

    :param gw_dic: dictionary with GW infos
    :param conf_vo: dictionary to be used to fill header of VO event
    :param what_com: dictionary with common infos needed for VO creation
    :param atable: astropy table with observation plan and meta data
    :return: voevent object
    """

    # get telescope name and role, will be used several time
    tel_name = what_com['tel_name']
    obs_mode = what_com['obs_mode']

    voevent = basic_gw_voevent(gw_dic, conf_vo, what_com)

    pixloc = vp.Param(name="Loc_url", value=str(gw_dic["Skymap"]),
                      ucd="meta.ref.url", dataType="string")

    pixloc.Description = "URL to retrieve location of the healpix skymap"
    voevent.What.append(pixloc)

    # add specific part for a GW event (nature, and physical properties)
    add_gw_voevent_content(voevent, gw_dic, obs_mode)

    if tel_name != "":
        name_tel = vp.Param(name="Name_tel", value=str(tel_name),
                            ucd="instr", dataType="string")
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
                tel_obsplan = np.transpose(np.array([gal_id, atable['RAJ2000'],
                                                     atable['DEJ2000'], atable['S']]))

            right_asc = objectify.SubElement(fields, "Field", name="RA", ucd="pos.eq.ra ",
                                             unit="deg", dataType="float")
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

    vp.add_where_when(voevent,
                      coords=vp.Position2D(ra=0., dec=0.,
                                           err=-1, units='deg',
                                           system=vp.definitions.sky_coord_system.utc_fk5_geo),
                      obs_time=gw_dic["Time"], observatory_location=gw_dic["Instruments"])

    # Check everything is schema compliant:
    vp.assert_valid_as_v2_0(voevent)

    return voevent


def add_gw_voevent_content(voevent, gw_dic, obsplan):
    """
    Add specific information for GW event nature and physical properties

    :param voevent: VO event object to be updated
    :param gw_dic: dictionary with GW information
    :param obsplan: type of Observation plan performed (tiling, galaxy)
    :return:
    """

    # Include FAR for the event
    proba = vp.Param(name="FAR", value=str(gw_dic["FAR"]),
                     ucd="arith.rate;stat.falsealarm", dataType="float")
    proba.Description = "False Alarm probability"
    voevent.What.append(proba)

    obs_plan = vp.Param(name="Obs_plan", value=obsplan, ucd="obs_plan", dataType="string")
    obs_plan.Description = "Observation plan done using either tiling or galaxy"
    voevent.What.append(obs_plan)

    eventpage = vp.Param(name="Quicklook_url", value=gw_dic["EventPage"],
                         ucd="meta.ref.url", dataType="string")
    eventpage.Description = "Web page for evolving status of this GW candidate"
    voevent.What.append(eventpage)

    lum = vp.Param(name="Distance", value=str(gw_dic["lum"]),
                   ucd="meta.number", dataType="float")
    lum.Description = "Luminosity distance (Mpc)"
    voevent.What.append(lum)

    errlum = vp.Param(name="Err_distance", value=str(gw_dic["errlum"]),
                      ucd="meta.number", dataType="float")
    errlum.Description = "Std for the luminosity distance (Mpc)"
    voevent.What.append(errlum)

    s50cr = vp.Param(name="50cr_skymap", value=str(gw_dic["50cr"]),
                     ucd="meta.number", dataType="float")
    s50cr.Description = "Sky localization area (50 pourcent confident region)"
    voevent.What.append(s50cr)

    s90cr = vp.Param(name="90cr_skymap", value=str(gw_dic["90cr"]),
                     ucd="meta.number", dataType="float")
    s90cr.Description = "Sky localization area (90 pourcent confident region)"
    voevent.What.append(s90cr)

    group = vp.Param(name="Group", value=gw_dic["Group"], ucd="meta.code", dataType="string")
    group.Description = "Data analysis working group"
    voevent.What.append(group)

    bns = vp.Param(name="BNS", value=str(gw_dic["BNS"]),
                   dataType="float", ucd="stat.probability")
    bns.Description = "Probability that the source is a binary neutron star merger"
    nsbh = vp.Param(name="NSBH", value=str(gw_dic["NSBH"]),
                    dataType="float", ucd="stat.probability")
    nsbh.Description = "Probability that the source is a neutron star - black hole merger"
    bbh = vp.Param(name="BBH", value=str(gw_dic["BBH"]),
                   dataType="float", ucd="stat.probability")
    bbh.Description = "Probability that the source is a binary black hole merger"
    massgap = vp.Param(name="MassGap", value=str(gw_dic["MassGap"]),
                       dataType="float", ucd="stat.probability")
    massgap.Description = "Probability that mass of one of the objects lies in 3-5 Ms"
    terrestrial = vp.Param(name="Terrestrial", value=str(gw_dic["Terrestrial"]),
                           dataType="float", ucd="stat.probability")
    terrestrial.Description = "Probability that the source is " \
                              "terrestrial (i.e., a background noise fluctuation or a glitch)"
    group_class = vp.Group(params=[bns, nsbh, bbh, terrestrial, massgap], name="Classification")
    group_class.Description = "Source classification: binary neutron star " \
                              "(BNS), neutron star-black hole (NSBH), " \
                              "binary black hole (BBH), or terrestrial (noise)"
    voevent.What.append(group_class)

    has_ns = vp.Param(name="HasNS", value=str(gw_dic["HasNS"]),
                      dataType="float", ucd="stat.probability")
    has_ns.Description = "Probability that at least one object " \
                         "in the binary has a mass that is less than 3 solar masses"
    has_remnant = vp.Param(name="HasRemnant", value=str(gw_dic["HasRemnant"]),
                           dataType="float", ucd="stat.probability")
    has_remnant.Description = "Probability that a nonzero mass was ejected " \
                              "outside the central remnant object"
    group_prop = vp.Group(params=[has_ns, has_remnant], name="Properties")
    group_prop.Description = "Qualitative properties of the source, " \
                             "conditioned on the assumption that the signal " \
                             "is an astrophysical compact binary merger"
    voevent.What.append(group_prop)


def create_dictionary(gw_dic, experiment, tel, role):
    """
    Create standard dictionary to fill info for VO event and name the VO files
    :param gw_dic: dictionary with GW information
    :param experiment: keyword for the experiment (defined in configuration file)
    :param tel: telescope name
    :param role: on-going role (test or observation), taken from VO input
    :return: dictionary with common infos
    """

    new_dic = {
        "event_type": "GW",
        "trigger_id": gw_dic["GraceID"],
        "event_status": gw_dic["AlertType"],
        "pkt_ser_num": gw_dic["Pkt_Ser_Num"],
        "inst": gw_dic["Instruments"],
        "tel_name": tel,
        "experiment": experiment,
        "obs_mode": role
    }

    return new_dic

def add_outputspace_request(output_space, gw_dic):
    """
    Function used to update the program file
    for the space segment and fill the header part
    of the observation plan json

    :param output_space: json object to complete
    :param gw_dic: dictionary with GW information

    """

    # create temporary array to fill information
    # add origin of the alert, here is GW,
    # add GraceDB ID,
    # add time of the alert,
    # add false alarm rate
    too_request = {
        "origin": "GW",
        "alert_id": gw_dic["GraceID"],
        "time_alert": str(gw_dic["Time"]),
        "far": gw_dic["FAR"]
    }

    # add possible remark in the json file
    if gw_dic["BNS"] > 0.5 or gw_dic["HasNS"]:
        too_request["remark"] = "At least one NS in the system"

    # add request part to json output file
    output_space["ToO_request"] = too_request

def add_outputspace_catalog(output_space, param):
    """
    Function used to update the program file
    for the space segment and fill catalog part of
    the observation plan json

    :param output_space: json object to complete
    :param param: gwemopt parameter dictionary

    """

    # retrieve info from catalog
    if param["doUseCatalog"]:
        catalog = {
            "name": param["galaxy_catalog"],
            # need to modify and retrieve from the catalog its version
            "cat_id": "v1"
        }
    else:
        catalog = {
            "name": "None",
            "cat_id": ""
        }

    output_space["Catalog"] = catalog

def add_outputspace_observationplan(output_space, atable, params):
    """
    Function to add the observation plan in output json for space segment

    :param output_space: json object to complete
    :param atable: astropy table with observation plan and meta data
    :param params: dictionary used to create the observation plan

    """

    # initalize full plan
    total_tiles = []

    # check that input astropy table is not empty
    if atable and atable[0]:
        for i in np.arange(0, len(atable)):
            tile = {}
            # add the needed information, start with coordinates
            # the next two lines will only work if we are using output of a tiling
            # if we use galaxy table we will need to use RAJ2000 and DECJ2000
            tile["RIGHT_ASCENSION"] = atable['RA'][i]
            tile["DECLINATION"] = atable['DEC'][i]
            # then score, add +1 on the id as the loop index start at 0
            tile["TILE_ID"] = int(i+1)
            tile["TILE_SCORE"] = atable['Prob'][i]
            # add duration, we used for the moment a default value ie 10 min
            tile["REQUESTED_OBS_DURATION_IN_MIN"] = 600
            # add ECLAIR conf, based on enum, to be updated later if needed
            tile["ECL_CONF"] = "0"
            # now GRM, based on enum, to be updated later if needed
            tile["GRM_CONF"] = "0"
            # MXT, for the moment we will always consider that we will use MXT
            # need to check if we need to add UV filter
            # based on enum, to be updated later if needed
            tile["MXT_CONF"] = "1"
            # VT conf is slightly different
            tile["VT_CONF"] = {
                # use default exposure time of 5 min
                "EXPOSURE_TIME" : 300,
                # window size may be smaller
                "WINDOW_SIZE" : 2048,
                # use 1 second between 2 images, we need to check possible numbers
                "INTERVAL_BETWEEN_IMG" : 1,
                # use a value for read_speed, need to see what are the possibilities
                "READ_SPEED" : 1
            }
            # next eleemtn is about plateform test ?
            tile["PF_CONF"] = {
                # need to check possible values
                "STABILITY" : "1",
                # Moon check need to come from the observation plan parameters
                "MOON_CHECK" : str(params["Moon_check"])
            }

            # two others paramaters can be added, we will see later
            total_tiles.append(tile)


    # add number of tiles
    output_space["Num_tiles"] = len(total_tiles)

    # then add the observation plan
    output_space["Tile_request"] = total_tiles

    # add iteration processing
    # keep only 1 for the moment
    output_space["Num_processing"] = 1

