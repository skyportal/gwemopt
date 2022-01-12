"""
Function to compute the score for the MM alert
This will then decide if we ask for ToO on space segment
There will be one function per alert type : GW, neutrinos, VHE

Author N. Leroy - 2020
Mofied S. Antier - 2022
"""


def gw_score(gw_dic, config_score):
    """
    Function specific to GW alerts

    :param gw_dic: dictionary with GW infos
    :param config_score : dictionary
    :return: score between 0 and 1, 1 is the best score
    """

    # init score to lowest value
    score = 0

    # need to check nature of the event
    # if not a BBH then add more interest to the event
    if gw_dic["BBH"] < config_score["BBH_thresh"]:
        score += 0.5

    return score
    
    
def swift_score(grb_dic):
	"""
	Function specific to BAT alerts

	:param gw_dic: dictionary with Swift infos
	:return: score between 0 and 1, 1 is the best score
	"""

	score=0
	#Astrophysics
	if grb_dic["defGRB"] == True:
		score=1
		#Observable
		if (float(grb_dic["moon_illum"]) < 70.0) and (float(grb_dic["moon_distance"]) > 20.0) and (float(grb_dic["sun_distance"]) > 20.0):
			score=2
			#Bright
			if (grb_dic["snr"] > 6.5):
					score=3

	return score
				
def fermi_score(grb_dic):
	"""
	Function specific to GBM alerts

	:param gw_dic: dictionary with GBM infos
	:return: score between 0 and 1, 1 is the best score
	"""

	score=0
	#Astrophysics
	if grb_dic["defGRB"] == True:
		score=1
		#Observable
		if (float(grb_dic["moon_illum"]) < 70.0) and (float(grb_dic["moon_distance"]) > 20.0) and (float(grb_dic["sun_distance"]) > 20.0):
			score=2
			#Bright
			if (grb_dic["snr"] > 8.0):
					score=3
					if (grb_dic["longshort"] != "Long") and (grb_dic["hratio"] > 1):
						score=5
	return score
