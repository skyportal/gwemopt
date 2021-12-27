from hop import Stream
from gcn import NoticeType
import logging
import logging.handlers 
#import TimedRotatingFileHandle


"""
Module to save gcn listener
"""
import sys
import os
import json
import time
import logging
from logging.handlers import TimedRotatingFileHandler

import voeventparse as vp
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
#from grandma_manager import online_processing




#Type GCN
#gcn.notice_types.LVC_INITIAL
#gcn.notice_types.LVC_UPDATE
#gcn.notice_types.LVC_TEST
#gcn.notice_types.LVC_PRELIMINARY
#gcn.notice_types.LVC_RETRACTION
#gcn.notice_types.ICECUBE_ASTROTRACK_GOLD
#gcn.notice_types.ICECUBE_ASTROTRACK_BRONZE


def filter_notices(notice_types):
	"""
	notice_types is a list of pygcn.NoticeType
	"""
	def filter_func(notice):
		# grab notice ID
		for param in notice.What["Param"]:
			if param["name"] == "Packet_Type":
				notice_id = int(param["value"])

		return notice_id in notice_types

	return filter_func

def process_gcn(payload, root):
	"""
	Process gcn function
	"""
	# Print the alert
	print('Got VOEvent:')
	print(payload)

	path = './temp/'
	time_r = time.time()
	filenameout = path + 'event_' + str(time_r) + '.json'
	print("filenameout: ", filenameout)
	# save the input xml file
	file_to_save = open(filenameout, 'wb')
	file_to_save.write(payload)
	file_to_save.close()

	voevent = vp.loads(payload)
 
 #To be defined differently 
    #online_processing(voevent, ROLE)


if __name__ == "__main__":
	# (killed or interrupted with control-C).

	# create directory where to store the inuput events
	if not os.path.exists('./temp/'):
		os.makedirs('./temp')

	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)


	# dir_path of the current directory
	dir_path = os.path.dirname(os.path.abspath(__file__)) + '/'
	print(dir_path)
	ROLE="test"


	
	if len(sys.argv) > 2:
		filename = str(sys.argv[1])
		print('config filename is ' + filename)
		ROLE = str(sys.argv[2])
		print('Run with role ' + ROLE)
	else:
		print('you need to provide a configuration file and a role (test,observation)')
		sys.exit()

	if not os.path.isabs(filename):
		# make assumption that the path is relative to the current directory
		filename = dir_path + filename

	with open(filename) as filein:
		server_config = json.load(filein)
		
	filter_func = filter_notices([NoticeType.LVC_TEST])
	stream = Stream(persist=True)
	with stream.open(server_config['scimma'], "r") as s:
		for message in filter(filter_func, s):
			print(message)
			process_gcn(message, root)

    #gcn.listen(host=server_config['in_host'], port=server_config['in_port'], handler=process_gcn)
