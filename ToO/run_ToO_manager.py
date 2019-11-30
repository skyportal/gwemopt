#!/bin/env/python
#script to test loading of VOEvent, treat them and send them to broker

#import and prepare setup
import sys
import voeventparse as vp

import ToO_manager as too
#import ToO_manager_David as too
#import ToO_manager_JG as too

#fileo= open('MS190425b-1-Preliminary.xml,0')

filename = str(sys.argv[1])
fileo= open(filename)
v = vp.load(fileo)

#too.online_processing(v,role_filter='test')
too.online_processing(v,role_filter='observation')

