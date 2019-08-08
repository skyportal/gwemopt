#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: David Corre

import numpy as np
import pymysql
from astropy.io import ascii
from collections import Counter
import json
import healpy as hp

class populate_DB:
    """ This class allows to write information in DB """

    def __init__(self, path_config, filename_config):
       
        # Get config params from json file
        with open(path_config+filename_config) as f:
            data = json.load(f)
            
        self.HOST = data["host"]
        self.USER = data["user"]
        self.PWD = data["password"]
        self.DB = data["database_name"]

    def connect_db(self):
        # Open database connection
        self.db = pymysql.connect(host=self.HOST, user=self.USER, passwd=self.PWD, db=self.DB)

        # prepare a cursor object using cursor() method
        self.cursor = self.db.cursor()

    def execute(self, sql):
        try:
            # Execute the SQL command
            self.cursor.execute(sql)
            # Commit your changes in the database
            self.db.commit()
        except pymysql.InternalError as error:
            code, message = error.args
            print (">>>>>>>>>>>>>", code, message)
            # Rollback in case there is any error
            self.db.rollback()

    def show_exact_query(self, sql):
        self.cursor.mogrify(sql)

    def close_db(self):
        # disconnect from server
        self.db.close()

    def fill_table(self, table, fields_dict):
        """ Generic method to write in a table """

        pass

    def request_id(self, getField, table, field, value):
        """ Generic method to get id from a specific field in a table """

        sql = "SELECT %s FROM %s WHERE %s = '%s'" % (str(getField).replace("'",''), 
                                                     str(table).replace("'",''),
                                                     str(field).replace("'",''),
                                                     str(value))
        # This is a bit dirty but will ensure that the whole pipeline 
        # will not crash if an id is not found
        try:
            #self.show_exact_query(sql)
            self.execute(sql)
            result = self.cursor.fetchone()
            return result[0]

        except:
            print ("No match found for '%s' in field '%s' of table '%s' in the database" % (value, field ,table))
            pass

    def request_id_2cond(self, getField, table, fields, values):
        """ Generic method to get id from a specific field in a table with 2 conditions"""

        sql = "SELECT %s FROM %s WHERE %s = '%s' AND %s = '%s'" % (str(getField).replace("'",''),
                                                                   str(table).replace("'",''),
                                                                   str(fields[0]).replace("'",''),
                                                                   str(values[0]),
                                                                   str(fields[1]).replace("'",''),
                                                                   str(values[1]))
        # This is a bit dirty but will ensure that the whole pipeline 
        # will not crash if an id is not found
        try:
            #self.show_exact_query(sql)
            self.execute(sql)
            result = self.cursor.fetchone()
            return result[0]

        except:
            print ("No match found for '%s' in field '%s'  or '%s' in field '%s' of table '%s' in the database" % (values[0], fields[0], values[1], fields[1], table))
            pass

    def request_event_id_from_name(self, event_name, event_status, event_revision):
        """ Generic method to get id from event name and status and revision ids"""

        pass

        
    def fill_events(self, VOEvent, path_skymap="", filename_skymap="", proba=0):
        """ """

        pass

    def fill_event_properties(self, VOEvent):
        """  """

        pass

    def fill_obs_plan(self,tiles_table_meta, status, revision):
        """ """
        
       pass 

    def fill_tiles(self, tiles_table, status, revision, path_skymap, filename_skymap, proba):
        """ """

        pass
                
    def fill_galaxies(self, galaxies_table, trigger_id, status, revision):
        """ """

        pass

    def fill_link_tiles_galaxies(self, tiles_table, galaxies_table, trigger_id, status, revision):
        """ """

        pass


    def get_tel_name_inDB(self, telname):
        """  """

        pass


    def get_tile_proba(self, pathskymap, skymap_file, corners_list):
        """returns the cumulative 2D probability of a tile defined by corners coordinate"""

        skymap_proba = hp.read_map(pathskymap+skymap_file, verbose = False)
        npix = len(skymap_proba)
        nside = hp.npix2nside(npix)   
    
        list_tile_proba = []
    
        for corners in corners_list:
            current_tile_pixels = self.get_tile_pixels(nside, corners)
            list_tile_proba.append(np.sum(skymap_proba[current_tile_pixels]))

        return list_tile_proba

    def get_tile_pixels(self, nside, corners):
        """Return pixels contained in a tile defined by its corner coordinates"""

        #conversion to vectors using healpy function
        xyz = []
        for i in range(len(corners)):
            xyz.append(hp.ang2vec(corners[i][0], corners[i][1], lonlat=True))
    
        tile_pixels = hp.query_polygon(nside, np.array(xyz), inclusive=False)

        return tile_pixels

