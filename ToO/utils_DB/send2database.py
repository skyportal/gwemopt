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
        print(self.HOST, self.USER, self.PWD, self.DB)
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

        field_list = []
        val_list = []
        #fmt_list = []
        # Keep only fields with not None values
        for key, val in fields_dict.items():
            if val['val'] is not None:
                field_list.append(key)
                val_list.append(val['val'])
                #fmt_list.append(val['fmt'])
        sql = "INSERT INTO %s (%s) VALUES (%s)" % (str(table).replace("'",''),
                                                   str(field_list).strip('[]').replace("'",''),
                                                   str(val_list).strip('[]'))
        self.execute(sql)


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

        sql = "SELECT id FROM events WHERE name = '%s' AND status_id = '%s' AND revision = '%s'" % (str(event_name).replace("'",''), str(event_status).replace("'",''), str(event_revision).replace("'",''))

        # This is a bit dirty but will ensure that the whole pipeline 
        # will not crash if an id is not found
        try:
            #self.show_exact_query(sql)
            self.execute(sql)
            result = self.cursor.fetchone()
            return result[0]

        except:
            print ("No match found for '%s' with status %s and revision %s in 'events' table" % (event_name, event_status, event_revision))
            pass

        
    def fill_events(self, VOEvent, path_skymap="", filename_skymap="", proba=0):
        """ Fill table events"""

        table_name = 'events'

        table_fields = {
            "event_type_id": {"val": None, "fmt": '%d'},
            "event_detector_id": {"val": None, "fmt": '%d'},
            "name": {"val": None, "fmt": '%s'},
            "alias": {"val": None, "fmt": '%s'},
            "status_id": {"val": None, "fmt": '%d'},
            "revision": {"val": None, "fmt": '%d'},
            "trigger_time": {"val": None, "fmt": '%s'},
            "RA_trigger": {"val": None, "fmt": '%.4f'},
            "DEC_trigger": {"val": None, "fmt": '%.4f'},
            "errorbox_surface": {"val": None, "fmt": '%.4f'},
            "errorbox_shape": {"val": None, "fmt": '%s'},
            "errorbox_proba": {"val": None, "fmt": '%.4f'},
            "duration": {"val": None, "fmt": '%f'},
            "SNR": {"val": None, "fmt": '%f'},
            "probability": {"val": None, "fmt": '%f'},
            "rate_id": {"val": None, "fmt": '%d'},
            "comments": {"val": None, "fmt": '%s'},
            "url_locmap": {"val": None, "fmt": '%s'},
            "url_quicklook": {"val": None, "fmt": '%s'}
        }

        
        if VOEvent['Event_type']['value'] == 'GW':
            table_fields['event_type_id']['val'] = self.request_id('id', 'event_type', 'type', VOEvent['Event_type']['value'])
            table_fields['event_detector_id']['val'] = self.request_id('id','telescope', 'name', VOEvent['Event_inst']['value'])
            table_fields['name']['val'] = VOEvent['Event_ID']['value']
            table_fields['status_id']['val'] = self.request_id('id', 'event_status', 'value',VOEvent['Event_status']['value'])
            table_fields['revision']['val'] = VOEvent['Revision']['value']
            table_fields['trigger_time']['val'] = VOEvent['Trigger_time']['value']
            #table_fields['RA_trigger']['val'] = VOEvent['Event_ID']['value']
            #table_fields['DEC_trigger']['val'] = VOEvent['Event_ID']['value']
            #table_fields['errorbox_surface']['val'] = VOEvent['Event_ID']['value']
            #table_fields['errorbox_proba']['val'] = VOEvent['Event_ID']['value']
            #table_fields['duration']['val'] = VOEvent['Event_ID']['value']
            #table_fields['SNR']['val'] = VOEvent['Event_ID']['value']
            table_fields['rate_id']['val'] = VOEvent['importance']['value']
            #able_fields['comments']['val'] = VOEvent['Event_ID']['value']
            table_fields['url_locmap']['val'] = VOEvent['Loc_url']['value']
            table_fields['url_quicklook']['val'] = VOEvent['Quicklook_url']['value']


        if filename_skymap:
            from MOCregion import MOC_confidence_region

            # get skymap
            skymap = MOC_confidence_region(path_skymap, filename_skymap, proba)
            skymap.create_moc()

            table_fields['errorbox_shape']['val'] = str(skymap.moc)
            table_fields['probability']['val'] = proba
            table_fields['RA_trigger']['val'] = skymap.RA
            table_fields['DEC_trigger']['val'] = skymap.DEC

        self.fill_table(table_name, table_fields)


    def fill_event_properties(self, VOEvent):
        """ Fill table event_properties """

        table_name = 'event_properties'

        table_fields = {
            "event_id": {"val": None, "fmt": '%d'},
            "FAR": {"val": None, "fmt": '%f'},
            "luminosity_distance": {"val": None, "fmt": '%f'},
            "error_luminosity_distance": {"val": None, "fmt": '%f'},
            "redshift": {"val": None, "fmt": '%f'},
            "err_redshift": {"val": None, "fmt": '%f'},
            "error_90": {"val": None, "fmt": '%.4f'},
            "error_50": {"val": None, "fmt": '%.4f'},
            "search_method_id": {"val": None, "fmt": '%d'},
            "search_pipeline_id": {"val": None, "fmt": '%d'},
            "proba_bns": {"val": None, "fmt": '%f'},
            "proba_bbh": {"val": None, "fmt": '%f'},
            "proba_nsbh": {"val": None, "fmt": '%f'},
            "proba_remnant": {"val": None, "fmt": '%f'},
            "proba_ns": {"val": None, "fmt": '%f'},
            "proba_terrestrial": {"val": None, "fmt": '%f'},
            "comment": {"val": None, "fmt": '%s'}
        }

        if VOEvent['Event_type']['value'] == 'GW':
            # Convert FAR in Hz to a string '1 per xxx years'
            sec_in_years = 86400 * 365
            FAR_Hz = float(VOEvent['FAR']['value'])
            FAR_yr = FAR_Hz*sec_in_years
            #FAR_yr_str = '{:,.1f}'.format(1 / (FAR_yr)).replace(',', ' ')
            FAR = "1 per %.1e years" % (1/FAR_yr)
            
            #table_fields['event_id']['val'] = self.request_id('id', 'events', 'name', VOEvent['Event_ID']['value'])
            status_id = self.request_id('id', 'event_status', 'value',VOEvent['Event_status']['value'])
            table_fields['event_id']['val'] = self.request_event_id_from_name(VOEvent['Event_ID']['value'], status_id, VOEvent['Revision']['value'])
            table_fields['FAR']['val'] = FAR
            table_fields['luminosity_distance']['val'] = float(VOEvent['Distance']['value'])
            table_fields['error_luminosity_distance']['val'] = float(VOEvent['Err_distance']['value'])
            #table_fields['redshift']['val'] = VOEvent['Revision']['value']
            #table_fields['err_redshift']['val'] = VOEvent['Event_ID']['value']
            table_fields['error_90']['val'] = float(VOEvent['90cr_skymap']['value'])
            table_fields['error_50']['val'] = float(VOEvent['50cr_skymap']['value'])
            table_fields['search_method_id']['val'] = VOEvent['Group']['value']
            table_fields['search_pipeline_id']['val'] = VOEvent['Pipeline']['value']
            table_fields['proba_bns']['val'] = float(VOEvent['BNS']['value'])
            table_fields['proba_bbh']['val'] = float(VOEvent['BBH']['value'])
            table_fields['proba_nsbh']['val'] = float(VOEvent['NSBH']['value'])
            table_fields['proba_remnant']['val'] = float(VOEvent['HasRemnant']['value'])
            table_fields['proba_ns']['val'] = float(VOEvent['HasNS']['value'])
            table_fields['proba_terrestrial']['val'] = float(VOEvent['Terrestrial']['value'])
            #table_fields['comment']['val'] = VOEvent['Quicklook_url']['value']

            self.fill_table(table_name, table_fields)


    def fill_obs_plan(self,tiles_table_meta, status, revision):
        """ Fill table observation_plan"""
        
        table_name = 'observation_plan'

        table_fields = {
            "event_id": {"val": None, "fmt": '%d'},
            "telescope_id": {"val": None, "fmt": '%d'},
            "status_id": {"val": None, "fmt": '%d'},
            "mode": {"val": None, "fmt": '%d'},
        }
        telname_DB = self.get_tel_name_inDB(tiles_table_meta['telescope_name'])

        #table_fields['event_id']['val'] = self.request_id('id', 'events', 'name', tiles_table_meta['trigger_id'])
        status_id = self.request_id('id', 'event_status', 'value', status)
        table_fields['event_id']['val'] = self.request_event_id_from_name(tiles_table_meta['trigger_id'], status_id, revision)

        table_fields['telescope_id']['val'] = self.request_id('id', 'telescope', 'name', telname_DB)
        # By default status is computed
        table_fields['status_id']['val'] = 2
        mode_id = self.request_id('id', 'observation_plan_mode', 'mode', tiles_table_meta['obs_mode'])
        table_fields['mode']['val'] = mode_id

        self.fill_table(table_name, table_fields)
        

    def fill_tiles(self, tiles_table, status, revision, path_skymap, filename_skymap, proba):
        """ Fill table tiles"""

        table_name = 'tiles'

        table_fields = {
            "trigger_id": {"val": None, "fmt": '%d'},
            #"observatory_id": {"val": None, "fmt": '%d'},
            "telescope_id": {"val": None, "fmt": '%d'},
            "obs_plan_id": {"val": None, "fmt": '%d'},
            "instrument_id": {"val": None, "fmt": '%d'},
            "filter_id": {"val": None, "fmt": '%d'},
            #"T_start": {"val": None, "fmt": '%s'},
            #"T_end": {"val": None, "fmt": '%s'},
            "field_center_RA": {"val": None, "fmt": '%.4f'},
            "field_center_dec": {"val": None, "fmt": '%.4f'},
            "field_roll_angle": {"val": None, "fmt": '%.4f'},
            "corners_coord": {"val": None, "fmt": '%s'},
            #"airmass": {"val": None, "fmt": '%.4f'},
            "public_result": {"val": None, "fmt": '%s'},
            "follow_up": {"val": None, "fmt": '%s'},
            "request_date": {"val": None, "fmt": '%s'},
            "status_id": {"val": None, "fmt": '%d'},
            "Filename": {"val": None, "fmt": '%s'},
            "rank_id": {"val": None, "fmt": '%f'},
            "proba_metric1": {"val": None, "fmt": '%f'},
            "err_metric1": {"val": None, "fmt": '%f'},
            "proba_metric2": {"val": None, "fmt": '%f'},
            "err_metric2": {"val": None, "fmt": '%f'},
            "proba_metric3": {"val": None, "fmt": '%f'},
            "err_metric3": {"val": None, "fmt": '%f'}
        }

        telname_DB = self.get_tel_name_inDB(tiles_table.meta['telescope_name'])
        telescope_id = self.request_id('id', 'telescope', 'name', telname_DB)
        #event_id = self.request_id('id', 'events', 'name', tiles_table.meta['trigger_id'])
        status_id = self.request_id('id', 'event_status', 'value', status)
        event_id = self.request_event_id_from_name(tiles_table.meta['trigger_id'], status_id, revision)
        obs_plan_id = self.request_id_2cond('id', 'observation_plan', ['event_id','telescope_id'], [event_id, telescope_id])

        table_fields['trigger_id']['val'] = event_id        
        table_fields['telescope_id']['val'] = telescope_id
        table_fields['obs_plan_id']['val'] = obs_plan_id

        # compute 2D probability for each tile
        tile_proba_list = self.get_tile_proba(path_skymap, filename_skymap, tiles_table['Corners_list'])

        #table_fields['observatory_id']['val'] = self.request_id('observatory_id', 'telescope', 'name', telname_DB)
        for i, row in enumerate(tiles_table):
            table_fields['rank_id']['val'] = int(row['rank_id'])
            table_fields['field_center_RA']['val'] = float(row['RA'])
            table_fields['field_center_dec']['val'] = float(row['DEC'])
            #table_fields['proba_metric1']['val'] = float(row['Prob'])
            table_fields['proba_metric1']['val'] = np.round(float(tile_proba_list[i]),decimals=5)
            table_fields['corners_coord']['val'] = row['Corners']
            table_fields['proba_metric2']['val'] = float(row['Prob'])
            table_fields['proba_metric3']['val'] = float(row['Prob'])

            self.fill_table(table_name, table_fields)

                
    def fill_galaxies(self, galaxies_table, trigger_id, status, revision):
        """ Fill table galaxies"""

        table_name = 'galaxies'

        table_fields = {
            "event_id": {"val": None, "fmt": '%s'},
            "name": {"val": None, "fmt": '%s'},
            "RA_center": {"val": None, "fmt": '%f'},
            "DEC_center": {"val": None, "fmt": '%f'},
            "redshift": {"val": None, "fmt": '%f'},
            "err_redshift": {"val": None, "fmt": '%f'},
            "distance": {"val": None, "fmt": '%f'},
            "err_distance": {"val": None, "fmt": '%f'},
            "rank_id": {"val": None, "fmt": '%f'},
            "Proba": {"val": None, "fmt": '%f'},
            "type_id": {"val": None, "fmt": '%d'},
            "catalog_name": {"val": None, "fmt": '%s'},
        }

        #table_fields['event_id']['val'] = self.request_id('id', 'events', 'name', trigger_id)
        status_id = self.request_id('id', 'event_status', 'value', status)
        table_fields['event_id']['val'] = self.request_event_id_from_name(trigger_id, status_id, revision)

        for row in galaxies_table:
            table_fields['rank_id']['val'] = row['id']
            table_fields['RA_center']['val'] = float(row['RAJ2000'])
            table_fields['DEC_center']['val'] = float(row['DEJ2000'])
            if row['z']:
                table_fields['redshift']['val'] = float(row['z'])           
            if row['Dist']:
                table_fields['distance']['val'] = float(row['Dist'])
            #if row['Dist_err'] != 'None':
            #    table_fields['err_distance']['val'] = float(row['Dist_err'])
            #else:
            #    table_fields['err_distance']['val'] = -1
            table_fields['Proba']['val'] = float(row['Sloc'])

            # Save only one galaxy identifier
            if (row['GWGC'] != '---') and (row['GWGC']):
                name = row['GWGC']
                catalog = 'GWGC'
            #elif (row['PGC'] != '--') and (row['PGC']):
            #    name = row['PGC']
            #    catalog = 'PGC'
            elif (row['HyperLEDA'] != '---') and (row['HyperLEDA']):
                name = row['HyperLEDA']
                catalog = 'HyperLEDA'
            elif (row['2MASS'] != '---') and (row['2MASS']):
                name = row['2MASS']
                catalog = '2MASS'
            elif (row['SDSS'] != '---') and (row['SDSS']):
                name = row['SDSS']
                catalog = 'SDSS-DR12'

            table_fields['name']['val'] = name
            table_fields['catalog_name']['val'] = catalog

            self.fill_table(table_name, table_fields)

    def fill_link_tiles_galaxies(self, tiles_table, galaxies_table, trigger_id, status, revision):
        """ Fill table galaxies"""

        table_name = 'link_galaxy_tile'

        table_fields = {
            "obs_plan_id": {"val": None, "fmt": '%d'},
            "tile_id": {"val": None, "fmt": '%d'},
            "galaxy_id": {"val": None, "fmt": '%d'},
        }

        telname_DB = self.get_tel_name_inDB(tiles_table.meta['telescope_name'])
        telescope_id = self.request_id('id', 'telescope', 'name', telname_DB)
        #event_id = self.request_id('id', 'events', 'name', tiles_table.meta['trigger_id'])
        status_id = self.request_id('id', 'event_status', 'value', status)
        event_id = self.request_event_id_from_name(tiles_table.meta['trigger_id'], status_id, revision)
        obs_plan_id = self.request_id_2cond('id', 'observation_plan', ['event_id','telescope_id'], [event_id, telescope_id])

        table_fields['obs_plan_id']['val'] = obs_plan_id

        FoV_sep = tiles_table.meta['FoV_sep']
        FoV_tel = tiles_table.meta['FoV_telescope']

        # Find galaxies within FoV_sep * FoV_telescope around center of the tile
        for row in tiles_table:
            table_fields['tile_id']['val'] = row['rank_id']
            tile_center_RA = float(row['RA'])
            tile_center_DEC = float(row['DEC'])
            mask = ((FoV_tel/2 * FoV_sep)**2 >= (galaxies_table['RAJ2000'] - tile_center_RA)**2 + (galaxies_table['DEJ2000'] - tile_center_DEC)**2)
            galaxies_id = list(galaxies_table['id'][mask])
            # Link each galaxy to a tile
            for galaxy_id in galaxies_id:
                table_fields['galaxy_id']['val'] = galaxy_id
                self.fill_table(table_name, table_fields)



    def get_tel_name_inDB(self, telname):
        """ Convert telescope name from gwemopt convention to GRANDMA convention """

        if telname == "F60":
            telnameDB = "GWAC/F60-A"
        elif telname == "GWAC":
            telnameDB = "GWAC"
        elif telname == "IRIS":
            telnameDB = "OHP-IRiS"
        elif telname == "Makes-60":
            telnameDB = "LesMakes-T60"
        elif telname == "NOWT":
            telnameDB = "NOWT"
        elif telname == "TCA":
            telnameDB = "TAROT-Calern (TCA)"
        elif telname == "TCH":
            telnameDB = "TAROT-Chili (TCH)"
        elif telname == "TRE":
            telnameDB = "TAROT-Reunion (TRE)"
        elif telname == "Zadko":
            telnameDB = "Zadko"
        elif telname == "TNT":
            telnameDB = "TNT"
        elif telname == "OAJ":
            telnameDB = "OAJ-T80"
        elif telname == "Abastunami-T70":
            telnameDB = "Abastunami-T70"
        elif telname == "Abastunami-T48":
            telnameDB = "Abastunami-T48"
        elif telname == "Lisnyky-AZT8":
            telnameDB = "Lisnyky-AZT8"
        elif telname == "OSN":
            telnameDB = "OSN-T150"
        elif telname == "CAHA":
            telnameDB = "CAHA"
        elif telname == "ShAO-T60":
            telnameDB = "ShAO-T60"
        elif telname == "OHP-T120":
            telnameDB = "OHP-T120"
        else:
            telnameDB = "Not found"

        return telnameDB


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

