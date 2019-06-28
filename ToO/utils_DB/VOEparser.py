#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: David Corre, Sarah Antier

import voeventparse as vp


class VOEparser:
    """This class is used to parse the VOE"""
    def __init__(self, VO_event, isfile = False):
        """
        params:
            VO_event: filename or voeventparse argument
                       
            isfile: boolean
                  True: VOE loaded from a file  
                  False: VOE passed as argument (default)
        """
        
        self.VO_event = VO_event
        self.isfile = isfile
        
        # Create dictionary that will contain VOE informations
        self.VOE_dict = {}

    def load_voevent(self):
        """ Load VOE """
        if self.isfile:
            with open(self.VO_event,'rb') as f:
                self.v = vp.load(f)
        else:
            self.v = self.VO_event
    

    def get_parameters(self):
        """ Extract information from VOE """

        for element in self.v.What.iterchildren():
            tag_type = str(element.tag)
            if tag_type == "Param":
                Key = element.attrib['name']
                Value = element.attrib['value']
                try:
                    Unit = element.attrib['unit']
                except KeyError:
                    Unit = "-"
                try:
                    Description = str(element.Description)
                except AttributeError:
                    Description = "-"
                self.VOE_dict[Key]={'value': Value, 'Unit': Unit, 'Description': Description}

            elif tag_type == "Group":
                for subelement in element.iterchildren():
                    tag_type = str(subelement.tag)
                    if tag_type == "Param":
                        Key = subelement.attrib['name']
                        Value = subelement.attrib['value']
                        try:
                            Unit = subelement.attrib['unit'] 
                        except KeyError:
                            Unit = "-"
                        try:
                            Description = str(subelement.Description)
                        except AttributeError:
                            Description = "-"
                        self.VOE_dict[Key]={'value': Value, 'Unit': Unit, 'Description': Description}

            
        # try to get importance set by GRANDMA
        try:
            Key = 'importance'
            Value = self.v.Why.attrib[Key]
            self.VOE_dict[Key]={'value': Value}
        except:
            self.VOE_dict['importance']={'value': ' '}
        
        # try to get trigger time
        try:
            Key = 'Trigger_time'
            Value = self.v.WhereWhen.ObsDataLocation.ObservationLocation.AstroCoords.Time.TimeInstant.ISOTime
            Value = str(Value).replace("T",' ')
            self.VOE_dict[Key]={'value': Value}
        except:
            pass
        
        return self.VOE_dict
