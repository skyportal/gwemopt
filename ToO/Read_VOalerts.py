#!/usr/bin/env python

import voeventparse as vp
import numpy as np
import pandas as pd


#######################################
#Load VO event
def load_voevent(VO_event):
 with open(VO_event,'rb') as f:
  v = vp.load(f)
 return v

#######################################

#Find emitter
def get_emitter(v): 
 Name=[]
 Entry=[]

 #Get ivorn
 Name.append("ivorn")
 Entry.append(v.attrib['ivorn'])
 
 #Get author info
 for element in v.Who.Author.iterchildren():
  Name.append(element.tag)
  Entry.append(element.text)

 df = pd.DataFrame({'key' : Name, 'content' : Entry})
 df.sort_values('key')
 print(df)
 return Name,Entry

#######################################

#Find dedicated GRANDMA receiver
def get_receiver(VO_event):
 receiver=VO_event.split("_")[2]
 print("VO event dedicated to: "+str(receiver))
 return receiver

#######################################

#Find the alert type, status and its internal GRANDMA priority
def get_alertype(v):
 Name=[]
 Entry=[]
 Description=[]

 #Type of the alert
 Name.append("Event_type")
 Entry.append(v.find(".//Param[@name='Event_type']").attrib['value'])
 Description.append(str(v.find(".//Param[@name='Event_type']").Description))

 #Internal ranking in GRANMDA
 Name.append("Importance")
 Entry.append(str(v.Why.attrib['importance']))
 Description.append(str(v.Why.Description))

 #Originated instrument
 Name.append("Event_inst")
 Entry.append(v.find(".//Param[@name='Event_inst']").attrib['value'])
 Description.append(str(v.find(".//Param[@name='Event_inst']").Description))
 

 #Status of the alert
 Name.append("Event_status")
 Entry.append(v.find(".//Param[@name='Event_status']").attrib['value'])
 Description.append(str(v.find(".//Param[@name='Event_status']").Description))

 Name.append("Iteration")
 Entry.append(v.find(".//Param[@name='Iteration']").attrib['value'])
 Description.append(str(v.find(".//Param[@name='Iteration']").Description))

 df = pd.DataFrame({'key' : Name, 'content' : Entry, 'description' : Description})
 df.sort_values('key')
 print(df)
 return Name,Entry,Description



#######################################

#GRANDMA Follow-up advocate on duty at the time of the alert
def get_FA(v):
 FA=v.find(".//Param[@name='FA']").attrib['value']
 print("Follow-up Advocate on duty at the time of the alert: FA")
 return FA

#######################################

#Access to parameters of the alert
def get_parameters(v):

 Name=[]
 Value=[]
 Unit=[]
 Description=[]
 for element in v.What.iterchildren():
  tag_type=str(element.tag)
  if tag_type=="Param":
   Name.append(element.attrib['name'])
   Value.append(element.attrib['value'])
   try:
    Unit.append(element.attrib['unit'])
   except KeyError:
    Unit.append("-")
   try:
    Description.append(str(element.Description))
   except AttributeError:
    Description.append("-")
   
  if tag_type=="Group":
   for subelement in element.iterchildren():
    tag_type=str(subelement.tag)
    if tag_type=="Param":
     Name.append(subelement.attrib['name'])
     Value.append(subelement.attrib['value'])
     try:
      Unit.append(subelement.attrib['unit'])
     except KeyError:
      Unit.append("-")
     try:
      Description.append(str(subelement.Description))
     except AttributeError:
      Description.append("-")
     

 df = pd.DataFrame({'key' : Name, 'value' : Value, 'unit' : Unit,  'description' : Description })
 df.sort_values('key')
 print(df)
 return Name,Value,Unit,Description

#Load Observation plan strategy
def get_obsplan(v,receiver):
 ID=[]
 Ra=[]
 Dec=[]
 Grade=[]
 Header=[]
 if receiver!="DB":
  for element in v.What.iterchildren():
   tag_type=str(element.tag)
   if tag_type=="Table":
    for subelement in element.iterchildren():
     tag_type=str(subelement.tag)
     if tag_type=="Field":
      Header.append(str(subelement.attrib['name']))
     if tag_type=="Data":
      for lines in subelement.iterchildren():
       ID.append(int(lines.TD[0]))
       Ra.append(float(lines.TD[1]))
       Dec.append(float(lines.TD[2]))
       Grade.append(float(lines.TD[3]))
      
  df = pd.DataFrame({Header[0] :ID, Header[1] : Ra, Header[2] : Dec,  Header[3] : Grade })
  print(df)
  return ID,Ra,Dec,Grade
 return "","","",""

def treatment_VO(VO_event):

 #Load VO event
 v=load_voevent(VO_event)

 #Find emitter
 get_emitter(v)

 #Find dedicated GRANDMA receiver
 receiver=get_receiver(VO_event)

 #Find the alert type, status and its internal GRANDMA priority
 get_alertype(v)

 #GRANDMA Follow-up advovate on duty at the time of the alert
 get_FA(v)

 #Access to parameters of the alert
 get_parameters(v)

 #Load Observation plan strategy
 [ID,Ra,Dec,Grade]=get_obsplan(v,receiver)

treatment_VO("./VOEVENTS/GRANDMA20190203_GWMS181101ab_GWAC_a.xml")
