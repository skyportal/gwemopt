#!/usr/bin/env python3


def VO_dic():
  VO = {
    "role" : "",
		  "stream" : "grandma.lal.in2p3.fr/GRANDMA_Alert",
		  "streamid" : "",
    "voivorn" : "ivo://svom.bao.ac.cn/LV#SFC_GW_",
		  "authorivorn" : "GRANDMA_Alert",
		  "shortName" : "GRANDMA",
		  "contactName": "Nicolas  Leroy",
    "contactPhone": "+33-1-64-46-83-73",
    "contactEmail": "leroy@lal.in2p3.fr",
    "description": "Selected by ",
    "vodescription": "VOEvent created in GRANDMA",
    "locationserver": "",
    "voschemaurl":"http://www.cacr.caltech.edu/~roy/VOEvent/VOEvent2-110220.xsd",
    "name":"TAROT-20180204-a",
    "FOV":"1.9",
    "Mag_limit":"18",
    "Filters_tel":"r",
    "airmass":"",
    "calib":"",
    "exposure":"30",
    "sequence":"1",
    "Ra_tile":"72.85",
    "Dec_tile":"-9.86",
    "Neargal_name":"04512626-1023199",
    "Neargal_dist":"1102.2",
    "obs":[["2019-01-01-00:00:00.0","2019-01-01-00:05:00.0","r","johnson","17","0.3","",""]],
    "utc":"2019-01-01-00:02:00.0",
    "tel":"TCA",
    "ra":"72.84",
    "dec":"-9.86",
    "errbox":"2",
    "imp":"1.0",
    "url":"",
    "class":"Kilonova",
    "prob":"1.0",
    "slope":"1.5",
    "slope_err":"1.5",
    "filts":"r",
    "filtsys":"",
    "magsym":"",
    "magsys":"",   
	 }
  return VO


def NewVOEvent(VO_dic,initxml,tag): 


    # Parse UTC
    print(VO_dic["utc"])
    utc_YY = int(VO_dic["utc"][:4])
    utc_MM = int(VO_dic["utc"][5:7])
    utc_DD = int(VO_dic["utc"][8:10])
    utc_hh = int(VO_dic["utc"][11:13])
    utc_mm = int(VO_dic["utc"][14:16])
    utc_ss = float(VO_dic["utc"][17:])
    t = Time('T'.join([VO_dic["utc"][:10], VO_dic["utc"][11:]]), scale='utc', format='isot')
    mjd = t.mjd
    
    now = Time.now()
    mjd_now = now.mjd
   
    ivorn = ''.join([VO_dic["name"], str(utc_hh), str(utc_mm), '/', str(mjd_now)]) 

    v = vp.Voevent(stream=VO_dic["stream"], stream_id=ivorn, role=vp.definitions.roles.observation)

    # Author origin information
    vp.set_who(v, date=datetime.datetime.utcnow(), author_ivorn=VO_dic["authorivorn"])

    # Author contact information
    vp.set_author(v, title="GRANDMA ALERT Optical transient")
    vp.set_author(v, contactName=VO_dic["contactName"])
    vp.set_author(v, shortName=VO_dic["shortName"])
    vp.set_author(v,contactPhone=VO_dic["contactPhone"])
    vp.set_author(v,contactEmail=VO_dic["contactEmail"])

    # Parameter definitions

    if tag=="photo":
        parse_param_OT(v,VO_dic)


    #WhereWhen


    vp.add_where_when(v, coords=vp.Position2D(ra=VO_dic["ra"], dec=VO_dic["dec"], err=VO_dic["errbox"], units='deg', system=vp.definitions.sky_coord_system.utc_fk5_geo),
        obs_time=datetime.datetime(utc_YY,utc_MM,utc_DD,utc_hh,utc_mm,int(utc_ss), tzinfo=pytz.UTC), observatory_location=VO_dic["tel"])

    #Why
    
    vp.add_why(v, importance=VO_dic["imp"])
    v.Why.Name = VO_dic["stream"]

    get_citation_inital_alert(initxml,v.Why)
    """
    if vp.valid_as_v2_0(v):
        with open('./VOEVENTS/%s.xml' % VO_dic["utc"], 'wb') as f:
            voxml = vp.dumps(v)
            xmlstr = minidom.parseString(voxml).toprettyxml(indent="   ")
            f.write(xmlstr)
            print(vp.prettystr(v.Who))
            print(vp.prettystr(v.What))
            print(vp.prettystr(v.WhereWhen))
            print(vp.prettystr(v.Why))
    else:
        print("Unable to write file %s.xml" % VO_dic["name"])

    """
    with open('./VOEVENTS/%s.xml' % VO_dic["utc"], 'wb') as f:
        vp.dump(v, f)

#Load VO event
def load_voevent(xml):
    with open(xml,'rb') as f:
        v = vp.load(f)
        return v

def get_citation_inital_alert(xml,h):
    oldvo=load_voevent(xml)
    ivorn=oldvo.attrib['ivorn']
    Citations = objectify.SubElement(h,"Citations",cite="followup>"+str(ivorn))
    h.append(Citations)
    #newvo.add_why(v,importance=VO_dic["voimportance"])
    #newvo.Why.EventIVORN
    
    #<EventIVORN cite="followup">ivo://org.hotwired/exciting_events#1</EventIVORN>


def parse_param_OT(newvo,Tel_dic):

    #Name of the OT
    Name = vp.Param(name="Name_ID", value=Tel_dic["name"],ucd="meta.id",dataType="string")
    Name.Description = "Name of the optical transient"
    newvo.What.append(Name)

    #Configuration of the telescope
    FOV_tel = vp.Param(name="FOV", value=Tel_dic["FOV"],ucd="instr.fov",dataType="float",unit="deg")
    FOV_tel.Description = "FOV of the telescope used for the observation strategy"
    magnitude_tel = vp.Param(name="Mag_limit", value=Tel_dic["Mag_limit"],ucd="phot.mag",dataType="float",unit="mag")
    magnitude_tel.Description = "Intrinsic magnitude limit of the telescope 5 sigma"
    filt_tel = vp.Param(name="Filters_tel", value=Tel_dic["Filters_tel"],ucd="inst.filt",dataType="string")
    filt_tel.Description = "Available filters"
    config_tel=vp.Group(params=[FOV_tel,magnitude_tel, filt_tel],name="Set-up_tel")
    config_tel.Description="Some characteristics of the telescope used for the observations"
    newvo.What.append(config_tel)    
    
  
    #Configuration of observations
    airmass = vp.Param(name="airmass", value=Tel_dic["airmass"],ucd="obs.airMass ")
    airmass.Description = "Airmass"
    calib = vp.Param(name="calib", value=Tel_dic["calib"],ucd="obs.calib")
    calib.Description = "Calibration for the observation"
    exposure = vp.Param(name="exposure", value=Tel_dic["exposure"],ucd="obs.exposure",dataType="float",unit="s")
    exposure.Description = "Exposure time"
    sequence=vp.Param(name="sequence", value=Tel_dic["sequence"],ucd="obs.sequence")
    sequence.Description="Number of same sequence down for the observation"
    config_obs=vp.Group(params=[airmass,calib,exposure,sequence])
    config_obs.Description="Observation parameters"
    newvo.What.append(config_obs)

    #Field params
    ra_tile = vp.Param(name="Ra_tile", ucd="pos.eq.ra ", unit="deg", dataType="float",value=Tel_dic["Ra_tile"]) 
    ra_tile.Description="The right ascension at center of fov in equatorial coordinates"
    dec_tile = vp.Param(name="Dec_tile", ucd="pos.eq.ra ", unit="deg", dataType="float",value=Tel_dic["Dec_tile"])      
    dec_tile.Description="The declination at center of the fov in equatorial coordinates"
    Field_param=vp.Group(params=[ra_tile,dec_tile])
    Field_param.Description="Coordinates of the center of the field observed by the telescope"
    newvo.What.append(Field_param)

    #Properties of the source

    quicklook=vp.Param(name="Url_Quicklook",value=Tel_dic["url"], ucd="meta.ref",dataType="string")
    quicklook.Description="Quicklook data"

    #Nature of the transient 
    classification=vp.Param(name="Classification",value=Tel_dic["class"], ucd="meta.id",dataType="string")
    classification.Description="Nature of the source: Supernovae, Flare star, Minor planet, Kilonovae, Orphan Afterglow ..."
    #newvo.What.append(classification)
    probability=vp.Param(name="Probability",value=Tel_dic["prob"], ucd="stat.probability",dataType="float")
    probability.Description="Indicator from 0 to 1 that nature of the transient is real"  
    #newvo.What.append(probability) 

    #Slope
    slope=vp.Param(name="Slope",value=Tel_dic["slope"], ucd="src.var",dataType="float")
    slope.Description="Fading of the transient in mag/day"
    #newvo.What.append(slope) 
    slope_error=vp.Param(name="Slope_error", value=Tel_dic["slope_err"],ucd="phot.mag",dataType="float") 
    slope_error.Description="Fading of the transient in mag/day (error)"
    #newvo.What.append(slope_error)
    filt_symbol=vp.Param(name="filt_symbol", value=Tel_dic["filts"], ucd="meta.number",dataType="string") 
    filt_symbol.Description="Symbol for the filter"
    #newvo.What.append(filt_symbol)
    filt_system=vp.Param(name="filt_system",value=Tel_dic["filtsys"], ucd="meta.number",dataType="string") 
    filt_system.Description="System for the filter"
    #newvo.What.append(filt_system)
    mag_symbol=vp.Param(name="mag_symbol",value=Tel_dic["magsym"],ucd="meta.number",dataType="string") 
    mag_symbol.Description="Symbol used for calculating the magnitude"
    #newvo.What.append(mag_symbol)
    mag_system=vp.Param(name="mag_system",value=Tel_dic["magsys"], ucd="meta.number",dataType="string") 
    mag_system.Description="System used for calculating the magnitude"
    #newvo.What.append(mag_system)
 
    #Nearby Galaxy
    gal_name = vp.Param(name="Neargal_name",value="", ucd="meta.id",dataType="string")
    gal_name.Description="Nearby Galaxy close to the transient"
    gal_dist = vp.Param(name="Neargal_dist",value="", ucd="meta.distance",dataType="float",unit="Mpc")
    gal_dist.Description="Distance of the near-by galaxy"
    Prop_param=vp.Group(params=[quicklook,classification,probability,slope, slope_error,filt_symbol,filt_system,mag_symbol,mag_system,gal_name,gal_dist])
    Prop_param.Description="Properties of the source"
    newvo.What.append(Prop_param)
    
    Table = objectify.Element("Table")
    Table.Description="Properties of the transient observation"

    #Tstart tstop for the full tile
    tstart=objectify.SubElement(Table, "Field", name="tstart", ucd="time.start",dataType="string") 
    tstart.Description="Start of the observations"
    tstop=objectify.SubElement(Table, "Field", name="tstop", ucd="time",dataType="string") 
    tstop.Description="Time end of the observations"

    #Filters params
    filt_symbol=objectify.SubElement(Table, "Field", name="filt_symbol", ucd="meta.number",dataType="string") 
    filt_symbol.Description="Symbol for the filter"
    filt_system=objectify.SubElement(Table, "Field", name="filt_system", ucd="meta.number",dataType="string") 
    filt_system.Description="System for the filter"

    #Magnitude params
    mag_value=objectify.SubElement(Table, "Field", name="mag_value", ucd="phot.mag",dataType="float") 
    mag_value.Description="Magnitude for the transient"
    mag_error=objectify.SubElement(Table, "Field", name="mag_error", ucd="phot.mag",dataType="float") 
    mag_error.Description="Error of the magnitude for the transient"
    mag_symbol=objectify.SubElement(Table, "Field", name="mag_symbol", ucd="meta.number",dataType="string") 
    mag_symbol.Description="Symbol used for calculating the magnitude"
    mag_system=objectify.SubElement(Table, "Field", name="mag_system", ucd="meta.number",dataType="string") 
    mag_system.Description="System used for calculating the magnitude"
    Data = objectify.SubElement(Table, "Data") 
    for i in np.arange(len(Tel_dic["obs"])):
      TR = objectify.SubElement(Data, "TR")
      for j in np.arange(len(Tel_dic["obs"][i])):
        #objectify.SubElement(TR, "TD",value=str(Tel_dic["OS"][i][j]))
         objectify.SubElement(TR, 'TD')
         TR.TD[-1]=str(Tel_dic["obs"][i][j])
    newvo.What.append(Table)
    

if __name__ == "__main__":
    
    import astropy.coordinates as coord
    import astropy.units as u
    from astropy.io import ascii
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    import voeventparse as vp
    import datetime
    import os
    import sys
    import pytz
    import numpy as np
    import argparse
    from xml.dom import minidom
    import lxml.objectify as objectify
    import Read_VOalerts as rvo


    #PHOTOMETRY
    initxml="./VOEVENTS/GRANDMA20190204_GWMS181101ab_TCH_a.xml"

    dic=VO_dic()

    NewVOEvent(dic,initxml,"photo")



