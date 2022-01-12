# Inspired by https://github.com/growth-astro/growth-too-marshal/blob/main/growth/too/gcn.py

import os
import numpy as np
import scipy
import gcn
from urllib.parse import urlparse

from astropy import table
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord

from astropy.coordinates import ICRS
from astropy_healpix import HEALPix, nside_to_level, pixel_resolution_to_nside
import ligo.skymap.io
import ligo.skymap.postprocess
import ligo.skymap.moc
from astropy.coordinates import Angle
from astropy.io import fits


def get_dateobs(root):
    """Get the UTC event time from a GCN notice, rounded to the nearest second,
    as a datetime.datetime object."""
    dateobs = Time(
        root.find(
            "./WhereWhen/{*}ObsDataLocation"
            "/{*}ObservationLocation"
            "/{*}AstroCoords"
            "[@coord_system_id='UTC-FK5-GEO']"
            "/Time/TimeInstant/ISOTime"
        ).text,
        precision=0,
    )

    # FIXME: https://github.com/astropy/astropy/issues/7179
    dateobs = Time(dateobs.iso)

    return dateobs.datetime
    
    
def get_contact(root):
    """Get the Author name from a GCN notice"""
    contact = root.find(
            "./Who/{*}Author"
            "/{*}contactName"
        ).text,
 
    return contact
    
def get_packet(root):
    """Get the packet type from a GCN notice"""
    packet_type = int(root.find(".//Param[@name='Packet_Type']").attrib['value'])
 
    return packet_type
    
    
def get_packetsernum(root):
    """Get the Pkt ser num e.g revision from a GCN notice"""
    pkt_ser_num = int(root.find(".//Param[@name='Pkt_Ser_Num']").attrib['value'])
 
    return pkt_ser_num
    
def get_howdes(root):
    """Get the collaboration name from a GCN notice"""
    how = root.find(
            "./How/{*}Description"
        ).text,
 
    return how
    
def get_gbmlc(root):
				"""Get the LC from a GCN notice of Fermi-GBM"""
				lcgbm = str(root.find(".//Param[@name='LightCurve_URL']").attrib['value'])

				return  lcgbm
    
#SOME ISSUE 
def get_trigname(root):
				"""Get the trig id from a GCN notice of SWIFT/BAT and Fermi/GBM"""
				swiftid = str(root.find(".//Param[@name='TrigID']").attrib['value'])

				return  swiftid
				
def get_swiftname(root):
				"""Get the name from a GCN notice of SWIFT/BAT"""
				swiftname = root.find(
            "./Why/{*}Inference"
            "/{*}Name"
        ).text,
				name=str(swiftname[0])
				if name[-1]!="B":
					name=name+"A"
				return name
    
def get_gbmname(root):
				"""Get GRB name from a GCN notice of Fermi-GBM"""
				lcgbm=get_gbmlc(root)
				gbmname = "GRB"+str(lcgbm).split("/")[9].split("bn")[1]

				return  gbmname

def get_gbmratesnr(root):
				"""Get rate snr from a GCN notice of Fermi-GBM"""
				ratesnr = float(root.find(".//Param[@name='Burst_Signif']").attrib['value'])
				return ratesnr


def get_swiftratesnr(root):
				"""Get rate snr from a GCN notice of Swift-BAT"""
				ratesnr = float(root.find(".//Param[@name='Rate_Signif']").attrib['value'])
				return ratesnr

def get_swiftimagesnr(root):
				"""Get image snr from a GCN notice of Swift-BAT"""
				imagesnr = float(root.find(".//Param[@name='Image_Signif']").attrib['value'])
				return imagesnr
				

def get_defGRB(root):
				"""Get def not GRB from a GCN notice of Swift or Fermi-GBM"""
				defGRB=False
				defnotGRB = root.find(".//Param[@name='Def_NOT_a_GRB']").attrib['value']
				if defnotGRB =="false":
							defGRB=True
				return defGRB


def get_gbmmoondistance(root):
				"""Get moon distance from a GCN notice of Fermi-GBM"""
				moon_distance = root.find(".//Param[@name='MOON_Distance']").attrib['value']
				return moon_distance

def get_gbmsundistance(root):
				"""Get sun distance from a GCN notice of Fermi-GBM"""
				sun_distance = root.find(".//Param[@name='Sun_Distance']").attrib['value']
				return sun_distance

def get_gbmmoonillum(root):
				"""Get moon illumination from a GCN notice of Fermi-GBM"""
				moon_illum = root.find(".//Param[@name='Moon_Illum']").attrib['value']
				return moon_illum

				
def get_swiftdateobs(root):
				"""Get def GRB from a GCN notice of Swift"""
				day = float(root.find(".//Param[@name='Burst_SOD']").attrib['value'])/(24.0*3600.0)
				dateobs = Time(float(root.find(".//Param[@name='Burst_TJD']").attrib['value'])+40000.0+day,format="mjd")
				
				return dateobs.datetime
				
def get_fermidateobs(root):
				"""Get def GRB from a GCN notice of GBM (v2)"""
				day = float(root.find(".//Param[@name='Burst_SOD']").attrib['value'])/(24.0*3600.0)
				dateobs = Time(float(root.find(".//Param[@name='Burst_TJD']").attrib['value'])+40000.0+day,format="mjd")
				
				return dateobs.datetime
				
				
def get_swiftgbmlocalization(root):
				loc = root.find('./WhereWhen/ObsDataLocation/ObservationLocation')
				ra = loc.find('./AstroCoords/Position2D/Value2/C1')
				dec = loc.find('./AstroCoords/Position2D/Value2/C2')
				error = loc.find('./AstroCoords/Position2D/Error2Radius')
				ra_formatted = Angle(float(ra.text) * u.degree).to_string(precision=2, sep=' ', unit=u.hour)
				dec_formatted = Angle(float(dec.text) * u.degree).to_string(precision=1, sep=' ', alwayssign=True)
				return ra,dec,error,ra_formatted,dec_formatted 
				
				
def get_longshort(root):
				longshort = root.find(".//Param[@name='Long_short']").attrib['value']
				return longshort

def get_tags(root):
    """Get source classification tag strings from GCN notice."""
    # Get event stream.
    mission = urlparse(root.attrib['ivorn']).path.lstrip('/')
    yield mission

    # What type of burst is this: GRB or GW?
    try:
        value = root.find("./Why/Inference/Concept").text
    except AttributeError:
        pass
    else:
        if value == 'process.variation.burst;em.gamma':
            yield 'GRB'
        elif value == 'process.variation.trans;em.gamma':
            yield 'transient'

    # LIGO/Virgo alerts don't provide the Why/Inference/Concept tag,
    # so let's just identify it as a GW event based on the notice type.
    notice_type = gcn.get_notice_type(root)
    if notice_type in {
        gcn.NoticeType.LVC_PRELIMINARY,
        gcn.NoticeType.LVC_INITIAL,
        gcn.NoticeType.LVC_UPDATE,
        gcn.NoticeType.LVC_RETRACTION,
    }:
        yield 'GW'

    # Is this a retracted LIGO/Virgo event?
    if notice_type == gcn.NoticeType.LVC_RETRACTION:
        yield 'retracted'

    # Is this a short GRB, or a long GRB?
    try:
        value = root.find(".//Param[@name='Long_short']").attrib['value']
    except AttributeError:
        pass
    else:
        if value != 'unknown':
            yield value.lower()

    # Gaaaaaah! Alerts of type FERMI_GBM_SUBTHRESH store the
    # classification in a different property!
    try:
        value = root.find(".//Param[@name='Duration_class']").attrib['value'].title()
    except AttributeError:
        pass
    else:
        if value != 'unknown':
            yield value.lower()

    # Get LIGO/Virgo source classification, if present.
    classifications = [
        (float(elem.attrib['value']), elem.attrib['name'])
        for elem in root.iterfind("./What/Group[@type='Classification']/Param")
    ]
    if classifications:
        _, classification = max(classifications)
        yield classification

    search = root.find("./What/Param[@name='Search']")
    if search is not None:
        yield search.attrib['value']

def get_gbmskymapini(root):
    # Try error cone
    loc = root.find('./WhereWhen/ObsDataLocation/ObservationLocation')
    if loc is None:
        return None

    ra = loc.find('./AstroCoords/Position2D/Value2/C1')
    dec = loc.find('./AstroCoords/Position2D/Value2/C2')
    error = loc.find('./AstroCoords/Position2D/Error2Radius')

    if None in (ra, dec, error):
        return None

    ra, dec, error = float(ra.text), float(dec.text), float(error.text)
    
    skymap,uniq=from_cone(ra, dec, error)
    
    skymap_tot = {
        'localization_name': None,
        'uniq': None,
        'probdensity': None,
        'distmu': None,
        'distsigma': None,
        'nside': -1,
        'distnorm': None,
        'skymap':skymap,
        'uniq':uniq
    }

    return skymap_tot
    
def get_gbmskymapupd(root):
    # Try Fermi GBM convention (not running on subthreshold convention)
    url = root.find("./What/Param[@name='LocationMap_URL']").attrib['value']
    url = url.replace('http://', 'https://')
    url = url.replace('_locplot_', '_healpix_')
    url = url.replace('.png', '.fit')
    return from_url(url)
    
def get_gbmquicklook(root):
    # Try Fermi GBM convention (not running on subthreshold convention)
    url = root.find("./What/Param[@name='LocationMap_URL']").attrib['value']
    url = url.replace('http://', 'https://')
    url = url.replace('_locplot_', '_lc_')
    url = url.replace('.png', '.gif')
    return url
    
def get_gbmhratio(root):
				"""Get Hardness_Ratio from a GCN notice of Fermi-GBM"""
				hardratio = root.find(".//Param[@name='Hardness_Ratio']").attrib['value']
				return hardratio
				
				
def get_skymap(root, gcn_notice):
    mission = urlparse(root.attrib['ivorn']).path.lstrip('/')

    # Try Fermi GBM convention
    if gcn_notice.notice_type == gcn.NoticeType.FERMI_GBM_FIN_POS:
        url = root.find("./What/Param[@name='LocationMap_URL']").attrib['value']
        url = url.replace('http://', 'https://')
        url = url.replace('_locplot_', '_healpix_')
        url = url.replace('.png', '.fit')
        return from_url(url)

    # Try Fermi GBM **subthreshold** convention. 
    if gcn_notice.notice_type == gcn.NoticeType.FERMI_GBM_SUBTHRESH:
        url = root.find("./What/Param[@name='HealPix_URL']").attrib['value']
        return from_url(url)

    # Try LVC convention
    skymap = root.find("./What/Group[@type='GW_SKYMAP']")
    if skymap is not None:
        children = skymap.getchildren()
        for child in children:
            if child.attrib['name'] == 'skymap_fits':
                url = child.attrib['value']
                break

        return from_url(url)

    retraction = root.find("./What/Param[@name='Retraction']")
    if retraction is not None:
        retraction = int(retraction.attrib['value'])
        if retraction == 1:
            return None

    # Try error cone
    loc = root.find('./WhereWhen/ObsDataLocation/ObservationLocation')
    if loc is None:
        return None

    ra = loc.find('./AstroCoords/Position2D/Value2/C1')
    dec = loc.find('./AstroCoords/Position2D/Value2/C2')
    error = loc.find('./AstroCoords/Position2D/Error2Radius')

    if None in (ra, dec, error):
        return None

    ra, dec, error = float(ra.text), float(dec.text), float(error.text)

    # Apparently, all experiments *except* AMON report a 1-sigma error radius.
    # AMON reports a 90% radius, so for AMON, we have to convert.
    if mission == 'AMON':
        error /= scipy.stats.chi(df=2).ppf(0.95)

    return from_cone(ra, dec, error)


def from_cone(ra, dec, error):
    localization_name = "%.5f_%.5f_%.5f" % (ra, dec, error)

    center = SkyCoord(ra * u.deg, dec * u.deg)
    radius = error * u.deg

    # Determine resolution such that there are at least
    # 16 pixels across the error radius.
    hpx = HEALPix(
        pixel_resolution_to_nside(radius / 16, round='up'), 'nested', frame=ICRS()
    )

    # Find all pixels in the 4-sigma error circle.
    ipix = hpx.cone_search_skycoord(center, 4 * radius)

    # Convert to multi-resolution pixel indices and sort.
    uniq = ligo.skymap.moc.nest2uniq(nside_to_level(hpx.nside), ipix.astype(np.int64))
    i = np.argsort(uniq)
    ipix = ipix[i]
    uniq = uniq[i]

    # Evaluate Gaussian.
    distance = hpx.healpix_to_skycoord(ipix).separation(center)
    probdensity = np.exp(
        -0.5 * np.square(distance / radius).to_value(u.dimensionless_unscaled)
    )
    probdensity /= probdensity.sum() * hpx.pixel_area.to_value(u.steradian)


    skymap=table.Table(
            [np.asarray(uniq), np.asarray(probdensity)],
            names=['UNIQ', 'PROBDENSITY'])

    return skymap,uniq


def from_url(url):
    def get_col(m, name):
        try:
            col = m[name]
        except KeyError:
            return None
        else:
            return col.tolist()


    filename = os.path.basename(urlparse(url).path)
    skymap = ligo.skymap.io.read_sky_map(url, moc=True)
    
    
    
    

    skymap_tot = {
        'localization_name': url,
        'uniq': get_col(skymap, 'UNIQ'),
        'probdensity': get_col(skymap, 'PROBDENSITY'),
        'distmu': get_col(skymap, 'DISTMU'),
        'distsigma': get_col(skymap, 'DISTSIGMA'),
        'nside': -1,
        'distnorm': get_col(skymap, 'DISTNORM'),
        'skymap':skymap
    }

    return skymap_tot


def get_contour(localization):

    # Calculate credible levels.
    prob = localization.flat_2d
    cls = 100 * ligo.skymap.postprocess.find_greedy_credible_levels(prob)

    # Construct contours and return as a GeoJSON feature collection.
    levels = [50, 90]
    paths = ligo.skymap.postprocess.contour(cls, levels, degrees=True, simplify=True)
    center = ligo.skymap.postprocess.posterior_max(prob)
    localization.contour = {
        'type': 'FeatureCollection',
        'features': [
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [center.ra.deg, center.dec.deg],
                },
                'properties': {'credible_level': 0},
            }
        ]
        + [
            {
                'type': 'Feature',
                'properties': {'credible_level': level},
                'geometry': {'type': 'MultiLineString', 'coordinates': path},
            }
            for level, path in zip(levels, paths)
        ],
    }

    return localization
