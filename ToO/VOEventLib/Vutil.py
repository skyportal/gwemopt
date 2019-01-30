import sys
from . import VOEvent


class VOEventExportClass(VOEvent.VOEvent):
    def __init__(self, event, schemaURL):
        self.event = event
        self.schemaURL = schemaURL

    def export(self, outfile, level, namespace_='', name_='VOEvent', namespacedef_=''):
        VOEvent.showIndent(outfile, level)
        added_stuff = 'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
        added_stuff += 'xmlns:voe="http://www.ivoa.net/xml/VOEvent/v2.0"\n'
        added_stuff += 'xsi:schemaLocation="http://www.ivoa.net/xml/VOEvent/v2.0 %s"\n' % self.schemaURL

        outfile.write('<%s%s%s %s' % (namespace_, name_,
            namespacedef_ and ' ' + namespacedef_ or '',
            added_stuff,
            ))
#        self.event.exportAttributes(outfile, level, [], namespace_)
        self.event.exportAttributes(outfile, level, [])
        if self.event.hasContent_():
            outfile.write('>\n')
#            self.event.exportChildren(outfile, level + 1, namespace_='', name_)
            self.event.exportChildren(outfile, level + 1, '', name_)
            VOEvent.showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write('/>\n')

try:
    from io import StringIO
except ImportError:
    from io import StringIO

def stringVOEvent(event, schemaURL = "http://www.ivoa.net/xml/VOEvent/VOEvent-v2.0.xsd"):
    '''
    Converts a VOEvent to a string suitable for output
    '''
    v = VOEventExportClass(event, schemaURL)
    out = StringIO()

    out.write('<?xml version="1.0" ?>\n')
    v.export(out, 0, namespace_='voe:')
    out.write('\n')
    return out.getvalue()

def paramValue(p):
    s1 = p.get_value()
    s2 = p.get_Value()
    if not s2: return s1
    if not s1: return s2
    if len(s1) > len(s2): return s1
    else: return s2

def htmlList(list):
    '''
    Converts a list of strings to an HTML <ul><li> structure.
    '''
    s = '<ul>'
    for x in list:
        s += '<li>' + str(x) + '</li>'
    s += '</ul>'
    return s

def htmlParam(g, p):
    '''
    Builds an HTML table row from a Param and its enclosing Group (or None)
    '''
    s = ''
    if g == None:
        s += '<td/>'
    else:
        s += '<td>' + g.get_name() + '</td>'
    s += '<td>' + str(p.get_name()) + '</td>'
    s += '<td>'
    for d in p.get_Description(): s += str(d)
    s += '</td>'
    s += '<td><b>' + str(paramValue(p)) + '</b></td>'
    s += '<td>' + str(p.get_ucd()) + '</td>'
    s += '<td>' + str(p.get_unit()) + '</td>'
    s += '<td>' + str(p.get_dataType()) + '</td>'
    return s

def parse(file):
    '''
    Parses a file and builds the VOEvent DOM.
    '''
    doc = VOEvent.parsexml_(file)
    rootNode = doc.getroot()
    rootTag, rootClass = VOEvent.get_root_tag(rootNode)
    v = rootClass.factory()
    v.build(rootNode)
    return v

def parseString(inString):
    '''
    Parses a string and builds the VOEvent DOM.
    '''
    from io import StringIO
    doc = VOEvent.parsexml_(StringIO(inString))
    rootNode = doc.getroot()
    rootTag, rootClass = VOEvent.get_root_tag(rootNode)
    rootObj = rootClass.factory()
    rootObj.build(rootNode)
    return rootObj

def getWhereWhen(v):
    '''
    Builds a dictionary of the information in the WhereWhen section:
    observatory: location of observatory (string);
    coord_system: coordinate system ID, for example UTC-FK5-GEO;
    time: ISO8601 representation of time, for example 1918-11-11T11:11:11;
    timeError: in seconds;
    longitude: in degrees, usually right ascension;
    latitiude: in degrees, usually declination;
    positionalError: positional error in degrees.
    '''
    wwd = {}
    ww = v.get_WhereWhen()
    if not ww:
        return wwd
    w = ww.get_ObsDataLocation()
    if not w: 
        return wwd
    ol = w.get_ObservatoryLocation()
    if ol: 
        wwd['observatory'] = ol.get_id()
    ol = w.get_ObservationLocation()
    if not ol:
        return wwd
    observation = ol.get_AstroCoords()
    if not observation: 
        return wwd
    wwd['coord_system'] = observation.get_coord_system_id()
    time = observation.get_Time()
    wwd['time'] = time.get_TimeInstant().get_ISOTime()
    wwd['timeError'] = time.get_Error()

    pos = observation.get_Position2D()
    if not pos:
        return wwd
    wwd['positionalError']  = pos.get_Error2Radius()
    v2 = pos.get_Value2()
    if not v2:
        return wwd
    wwd['longitude'] = v2.get_C1()
    wwd['latitude']  = v2.get_C2()
    return wwd

def makeWhereWhen(wwd):
    '''
    Expects a dictionary of the information in the WhereWhen section, and makes a 
    VOEvent.WhereWhen object suitable for set_WhereWhen().
    observatory: location of observatory (string);
    coord_system: coordinate system ID, for example UTC-FK5-GEO;
    time: ISO8601 representation of time, for example 1918-11-11T11:11:11;
    timeError: in seconds;
    longitude: in degrees, usually right ascension;
    latitiude: in degrees, usually declination;
    positionalError: positional error in degrees.
    '''

    if 'observatory' not in wwd:     wwd['observatory'] = 'unknown'
    if 'coord_system' not in wwd:    wwd['coord_system'] = 'UTC-FK5-GEO'
    if 'timeError' not in wwd:       wwd['timeError'] = 0.0
    if 'positionalError' not in wwd: wwd['positionalError'] = 0.0

    if 'time' not in wwd: 
        print("Cannot make WhereWhen without time")
        return None
    if 'longitude' not in wwd:
        print("Cannot make WhereWhen without longitude")
        return None
    if 'latitude' not in wwd:
        print("Cannot make WhereWhen without latitude")
        return None

    ac = VOEvent.AstroCoords(coord_system_id=wwd['coord_system'])

    ac.set_Time(
        VOEvent.Time(
            TimeInstant = VOEvent.TimeInstant(wwd['time'])))

    ac.set_Position2D(
        VOEvent.Position2D(
            Value2 = VOEvent.Value2(wwd['longitude'], wwd['latitude']),
            Error2Radius = wwd['positionalError']))

    acs = VOEvent.AstroCoordSystem(id=wwd['coord_system'])

    onl = VOEvent.ObservationLocation(acs, ac)
    oyl = VOEvent.ObservatoryLocation(id=wwd['observatory'])
    odl = VOEvent.ObsDataLocation(oyl, onl)
    ww = VOEvent.WhereWhen()
    ww.set_ObsDataLocation(odl)
    return ww

def getParamNames(v):
    '''
    Takes a VOEvent and produces a list of pairs of group name and param name.
    For a bare param, the group name is the empty string.
    '''
    list = []
    w = v.get_What()
    if not w: return list
    for p in v.get_What().get_Param():
        list.append(('', p.get_name()))
    for g in v.get_What().get_Group():
        for p in v.get_What().get_Param():
            list.append((g.get_Name(), p.get_Name()))
    return list

def findParam(event, groupName, paramName):
    '''
    Finds a Param in a given VOEvent that has the specified groupName
    and paramName. If it is a bare param, the group name is the empty string.
    '''
    w = event.get_What()
    if not w:
        print("No <What> section in the event!")
        return None
    if groupName == '':
        for p in event.get_What().get_Param():
            if p.get_name() == paramName:
                return p
    else:
        for g in event.get_What().get_Group():
            if g.get_Name == groupName:
                for p in event.get_What().get_Param():
                    if p.get_name() == paramName:
                        return p
    print('Cannot find param named %s/%s' % (groupName, paramName))
    return None

######## utilityTable ########################
class utilityTable(VOEvent.Table):
    '''
    Class to represent a simple Table from VOEvent
    '''
    def __init__(self, table):
        self.table = table
        self.colNames = []
        self.default = []
        col = 0
        for f in table.get_Field():
            if f.get_name():
                self.colNames.append(f.get_name())
                type = f.get_dataType()
                if type == 'float': self.default.append(0.0)
                elif type == 'int': self.default.append(0)
                else:               self.default.append('')

    def getTable(self):
        return self.table

    def blankTable(self, nrows):
        '''
        From a table template, replaces the Data section with nrows of empty TR and TD
        '''
        data = VOEvent.Data()
        ncol = len(self.colNames)

        for i in range(nrows):
            tr = VOEvent.TR()
            for col in range(ncol):
                tr.add_TD(self.default[col])
            data.add_TR(tr)
        self.table.set_Data(data)

    def getByCols(self):
        '''
        Returns a dictionary of column vectors that represent the table.
        The key for the dict is the Field name for that column.
        '''
        d = self.table.get_Data()
        nrow = len(d.get_TR())
        ncol = len(self.colNames)

# we will build a matrix nrow*ncol and fill in the values as they
# come in, with col varying fastest.  The return is a dictionary,
# arranged by column name, each with a vector of
# properly typed values.
        data = []
        for col in range(ncol):
            data.append([self.default[col]]*nrow)

        row = 0
        for tr in d.get_TR():
            col = 0
            for td in tr.get_TD():
                data[col][row] = td
                col += 1
            row += 1

        dict = {}
        col = 0
        for colName in self.colNames:
            dict[colName] = data[col]
            col += 1
        return dict

    def setValue(self, name, irow, value, out=sys.stdout):
        '''
        Copies a single value into a cell of the table.
        The column is identified by its name, and the row by an index 0,1,2...
        '''
        if name in self.colNames:
            icol = self.colNames.index(name)
        else:
            print("setTable: Unknown column name %s. Known list is %s" % (name, str(self.colNames)), file=out)
            return False

        d = self.table.get_Data()
        ncols = len(self.colNames)
        nrows = len(d.get_TR())

        if nrows <= irow:
            print("setTable: not enough rows -- you want %d, table has %d. Use blankTable to allocate the table." % (irow+1, nrows), file=out)
            return False

        tr = d.get_TR()[irow]
        row = tr.get_TD()
        row[icol] = value
        tr.set_TD(row)

    def toString(self):
        '''
        Makes a crude string representation of a utilityTable
        '''
        s = ' '
        for name in self.colNames:
            s += '%9s|' % name[:9]
        s += '\n\n'

        d = self.table.get_Data()
        for tr in d.get_TR():
            for td in tr.get_TD():
                s += '%10s' % str(td)[:10]
            s += '\n'
        return s
