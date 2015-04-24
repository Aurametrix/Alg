import xml.etree.ElementTree as ET
tree = ET.parse('sample.xml')
root = tree.getroot()

# reading from a string
# root = ET.fromstring(sample_as_string)

for child in root:
     print child.tag, child.attrib

# accessing specific children by index
print "2nd sub-element of the first element:", root[0][1].text
print "3rd sub-element of the second element:", root[1][2].text

#e = xml.etree.ElementTree.parse('sample.xml').getroot()


#for atype in e.findall('type'):
#    print(atype.get('foobar'))
