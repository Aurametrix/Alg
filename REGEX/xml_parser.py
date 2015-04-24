import xml.etree.ElementTree as ET
tree = ET.parse('sample.xml')
root = tree.getroot()

# reading from a string
# root = ET.fromstring(sample_as_string)

for child in root:
     print child.tag, child.attrib

#e = xml.etree.ElementTree.parse('sample.xml').getroot()


#for atype in e.findall('type'):
#    print(atype.get('foobar'))
