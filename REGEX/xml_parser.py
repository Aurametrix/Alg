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


#finding interesting elements
#Element.iter() is a method that helps iterate recursively over all the sub-trees below it - children, their children, etc 

for neighbor in root.iter('neighbor'):
   print neighbor.attrib

# Element.findall() finds only elements with a tag which are direct children of the current element. 
# Element.find() finds the first child with a particular tag
# Element.text accesses the element's text content. 
# Element.get() accesses the element's attributes

for country in root.findall('country'):
    rank = country.find('rank').text
    name = country.get('name')
    print name, rank


