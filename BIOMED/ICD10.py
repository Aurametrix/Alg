import xml.etree.ElementTree as ET
import csv
 
csvwriter = csv.writer(open('diagnostics.csv', 'wb'))
 
tree = ET.parse('ICD10CM_FY2014_Full_XML_Tabular.xml')
root = tree.getroot()
 
for diag in root.iter('diag'):           # Loop through every diagnostic tree
   name = diag.find('name').text.encode('utf8')  # Extract the diag code
   desc = diag.find('desc').text.encode('utf8')  # Extract the description
   csvwriter.writerow((name,desc))       # write to a .csv file
