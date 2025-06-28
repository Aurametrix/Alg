# EuropePMC XML to Dimensions.AI type CSV 

import xml.etree.ElementTree as ET
import pandas as pd

# Set your file names here
input_file = 'europepmc.xml'
output_file = 'europepmc.csv'

def parse_xml(input_file):
    tree = ET.parse(input_file)
    root = tree.getroot()
    records = []
    for result in root.findall('.//result'):
        rec = {
            'DOI': result.findtext('doi', 'N/A'),
            'PMID': result.findtext('pmid', 'N/A'),
            'PMCID': result.findtext('pmcid', 'N/A'),
            'Title': result.findtext('title', 'N/A'),
            'Abstract/Summary': result.findtext('abstractText', 'N/A'),
            'Acknowledgements': 'N/A',
            'Funding': '; '.join([
                f"{g.findtext('grantId','N/A')}: {g.findtext('agency','N/A')}"
                for g in result.findall('.//grantsList/grant')
            ]) or 'N/A',
            'Source title': result.findtext('journalInfo/journal/title') 
                            or result.findtext('bookOrReportDetails/publisher','N/A'),
            'Anthology title': 'N/A',
            'Book editors': 'N/A',
            'MeSH terms': '; '.join([
                mh.findtext('descriptorName','') 
                for mh in result.findall('.//meshHeadingList/meshHeading')
            ]) or 'N/A',
            'Publication date': result.findtext('firstPublicationDate','N/A'),
            'PubYear': result.findtext('pubYear','N/A'),
            'Publication date (online)': result.findtext('electronicPublicationDate','N/A'),
            'Publication date (print)': result.findtext('printPublicationDate','N/A'),
            'Volume': result.findtext('journalInfo/volume','N/A'),
            'Issue': result.findtext('journalInfo/issue','N/A'),
            'Pagination': result.findtext('pageInfo','N/A'),
            'Open Access': result.findtext('isOpenAccess','N/A'),
            'Publication Type': '; '.join([
                pt.text for pt in result.findall('.//pubTypeList/pubType')
            ]) or 'N/A',
            'Authors': '; '.join([
                a.findtext('fullName','') 
                for a in result.findall('.//authorList/author')
            ]) or 'N/A',
            'Authors (Raw Affiliation)': '; '.join([
                f"{a.findtext('fullName')}: " + '; '.join([
                    aff.text for aff in a.findall('.//affiliation') if aff.text
                ])
                for a in result.findall('.//authorList/author')
                if a.findall('.//affiliation')
            ]) or 'N/A',
            'Corresponding Authors': 'N/A',
            'Authors Affiliations': '; '.join([
                aff.text for aff in result.findall('.//affiliation') 
                if aff.text
            ]) or 'N/A',
            'Times cited': result.findtext('citedByCount','N/A'),
            'Recent citations': 'N/A',
            'RCR': 'N/A',
            'FCR': 'N/A',
            'Source Linkout': next((u.text for u in result.findall('.//fullTextUrlList/fullTextUrl/url')), 'N/A'),
        }
        records.append(rec)
    return records

# Run the parser and export to CSV
records = parse_xml(input_file)
df = pd.DataFrame(records)
df.to_csv(output_file, index=False)
print(f'Exported {len(records)} records to {output_file}')

