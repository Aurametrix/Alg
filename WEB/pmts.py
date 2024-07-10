import requests
from PyPDF2 import PdfFileReader
from io import BytesIO
import pandas as pd
import re
from datetime import datetime, timedelta

# Generate list of URLs based on the pattern
base_url = "..."
date_format = "%m-%d-%y"

# Generate dates for first and third Thursday of each month
def generate_dates(start_year, end_year):
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # First Thursday
            first_day = datetime(year, month, 1)
            first_thursday = first_day + timedelta(days=(3 - first_day.weekday() + 7) % 7)
            dates.append(first_thursday.strftime(date_format))
            
            # Third Thursday
            third_thursday = first_thursday + timedelta(weeks=2)
            dates.append(third_thursday.strftime(date_format))
    return dates

dates = generate_dates(2021, 2024)

# Generate URLs
pdf_urls = [f"{base_url}{date}-Minutes.pdf" for date in dates]

# Step 2: Download PDFs and extract text
def download_pdf(url):
    response = requests.get(url)
    return PdfFileReader(BytesIO(response.content))

def extract_text_from_pdf(pdf):
    text = ""
    for page_num in range(pdf.getNumPages()):
        page = pdf.getPage(page_num)
        text += page.extractText()
    return text


def parse_sf_paragraphs(text):
    pattern = r'SF# \d+:.*?(?=SF# \d+:|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    data = []
    for match in matches:
        lines = match.strip().split('\n')
        owner = lines[0]
        lot = next((line for line in lines if line.startswith("Lot")), "")
        footage = next((line for line in lines if line.startswith("Total Footage")), "")
        contractor = next((line for line in lines if line.startswith("Contractor")), "")
        data.append([ower, lot, footage, contractor])
    return data

def extract_permit_data(text):
    permit_data = []
    date = extract_date(text)
    print(f"Extracted date: {date}")  # Add this line to verify the extracted date
    
    sf_paragraphs = re.findall(r'SF#.*?(?=SF#|\Z)', text, re.DOTALL)
    
    for i, paragraph in enumerate(sf_paragraphs, 1):
        try:
            permit_match = re.search(r'SF# (\d+):', paragraph)
            owner_match = re.search(r'SF# \d+: (.+?)\s+Assign:', paragraph)
            lot_match = re.search(r'Lot (\d+), Block \d+, (.+?);', paragraph)
            address_match = re.search(r'; (\d+.+?)(?=\s+Total Footage:)', paragraph)
            footage_match = re.search(r'Footage: (\d+);', paragraph)
            contractor_match = re.search(r'Contractor: (.+?)(?:;|$)', paragraph)
            
            if all([permit_match, owner_match, lot_match, address_match, footage_match, contractor_match]):
                permit_data.append({
                    'Date': date,
                    'Permit': permit_match.group(1),
                    'Owner': owner_match.group(1).strip(),
                    'Lot': lot_match.group(1),
                    'Neighborhood': lot_match.group(2).strip(),
                    'Address': address_match.group(1).strip(),
                    'Footage': footage_match.group(1),
                    'Contractor': contractor_match.group(1).strip()
                })
            else:
                print(f"Warning: Not all fields found in paragraph {i}")
        except Exception as e:
            print(f"Error processing paragraph {i}: {str(e)}")
            print(f"Paragraph content: {paragraph}")
    
    return permit_data

              
def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Date', 'Permit', 'Owner', 'Lot', 'Neighborhood', 'Address', 'Footage', 'Contractor']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# Export data to CSV
all_data = []
for url in pdf_urls:
    try:
        pdf = download_pdf(url)
        text = extract_text_from_pdf(pdf)
        data = parse_sf_paragraphs(text)
        all_data.extend(data)
    except Exception as e:
        print(f"Failed to process {url}: {e}")

df = pd.DataFrame(all_data, columns=["Ower", "LOT", "Footage", "Contractor"])
df.to_csv("export.csv", index=False)
