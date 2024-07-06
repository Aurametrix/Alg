import requests
from PyPDF2 import PdfFileReader
from io import BytesIO
import pandas as pd
import re
from datetime import datetime, timedelta

# Step 1: Generate list of URLs based on the pattern
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

# Step 3: Parse relevant information
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

# Step 4: Export data to CSV
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
