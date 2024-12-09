import re
from datetime import datetime

def extract_closest_date(text):
    # Extract all dates from the text
    dates = re.findall(r'\d{1,2}/\d{1,2}/\d{4}', text)

    # Convert the dates to datetime objects
    date_objects = [datetime.strptime(date, '%m/%d/%Y') for date in dates]

    # Filter out dates more recent than 2021
    date_objects = [date for date in date_objects if date.year <= 2021]

    # Find the date closest to 2021
    if date_objects:
        closest_date = max(date_objects, key=lambda date: (date.year, date.month, date.day))
        return closest_date.strftime('%m/%d/%Y')
    else:
        return None

text = "5/30/2024, $0, 474, 290, , -, -, 10/1/2021, $292,000, 440, 497, I - IMPROVED, WD - WARRANTY DEED, A - ACCEPTED, 8/4/2008, $201,000, 329, 111, I - IMPROVED, WD - WARRANTY DEED, A - ACCEPTED, 10/29/2004, $144,900, 289, 501, I - IMPROVED, WD - WARRANTY DEED, A - ACCEPTED, 6/25/1993, $126,000, 206, 518, I - IMPROVED, WD - WARRANTY DEED, A - ACCEPTED, 11/4/1989, $107,000, 183, 333, I - IMPROVED, WD - WARRANTY DEED, H - BUSINESS/CORPORATE SALE, 3/23/1988, $20,000, 172, 759, V - VACANT, WD - WARRANTY DEED, H - BUSINESS/CORPORATE SALE, 12/16/1985, $0, 158, 656, , -, -"
print(extract_closest_date(text))
