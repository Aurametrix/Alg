"""
Queries arxiv API and downloads papers (the query is a parameter).
The script is intended to enrich an existing database pickle (by default db.p),
so this file will be loaded first, and then new results will be added to it.
"""

import urllib
import time
import feedparser
import os
import cPickle as pickle
import argparse
import random
import utils

def encode_feedparser_dict(d):
  """ 
  helper function to get rid of feedparser bs with a deep copy. 
  I hate when libs wrap simple things in their own classes.
  """
  if isinstance(d, feedparser.FeedParserDict) or isinstance(d, dict):
    j = {}
    for k in d.keys():
      j[k] = encode_feedparser_dict(d[k])
    return j
  elif isinstance(d, list):
    l = []
    for k in d:
      l.append(encode_feedparser_dict(k))
    return l
  else:
    return d

def parse_arxiv_url(url):
  """ 
  examples is http://arxiv.org/abs/1512.08756v2
  we want to extract the raw id and the version
  """
  ix = url.rfind('/')
  idversion = j['id'][ix+1:] # extract just the id (and the version)
  parts = idversion.split('v')
  assert len(parts) == 2, 'error parsing url ' + url
  return parts[0], int(parts[1])

if __name__ == "__main__":

  # parse input arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--db_path', dest='db_path', type=str, default='db.p', help='database pickle filename that we enrich')
  parser.add_argument('--search_query', dest='search_query', type=str,
                      default='cat:cs.CV+OR+cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CL+OR+cat:cs.NE+OR+cat:stat.ML',
                      help='query used for arxiv API. See http://arxiv.org/help/api/user-manual#detailed_examples')
  parser.add_argument('--start_index', dest='start_index', type=int, default=0, help='0 = most recent API result')
  parser.add_argument('--max_index', dest='max_index', type=int, default=10000, help='upper bound on paper index we will fetch')
  parser.add_argument('--results_per_iteration', dest='results_per_iteration', type=int, default=100, help='passed to arxiv API')
  parser.add_argument('--wait_time', dest='wait_time', type=float, default=5.0, help='lets be gentle to arxiv API (in number of seconds)')
  parser.add_argument('--break_on_no_added', dest='break_on_no_added', type=int, default=1, help='break out early if all returned query papers are already in db? 1=yes, 0=no')
  args = parser.parse_args()

  # misc hardcoded variables
  base_url = 'http://export.arxiv.org/api/query?' # base api query url
  print 'Searching arXiv for %s' % (args.search_query, )

  # lets load the existing database to memory
  try:
    db = pickle.load(open(args.db_path, 'rb'))
  except Exception, e:
    print 'error loading existing database:'
    print e
    print 'starting from an empty database'
    db = {}

  # -----------------------------------------------------------------------------
  # main loop where we fetch the new results
  print 'database has %d entries at start' % (len(db), )
  num_added_total = 0
  for i in range(args.start_index, args.max_index, args.results_per_iteration):

    print "Results %i - %i" % (i,i+args.results_per_iteration)
    query = 'search_query=%s&sortBy=lastUpdatedDate&start=%i&max_results=%i' % (args.search_query,
                                                         i, args.results_per_iteration)
    response = urllib.urlopen(base_url+query).read()
    parse = feedparser.parse(response)
    num_added = 0
    num_skipped = 0
    for e in parse.entries:

      j = encode_feedparser_dict(e)

      # extract just the raw arxiv id and version for this paper
      rawid, version = parse_arxiv_url(j['id'])
      j['_rawid'] = rawid
      j['_version'] = version

      # add to our database if we didn't have it before, or if this is a new version
      if not rawid in db or j['_version'] > db[rawid]['_version']:
        db[rawid] = j
        print 'updated %s added %s' % (j['updated'].encode('utf-8'), j['title'].encode('utf-8'))
        num_added += 1
      else:
        num_skipped += 1

    # print some information
    print 'Added %d papers, already had %d.' % (num_added, num_skipped)

    if len(parse.entries) == 0:
      print 'Received no results from arxiv. Rate limiting? Exiting. Restart later maybe.'
      print response
      break

    if num_added == 0 and args.break_on_no_added == 1:
      print 'No new papers were added. Assuming no new papers exist. Exiting.'
      break

    print 'Sleeping for %i seconds' % (args.wait_time , )
    time.sleep(args.wait_time + random.uniform(0, 3))

  # save the database before we quit
  print 'saving database with %d papers to %s' % (len(db), args.db_path)
  utils.safe_pickle_dump(db, args.db_path)

=======================
# simplest  @NL2G
import requests
import pandas as pd
import xmltodict

# Parse Arxiv Papers into a TSV file. Currently without pagination as there were less arxiv papers at query time
r = xmltodict.parse(requests.get(
    "http://export.arxiv.org/api/query?search_query=all:chatgpt&max_results=50"
).content)["feed"]["entry"]

df = pd.DataFrame(r)
df.to_csv("chatgpt.tsv", sep="\t")
======================================
# Latest @daveshap
import requests
from time import time
from xml.etree import ElementTree as ET


# get search term
a = input("Whatcha wanna lookup? ")


# URL of the XML object
url = "https://export.arxiv.org/api/query?search_query=all:%s&sortBy=lastUpdatedDate&sortOrder=descending&max_results=200" % a.lower().replace(' ','%20')

# Send a GET request to the URL
response = requests.get(url)

# Parse the XML response
root = ET.fromstring(response.content)

# Namespace dictionary to find elements
namespaces = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

# Open the output file with UTF-8 encoding
with open("output-%s-%s.md" % (time(), a), "w", encoding='utf-8') as file:
    # Iterate over each entry in the XML data
    for entry in root.findall('atom:entry', namespaces):
        # Extract and write the title
        title = entry.find('atom:title', namespaces).text
        title = ' '.join(title.split())  # Replace newlines and superfluous whitespace with a single space
        file.write(f"# {title}\n\n")

        # Extract and write the link to the paper
        id = entry.find('atom:id', namespaces).text
        file.write(f"[Link to the paper]({id})\n\n")

        # Extract and write the authors
        authors = entry.findall('atom:author', namespaces)
        file.write("## Authors\n")
        for author in authors:
            name = author.find('atom:name', namespaces).text
            file.write(f"- {name}\n")
        file.write("\n")

        # Extract and write the summary
        summary = entry.find('atom:summary', namespaces).text
        file.write(f"## Summary\n{summary}\n\n")


#### for Dermatology paper
import requests
import pandas as pd
import xmltodict
import urllib.parse

# Define the dermatology-related keywords
dermatology_terms = [
    "Dermatology",
    "skin",
    "hair",
    "nails",
    "dermatitis",
    "acne",
    "vitiligo",
    "alopecia",
    "lichen planus",
    "cosmetics",
    "eczema",
    "malodor",
    "hyperhidrosis",
    "bromhidrosis",
    "olfactory reference syndrome",
    "rosacea",
    "sunburn"
]


# Define the ChatGPT-related keywords
chatgpt_terms = [
    "ChatGPT",
    "chat GPT"
]

# Construct the query string
# arXiv uses a specific query syntax: (term1 OR term2 OR ...) AND (termA OR termB)
dermatology_query = " OR ".join([f'"{term}"' if " " in term else term for term in dermatology_terms])
chatgpt_query = " OR ".join([f'"{term}"' if " " in term else term for term in chatgpt_terms])

# Combine both parts using AND
full_query = f'({dermatology_query}) AND ({chatgpt_query})'

# URL encode the query
encoded_query = urllib.parse.quote(full_query)

# Define the maximum number of results you wish to retrieve
max_results = 100  # Adjust as needed

# Construct the complete API URL
api_url = f'http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={max_results}'

# Make the API request
response = requests.get(api_url)

# Check if the request was successful
if response.status_code != 200:
    raise Exception(f"Error fetching data from arXiv API: {response.status_code}")

# Parse the XML response
try:
    parsed = xmltodict.parse(response.content)
    entries = parsed["feed"]["entry"]
except KeyError:
    # No entries found
    entries = []
except Exception as e:
    raise Exception(f"Error parsing XML: {e}")

# Handle cases where there is only one entry (dict) or multiple entries (list)
if isinstance(entries, dict):
    entries = [entries]

# If no entries are found, notify the user
if not entries:
    print("No results found for the given query.")
else:
    # Extract relevant information from each entry
    data = []
    for entry in entries:
        # Handle the 'category' field which could be a list or a single dictionary
        categories = ""
        if isinstance(entry.get("category", []), list):
            # If it's a list, join all the terms
            categories = ", ".join([cat.get("@term", "") for cat in entry["category"]])
        else:
            # If it's a single dictionary, get the term directly
            categories = entry.get("category", {}).get("@term", "")
        
        # Handle the 'link' field which might also be a list or single dictionary
        links = ""
        if isinstance(entry.get("link", []), list):
            # If it's a list, extract the first relevant link (usually to the PDF or abstract)
            links = next((lnk.get("@href", "") for lnk in entry["link"] if lnk.get("@href", "")), "")
        else:
            links = entry.get("link", {}).get("@href", "")
        
        # Create the entry_data dictionary inside the loop
        entry_data = {
            "Title": entry.get("title", "").replace("\n", " ").strip(),
            "Authors": ", ".join([author["name"] for author in entry.get("author", [])]) if isinstance(entry.get("author", []), list) else entry.get("author", {}).get("name", ""),
            "Summary": entry.get("summary", "").replace("\n", " ").strip(),
            "Published": entry.get("published", ""),
            "Updated": entry.get("updated", ""),
            "Primary Category": entry.get("arxiv:primary_category", {}).get("@term", ""),
            "Categories": categories,  # Use the corrected categories value
            "Link": links  # Use the corrected link value
        }
        data.append(entry_data)

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Save the DataFrame to a TSV file
    output_filename = "dermatology_chatgpt_arxiv.tsv"
    df.to_csv(output_filename, sep="\t", index=False)
    print(f"Successfully saved {len(df)} records to '{output_filename}'.")
