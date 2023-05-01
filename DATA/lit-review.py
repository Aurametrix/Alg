# ChatGPT: A Meta-Analysis after 2.5 Months

# paper_crawler_scholar.py

import pandas as pd
import requests, json

# Write papers from Scholar to tsv. Manual pagination, as it only requires 2 calls (less than 200 papers at query time)
url = "https://api.semanticscholar.org/graph/v1/paper/search?query=chatgpt&limit=100&fields=authors,title,abstract,venue,publicationDate"
r = requests.get(url)
content = json.loads(r.content)["data"]

url = "https://api.semanticscholar.org/graph/v1/paper/search?query=chatgpt&limit=100&offset=100&fields=authors,title,abstract,venue,publicationDate"
r = requests.get(url)
content += json.loads(r.content)["data"]

df = pd.DataFrame(content)
df.to_csv("scholar_results_aes", sep="\t")

# paper_crawler_arxiv.py

import requests
import pandas as pd
import xmltodict

# Parse Arxiv Papers into a TSV file. Currently without pagination as there were less arxiv papers at query time
r = xmltodict.parse(requests.get(
    "http://export.arxiv.org/api/query?search_query=all:chatgpt&max_results=50"
).content)["feed"]["entry"]

df = pd.DataFrame(r)
df.to_csv("chatgpt.tsv", sep="\t")
