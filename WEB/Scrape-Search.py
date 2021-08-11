import re
import time
import requests
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch

es_client = Elasticsearch(['http://10.0.1.10:9200'])

drop_index = es_client.indices.create(index='blog-sysadmins', ignore=400)
create_index = es_client.indices.delete(index='blog-sysadmins', ignore=[400, 404])

def urlparser(title, url):
    # scrape title
    p = {}
    post = title
    page = requests.get(post).content
    soup = BeautifulSoup(page, 'lxml')
    title_name = soup.title.string

    # scrape tags
    tag_names = []
    desc = soup.findAll(attrs={"property":"article:tag"})
    for x in xrange(len(desc)):
        tag_names.append(desc[x-1]['content'].encode('utf-8'))

    # payload for elasticsearch
    doc = {
        'date': time.strftime("%Y-%m-%d"),
        'title': title_name,
        'tags': tag_names,
        'url': url
    }

    # ingest payload into elasticsearch
    res = es_client.index(index="blog-sysadmins", doc_type="docs", body=doc)
    time.sleep(0.5)

sitemap_feed = 'https://sysadmins.co.za/sitemap-posts.xml'
page = requests.get(sitemap_feed)
sitemap_index = BeautifulSoup(page.content, 'html.parser')
urls = [element.text for element in sitemap_index.findAll('loc')]

for x in urls:
    urlparser(x, x)
    
    
import requests 
from bs4 import BeautifulSoup 
import queue 
from threading import Thread 
 
starting_url = 'https://scrapeme.live/shop/page/1/' 
visited = set() 
max_visits = 100 # careful, it will crawl all the pages 
num_workers = 5 
data = [] 
 
def get_html(url): 
	try: 
		response = requests.get(url) 
		# response = requests.get(url, headers=headers, proxies=proxies) 
		return response.content 
	except Exception as e: 
		print(e) 
		return '' 
 
def extract_links(soup): 
	return [a.get('href') for a in soup.select('a.page-numbers') 
			if a.get('href') not in visited] 
 
def extract_content(soup): 
	for product in soup.select('.product'): 
		data.append({ 
			'id': product.find('a', attrs={'data-product_id': True})['data-product_id'], 
			'name': product.find('h2').text, 
			'price': product.find(class_='amount').text 
		}) 
 
def crawl(url): 
	visited.add(url) 
	print('Crawl: ', url) 
	html = get_html(url) 
	soup = BeautifulSoup(html, 'html.parser') 
	extract_content(soup) 
	links = extract_links(soup) 
	for link in links: 
		if link not in visited: 
			q.put(link) 
 
def queue_worker(i, q): 
	while True: 
		url = q.get() # Get an item from the queue, blocks until one is available 
		if (len(visited) < max_visits and url not in visited): 
			crawl(url) 
		q.task_done() # Notifies the queue that the item has been processed 
 
q = queue.Queue() 
for i in range(num_workers): 
	Thread(target=queue_worker, args=(i, q), daemon=True).start() 
 
q.put(starting_url) 
q.join() # Blocks until all items in the queue are processed and marked as done 
 
print('Done') 
print('Visited:', visited) 
print('Data:', data) 
