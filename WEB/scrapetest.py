# Web Scraping with Python: Collecting Data from the Modern Web. Ryan Mitchell, 2015
# https://books.google.com/books?id=7z_fCQAAQBAJ
# https://www.safaribooksonline.com/library/view/web-scraping-with/9781491910283/ch01.html

from urllib.request import urlopen
html = urlopen("http://pythonscraping.com/pages/page1.html")
print(html.read())
