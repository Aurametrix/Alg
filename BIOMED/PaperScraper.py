from paperscraper import PaperScraper
scraper = PaperScraper()

# print(scraper.extract_from_url("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3418173/"))

print(scraper.extract_from_pmid("22915848"))
