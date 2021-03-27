### https://bart.degoe.de/building-a-full-text-search-engine-150-lines-of-code/
### https://github.com/bartdegoede/python-searchengine/

# index.py
import math

def document_frequency(self, token):
    return len(self.index.get(token, set()))

def inverse_document_frequency(self, token):
    # Manning, Hinrich and Schütze use log10, so we do too, even though it
    # doesn't really matter which log we use anyway
    # https://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html
    return math.log10(len(self.documents) / self.document_frequency(token))

def rank(self, analyzed_query, documents):
    results = []
    if not documents:
        return results
    for document in documents:
        score = 0.0
        for token in analyzed_query:
            tf = document.term_frequency(token)
            idf = self.inverse_document_frequency(token)
            score += tf * idf
        results.append((document, score))
    return sorted(results, key=lambda doc: doc[1], reverse=True)
