import spacy
nlp = spacy.load('en')

doc = nlp("Tea is healthy and calming, don't you think?")

for token in doc:
    print(token)
    
print(f"Token ttLemma ttStopword".format('Token', 'Lemma', 'Stopword'))
print("-"*40)
for token in doc:
    print(f"{str(token)}tt{token.lemma_}tt{token.is_stop}")
    
# pattern matching
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

terms = ['Galaxy Note', 'iPhone 11', 'iPhone XS', 'Google Pixel']
patterns = [nlp(text) for text in terms]
matcher.add("TerminologyList", patterns)

text_doc = nlp("Glowing review overall, and some really interesting side-by-side "
               "photography tests pitting the iPhone 11 Pro against the "
               "Galaxy Note 10 Plus and last year’s iPhone XS and Google Pixel 3.") 
matches = matcher(text_doc)
match_id, start, end = matches[0]
print(nlp.vocab.strings[match_id], text_doc[start:end])

# matching with Lemma attribute
run_matcher = Matcher(nlp.vocab)
pattern = [{"LEMMA": "run"}]
run_matcher.add("RUN", None, pattern)

doc = nlp("Only when it dawned on him that he had nowhere left to run to, he finally stopped running.")
matches = run_matcher(doc)
for match_id, start, end in matches:
    span = doc[start:end]  
    print(start, end, span.text)
    
# Output
# 12 13 run
# 18 19 running

regex_matcher = Matcher(nlp.vocab)
pattern = [{"LOWER": {"REGEX": "colou?r"}}]
regex_matcher.add("BUY", None, pattern)

doc = nlp("Color is the spelling used in the United States. Colour is used in other English-speaking countries.")
matches = regex_matcher(doc)
for match_id, start, end in matches:
    span = doc[start:end]  
    print(start, end, span.text)
    
# Output
# 0 1 Color
# 10 11 Colour

matcher.add("Product", None, [{"TEXT": {"REGEX": "(?i)CAT"}},{"TEXT":"-"},{"TEXT": {"REGEX": r"(?i)[A-Z]+-\d+"}}])
matches = matcher(doc)
for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]  # Get string representation
    span = doc[start:end]  # The matched span
    print(match_id, string_id, start, end, span.text)

# => 16898055450696666743 Product 6 9 CAT-POS-2299
