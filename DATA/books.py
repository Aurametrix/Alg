import re
from collections import Counter
import operator
# clean special characters
text_clean = [re.sub(r"[^a-zA-Z0-9]+", ' ', k)  for t in text for k in t.split("\n")]
# count occurrences of bigrams in different posts
countsb = Counter()
words = re.compile(r'\w+')
for t in text_clean:
    w = words.findall(t.lower())
    countsb.update(zip(w,w[1:]))
# sort results
bigrams = sorted(
    countsb.items(),
    key=operator.itemgetter(1),
    reverse=True
)
