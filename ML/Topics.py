import pandas as pd
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

df = pd.read_csv('data.csv')

# convert the text to lowercase, tokenize it, remove stopwords, and create a dictionary and corpus for LDA.

stop_words = set(stopwords.words('english'))

def preprocess(text):
   tokens = word_tokenize(text.lower())
   return [token for token in tokens if token.isalpha() and token not in stop_words]

comments = df['Comments'].tolist()
processed_comments = [preprocess(comment) for comment in comments]

dictionary = corpora.Dictionary(processed_comments)
corpus = [dictionary.doc2bow(comment) for comment in processed_comments]

lda_model = models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=2)

# print 5 most important words for each topic
topics = lda_model.print_topics(num_words=5)
for topic in topics:
   print(topic)
