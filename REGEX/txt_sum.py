# Creating a dictionary for the word frequency table
frequency_table = _create_dictionary_table(article)

# Tokenizing the sentences
sentences = sent_tokenize(article)

# Algorithm for scoring a sentence by its words
sentence_scores = _calculate_sentence_scores(sentences, frequency_table)

# Getting the threshold
threshold = _calculate_average_score(sentence_scores)

# Producing the summary
article_summary = _get_article_summary(sentences, sentence_scores, 1.5 * threshold)

print(article_summary)
