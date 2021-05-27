from sklearn import feature_extraction


def bag_of_words(corpus, ngram_range=(1, 2)):
    transformer = feature_extraction.text.CountVectorizer(ngram_range=ngram_range)
    transformer.fit(corpus)
    return transformer


def tf_idf(corpus, ngram_range=(1, 2)):
    transformer = feature_extraction.text.TfidfVectorizer(ngram_range=ngram_range)
    transformer.fit(corpus)
    return transformer
