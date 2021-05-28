from sklearn import feature_extraction
import gensim
from gensim.models import Word2Vec

def bag_of_words(corpus, ngram_range=(1, 2)):
    transformer = feature_extraction.text.CountVectorizer(ngram_range=ngram_range)
    transformer.fit(corpus)
    return transformer


def tf_idf(corpus, ngram_range=(1, 2)):
    transformer = feature_extraction.text.TfidfVectorizer(ngram_range=ngram_range)
    transformer.fit(corpus)
    return transformer


def w2vec(corpus): 
    transformer = Word2Vec(corpus)
    return transformer