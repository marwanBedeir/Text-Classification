import matplotlib.pyplot as plt
import data_preparation as dp
import transformation as tr
import nltk

if __name__ == '__main__':
    data_1 = dp.get_prepared_data(number_of_books=2, rm_stop_words=False, stemming=False, lemmatisation=False)
    data_2 = dp.get_prepared_data(number_of_books=2, rm_stop_words=True, stemming=False, lemmatisation=False)
    data_3 = dp.get_prepared_data(number_of_books=2, rm_stop_words=True, stemming=True, lemmatisation=True)

    transformer_1 = tr.bag_of_words(data_1.clean_text, ngram_range=(1, 1))
    transformer_2 = tr.bag_of_words(data_2.clean_text, ngram_range=(1, 1))
    transformer_3 = tr.bag_of_words(data_3.clean_text, ngram_range=(1, 1))

    names = ["without any change", "stopwords removed", "after stemming and lemmatisation"]
    values = [len(transformer_1.vocabulary_), len(transformer_2.vocabulary_), len(transformer_3.vocabulary_)]
    plt.bar(names, values, color='maroon', width=0.4)
    plt.show()

    number_of_words_to_plot = 50

    words = ""
    for sample in data_1.clean_text:
        words += " " + sample

    fd = nltk.FreqDist(words.split(" "))
    fd.plot(number_of_words_to_plot, cumulative=False)

    words = ""
    for sample in data_2.clean_text:
        words += " " + sample

    fd = nltk.FreqDist(words.split(" "))
    fd.plot(number_of_words_to_plot, cumulative=False)

    words = ""
    for sample in data_3.clean_text:
        words += " " + sample

    fd = nltk.FreqDist(words.split(" "))
    fd.plot(number_of_words_to_plot, cumulative=False)
