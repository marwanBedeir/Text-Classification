import nltk
nltk.download("gutenberg")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download('wordnet')

from nltk.corpus import gutenberg
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
import random
stop_words = stopwords.words('english')
COLUMNS_NAMES = ['clean_text', 'book_name', 'book_id']


def get_random_sample(lst, sample_size):
    random_index = random.randint(0, len(lst) - (sample_size + 2))
    return lst[random_index:random_index + sample_size]


def get_prepared_data(number_of_books=18, list_of_books=None, number_of_samples_per_book=200,
                      number_of_words_per_sample=100, rm_stop_words=True, stemming=True, lemmatisation=True):
    all_books_names = get_books_names()
    books_names = []

    if list_of_books:
        for book_id, book_name in enumerate(all_books_names):
            if book_name in list_of_books:
                books_names.append((book_id, book_name))
    else:
        books_names = [(book_id, book_name) for book_id, book_name in enumerate(all_books_names[:number_of_books])]

    # Create an empty DataFrame which will hold our data set.
    df = pd.DataFrame(columns=COLUMNS_NAMES)
    for book_id, book_name in books_names:
        # Get all words for a specific book.
        words = gutenberg.words(book_name)

        # Clean the data remove: (numbers, special characters) and make words in lower case.
        cleaned_words = [word.lower() for word in words if re.match("\w", word)]

        # Remove stopwords if needed
        if rm_stop_words:
            cleaned_words = [word for word in cleaned_words if word not in stop_words]

        # Do stemming if needed
        if stemming:
            ps = PorterStemmer()
            cleaned_words = [ps.stem(word) for word in cleaned_words]

        # Do lemmatisation if needed
        if lemmatisation:
            lem = WordNetLemmatizer()
            cleaned_words = [lem.lemmatize(word) for word in cleaned_words]

        # Create random samples.
        samples = []
        for _ in range(number_of_samples_per_book):
            samples.append((" ".join(get_random_sample(cleaned_words, number_of_words_per_sample)), book_name, book_id))

        # Append the new data to the DataFrame.
        df = df.append(pd.DataFrame(samples, columns=COLUMNS_NAMES), ignore_index=True)
    return df


def get_books_names():
    return gutenberg.fileids()
