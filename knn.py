import data_preparation as dp
import transformation as tr
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import matplotlib.pyplot as plt


def knn(num_books=5, words_num=100, words_increase_num=20, books_num=200, books_increase_num=None,
                transform_knn="tf-idf", r=1, cv=True):
    scores = []
    words_count = []
    books_count = []
    for i in range(0, r):

        df = dp.get_prepared_data(number_of_books=num_books, number_of_samples_per_book=books_num,
                               number_of_words_per_sample=words_num)

        if (transform_knn == "bow"):
            transformer = tr.bag_of_words(df['clean_text'])

        elif (transform_knn == "tf-idf"):
            transformer = tr.tf_idf(df['clean_text'])

        if words_increase_num:
            words_count.append(words_num)
            words_num += words_increase_num

        if books_increase_num:
            books_count.append(books_num)
            books_num += books_increase_num

        Encoder = LabelEncoder()

        if cv:

            x = transformer.transform(df['clean_text'])
            y = Encoder.fit_transform(df['book_id'])

            knn = KNeighborsClassifier(n_neighbors=5)
            score = cross_val_score(knn, x, y, cv=10)
            scores.append(score.mean())

        else:

            x_train, x_test, y_train, y_test = train_test_split(df['clean_text'], df['book_id'], test_size=0.2)

            y_train = Encoder.fit_transform(y_train)
            y_test = Encoder.fit_transform(y_test)

            x_train_with_transformation = transformer.transform(x_train)
            x_test_with_transformation = transformer.transform(x_test)

            knn = KNeighborsClassifier(n_neighbors=5)

            knn.fit(x_train_with_transformation, y_train)
            # Predict the response for test dataset
            y_pred = knn.predict(x_test_with_transformation)
            scores.append(metrics.accuracy_score(y_pred, y_test))

    return {"words_count": words_count, "scores": scores, "books_count": books_count}


# tring number of k-neighbors to show error rate when k increased
def accurecy_for_k():
    df = dp.get_prepared_data(number_of_books=5, number_of_samples_per_book=200,number_of_words_per_sample=100)
    TF_IDF_Transformer = tr.tf_idf(df)

    x_train, x_test, y_train, y_test = train_test_split(df['clean_text'],df['book_id'],test_size=0.2)

    Encoder = LabelEncoder()
    y_train = Encoder.fit_transform(y_train)
    y_test = Encoder.fit_transform(y_test)

    TF_IDF_Transformer.fit(df['clean_text'])

    x_train_with_tfidf = TF_IDF_Transformer.transform(x_train)
    x_test_with_tfidf = TF_IDF_Transformer.transform(x_test)

    n = 40
    knn_accuracy = []
    # Calculating error for K values between 1 and n
    for i in range(1, n):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train_with_tfidf, y_train)
        y_pred = knn.predict(x_test_with_tfidf)
        knn_accuracy.append(metrics.accuracy_score(y_test, y_pred))

    plt.plot(range(1,n),knn_accuracy)
    plt.title('Accuracy to K Values')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')