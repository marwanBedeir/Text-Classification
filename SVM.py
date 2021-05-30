import data_preparation as dp
import transformation as tr
from sklearn.metrics import accuracy_score
from sklearn import model_selection, svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


def SVM_TIDF():
    df = dp.get_prepared_data()
    df.sample(frac=1)
    TF_IDF_Transformer = tr.tf_idf(df)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(df['clean_text'],df['book_id'],test_size=0.2)
    Encoder = LabelEncoder()
    y_train = Encoder.fit_transform(y_train)
    y_test = Encoder.fit_transform(y_test)
    TF_IDF_Transformer.fit(df['clean_text'])
    x_train_with_tfidf = TF_IDF_Transformer.transform(x_train)
    x_test_with_tfidf = TF_IDF_Transformer.transform(x_test)
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    scores = cross_val_score(SVM, x_train_with_tfidf, y_train, cv=10)
    SVM.fit(x_train_with_tfidf,y_train)
    predictions_SVM = SVM.predict(x_test_with_tfidf)
    print("SVM Accuracy With TF-IDF IS :  ",accuracy_score(predictions_SVM, y_test)*100)
    print(classification_report(predictions_SVM , y_test))
    print("Cross validation score for TF-IDF is : " ,scores )



def SVM_BoW():
    df = dp.get_prepared_data()
    df.sample(frac=1)
    bag = tr.bag_of_words(df)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(df['clean_text'],df['book_id'],test_size=0.2)
    Encoder = LabelEncoder()
    y_train = Encoder.fit_transform(y_train)
    y_test = Encoder.fit_transform(y_test)
    bag.fit(df['clean_text'])
    x_train_with_tfidf = bag.transform(x_train)
    x_test_with_tfidf = bag.transform(x_test)
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    scores = cross_val_score(SVM, x_train_with_tfidf, y_train, cv=10)
    SVM.fit(x_train_with_tfidf,y_train)
    predictions_SVM = SVM.predict(x_test_with_tfidf)
    print("SVM Accuracy With BoW IS :  ",accuracy_score(predictions_SVM, y_test)*100)
    print(classification_report(predictions_SVM , y_test))
    print("Cross validation score for BoW is : " ,scores )