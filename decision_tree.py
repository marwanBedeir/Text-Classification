from sklearn.tree import DecisionTreeClassifier
import data_preparation as dp
import transformation as trans
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    data = dp.get_prepared_data(number_of_books=5)
    # transformer = tr.tf_idf(data.clean_text)
    # X_train, X_test, y_train, y_test = train_test_split(data.clean_text, data.book_id, test_size=0.3)
    x = data.clean_text
    y = data.book_id
    acc_score = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        tf_idf = trans.tf_idf(X_train)
        x_train_trans = tf_idf.transform(X_train)
        x_test_trans = tf_idf.transform(X_test)

        Encoder = LabelEncoder()
        y_train = Encoder.fit_transform(y_train)
        y_test = Encoder.fit_transform(y_test)

        model = DecisionTreeClassifier(random_state=0)
        model.fit(x_train_trans, y_train)
        y_pred = model.predict(x_test_trans)
        acc = accuracy_score(y_pred, y_test)
        acc_score.append(acc)

    avg_acc_score = sum(acc_score) / 10

    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))
