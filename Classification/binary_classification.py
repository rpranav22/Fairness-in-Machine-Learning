import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def preprocess(data):
    preprocess = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    for feature in preprocess:
        data[feature] = pd.cut(data[feature], 5, labels=(1, 2, 3, 4, 5))
    # data["age"] = pd.cut(data['age'] , 5, labels=(1,2,3,4,5))
    print(data.head())
    return data

def split_data(data):
    msk = np.random.rand(len(data)) < 0.8
    train = data[msk]
    test = data[~msk]

    Y_train = train['target'].to_frame()
    X_train = train.drop('target', axis=1)
    Y_test = test['target'].to_frame()
    X_test = test.drop('target', axis=1)
    return X_train,Y_train,X_test,Y_test

def Main():
    raw_data = pd.read_csv("heart.csv")
    data = preprocess(raw_data)
    classifiers = ['Linear Svm', 'Radial Svm', 'Logistic Regression', 'KNN']

    X_train, Y_train, X_test, Y_test = split_data(data)

    abc4 = []
    print(X_train.head())
    models = [svm.SVC(kernel='linear'), svm.SVC(kernel='rbf'), LogisticRegression(),
              KNeighborsClassifier(n_neighbors=3)]
    for i in models:
        model = i
        model.fit(X_train, Y_train)
        prediction = model.predict(X_test)
        abc4.append(metrics.accuracy_score(prediction, Y_test))

    y = models[0].decision_function(X_train)

    dists = np.array(y)
    raw_data['dist'] = pd.DataFrame(dists)

    absdists = np.abs(dists)
    raw_data['absdist'] = pd.DataFrame(absdists)
    raw_data = raw_data.sort_values(['absdist'])
    print(raw_data.head())
    acc = pd.DataFrame(abc4, index=classifiers)
    print(acc)

if __name__ == '__main__':
    Main()