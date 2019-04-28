import pickle
import random
# from __builtin__ import xrange

import numpy as np
import pandas as pd
from sklearn import svm, metrics
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import paired_euclidean_distances, euclidean_distances

def load_data(dataset):

    if dataset == "german":
        data_german = pd.read_csv('german.numer.txt', sep=" ", header=None)
        data_german.columns = ["Target", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "0"]
        data_german = data_german.drop("0", axis=1)
        for i in range(1,25):
            if i >9:
                data_german[str(i)] = data_german[str(i)].apply(lambda x: float(str(x)[3:]))
            else:
                data_german[str(i)] = data_german[str(i)].apply(lambda x: float(str(x)[2:]))
        # print(data_german.head())
        return data_german

    elif dataset == "cod":
        data_cod = pd.read_csv('cod-rna.txt', sep=" ", header=None)
        data_cod.columns = ["Target", "1", "2", "3", "4", "5", "6", "7", "8"]
        for i in range(1,9):
            data_cod[str(i)] = data_cod[str(i)].apply(lambda x: float(str(x)[2:]))

        # print(data_cod.head())
        return data_cod


def plot_decision_surface_sklearn(clf, X, y):
    X0 = X[np.where(y == 0)]
    X1 = X[np.where(y == 1)]


    print(X.shape[1])
    plt.figure()

    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)

    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])
    plt.scatter(X0[:, 0], X0[:, 1], c='r', s=50)
    plt.scatter(X1[:, 0], X1[:, 1], c='b', s=50)
    plt.show()


def Main():
    data = load_data("cod")

    train_values = data.sample(frac=0.6).astype(float)
    test_values = data.sample(frac=0.4).astype(float)
    y_train = train_values['Target'].values.astype(int)
    y_test = test_values['Target'].values.astype(int)
    train_values = train_values.drop('Target', axis=1)
    test_values = test_values.drop('Target', axis=1)

    train_values = train_values.as_matrix().astype(float)
    test_values = test_values.as_matrix().astype(float)

    # For 2d representation
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(train_values)
    pca_2d = pca.transform(train_values)

    print(data.head())

    # model = svm.SVC(kernel='linear')
    # model.fit(train_values, y_train)
    #
    filename = 'cod_svm.sav'
    # pickle.dump(model, open(filename, 'wb'))

    model = pickle.load(open(filename, 'rb'))
    train_predicted_values = model.predict(train_values)
    # print(y_train)
    # print(len(train_predicted_values))

    accuracy = metrics.accuracy_score(y_train, train_predicted_values) * 100
    print('Training Accuracy {0:0.4f}%'.format(accuracy))

    test_predicted_values = model.predict(test_values)

    x1_train = []
    x0_train = []
    # print(train_values)
    for i in range(len(train_values)):
        # print(train_values[i])
        if (y_train[i] == 1):
            x1_train.append(train_values[i])
        else:
            x0_train.append(train_values[i])

    x0_test = []
    x1_test = []
    for i in range(len(test_values)):
        if (test_predicted_values[i] == 1):
            x1_test.append(test_values[i])
        else:
            x0_test.append(test_values[i])

    # print("Finished x_0 {} and \n x_1{}".format(x0_train, x1_train))

    # Distances from the decision boundar for points of each class
    d0 = model.decision_function(x0_train)
    d1 = model.decision_function(x1_train)

    dists1 = np.array(d1)
    dists0 = np.array(d0)

    absdists1 = np.abs(dists1)
    absdists0 = np.abs(dists0)
    # print(absdists0)

    # sorted order of points that are closest to the decision boundary from each class
    req1 = np.argsort(absdists1)
    req0 = np.argsort(absdists0)


    print()

    # /print("req1: {}".format(req1))
    # top 5 points from each class to the decision boundary
    a1_pts = [x for x in req1[:10]]
    a1_dist = [absdists1[x] for x in req1[:10]]
    print("a1 pts {} \n {}".format( a1_pts, a1_dist))
    a1 = [x1_train[x] for x in req1[:10]]
    a0 = [x0_train[x] for x in req0[:10]]

    a1_pts = [1727, 7503, 2593, 586, 7286, 7527, 6548, 10086, 10902, 8134]
    a1 = [train_values[x] for x in a1_pts]
    # print("closest points: {}".format(a1))
    # my_randoms = [17203, 8555, 6896, 4614, 23141, 12962, 7415, 3653, 733, 33672]
    # rand_pts = [train_values[x] for x in my_randoms]
    # print(my_randoms)
    # Running only for points in train_values
    dist_pts = euclidean_distances(a1, train_values)
    # print("closest: {}".format(dist_pts))
    close = []
    for x in dist_pts:
        np_x = np.array(x)
        re = np.argsort(np_x)
        close.append(re[:20]) #NUMBER OF NEIGHBOOURS CONSIDERED
        # print(re)

    # Check if test_predicts will have same index as x_test in this case
    close_preds = []
    # print(train_predicts)
    # classes of the top k points
    i =0
    for x in close:
        # print(x, a1_pts[i])
        check = []
        for t in x:
            # if i == 4:
            #     # print("y_tra", y_train[t], t)
            #     if y_train[t] == 1:
            #         print("Printing points 1")
            #         p1 = plt.scatter(pca_2d[t, 0], pca_2d[t, 1], c='y', marker='o', zorder = 10)
            #         # print(type(p1))
            #     else:
            #         print("Printing points")
            #         p2 = plt.scatter(pca_2d[t, 0], pca_2d[t, 1], c='b', marker='o', zorder = 10)
            check.append(y_train[t])
        #     print(test_predicts[t])
        # print(check)

        close_preds.append(check)
        # print(len(close_preds))
        i+=1

    i=-1
    fair =0
    unfair =0
    for pts in close_preds:
        i += 1
        n1 = 0
        # t = a1_pts[i]
        for x in pts:
            if x == 1:
                n1 += 1
        # print(n1)
        if n1 >= len(pts) / 2:
            # print("This point is fairly classified: {}".format(pts))
            fair +=1
            # p1 = plt.scatter(pca_2d[t, 0], pca_2d[t, 1], c='y', marker='o', zorder=10)
        else:
            # print("This point s unfairly classified {}".format(pts))
            unfair += 1
            # p2 = plt.scatter(pca_2d[t, 0], pca_2d[t, 1], c='b', marker='o', zorder=10)

    print("Number of Neighbours = 20 \n"
          "Number of points fairly classified: {}\n Number of points Unfairly Classified {}".format(fair,unfair))

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(train_values)
    pca_2d = pca.transform(train_values)
    # print(pca_2d)
    # plt.scatter(train_values[:, 0], train_values[:, 1], c=y_train, s=50, cmap='autumn')
    # print(pca_2d.shape[0])
    for i in range(0, pca_2d.shape[0],200):
        # print(i)
        if y_train[i] == 1:
            c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+', zorder = 0)
        elif y_train[i] == -1:
            c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='x', zorder =5)

    # print("p1 p2: ", p1, p2)
    # ax = plt.axes(autoscale_on=True)
    plt.legend([c1, c2], ['Positive', 'Negative', 'Fairly Classified Points', 'Unfairly Classified Points'])
    print(y_train.shape[0])

    pl.title('cod-rna dataset : 5 Neighbours')
    # plt.show()
if __name__ == '__main__':
    Main()