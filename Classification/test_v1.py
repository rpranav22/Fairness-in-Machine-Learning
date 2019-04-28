import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

def preprocess(data):
    preprocess = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    for feature in preprocess:
        data[feature] = pd.cut(data[feature], 5, labels=(1, 2, 3, 4, 5))
    # data["age"] = pd.cut(data['age'] , 5, labels=(1,2,3,4,5))
    
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

def Clusters(data,k):
    
    dataset = data.values[:,0:-1]
    
    km = KMeans(n_clusters=k).fit(dataset)
    
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, dataset)
    
    a=[]
    
    for x in closest:
        a.append(dataset[x])
    
    return(a)

def Main():
    raw_data = pd.read_csv("heart.csv")
    data = preprocess(raw_data)
    classifiers = ['Linear Svm', 'Radial Svm', 'Logistic Regression', 'KNN']

    X_train, Y_train, X_test, Y_test = split_data(data)

    abc4 = []
    
    models = [svm.SVC(kernel='linear'), svm.SVC(kernel='rbf'), LogisticRegression(),
              KNeighborsClassifier(n_neighbors=3)]
    
    model = models[0]
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    abc4.append(metrics.accuracy_score(prediction, Y_test))
    
    
    X1=pd.DataFrame(columns=list(X_test.columns.values))
    X0=pd.DataFrame(columns=list(X_test.columns.values))
    
    for x in range(len(prediction)):
        if(prediction[x]==1):
           X1 = X1.append(X_test.iloc[x])
        else:
           X0 = X0.append(X_test.iloc[x])
            
    
    Centers1= Clusters(X1,3)
    Centers0= Clusters(X0,3)
    Centers = Centers1.extend(Centers0)
    
    y1=models[0].decision_function(X1)
    y0 = models[0].decision_function(X0)

    dists1 = np.array(y1)
    dists0 = np.array(y0)
    X1['dist'] = pd.DataFrame(dists1)
    X0['dist'] = pd.DataFrame(dists0)

    absdists1 = np.abs(dists1)
    absdists0 = np.abs(dists0)
    X1['abs'] = pd.DataFrame(absdists1)
    X1_near = X1.sort_values(['abs'])[:5]
    X0['abs'] = pd.DataFrame(absdists0)
    X0_near = X0.sort_values(['abs'])[:5]
    X0_near = X0_near.drop(['dist','abs'],axis=1)
    X1_near = X1_near.drop(['dist','abs'],axis=1)
    
    
    
    closest0, _ = pairwise_distances_argmin_min(X0_near.values,Centers)      
    closest1, _ = pairwise_distances_argmin_min(X1_near.values,Centers)
    
    for i in range(5):
        print("Unfair 0:")
        if(closest0[i]<5):
           print(X0_near.iloc[[i]])
        print("Unfair 1:")
        if(closest1[i]>5):
           print(X1_near.iloc[[i]])
        
           
     
    

if __name__ == '__main__':
    Main()
