import matplotlib.pyplot as plot
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Podzielenie zbioru na część uczącą i testującą
data = pandas.read_csv("data/yeast.csv")
data.columns = ['Name', 'mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc']

data["nuc"] = pandas.factorize(data["nuc"])[0] + 1

X = data.iloc[:, 0:7].values
y = data.iloc[:, 8].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

metric = 'euclidean'
print(metric)

for k in range(3, 7):
    kf = KFold(n_splits=k, random_state=None)
    classifier = KNeighborsClassifier(n_neighbors=5, metric=metric)
    acc = cross_val_score(classifier, X, y, cv=kf)

    print(' k = {} | '.format(k) + 'Fold accuracy: {} | '.format(acc) + 'Average accuracy: {}'.format(acc.mean()))

metric = 'cosine'
print(metric)

for k in range(3, 7):
    kf = KFold(n_splits=k, random_state=None)
    classifier = KNeighborsClassifier(n_neighbors=5, metric=metric)
    acc = cross_val_score(classifier, X, y, cv=kf)

    print(' k = {} | '.format(k) + 'Fold accuracy: {} | '.format(acc) + 'Average accuracy: {}'.format(acc.mean()))
