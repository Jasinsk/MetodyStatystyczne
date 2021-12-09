import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.dates as mdates

data = pd.read_csv("data/owid-covid-data.csv")
data = data.loc[data['location'] == 'Poland']

# I
df_worthness = data.isna().sum()

data = data[['date', 'new_cases', 'new_deaths']]
meta = data.describe()

# II
data_knn = data.copy()
data_knn.iloc[:, 1:] = KNNImputer(n_neighbors=3).fit_transform(data.iloc[:, 1:])
data_knn['date'] = pd.to_datetime(data['date'])

# III
for (column_name, column_data) in data_knn.iloc[:, 1:].iteritems():
    plt.plot(data_knn['date'], column_data)
    plt.scatter(data_knn['date'][column_data[np.abs(stats.zscore(column_data)) > 3].index], column_data[np.abs(stats.zscore(column_data)) > 3], color = "r")
    plt.title(column_name)
    plt.show()

# IV
clf = LocalOutlierFactor(n_neighbors=2)
mask = clf.fit_predict(data_knn.iloc[:, 1:])
for (column_name, column_data) in data_knn.iloc[:, 1:].iteritems():
    plt.plot(data_knn['date'], column_data)
    plt.scatter(data_knn['date'][column_data[mask == -1].index], column_data[mask == -1], color="r")
    plt.title(column_name)
    plt.show()