import matplotlib.pyplot as plot
import pandas
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# Import danych
data = pandas.read_csv("data/Seed_Data.csv")
data.columns = ['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry', 'length_groove', 'variety']

# Klastryzacja hierarchiczna
aggloclust = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(data.iloc[:, :7])

plot_dendrogram(aggloclust, truncate_mode="level", p=5)
plot.show()

# Uśrednianie wartości atrybutów wewnątrz klastrów
cluster = AgglomerativeClustering(n_clusters=3).fit_predict(data.iloc[:, :-1])
mean_cluster = data.iloc[:, :-1].groupby(cluster).mean()

# Klastryzacja po zmniejszeniu wymiarowości
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data.iloc[:, :-1])

plot.scatter(data_pca[:, 0], data_pca[:, 1], c=data['variety'])
plot.show()

plot.scatter(data_pca[:, 0], data_pca[:, 1], c=cluster)
plot.show()

# Powtórzenie zadania dla klastryzacji k-means
k_means = KMeans(n_clusters=3)
clustering_k_means = k_means.fit_predict(data.iloc[:, :-1])
means_kmeans = data.iloc[:, :-1].groupby(clustering_k_means).mean()

plot.scatter(data_pca[:, 0], data_pca[:, 1], c=data['variety'])
plot.show()

plot.scatter(data_pca[:, 0], data_pca[:, 1], c=clustering_k_means)
plot.show()