import matplotlib.pyplot as plot
import pandas
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Import danych
data = pandas.read_csv("data/Seed_Data.csv")
print(data)

# I - Redukcja wymiarów PCA
pca = PCA(n_components=2)
pca.fit(data.iloc[:, :7].T)

# II - Rysowanie wykresu w zależności od klasy
data_pca = pandas.DataFrame(pca.components_.T)
data_pca.insert(loc=2, column="target", value=data.iloc[:, 7])
print(data_pca)

colors = data_pca["target"][1:].index
plot.scatter(data_pca[0][1:], data_pca[1][1:], c=colors)
plot.show()

# III - Obserwacja wkładu poszczególnych atrybutów
pca2 = PCA(n_components=7)
pca2.fit(data.iloc[:, :7].T)
print(pca2.explained_variance_ratio_)

# [9.82710540e-01 1.24463525e-02 4.57364807e-03 2.30574783e-04 3.41887320e-05 4.69584297e-06 2.67728747e-32]

# IV - Ocena sprawności t-SNE
tsne = TSNE(n_components=2)
tsne_result = pandas.DataFrame(tsne.fit_transform(data.iloc[:, :7]))
tsne_result.insert(loc=2, column="target", value=data.iloc[:, 7])

colors = tsne_result["target"][1:].index
plot.scatter(tsne_result[0][1:], tsne_result[1][1:], c=colors)
plot.show()
