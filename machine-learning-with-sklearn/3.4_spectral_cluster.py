import matplotlib.pyplot as plt
from sklearn import datasets
X, y = datasets.make_blobs(n_samples=500, n_features=6, centers=5,
                           cluster_std=[0.4, 0.3, 0.4, 0.3, 0.4], random_state=11)

from sklearn.cluster import SpectralClustering
y_pred = SpectralClustering().fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

from sklearn import metrics
print ("Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred))

for index, gamma in enumerate((0.01,0.1,1,10)):
    for index1, k in enumerate((3,4,5,6)):
        plt.subplot(4, 4, index*4 + index1 + 1)
        y_pred = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        print ("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", k,"score:", metrics.calinski_harabaz_score(X, y_pred))

plt.show()