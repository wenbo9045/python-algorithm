import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，
# 簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]],
                  cluster_std=[0.4, 0.2, 0.2, 0.2],
                  random_state =9)
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()

from sklearn.cluster import Birch
y_pred = Birch(n_clusters = 4, threshold = 0.3, branching_factor = 10).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
from sklearn import metrics
print ("Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred))

from sklearn.cluster import DBSCAN
y_pred = DBSCAN(eps = 0.3, min_samples = 100).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()