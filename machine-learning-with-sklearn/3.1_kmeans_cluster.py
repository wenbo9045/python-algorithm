import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，
# 簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]],
                  cluster_std=[0.4, 0.2, 0.2, 0.2],
                  random_state =9)
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()

from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

from sklearn import metrics
metrics.calinski_harabaz_score(X, y_pred)


from sklearn.cluster import MiniBatchKMeans

for index, k in enumerate((2,3,4,5)):
    plt.subplot(2,2,index+1)
    y_pred = MiniBatchKMeans(n_clusters=k, batch_size = 200, random_state=9).fit_predict(X)
    score= metrics.calinski_harabaz_score(X, y_pred)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.text(.99, .01, ('k=%d, score: %.2f' % (k,score)), transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
plt.show()

from sklearn.cluster import SpectralClustering
y_pred = SpectralClustering(n_clusters=4, gamma=1).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()