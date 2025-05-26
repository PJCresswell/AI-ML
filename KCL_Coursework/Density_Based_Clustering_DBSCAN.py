import sklearn.datasets as data
import matplotlib.pyplot as plt
import sklearn.neighbors as neighbor

X, clusters = data._samples_generator.make_blobs(n_samples=1000, n_features=2, cluster_std=1.0)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

import sklearn.cluster as cluster

nn = neighbor.NearestNeighbors(n_neighbors=2, metric='euclidean')
nn.fit(X)
dist, ind = nn.kneighbors(X, n_neighbors=2)
print(dist[:, 1])
result = sorted(dist[:, 1])
plt.plot(result)
plt.show()

db = cluster.DBSCAN(eps=0.4, min_samples=5)
db.fit(X)
plt.scatter(X[:, 0], X[:, 1], c=db.labels_)
plt.show()