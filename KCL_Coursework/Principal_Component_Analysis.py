import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

iris = datasets.load_iris()
X_data = iris.data
y_target = iris.target
keys = iris.feature_names

num_features = X_data.shape[1]
the_classes = np.unique(y_target)
num_classes = len(the_classes)
prop_cycle = plt.rcParams['axes.prop_cycle']

plotnum = 1
for row in range(num_features):
    for col in range(num_features):
        axs = plt.subplot(num_features, num_features, plotnum)
        axs.set_xlabel(keys[col], fontsize=8)
        axs.set_ylabel(keys[row], fontsize=8)
        for n, v in zip (range(num_classes), prop_cycle):
            data = X_data[y_target == n]
            plt.plot(data[:, row], data[:, col], '.', color=v['color'])
        plotnum += 1
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.2, random_state=0)
clf = tree.DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
print('Decision Tree Classifier : pre PCA')
print('Test score = ' + str(clf.score(X_test, y_test)))

from sklearn import decomposition
pca = decomposition.PCA(n_components=num_features-2)
pca.fit(X_data)
pca_X = pca.transform(X_data)
pca_num_features = pca_X.shape[1]

plotnum = 1
for row in range(pca_num_features):
    for col in range(pca_num_features):
        axs = plt.subplot(pca_num_features, pca_num_features, plotnum)
        axs.set_xlabel('PC-'+ str(col+1), fontsize=8)
        axs.set_ylabel('PC-'+ str(row+1), fontsize=8)
        for n, v in zip (range(num_classes), prop_cycle):
            data = pca_X[y_target == n]
            plt.plot(data[:, row], data[:, col], '.', color=v['color'])
        plotnum += 1
plt.show()

X_train, X_test, y_train, y_test = train_test_split(pca_X, y_target, test_size=0.2, random_state=0)
clf = tree.DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
print('Decision Tree Classifier : post PCA')
print('Test score = ' + str(clf.score(X_test, y_test)))