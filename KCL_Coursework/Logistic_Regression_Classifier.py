import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics

# Final model : Constructing a logistic regression classifier
# Back to the iris dataset
# Start with the sklearn approach

import sklearn.datasets as data
import sklearn.linear_model as linear_model

iris = data.load_iris()
X = iris.data[:, :2]
y = iris.target
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)

# Scoring
print('Logistic Regression Classifier model example')
print('Training score = ' + str(clf.score(X_train, y_train)))
print('Test score = ' + str(clf.score(X_test, y_test)))
print('Accuracy score = ' + str(metrics.accuracy_score(y_test, y_hat)))
cm = metrics.confusion_matrix(y_test, y_hat)
print('Confusion matrix '+ str(cm))
precision = metrics.precision_score(y_test, y_hat, average=None)
print('Precision '+ str(precision))
recall = metrics.recall_score(y_test, y_hat, average=None)
print('Recall '+ str(recall))
f1 = metrics.f1_score(y_test, y_hat, average=None)
print('f1 score '+ str(f1))

# Plot the decision boundaries

min_s_length = 4.2
max_s_length = 8.0
min_s_width = 1.9
max_s_width = 4.5

X_pairs = []
x0_range = np.arange(min_s_length, max_s_length, 0.01)
x1_range = np.arange(min_s_width, max_s_width, 0.01)
for i in range(len(x1_range)):
    for j in range(len(x0_range)):
        X_pairs.append([x0_range[j], x1_range[i]])
y_hat_pairs = clf.predict(X_pairs)
print('mesh score = ' + str(clf.score(X_pairs, y_hat_pairs)))
x0_mesh, x1_mesh = np.meshgrid(x0_range, x1_range)
y_hat_mesh = y_hat_pairs.reshape(x0_mesh.shape)
plt.pcolormesh(x0_mesh, x1_mesh, y_hat_mesh, shading='auto')
plt.set_cmap('Blues')
plt.show()

# Plot the ROC curve
import sklearn.preprocessing as preprocess

num_classes = 3
conf_scores = clf.decision_function(X_pairs)
y_binary = preprocess.label_binarize(y_hat_pairs, classes=sorted(set(y)))
fpr = dict()
tpr = dict()
for c in range(0, num_classes):
    fpr[c], tpr[c], tmp = metrics.roc_curve(y_binary[:, c], conf_scores[:, c])
for c in range(0, num_classes):
    plt.plot(fpr[c], tpr[c], label=iris.target_names[c])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()