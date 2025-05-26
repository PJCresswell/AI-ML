import numpy as np
import pandas as pd
from pylab import mpl, plt

# Load the data
raw = pd.read_csv('datasets/london-borough-profiles-jan2018.csv', encoding='ISO-8859-1')
x = raw['Male life expectancy, (2012-14)'].to_numpy()
y = raw['Female life expectancy, (2012-14)'].to_numpy()
#plt.scatter(x[2:], y[2:])
#plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x[2:], y[2:], test_size=0.1)
new_X_train = X_train.reshape(-1, 1)
new_X_test = X_test.reshape(-1, 1)

from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
x, y = datasets.make_classification(n_features=1, n_redundant=0, n_informative=1, n_classes=2, n_clusters_per_class=1, n_samples=100)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
per = linear_model.Perceptron()
per.fit(X_train, y_train)
y_hat = per.predict(X_test)
print('Perceptron accuracy : %f' % (metrics.accuracy_score(y_test, y_hat, normalize=True)))
y_plot = np.zeros((len(y_test), 1))
for j in range(0, len(X_test)):
    y_plot[j] = per.intercept_ + X_test[j] * per.coef_[0,0]
plt.scatter(X_test, X_test, c=y_test)
plt.plot(X_test, y_plot)
plt.show()