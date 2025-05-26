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

#plt.scatter(X_train, y_train, color="black")
#plt.scatter(X_test, y_test, color="red")
#plt.show()

#from sklearn import datasets
#a, b, p = datasets.make_regression(n_samples=100, n_features=2, n_informative=1, noise=10, coef=True)
#plt.scatter(a, b)
#plt.show()
#X_train, X_test, y_train, y_test = train_test_split(a, b, test_size=0.1)
#plt.scatter(X_train, y_train, color="black")
#plt.scatter(X_test, y_test, color="red")
#plt.show()

def gradient_descent_2 (M, X, W, y, a):
    for j in range (0, M):
        xval = float(X[j][0])
        y_hat = W[0] + W[1] * xval
        # print('Comparing ' + str(y[j]) + ' and ' + str(y_hat))
        diff = float(y[j]) - y_hat
        W[0] += a * diff * (1/M)
        # print('W0 ' + str(W[0]))
        W[1] += a * diff * xval * (1/M)
        # print('W1 ' + str(W[1]))
    return W

def compute_error(M, X, W, y):
    error = 0
    for j in range (0, M):
        xval = float(X[j][0])
        y_hat = W[0] + W[1] * xval
        # print('Comparing ' + str(y[j]) + ' and ' + str(y_hat))
        diff = float(y[j]) - y_hat
        error += (diff ** 2)
    error = error / M
    return error

def compute_r2(M, X, W, y):
    u = 0
    v = 0
    y_float = np.asarray(y, dtype=float)
    for j in range(0, M):
        y_hat = W[0] + W[1] * float(X[j])
        diff = float(y[j]) - y_hat
        u += diff ** 2
        diff2 = float(y[j]) - np.mean(y_float)
        v += diff2 ** 2
    r2 = 1 - (u / v)
    return r2

M_train = len(X_train)
M_test = len(X_test)
old_weights = np.array([0., 0.])

for i in range (1, 10):
    new_weights = gradient_descent_2(M_train, new_X_train, old_weights, y_train, 0.001)
    # Evaluate the model
    new_error = compute_error(M_test, new_X_test, new_weights, y_test)
    new_r2 = compute_r2(M_test, X_test, new_weights, y_test)
    print('Cycle ' + str(i))
    print('Error : ' + str(new_error))
    print('R2 : ' + str(new_r2))
    old_weights = new_weights

from sklearn import linear_model
from sklearn import metrics
lr = linear_model.LinearRegression()
lr.fit(new_X_train, y_train)
print('Equation : y = ' + str(lr.intercept_) + ' + ' + str(lr.coef_[0]) + ' x')
pred = lr.predict(new_X_test)
error = metrics.mean_squared_error(y_test, pred)
score = metrics.r2_score(y_test, pred)
print('Now using sklearn')
print('Error : ' + str(error))
print('R2 : ' + str(score))