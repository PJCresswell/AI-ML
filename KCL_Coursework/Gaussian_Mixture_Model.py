import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sklearn.mixture as mixture
from sklearn import metrics

def numeric_scalar_diff(a, b):
    #print('Comparing ' + str(a) + ' with ' + str(b))
    result = (a - b) ** 2
    #print('Difference ' + str(result))
    return result

def abs_numeric_scalar_diff(a, b):
    #print('Comparing ' + str(a) + ' with ' + str(b))
    result = abs((a - b) ** 2)
    #print('Difference ' + str(result))
    return result

def nominal_scalar_diff(a, b):
    #print('Comparing ' + str(a) + ' with ' + str(b))
    if a == b:
        result = 0
    else:
        result= 1
    #print('Difference ' + str(result))
    return result

def euclidean_diff(df, a, b):
    cum_dist = 0
    N = len(df.columns)
    column_types = df.dtypes
    for i in range (0, N):
        if column_types[i] == 'int64' :
            dist = numeric_scalar_diff(df.iloc[a, i], df.iloc[b, i])
        else:
            dist = nominal_scalar_diff(df.iloc[a, i], df.iloc[b, i])
        cum_dist = cum_dist + dist
    result = math.sqrt(cum_dist)
    return result

def manhattan_diff(df, a, b):
    cum_dist = 0
    N = len(df.columns)
    column_types = df.dtypes
    for i in range (0, N):
        if column_types[i] == 'int64' :
            dist = abs_numeric_scalar_diff(df.iloc[a, i], df.iloc[b, i])
        else:
            dist = nominal_scalar_diff(df.iloc[a, i], df.iloc[b, i])
        cum_dist = cum_dist + dist
    result = math.sqrt(cum_dist)
    return result

def show_mesh(M, final_grid):
    x0_range = np.arange(M)
    x1_range = np.arange(M)
    x0_mesh, x1_mesh = np.meshgrid(x0_range, x1_range)
    plt.figure()
    plt.set_cmap('Blues')
    plt.pcolormesh(x0_mesh, x1_mesh, final_grid, shading='auto')
    plt.show()

raw = pd.read_csv('datasets/house-prices.csv', encoding='latin-1', date_parser='')
raw.drop(columns='HomeID', inplace=True)

print('Number of instances : ' + str(len(raw)))
print('Number of columns : ' + str(len(raw.columns)))
#for column in raw.columns:
#    print('Attribute : ' + str(column))
#    print('Attribute data type : ' + str(raw[column].dtype))

M = len(raw) + 1
euc_grid = np.zeros((M, M))

for i in range (0, M-1):
    #print('Calculating for row ' + str(i))
    for j in range (0, i+1):
        #print('Calculating for ' + str(i) + ' ' + str(j))
        diff = euclidean_diff(raw, i, j)
        euc_grid[i, j] = diff
        euc_grid[j, i] = diff
#print(final_grid)
#show_mesh(M, euc_grid)

man_grid = np.zeros((M, M))
for i in range (0, M-1):
    #print('Calculating for row ' + str(i))
    for j in range (0, i+1):
        #print('Calculating for ' + str(i) + ' ' + str(j))
        diff = manhattan_diff(raw, i, j)
        man_grid[i, j] = diff
        man_grid[j, i] = diff
#print(final_grid)
#show_mesh(M, man_grid)

#plt.scatter(raw['Price'], raw['SqFt'])
#plt.show()

K = 6
gmm = mixture.GaussianMixture(n_components=K, covariance_type='spherical')
gmm.fit(raw[['Price','SqFt']])

print('Cluster centres')
print(gmm.means_)

labels = gmm.predict(raw[['Price','SqFt']])

plt.scatter(raw['Price'], raw['SqFt'], c=labels)
plt.show()

ll_score = gmm.score(raw[['Price','SqFt']])
sc_score = metrics.silhouette_score(raw[['Price','SqFt']], labels, metric='euclidean')
ch_score = metrics.calinski_harabasz_score(raw[['Price','SqFt']], labels)

print('Performance scores : goal to maximise')
print('Log likelihood score ' + str(ll_score))
print('Silhouette score ' + str(sc_score))
print('Calinski_harabasz score ' + str(ch_score))
