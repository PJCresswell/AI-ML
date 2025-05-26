# https://archive.ics.uci.edu/dataset/267/banknote+authentication

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split

datafilename = 'datasets/data_banknote_authentication.csv'

variance   = 0        # column indexes in input file
skewness   = 1
curtosis   = 2
entropy    = 3
note_class = 4        # this is the thing we are trying to predict

# Since feature names are not in the data file, we code them here
feature_names = [ 'variance', 'skewness', 'curtosis', 'entropy', 'note_class' ]

num_samples = 1372    # size of the data file.
num_features = 4

#
# Open and read data file in csv format
#
# After processing:
#
# data   is the variable holding the features;
# target is the variable holding the class labels.

try:
    with open( datafilename ) as infile:
        indata = csv.reader( infile )
        data = np.empty(( num_samples, num_features ))
        target = np.empty(( num_samples,), dtype=np.int64 )
        i = 0
        for j, d in enumerate( indata ):
            ok = True
            for k in range(0,num_features): # If a feature has a missing value
                if ( d[k] == "?" ):         # we don't use that record.
                    ok = False
            if ( ok ):
                data[i] = np.asarray( d[:-1], dtype=np.float64 )
                target[i] = np.asarray( d[-1], dtype=np.int64 )
                i = i + 1
except IOError as iox:
    print('there was an I/O error trying to open the data file: ' + str( iox ))
    sys.exit()
except Exception as x:
    print('there was an error: ' + str( x ))
    sys.exit()

# How many records do we have?
num_samples = i
print("Number of samples:", num_samples)

# Adjust the size of data and target so that they only hold the values
# loaded from the CSV file

data = data[:num_samples]
target = target[:num_samples]

from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
kernel_functions = ['linear', 'rbf', 'poly', 'sigmoid']

fig, ax = plt.subplots(1, 4, figsize=(15, 4))
plotnum = 0

# columns = data[:, 1:3]

columns = data[:, [1, 3]]

for function in kernel_functions:
    clf1 = svm.SVC(kernel=function)
    clf1.fit(columns, target)
    # print(clf1.coef_)
    # print(clf1.intercept_)
    disp = DecisionBoundaryDisplay.from_estimator(estimator=clf1, X=columns, response_method="predict", ax=ax[plotnum], xlabel=feature_names[3], ylabel=feature_names[1])
    disp.ax_.scatter(columns[:, 0], columns[:, 1], c=target, edgecolor="k")
    disp.ax_.set_title(function)
    plotnum += 1
plt.show()