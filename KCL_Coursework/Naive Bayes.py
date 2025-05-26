# https://archive.ics.uci.edu/ml/datasets/Heart+Disease

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split

datafilename = 'datasets/cleveland.csv'

age      = 0                              # column indexes in input file
sex      = 1
cp       = 2
trestbps = 3
chol     = 4
fbs      = 5
restecg  = 6
thalach  = 7
exang    = 8
oldpeak  = 9
slope    = 10
ca       = 11
thal     = 12
num      = 14 # this is the thing we are trying to predict

# Since feature names are not in the data file, we code them here
feature_names = [ 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num' ]

num_samples = 303 # size of the data file.
num_features = 13

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

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0)

from sklearn import tree
dtc = tree.DecisionTreeClassifier()
dtc.fit(X_train, y_train)
print('Decision tree score: ' + str(dtc.score(X_test, y_test)))

from sklearn import naive_bayes
nbc = naive_bayes.MultinomialNB()
nbc.fit(X_train, y_train)
print('Multinomial Naive Bayes score: ' + str(nbc.score(X_test, y_test)))

