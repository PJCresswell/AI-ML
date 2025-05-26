import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics

iris = load_iris()
print('Classes = ' + str(iris.target_names))
print('Attributes = ' + str(iris.feature_names))
M = len(iris.data)
print('Number of instances = ' + str(M))

# Classification rules using the OneR method
# Uses the iris dataset - two features : sepal length and sepal width
min_s_length = 43
max_s_length = 80
new_grid = np.arange(min_s_length, max_s_length, 1)
N = len(new_grid)
length_grid = np.c_[new_grid, np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)]

# First, we go through the data to find out the class frequencies and the most frequent class
# Starting with the length data
# First column is the feature value. Next 3 columns hold the class frequencies for 0,1,2. Final column holds the most frequent
for i in range(0, M):
    length = iris.data[i][0] * 10
    example_class = iris.target[i]
    for j in range(0, len(length_grid)) :
        if length_grid[j][0] == length :
            length_grid[j][1 + example_class] += 1
for j in range(0, len(length_grid)) :
    max_val = max(length_grid[j][1], length_grid[j][2], length_grid[j][3])
    for k in range(0, 3):
        if length_grid[j][1 + k] == max_val :
            length_grid[j][4] = k
# And now for the width data
# First column is the feature value. Next 3 columns hold the class frequencies for 0,1,2. Final column holds the most frequent
min_s_width = 20
max_s_width = 45
new_grid = np.arange(min_s_width, max_s_width, 1)
N = len(new_grid)
width_grid = np.c_[new_grid, np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)]
for i in range(0, M):
    width = iris.data[i][1] * 10
    example_class = iris.target[i]
    for j in range(0, len(width_grid)) :
        if width_grid[j][0] == width :
            width_grid[j][1 + example_class] += 1
for j in range(0, len(width_grid)) :
    max_val = max(width_grid[j][1], width_grid[j][2], width_grid[j][3])
    for k in range(0, 3):
        if width_grid[j][1 + k] == max_val :
            width_grid[j][4] = k

# Next we create a prediction grid - for every combination, what class
# If both features predict the same class then easy
# Otherwise we select the one with the highest frequency. If both the same, we go random
pred_grid = []
for i in range(0, len(length_grid)):
    for j in range(0, len(width_grid)):
        len_class = int(length_grid[i][4])
        freq_len_class = length_grid[i][int(len_class) + 1]
        wid_class = int(width_grid[j][4])
        freq_wid_class = width_grid[j][int(wid_class) + 1]
        if len_class == wid_class :
            pred_class = len_class
        elif freq_wid_class > freq_len_class :
            pred_class = wid_class
        elif freq_len_class > freq_wid_class :
            pred_class = len_class
        else :
            pred_class = len_class
        pred_grid.append((length_grid[i][0]/10, width_grid[j][0]/10, pred_class))

# Finally we work out how good our predictions are
# We don't split the data out into training and testing sets
correct_count = 0
for i in range(0, M):
    length = iris.data[i][0]
    width = iris.data[i][1]
    actual_class = iris.target[i]
    #print('Actual : Length= ' + str(length) + ' Width= ' + str(width) + ' Class= ' + str(actual_class))
    for j in range(0, len(pred_grid)):
        if (pred_grid[j][0] == length) and (pred_grid[j][1] == width):
            #print('Predicted : Length= ' + str(pred_grid[j][0]) + ' Width= ' + str(pred_grid[j][1]) + ' Class= ' + str(pred_grid[j][2]))
            if (actual_class == pred_grid[j][2]):
                correct_count += 1
print('Classification rules model example')
print('Score : Correct ' + str(correct_count) + ' from ' + (str(M)) + ' which is ' + (str(correct_count/M)))
