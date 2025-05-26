import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin

# Applying a function to an array
x = np.array([0, 1, 2, 3, 4])
y = x * 2
print(y[3])

# Populating an array with a range of values
x = np.arange(0, 2*pi, 0.01)
y = sin(x)
plt.plot(x, y)
plt.show()

# Need to be careful comparing floating point numbers - finite precision
import sys
print(sys.float_info)

x = 0.1 +0.2
y = 0.3
print(x == y)
print(np.allclose(x, y))

# Vectorization. Most functions are already vectorised in Numpy
# Can write your own though
def myfunc(x):
    if x >= 0.5: return x
    else: return 0.0
fv = np.vectorize(myfunc)
x = np.arange(0, 1, 0.1)
print(x)
y = fv(x)
print(y)

# Multi-dimensional array
x = np.array([[1, 2], [3, 4]])
print(x[1])
print(x[1][0])

# Creating a matrix + matrix operations
M = np.matrix(x)
print(M)
M_transposed = M.T
print(M_transposed)
M_inverse = M.I
print(M_inverse)

I2 = np.matrix('1 0; 0 2')
matrix_sum = M * I2
print(matrix_sum)
print(M[:, 1])          # Slicing a matrix is the same as a list - just a reference
# To copy a matrix
V = np.copy(M[:, 1])

# Summing across rows and columns
matrix = np.matrix('1 2 3; 4 5 6; 7 8 9')
print(matrix)
row_sum = np.sum(matrix, axis=0)
print(row_sum)
col_sum = np.sum(matrix, axis=1)
print(col_sum)
y1 = np.cumsum(matrix)
print(y1)
y2 = np.cumsum(matrix, axis=0)
print(y2)
y3 = np.cumsum(matrix, axis=1)
print(y3)
y4 = np.cumprod(matrix)
print(y4)

# Random numbers
from numpy.random import normal, uniform, exponential, randint
from numpy.random import seed
# Set the seed to reliably reproduce the same numbers
seed(5)
print('Random numbers ' + str(normal()) + ' and ' + str(normal()))
seed(5)
print('Random numbers ' + str(normal()) + ' and ' + str(normal()))
multi_samples = normal(size=(5, 5))
print(multi_samples)

data = normal(size=10000)
ax = plt.hist(data)
plt.show()
bins = np.histogram(data)
print(bins)
print('Mean ' + str(np.mean(data)))
print('Variance ' + str(np.var(data)))

# Discrete random numbers
die_roll = randint(low=0, high=6, size=20) + 1
print(die_roll)