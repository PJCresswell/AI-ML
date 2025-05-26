import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, cos

# 1 Array of FP numbers from 0-10 in increments of 0.2
x = np.arange(0, 10, 0.2)
# 2 New array with the square of the above
y = x ** 2
# 3 Plot function y = x squared
#plt.plot(x, y)
#plt.show()
# 4 Plot function y = cos
x = np.arange(0, 2*pi, 0.05)
y = cos(x)
#plt.plot(x, y)
#plt.show()
# 5 Plot function y = 100x - x^2
x = np.arange(0, 200, 0.01)
y = (100 * x) - (x ** 2)
#plt.plot(x, y)
#plt.show()
# 6 Change the colours and stuff
#plt.plot(x, y, color='g')
#plt.show()
#plt.plot(x, y, 'ro')
#plt.show()
# 7 & 8 label axis and use mathematical notation
#plt.plot(x, y); plt.xlabel('x'); plt.ylabel('y'); plt.title('$y = 100x - x^2$')
#plt.show()
#9 Matrix inverse
matrix = np.matrix('1 2; 3 4')
Y = matrix.I
# 10 Matrix multiplication
B = np.matrix('1; 2')
matrix_sum = Y * B
# print(matrix_sum)
# 11 and 12 Standard Normal Distribution
from numpy.random import normal, randint
new_array = normal(size=100)
#ax = plt.hist(new_array)
#plt.show()
# 13 Bernoulli sequence
bern = randint(2, size=100)
#print(bern)
# 14 Replace 0 with -1
new_bern = np.where(bern==1, bern, -1)
#print(new_bern)
# 15 Integer sequence with cumulative sum
new_cum_sum = np.cumsum(new_bern)
x = np.arange(0, 100)
#ax = plt.plot(x, new_cum_sum, 'r-')
#plt.show()
# 16 Twenty version of the same process
bern_array = randint(2, size=(100,20))
new_bern_array = np.where(bern_array==1, bern_array, -1)
bern_array_sum = np.cumsum(new_bern_array, axis=0)
#ax = plt.plot(x, bern_array_sum)
#plt.show()
# 17 Doing all of the above at once for a new stochactic process
# Mean is 0 and Variance is 0.005
x = np.arange(0, 100)
new_array = normal(loc=0, scale=0.005, size=(100,20))
new_array_sum = np.cumsum(new_array, axis=0)
new_array_exp = np.exp(new_array_sum)
final_array = 100 * new_array_exp
#ax = plt.plot(x, final_array)
#plt.show()
# 18 Finding the roots of a quadratic function
# Formula : ax^2 + bx + c = 0 where a, b and c are real numbers and a is not equal to 0
# Solution : -b plus or minus square root of (b^2  - 4 a c) all over 2 a
a = 1
b = -3
c = 2
d = (b ** 2) - (4 * a * c)
# Normal approach
solution1 = (-b + np.sqrt(d)) / (2 * a)
solution2 = (-b - np.sqrt(d)) / (2 * a)
print('Solutions are ' + str(solution1) + ' and ' + str(solution2))
# Alternative formula
# By calculating one root using addition and the other using division, we avoid direct subtraction of nearly equal numbers.
# The modified formula ensures numerical stability even when the discriminant is small.
solution1 = (-b + np.sqrt(d)) / (2 * a)
solution2 = (2 * c) / (-b + np.sqrt(d))
print('Solutions are ' + str(solution1) + ' and ' + str(solution2))