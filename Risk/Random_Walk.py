import numpy
import numpy as np
from numpy import log, exp, cumsum
import matplotlib.pyplot as plt
from numpy.random import randint, normal, uniform, exponential

# Simulating stock price behaviour

# Start with a simple counter moving up or down based on the flip of a coin
# Bernoulli trial = flip of the coin. 50% probability for both outcomes

t_max = 100     # number of timesteps = discrete = integers
movement = randint(0, 2, size=t_max)
# Each timestep we want to move the counter upwards if a 1 and dowmwards for a 0
# Replace the 0's with a -1 so can then take the cumulative sum
movements = np.where(movement==1, movement, -1)
path = np.cumsum(movements)

# Set the initial position at the centre of the grid
new_path = np.concatenate(([0], path))
#ax = plt.plot(new_path)
#plt.show()

# Now extend to show 10 paths
movement = randint(0, 2, size=(t_max, 10))
movements = np.where(movement==1, movement, -1)
path = np.cumsum(movements, axis=0)
# Add zero's to the start for every path
new_path = np.concatenate((np.matrix(np.zeros(10)), path), axis=0)
#ax = plt.plot(new_path)
#plt.show()

# This isn't how stock prices change over time - we need to MULTIPLY rather than add
# Prices change by a multiple of the price
# Start with a positive initial value
initial_value = 100.0
# Change comes from the normal distribution
rand = normal(size=t_max) * 0.005
# Add to 1 to give the multiplier for each step
multiplier = 1 + rand
# Subsequent values are the old value * the change
values = initial_value * np.cumprod(multiplier)
# Add the initial value into the path
path = np.concatenate(([initial_value], values))
#ax = plt.plot(path)
#plt.show()

##############
# Returns
##############

# Simple return = percentage change = the random number (eg 0.01, -0.01)
# Gross return = 1 + the simple return (eg 1.01, 0.99)
# Asset price simple return = (todays price - yesterdays price) / yesterdays price = (todays price / yesterdays price) - 1
# Asset price gross return  =  todays price / yesterdays price
# Simple returns additively aggregate across assets FOR ONE DAY
# Can be expressed as a weighted average of the simple returns of the individual assets
# Value of portfolio = sum of number of shares held * price
# Weight of a specific stock = number of shares held * price / Value of portfolio

# Log returns additively aggregate ACROSS TIME
# Continuously compounded log return = log(todays price) - log(yesterdays price)
# Not much difference between simple and log returns unless the returns are very large

from numpy import diff
prices = values
log_returns = diff(log(prices))

# Random walk with log returns
t_max = 100
volatility = 1e-2
initial_value = 100.0
r = normal(size=t_max) * np.sqrt(volatility)
# Treating these now as log returns - so apply the exponential function
y = initial_value * exp(np.cumsum(r))
#ax = plt.plot(np.concatenate(([initial_value], y)))
#plt.show()

# Same as above but for 10 paths
t_max = 100
volatility = 0.005
initial_value = 100.0
r = normal(size=(t_max, 10)) * np.sqrt(volatility)
# Treating these now as log returns - so apply the exponential function
y = initial_value * exp(np.cumsum(r, axis=0))
#ax = plt.plot(np.concatenate((np.matrix([initial_value] * 10), y)))
#plt.show()

# Adapting this model so works with arbitrary timesteps
# Brownian motion = How small particles move in air. Always being hit by multiple things eg dust particles
# Good analogy for stock prices. Being influenced by many external forces

# Divide each unit of time into k discrete steps
# At each step the particle moves 1 / sqrt(k) units of distance either up (+1) or down (-1)
# Position at time t = 1 / sqrt(k) * sum of the steps moved across all steps t * k

def random_walk(t, k, n=1):
    return np.cumsum(np.where(np.random.randint(0, 2, size=(t*k, n)) == 0, -1, +1), axis=0) * 1.0 / np.sqrt(k)

t_max = 10
k = 1000
T = np.arange(0, t_max, 1.0/k)
path = random_walk(t_max, k, n=100)
plt.plot(T, path)
plt.show()

# Standard properties of Brownian Motion : Paths generated have mean of 0 and variance of t
# The process has stationary and independent increments - so can calculate easily as below
t = 5.25
brownian_motion = np.sqrt(t) * np.random.normal()
print(brownian_motion)

# Simulating a standard brownian motion path
k = 10; t_max = 20.
z = np.random.normal(size=k-1)
t = np.arange(0, t_max, t_max/k)
dt = np.diff(t)
plt.plot(t, np.concatenate([[0.0], np.cumsum(np.sqrt(dt) * z)]), '+-', 'b')
plt.show()

# Brownian motion with drift and volatility
# Take a standard brownian motion (B)
# Multiply by a constant volatility parameter - sigma
# Add on a linear function over time - mu * t - where mu is the drift
# Xt = sigma * B + mu * t

# Brownian motion with positive drift (mu)
sigma = 0.005; mu = 0.005; k = 100; T = 10.
z = np.random.normal(size=k-1)
t = np.arange(0, T, T/k)
dt = np.diff(t)
plt.plot(t[1:], np.cumsum(sigma * np.sqrt(dt) * z + mu * dt))
# Using the $ sign is the LaTeX for maths
plt.xlabel('$t$'); plt.ylabel('$X_t$'); plt.title('$\mu=+0.005$')
plt.show()

# So we have a model for particle position. Want to model stock prices
# Geometric Brownian Motion

# Variables whose logarithm is normally distributed have a log normal distribution
# To draw from a log-normal distribution we draw from a normal distribution then take the exponential
# We then have a random variable whose logarithm is normally distributed

# Can see the distribution. Can see is skewed & the mean
z = np.random.normal(size=10000)
plt.hist(np.exp(z), bins=100)
plt.show()
from scipy.stats import skew
print(skew(np.exp(z)))
print(np.mean(np.exp(z)))

# Not good. So change the standard deviation
z = np.random.normal(0., scale=0.25, size=10000)
plt.hist(np.exp(z), bins=100)
plt.show()
from scipy.stats import skew
print(skew(np.exp(z)))
print(np.mean(np.exp(z)))

# Geometric Brownian Motion
# New variable St which is the stock price at time t
# St = S0 * e to the power of the standard brownian motion at time t (with drift mu and volatility sigma)
sigma = 0.4; mu = 0.1; k = 1000; t_max = 12.0; S0 = 100.0
dt = t_max / k
T = np.arange(0, t_max, dt)
z = np.random.normal(size=len(T))
S = S0 * np.cumprod(np.exp(sigma*np.sqrt(dt)*z + mu*dt))
plt.plot(T, S, '+-')
# Using the $ sign is the LaTeX for maths
plt.xlabel('$t$'); plt.ylabel('$S_t$')
plt.show()

# Can see that the gross returns are skewed
R = np.array([S[t] / S[t-1] for t in range(1, len(T))])
plt.hist(R, bins=20)
plt.show()
# But you can see that the log of the gross returns is normally distributed
# Makes sense as that's where came from
plt.hist(np.log(R), bins=20)
plt.show()