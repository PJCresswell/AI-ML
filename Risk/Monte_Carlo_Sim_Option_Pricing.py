'''
Monte Carlo methods can be used to analyse quantitative models
Any problem in which the solution can be written as an expectation of random variable(s) can be solved using a Monte Carlo approach
We write down an estimator for the problem = a variable whose expectation represents the solution
We then repeatedly sample input variables, and calculate the estimator numerically
The sample mean of this variable can be used as an approximation of the solution; that is, it is an estimate
The larger the sample size, the more accurate the estimate
There is an inverse-square relationship between sample size and the estimation error

sample = []
for i in range(n):
    x = draw_random_value(distribution)
    y = f(x)
    sample.append(y)
result = mean(sample)
'''

# Imagine a circle in a 2x2 square centered at (0, 0) and we want to find pi
import numpy as np
# Randomly generate points (X,Y) in the upper right quarter of the square
n = 1000000
X = np.random.random(size=n)
Y = np.random.random(size=n)
def f(x, y):
    # Determine the number of points in the square that are also in the circle
    if x*x + y*y < 1: return 1.
    else: return 0.
# If R is the number of points in the circle divided by the number of points in the square, then  π=4E[R]
pi_approx = 4 * np.mean([f(x, y) for (x, y) in zip(X,Y)])
print("Pi is approximately %f" % pi_approx)

'''
Option Pricing : Right to buy (call) or sell (put) an underlying stock at the strike price
    European option : Single date
    American option : Across a period of dates
Payoff depends on the security price when the option is exercised
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

K = 8000                            # Strike price
S = np.arange(7000, 9000, 100)      # Security values
h = np.maximum(S - K, 0)            # Intrinsic value of call option - if security price below strike then worthless
plt.plot(S, h, lw=2.5)        # Plot intrinsic values at maturity
plt.xlabel('underlying value $S_t$ at maturity'); plt.ylabel('intrinsic value of European call option')
plt.show()

'''
Parameters that Affect the Intrinsic Value
1. Initial price level of the security S0
2. Volatility of the security σ
3. The return(s) of the security
4. Strike time T
5. Strike price K

Risk-Neutral Pricing

When all of the following assumptions hold :
1. No arbitrage
2. Complete markets (no transaction costs and perfect information)
3. Law of one price (assets with identical risk and return have a unique price),
4. S follows a geometric Brownian motion
5. There is a risk-free security with interest rate r

Then:
1. It is possible to dynamically and continuously adapt a portfolio P of long and short positions in the underlying 
   security and the risk-free asset, such that the value of the portfolio at any time is the intrinsic value of the
   option at time t with probability 1. This is called a replicating portfolio.
2. The price of the option therefore should be set to the value of P at time 0, otherwise we would introduce arbitrage opportunities.
3. It is possible to prove that the value of P at time 0 equals the value of the option at time T if the underlying security
   would have drift r (instead of its "real" percentage drift), discounted back to time 0 according to the rate r
   (i.e. multiplied by e to the power −rT
4. This alternative probability measure obtained by replacing the drift parameter is called the risk-neutral probability measure
'''

# For a European call option
# NOT path dependent
from numpy import sqrt, exp, cumsum, sum, maximum, mean
from numpy.random import standard_normal

# Parameters
S0 = 100.; K = 105.; T = 1.0; r = 0.02; sigma = 0.1; I = 100000

# Draw I random numbers from the standard normal distribution
# Calculate the underlying security's value at the strike time T by simulating geometric Brownian motion with drift
# r and volatility σ using the given equation
S = S0 * exp((r - 0.5 * sigma ** 2) * T + sigma * sqrt(T) * standard_normal(I))

# Compute the intrinsic value of the option (the max part)
# Discount back to the present at the risk-free rate (the exp bit)
# Final result is the mean
C0 = exp(-r * T) * mean(maximum(S - K, 0))
print("Estimated present value is %f" % C0)

# Now for an Asian (Average Value) Call Option
# The payoff is determined by the average of the price of the underlying over a pre-defined period of time
# The payoff is path dependent, so now we need to simulate intermediate values of St

# Parameters
S0 = 100.; T = 1.0; K = 50; r = 0.02; sigma = 0.1; M = 200; dt = T / M; I = 100000

def inner_value(S):
    return np.array([max(V - K,0) for V in mean(S, axis=0)])

# Draw I x M random numbers from the standard normal distribution
# Calculate the underlying security's value at each time interval by simulating geometric Brownian motion with drift
# r and volatility σ using the given equation
S = S0 * exp(cumsum((r - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * standard_normal((M, I)), axis=0))

# Estimate the average value of the underlying (the mean of the sum bit)
# Compute the intrinsic value of the option (the max bit)
# Discount back to the present at the risk-free rate (the exp bit)
# Final result is the mean
C0 = exp(-r * T) * mean(inner_value(S))
print("Estimated present value is %f" % C0)