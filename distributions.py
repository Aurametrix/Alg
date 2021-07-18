import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# binomial
n=10
p =0.3
k = np.arange(0,21)
binomial = stats.binom.pmf(k,n,p)
binomial

# Poisson
rate = 2
n = np.arange(0,10)
y = stats.poisson.pmf(n, rate)
y

plt.plot(n, y, '0-')
plt.title(Poisson: $\lambda$=%i' % rate)
          plt.xlabel('Number of accidets')
          ply.ylabel('Probability of this number of accidents')
plt.show

data = stats.poisson.rvs(mu=2, loc=0, size = 1000)
