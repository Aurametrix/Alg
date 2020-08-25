import numpy as np
from skbio.stats.composition import clr
x = np.array([.1, .3, .4, .2])
clr_inv(x)

# array([ 0.21383822,  0.26118259,  0.28865141,  0.23632778])

### OR

from scipy.stats import linregress
from scipy.stats.mstats import gmean
    
def clr(x):
    return np.log(x) - np.log(gmean(x))

x = np.array([.1, .3, .4, .2])

transformed = clr(x)

print(clr(x))

# [-0.79451346  0.30409883  0.5917809  -0.10136628]

#### mean-centering

df.apply(lambda x: x-x.mean())

%timeit df.apply(lambda x: x-x.mean())
1000 loops, best of 3: 2.09 ms per loop

df.subtract(df.mean())

%timeit df.subtract(df.mean())
1000 loops, best of 3: 902 µs per loop
