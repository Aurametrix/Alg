import numpy as np
import scipy.optimize as optimize


def f(x):

     V1 = 0.4104530886
     V2 = 0.3754395084
     alpha = 0.9230769231
     f = abs(((V2 - x[0])/(x[1] - x[0]))**alpha - (V1 - x[0])/(x[1] - x[0]))
     return f

#result = optimize.minimize(f, [0,1])

bnds = ((0, 1.), (0.3, 999.))
result = optimize.minimize(f, (0, 1), method='TNC', bounds=bnds, tol=1e-6)
print (result['x'], result['fun'])
