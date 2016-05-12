# Import APM package
from apm import *

# Solve optimization problem
sol = apm_solve('hs71',3)

print '--- Results of the Optimization Problem ---'
print 'x[1]: ', sol['x[1]']
print 'x[2]: ', sol['x[2]']
print (((0.5166406610 - sol['x[1]'])/(sol['x[2]'] - sol['x[1]']))**0.9 - (0.5847163622 - sol['x[1]'])/(sol['x[2]'] - sol['x[1]']))
