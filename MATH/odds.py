## Simulation study for the log odds ratio

from scipy.stats import rv_discrete

## Cell probabilities
P = np.array([[0.3,0.2],[0.1,0.4]])

## The population log odds ratio
PLOR = np.log(P[0,0]) + np.log(P[1,1]) - np.log(P[0,1]) - np.log(P[1,0])

## Sample size
n = 100

## ravel vectorizes by row
m = rv_discrete(values=((0,1,2,3), P.ravel()))

## Generate the data
D = m.rvs(size=(nrep,n))

## Convert to cell counts
Q = np.zeros((nrep,4))
for j in range(4):
    Q[:,j] = (D == j).sum(1)

## Calculate the log odds ratio
LOR = np.log(Q[:,0]) + np.log(Q[:,3]) - np.log(Q[:,1]) - np.log(Q[:,2])

## The standard error
SE = np.sqrt((1/Q.astype(np.float64)).sum(1))

print "The mean estimated standard error is %.3f" % SE.mean()
print "The standard deviation of the estimates is %.3f" % LOR.std()

## 95% confidence intervals
LCL = LOR - 2*SE
UCL = LOR + 2*SE

## Coverage probability
CP = np.mean((PLOR > LCL) & (PLOR < UCL))

print "The population LOR is %.2f" % PLOR
print "The expected value of the estimated LOR is %.2f" % LOR[np.isfinite(LOR)].mean()
print "The coverage probability of the 95%% CI is %.3f" % CP
