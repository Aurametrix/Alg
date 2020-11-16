import math
import random
 
def GammaInc_Q( a, x):
    a1 = a-1
    a2 = a-2
    def f0( t ):
        return t**a1*math.exp(-t)
 
    def df0(t):
        return (a1-t)*t**a2*math.exp(-t)
 
    y = a1
    while f0(y)*(x-y) >2.0e-8 and y < x: y += .3
    if y > x: y = x
 
    h = 3.0e-4
    n = int(y/h)
    h = y/n
    hh = 0.5*h
    gamax = h * sum( f0(t)+hh*df0(t) for t in ( h*j for j in xrange(n-1, -1, -1)))
 
    return gamax/gamma_spounge(a)
 
c = None
def gamma_spounge( z):
    global c
    a = 12
 
    if c is None:
       k1_factrl = 1.0
       c = []
       c.append(math.sqrt(2.0*math.pi))
       for k in range(1,a):
          c.append( math.exp(a-k) * (a-k)**(k-0.5) / k1_factrl )
          k1_factrl *= -k
 
    accm = c[0]
    for k in range(1,a):
        accm += c[k] / (z+k)
    accm *= math.exp( -(z+a)) * (z+a)**(z+0.5)
    return accm/z;
 
def chi2UniformDistance( dataSet ):
    expected = sum(dataSet)*1.0/len(dataSet)
    cntrd = (d-expected for d in dataSet)
    return sum(x*x for x in cntrd)/expected
 
def chi2Probability(dof, distance):
    return 1.0 - GammaInc_Q( 0.5*dof, 0.5*distance)
 
def chi2IsUniform(dataSet, significance):
    dof = len(dataSet)-1
    dist = chi2UniformDistance(dataSet)
    return chi2Probability( dof, dist ) > significance
 
# dset1 = [ 199809, 200665, 199607, 200270, 199649 ]
# dset2 = [ 522573, 244456, 139979,  71531,  21461 ]

dset1 = [ 1.3, 0.7, 0.5, 1, 0.8, 0.9, 0.5, 0.5, 0.6, 1.2, 0.6, 0.4]
dset2 = [ 1.2, 0.8, 1, 1.3, 1.2, 0.8, 1, 1.3, 1.2, 0.8, 1, 1.3]
 
for ds in (dset1, dset2):
    print "Data set:", ds
    dof = len(ds)-1
    distance =chi2UniformDistance(ds)
    print "dof: %d distance: %.4f" % (dof, distance),
    prob = chi2Probability( dof, distance)
    print "probability: %.4f"%prob,
    print "uniform? ", "Yes"if chi2IsUniform(ds,0.05) else "No"
    
    
import pandas as pd
import researchpy as rp
import scipy.stats as stats

# To load a sample dataset  - from stata documentation
import statsmodels.api as sm

df = sm.datasets.webuse("citytemp2")

df.info()
# 956 entries, 0 to 955; 7 columns
# is there a relationship between region and age (ageecat = 3 categories)
rp.summary_cat(df[["agecat", "region"]])
# CHI-SQUARE TEST OF INDEPENDENCE WITH SCIPY.STATS
# scipy.stats.chi2_contingency : https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
# This method requires to pass a crosstabulation table, this can be accomplished using pandas.crosstab
crosstab = pd.crosstab(df["region"], df["agecat"])
crosstab
stats.chi2_contingency(crosstab)
# 2st value returned is chi-square 61.29, second pval< 0.0001

# independece test with Researchpy
crosstab, test_results, expected = rp.crosstab(df["region"], df["agecat"],
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")

crosstab
test_results
# Pearson chi-square is 61.29, pval also small
# dditionally calculated: strength of association - Cramer's V
# 0.17 here - strong but not very strong
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6107969/
