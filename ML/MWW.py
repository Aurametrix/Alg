from pandas import *

dataF = read_csv(“data_original.csv”, na_values=[” “])

print dataF

dataF.describe()

# print dataF[“1st_set”]
# print dataF[“2nd_set”][15]
# print dataF[dataF[“2nd_set”] > 27]

from scipy import stats

# print stats.sem(dataF[“3rd_set”])
# To correct nan use “dropna” 
# dataF[“2nd_set”].dropna()  
#dataF.fillna(0.0)[“2nd_set”]
#dataF[“2nd_set”].dropna().mean()

# dataF.fillna(0.0)[“2nd_set”].mean()

# dataF[dataF[“1st_set”] == 25][“3rd_set”].mean()


# dataF.fillna(0.0)[dataF[“1st_set”] == 25][“3rd_set”].var()

# standard error of the mean for “3rd_set” column when “1st_set” equals 25: 

 from scipy import stats

# stats.sem(dataF[dataF[“1st_set”] == 25][“3rd_set”])

# MWW Rank Sum test  - if two sets of data are significantly different -  vs t-test normally distributed.
# the Rank Sum test could provide a more accurate assessment on data.

sample1=dataF[“1st_set”].dropna()

sample2=dataF[“2nd_set”].dropna()

sample3=dataF[“3rd_set”].dropna()

stats.ranksums(sample2, sample3)

sample1=dataF[“1st_set”].dropna()

sample2=dataF[“2nd_set”].dropna()

sample3=dataF[“3rd_set”].dropna()

stats.ranksums(sample2, sample3)



stats.ranksums(sample1, sample2)


import pandas as pd
data=pd.read_csv("dataset/mwt.csv", skipinitialspace= True )
data['Speaker'] = data['Speaker'].map(lambda x: x.strip())
# data.shape
# test of stochastic equality
import scipy.stats as stats
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.style.use('bmh')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 20, 10
Pooh=data[data['Speaker'] == 'Pooh']
Piglet=data[data['Speaker'] == 'Piglet']
Pooh.hist(figsize=(5, 5), bins=5, xlabelsize=8, ylabelsize=8);
Piglet.hist(figsize=(5, 5), bins=5, xlabelsize=8, ylabelsize=8);

stats.mannwhitneyu(Pooh.Likert, Piglet.Likert)

# If the p-value>0.05 we accept alternate hypothesis which is The two groups do not exhibit stochastic equality.

# The Mann-Whitney U test allows comparison of two groups of data where the data is not normally distributed.

import numpy as np
import scipy.stats as stats

# Create two groups of data

group1 = [1, 5 ,7 ,3 ,5 ,8 ,34 ,1 ,3 ,5 ,200, 3]
group2 = [10, 18, 11, 12, 15, 19, 9, 17, 1, 22, 9, 8]

# Calculate u and probability of a difference

u_statistic, pVal = stats.mannwhitneyu(group1, group2)

# Print results

print ('P value:')
print (pVal)


# Confidence intervl for t-test averages - ds1 and ds2
import statsmodels.api as sm
tstat, p_value, dof = sm.stats.ttest_ind(ds1, ds2)
CI = sm.stats.CompareMeans.from_data(ds1, ds2).tconfint_diff()

### Confidence interval for Mann-Whitney medians
from scipy.stats import norm

ct1 = ds1.count()  #items in dataset 1
ct2 = ds2.count()  #items in dataset 2
alpha = 0.05       #95% confidence interval
N = norm.ppf(1 - alpha/2) # percent point function - inverse of cdf

# The confidence interval for the difference between the two population
# medians is derived through these nxm differences.
diffs = sorted([i-j for i in ds1 for j in ds2])

# For an approximate 100(1-a)% confidence interval first calculate K:
k = int(round(ct1*ct2/2 - (N * (ct1*ct2*(ct1+ct2+1)/12)**0.5)))

# The Kth smallest to the Kth largest of the n x m differences 
# ct1 and ct2 should be > ~20
CI = (diffs[k], diffs[len(diffs)-k])
