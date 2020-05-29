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

