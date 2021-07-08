import pandas as pd
import numpy as np
# https://www.kaggle.com/ronitf/heart-disease-uci
df = pd.read_csv('Heart.csv')
df

df['Sex1'] = df.Sex.replace({1: "Male", 0: "Female"})
dx = df[["AHD", "Sex1"]].dropna()

# number of females with heart disease
pd.crosstab(dx.AHD, dx.Sex1)
# population proportion - 72 healthy, 25 with heart disease
p_fm = 25/(72+25)
# standard error
se_female = np.sqrt(p_fm * (1 - p_fm) / n)

# The z-score is 1.96 for a 95% confidence interval.

z_score = 1.96
lcb = p_fm - z_score* se_female #lower limit of the CI
ucb = p_fm + z_score* se_female #upper limit of the CI

import statsmodels.api as sm
sm.stats.proportion_confint(n * p_fm, n)

p_male = 114/(114+92)  #male population proportion
n = 114+92             #total male population
se_male = np.sqrt(p_male * (1 - p_male) / n)

se_diff = np.sqrt(se_female**2 + se_male**2)

d = 0.55 - 0.26
lcb = d - 1.96 * se_diff  #lower limit of the CI
ucb = d + 1.96 * se_diff  #upper limit of the CI
