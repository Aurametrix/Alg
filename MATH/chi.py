import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

data = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
data.head()

data.shape
data.dtypes
data.describe()

data.isna().any()

attri_count = data['Attrition'].value_counts()

fig = plt.figure(figsize=(6, 6))
ax = sns.barplot(attri_count.index, attri_count.values)
plt.title("Attrition Distribution",fontsize = 20)
plt.ylabel('Number of Instances', fontsize = 12)
plt.xlabel('Attrition', fontsize = 12);

data['JobSatisfaction'].value_counts()

# Chi Square Test of Independence.
 
obs = np.append(ct.iloc[0][0:4].values, ct.iloc[1][0:4].values)
obs
row_sum = ct.iloc[0:2,4].values
exp = []
for j in range(2):
    for val in ct.iloc[2,0:4].values:
        exp.append(val * row_sum[j] / ct.loc['All', 'All'])
exp

chi_sq_stats = ((obs - exp)**2/exp).sum()
chi_sq_stats

dof = (len(row_sum)-1)*(len(ct.iloc[2,0:4].values)-1)
dof

1 - stats.chi2.cdf(chi_sq_stats, dof)

### Use SciPy to double check:

obs = np.array([ct.iloc[0][0:4].values,
                  ct.iloc[1][0:4].values])
stats.chi2_contingency(obs)[0:3]

ct = pd.crosstab(data.Attrition, data.WorkLifeBalance, margins=True)
ct

obs = np.array([ct.iloc[0][0:4].values,
                  ct.iloc[1][0:4].values])
stats.chi2_contingency(obs)[0:3]

# if p-val is high, Attrition is independent of Education
ct = pd.crosstab(data.Attrition, data.Education, margins=True)
ct

obs = np.array([ct.iloc[0][0:5].values,
                  ct.iloc[1][0:5].values])
stats.chi2_contingency(obs)[0:3]
