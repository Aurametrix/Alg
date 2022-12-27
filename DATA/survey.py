import pandas as pd

df = pd.read_csv('survey_results_public.csv')
df.head()
df.shape
# multiple choice
df['BetterLife'].value_counts()
df['BetterLife'].value_counts(normalize=True)
df['MgrMoney'].value_counts(normalize=True)
df['SocialMedia'].value_counts().plot(kind="bar")
df['SocialMedia'].value_counts().plot(kind="bar", figsize=(15,7), color="#61d199")

said_no = df[df['BetterLife'] == 'No']
said_no.head(3)

print(said_no['Age'].mean(),
      said_yes['Age'].mean(),
      said_no['Age'].median(),
      said_yes['Age'].median()
     )
     
filtered_1 = df[(df['BetterLife'] == 'Yes') & (df['Country'] == 'India')]

filtered = df[(df['BetterLife'] == 'Yes') & (df['Age'] >= 50) & (df['Country'] == 'India') &~ (df['Hobbyist'] == "Yes") &~ (df['OpenSourcer'] == "Never")]
filtered

python_bool = df["LanguageWorkedWith"].str.contains('Python')
python_bool.value_counts(normalize=True)
lang_df.stack().value_counts().plot(kind='bar', figsize=(15,7), color="#61d199")


def CronbachAlpha(itemscores):
    itemscores = numpy.asarray(itemscores)
    itemvars = itemscores.var(axis=1, ddof=1)
    tscores = itemscores.sum(axis=0)
    nitems = len(itemscores)

    return nitems / (nitems-1.) * (1 - itemvars.sum() / tscores.var(ddof=1))
    
from tcistats import cronbach_alpha
#cronbach_alpha([[1, 2, 3], [2, 3, 4], [3, 4, 5]])

alphas = cronbach_alpha(my_items)
print('Cronbach alpha results: ', alphas)


=========
import pandas as pd
import numpy as np
from pgmpy.estimators import BayesianEstimator
from pgmpy.models import BayesianNetwork

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('file.csv')

# Extract the column names from the DataFrame
columns = list(df.columns)

# Create a Bayesian network model
model = BayesianNetwork()

# Add nodes to the model for each column in the DataFrame
for column in df.columns:
    model.add_node(column)

from pgmpy.estimators import MaximumLikelihoodEstimator




================================================
# Create a maximum likelihood estimator
estimator = MaximumLikelihoodEstimator(model, data)

# Create a Bayesian Estimator object
estimator = BayesianEstimator(model, df)

================================================

# Estimate the CPDs for each node in the model
cpds = estimator.estimate_cpds()

# Add the estimated CPDs to the model
model.add_cpds(*cpds)

# Fit the model to the data
model.fit(df)

# Test the fit of the model to the data
model.test_fit(df)

==============================================

# Learn the structure of the Bayesian Network
model = estimator.estimate(prior_type='BDeu', equivalent_sample_size=10)

# Print the structure of the learned Bayesian Network
print(model.edges())

