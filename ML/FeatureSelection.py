import pandas as pd
import numpy as np

data = pd.read_csv('MB21.csv', header=0)
data.head()

# list(data) or 
names=list(data.columns) 

#print(names)  # Symptoms; Microbes
#print(data.columns)  

array = data.values
X = array[:,1:1704]
Y = array[:,0:1]

# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)

# Summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)

features = fit.transform(X)
# Summarize selected features
print(features[0:5,:])

# Import your necessary dependencies
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Feature extraction
#model = LogisticRegression()
model = LogisticRegression(solver='lbfgs', max_iter=5000)

# how to retrieve the 15 most informative features
rfe = RFE(model, 15)
#fit = rfe.fit(X, Y)
#print(Y.ravel())
fit = rfe.fit(X, Y.ravel())

print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))

new = pd.DataFrame(list(zip(*[names, fit.support_, fit.ranking_]))).add_prefix('Col')

new.to_csv('features.csv', index=False)
