from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB

dataset = datasets.load_iris() ## Loading the dataset


## Importing 2 features for visualizing
X = dataset.data[:,2:4]
Y = dataset.target;
plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.Paired)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.show()
