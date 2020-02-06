from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

data_X = pd.read_csv('MB21.csv', header=0)
data_X = data_X.loc[:, data_X.columns != 'Symptoms'] #all except Symptoms column

data_X = data_X.dropna()     # drop null values

class_label = pd.read_csv('MB21.csv', sep=',', header=0, usecols=[0])  #first column 
class_label = class_label.values.reshape(-1, 1)

trainX, testX, trainy, testy = train_test_split(data_X, class_label, test_size=0.5, random_state=1)

model = RandomForestClassifier(n_estimators=200)

model.fit(trainX, trainy.ravel())
#fit(self, X, y, sample_weight=None)[source]


probs = model.predict_proba(testX)
probs = probs[:, 1]

auc = roc_auc_score(testy, probs)
print('AUC: %.2f' % auc)

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
feature_list = list(pd.read_csv('MB21.csv').head(0))
feature_list.pop(0)
tree = model.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')
