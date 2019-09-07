# cohen_kappa_score(rater1, rater2)

sklearn.metrics import cohen_kappa_score
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
cohen_kappa_score(y_true, y_pred)



from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC

kappa_scorer = make_scorer(cohen_kappa_score)
grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=kappa_scorer)
