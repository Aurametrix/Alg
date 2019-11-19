# Setup A
actual_a = [1 for n in range(10)] + [0 for n in range(10)]
predicted_a = [1 for n in range(9)] + [0, 1, 1] + [0 for n in range(8)]
print(actual_a)
print(predicted_a)

# Confusion Matrix
from sklearn.metrics import confusion_matrix

def my_confusion_matrix(actual, predicted):
    true_positives = len([a for a, p in zip(actual, predicted) if a == p and p == 1])
    true_negatives = len([a for a, p in zip(actual, predicted) if a == p and p == 0])
    false_positives = len([a for a, p in zip(actual, predicted) if a != p and p == 1])
    false_negatives = len([a for a, p in zip(actual, predicted) if a != p and p == 0])
    return "[[{} {}]\n  [{} {}]]".format(true_negatives, false_positives, false_negatives, true_positives)

print("my Confusion Matrix A:\n", my_confusion_matrix(actual_a, predicted_a))
print("sklearn Confusion Matrix A:\n", confusion_matrix(actual_a, predicted_a))

# Accuracy
from sklearn.metrics import accuracy_score

# Accuracy = TP + TN / TP + TN + FP + FN
def my_accuracy_score(actual, predicted): #threshold for non-classification?  
    true_positives = len([a for a, p in zip(actual, predicted) if a == p and p == 1])
    true_negatives = len([a for a, p in zip(actual, predicted) if a == p and p == 0])
    false_positives = len([a for a, p in zip(actual, predicted) if a != p and p == 1])
    false_negatives = len([a for a, p in zip(actual, predicted) if a != p and p == 0])
    return (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

print("my Accuracy A:", my_accuracy_score(actual_a, predicted_a))
print("sklearn Accuracy A:", accuracy_score(actual_a, predicted_a))


# Precision
from sklearn.metrics import precision_score

# Precision = TP / TP + FP
def my_precision_score(actual, predicted):
    true_positives = len([a for a, p in zip(actual, predicted) if a == p and p == 1])
    false_positives = len([a for a, p in zip(actual, predicted) if a != p and p == 1])
    return true_positives / (true_positives + false_positives)

print("my Precision A:", my_precision_score(actual_a, predicted_a))
print("sklearn Precision A:", precision_score(actual_a, predicted_a))

# Recall
from sklearn.metrics import recall_score

def my_recall_score(actual, predicted):
    true_positives = len([a for a, p in zip(actual, predicted) if a == p and p == 1])
    false_negatives = len([a for a, p in zip(actual, predicted) if a != p and p == 0])
    return true_positives / (true_positives + false_negatives)

print("my Recall A:", my_recall_score(actual_a, predicted_a))
print("sklearn Recall A:", recall_score(actual_a, predicted_a))

#Precision-Recall
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, _ = precision_recall_curve(actual_a, predicted_a)
plt.step(recall, precision, color='g', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='g', step='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve')
plt.show()

# F1 Score
from sklearn.metrics import f1_score

# Harmonic mean of (a, b) is 2 * (a * b) / (a + b)
def my_f1_score(actual, predicted):
    return 2 * (my_precision_score(actual, predicted) * my_recall_score(actual, predicted)) / (my_precision_score(actual, predicted) + my_recall_score(actual, predicted))

print("my F1 Score A:", my_f1_score(actual_a, predicted_a))
print("sklearn F1 Score A:", f1_score(actual_a, predicted_a))

#ROC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

print("sklearn ROC AUC Score A:", roc_auc_score(actual_a, predicted_a))
fpr, tpr, _ = roc_curve(actual_a, predicted_a)
plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') #center line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# Accuracy for non-binary predictions
def my_general_accuracy_score(actual, predicted):
    correct = len([a for a, p in zip(actual, predicted) if a == p])
    wrong = len([a for a, p in zip(actual, predicted) if a != p])
    return correct / (correct + wrong)

print("my Accuracy B:", my_general_accuracy_score(actual_b, predicted_b))
print("sklearn Accuracy B:", accuracy_score(actual_b, predicted_b))

def my_general_precision_score(actual, predicted, value):
    true_positives = len([a for a, p in zip(actual, predicted) if a == p and p == value])
    false_positives = len([a for a, p in zip(actual, predicted) if a != p and p == value])
    return true_positives / (true_positives + false_positives)

print("my Precision B:", my_general_precision_score(actual_b, predicted_b, 2))

# Accuracy for continuous with threshold
def my_threshold_accuracy_score(actual, predicted, threshold):
    a = [0 if x >= threshold else 1 for x in actual]
    p = [0 if x >= threshold else 1 for x in predicted]
    return my_accuracy_score(a, p)

print("my Accuracy C:", my_threshold_accuracy_score(actual_c, predicted_c, 5))
