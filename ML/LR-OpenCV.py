from cv2 import ml
from sklearn.datasets import make_blobs
from sklearn import model_selection as ms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from numpy import float32
from matplotlib.pyplot import show
from digits_dataset import split_images, split_data
 
# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)
 
# Obtain training and testing datasets from the digits image
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = split_data(20, sub_imgs, 0.8)
 
# Create an empty logistic regression model
lr_digits = ml.LogisticRegression_create()
 
# Check the default training method
print('Training Method:', lr_digits.getTrainMethod())
 
# Set the training method to Mini-Batch Gradient Descent and the size of the mini-batch
lr_digits.setTrainMethod(ml.LogisticRegression_MINI_BATCH)
lr_digits.setMiniBatchSize(400)
 
# Set the number of iterations
lr_digits.setIterations(10)
 
# Train the logistic regressor on the set of training data
lr_digits.train(digits_train_imgs.astype(float32), ml.ROW_SAMPLE, digits_train_labels.astype(float32))
 
# Print the number of learned coefficients, and the number of input features
print('Number of coefficients:', len(lr_digits.get_learnt_thetas()[0]))
print('Number of input features:', len(digits_train_imgs[0, :]))
 
# Predict the target labels of the testing data
_, y_pred = lr_digits.predict(digits_test_imgs.astype(float32))
 
# Compute and print the achieved accuracy
accuracy = (sum(y_pred[:, 0] == digits_test_labels[:, 0]) / digits_test_labels.size) * 100
print('Accuracy:', accuracy, '%')
 
# Generate and plot confusion matrix
cm = confusion_matrix(digits_test_labels, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
show()
