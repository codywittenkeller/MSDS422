# MSDS 422 Assignment 5 for Cody Wittenkeller

# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1

# although we standardize X and y variables on input,
# we will fit the intercept term in the models
# Expect fitted values to be close to zero
SET_FIT_INTERCEPT = True

# import base packages into the namespace for this program
import pandas as pd  # data frame operations  
import numpy as np  # arrays and math functions

import matplotlib.pyplot as plt # plotting

import seaborn as sns # plotting pretty
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import time



#import MNIST data
mnist_X = pd.read_csv('mnist_X.csv')
mnist_y = pd.read_csv('mnist_y.csv')

# define array data
X = np.array(mnist_X)
y = np.array(mnist_y)


# (1) produce an RFC on the 60k training set and use F1 scoring, record run time

start_time_1 = time.time()

# define train and test data
X_train,X_test,y_train,y_test = X[:60000],X[60000:],y[:60000],y[60000:]

# create the RFC model
prelim_model = RandomForestClassifier(n_estimators = 10,max_features='sqrt',
                          n_jobs=-1, bootstrap=True, random_state=RANDOM_SEED)

# make predictions on the training set
prelim_model.fit(X_train,y_train.ravel())
y_pred = prelim_model.predict(X_test)
prelim_model.feature_importances_

# evaluate the f1 score of the RFC on all 787 variables
print(f1_score(y_test.ravel(),y_pred,average='micro'))

# create confusion matrix for the RFC model
conf_mx_1 = confusion_matrix(y_test.ravel(),y_pred)
sns.heatmap(conf_mx_1, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

print("--- %s seconds ---" % (time.time() - start_time_1))
print('\n----------------------------------------------')
# run time was ~5.5 minutes for the entire section of code
# model performs with 100% accuracy on the training data

# (2) Execute PCA on all 70k rows with 95% of the variability 
start_time_23 = time.time()

# create the reduced X data
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)

# the reduced model now has only 154 features!
X_reduced.shape


# run time was less than 10 seconds to perform

# (3) Create a RFC model on the 60k training samples using the 154 feature data

# define train and test data
X_train_reduced,X_test_reduced,y_train,y_test = X_reduced[:60000],X_reduced[60000:],y[:60000],y[60000:]

prelim_model.fit(X_train_reduced,y_train.ravel())

y_pred_reduced = prelim_model.predict(X_test_reduced)

# evaluate the f1 score of the RFC on reduced data
print(f1_score(y_test.ravel(),y_pred_reduced,average='micro'))

# create confusion matrix for the PCA model
conf_mx_2 = confusion_matrix(y_test.ravel(),y_pred_reduced)
sns.heatmap(conf_mx_2, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label reduced')
plt.show()

# print the run time for steps 2 and 3 combined
print("--- %s seconds ---" % (time.time() - start_time_23))
print('\n----------------------------------------------')

# 5 fix the model flaw

# run time was greater for steps 2 and 3 and at a decreased accuracy
# the decreased accuracy is expected due to less features. 
# the confusion matricies indicated the the less features cause more confustion
# for the digits 4 and 9 on the PCA model. 

# experiment is unfair because PCA transformation makes non-binary numbers, 
# these makes it more difficult for the PCA model than the original model 

# fix the experiment

start_time_5 = time.time()

from sklearn.pipeline import Pipeline

# create the reduced X data
pipe = Pipeline([('pca', PCA(n_components=0.95)),
                 ('tree', RandomForestClassifier(n_estimators = 10,max_features='sqrt',
                          n_jobs=-1, bootstrap=True, random_state=RANDOM_SEED))])

pipe.fit(X_train, y_train.ravel())

y_pred_5 = pipe.predict(X_test)


# evaluate the f1 score of the RFC on reduced data
print(f1_score(y_test.ravel(),y_pred_5,average='micro'))

# create confusion matrix for the PCA model
conf_mx_5 = confusion_matrix(y_test.ravel(),y_pred_5)
sns.heatmap(conf_mx_5, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label 5')
plt.show()

# print the run time for steps 2 and 3 combined
print("--- %s seconds ---" % (time.time() - start_time_5))
print('\n----------------------------------------------')


