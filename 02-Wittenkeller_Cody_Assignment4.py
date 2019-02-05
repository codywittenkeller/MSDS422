# MSDS 422 Assignment 4 for Cody Wittenkeller

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

# modeling routines from Scikit Learn packages
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score  
from math import sqrt  # for root mean-squared error calculation


# read data for the Boston Housing Study
# creating data frame restdata
boston_input = pd.read_csv('boston.csv')

# check the pandas DataFrame object boston_input
print('\nboston DataFrame (first and last five rows):')
print(boston_input.head())
print(boston_input.tail())

print('\nGeneral description of the boston_input DataFrame:')
print(boston_input.info())

# drop neighborhood from the data being considered
boston = boston_input.drop('neighborhood', 1)
print('\nGeneral description of the boston DataFrame:')
print(boston.info())

print('\nDescriptive statistics of the boston DataFrame:')
print(boston.describe())

# set up preliminary data for data for fitting the models 
# the first column is the median housing value response
# the remaining columns are the explanatory variables
model_data = np.array([boston.mv,\
    boston.crim,\
    boston.zn,\
    boston.indus,\
    boston.chas,\
    boston.nox,\
    boston.rooms,\
    boston.age,\
    boston.dis,\
    boston.rad,\
    boston.tax,\
    boston.ptratio,\
    boston.lstat]).T

# dimensions of the polynomial model X input and y response
# preliminary data before standardization
print('\nData dimensions:', model_data.shape)

# standard scores for the columns... along axis 0
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# dimensions of the polynomial model X input and y response
# all in standardized units of measure
print('\nDimensions for model_data:', model_data.shape)

# create names of regression modeling methods
names = ["Elastic Net","SVM","Random Forest","Gradient Boost"]

model_methods = [ElasticNet(alpha=.01, l1_ratio=0.5,
                    fit_intercept=SET_FIT_INTERCEPT, normalize=False, 
                    precompute=False, max_iter=1000,copy_X=True, tol=0.0001, 
                    warm_start=False,positive=False, random_state=RANDOM_SEED),               
                LinearSVR(C=10,epsilon=0,fit_intercept=SET_FIT_INTERCEPT,
                     random_state=RANDOM_SEED),
                RandomForestRegressor(n_estimators = 100,max_features="log2",
                          n_jobs=-1, bootstrap=True, random_state=RANDOM_SEED),
                GradientBoostingRegressor(n_estimators = 100,max_features="log2",
                          random_state=RANDOM_SEED)]

# Ten-fold cross-validation employed here
N_FOLDS = 10

# set up numpy array for storing results
cv_results = np.zeros((N_FOLDS, len(names)))

# set up cross-validation variable
from sklearn.model_selection import KFold
kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)

# check the splitting process by looking at fold observation counts
index_for_fold = 0  # fold count initialized 

for train_index, test_index in kf.split(model_data):
    print('\nFold index:', index_for_fold,
          '------------------------------------------')
#   note that 0:model_data.shape[1]-1 slices for explanatory variables
#   and model_data.shape[1]-1 is the index for the response variable    
    X_train = scaler_X.fit_transform(model_data[train_index, 0:model_data.shape[1]-1])
    X_test = scaler_X.fit_transform(model_data[test_index, 0:model_data.shape[1]-1])
    y_train = scaler_y.fit_transform(model_data[train_index, model_data.shape[1]-1].reshape(-1, 1))
    y_test = scaler_y.fit_transform(model_data[test_index, model_data.shape[1]-1].reshape(-1, 1))

    index_for_method = 0  # initialize
    for name, reg_model in zip(names, model_methods):
        print('\nRegression model evaluation for:', name)
        print('  Scikit Learn method:', reg_model)
        reg_model.fit(X_train, y_train)  # fit on the train set for this fold
 
        # evaluate on the test set for this fold
        y_test_predict = reg_model.predict(X_test)
        print('Coefficient of determination (R-squared):',
              r2_score(y_test, y_test_predict))
        fold_method_result = sqrt(mean_squared_error(y_test, y_test_predict))
        print(reg_model.get_params(deep=True))
        print('Root mean-squared error:', fold_method_result)
        cv_results[index_for_fold, index_for_method] = fold_method_result
        index_for_method += 1

    index_for_fold += 1
    
cv_results_df = pd.DataFrame(cv_results)
cv_results_df.columns = names

print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      'in standardized units (mean 0, standard deviation 1)\n',
      '\nMethod               RMSE', sep = '')     
print(cv_results_df.mean())  

# --------------------------------------------------------
# Based on the results the Random Forest is the best performing model

# --------------------------------------------------------
 
# fit linear regression to full data set for predictions and evaluation
my_X_train = scaler_X.fit_transform(model_data[:, 0:model_data.shape[1]-1])
my_X_test = scaler_X.fit_transform(model_data[:, 0:model_data.shape[1]-1])
my_y_train = scaler_y.fit_transform(model_data[:, model_data.shape[1]-1].reshape(-1, 1))
my_y_test = scaler_y.fit_transform(model_data[:, model_data.shape[1]-1].reshape(-1, 1))

# define the model to be a linear regression model
model = RandomForestRegressor(n_estimators = 100,max_features="log2",
                          n_jobs=-1, bootstrap=True, random_state=RANDOM_SEED)

model = model.fit(my_X_train,my_y_train.ravel())

# define feature variables
features = ['Crime Rate', 'Zoned for Lots', 'Percent Industrial', 
              'Near Charles River', 'Air Pollution', 'Average Rooms', 
              'Average Age','Distance to Employment', 'Near Radial Highways',
              'Tax Rate', 'Pupil/Teacher Ratio', 'Percent of Lower Status']

model_data = pd.DataFrame(list((zip(features,model.feature_importances_))),
                          columns=["Features","Importance"])

model_data_fig, ax = plt.subplots()
ax = sns.barplot(x='Features',y='Importance',data=model_data,color="b")
plt.setp(ax.get_xticklabels(),rotation=90)
ax.set_title("Summary of Model Feature Scores")
model_data_fig.savefig('Model_Data_Fig' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  


# create predictions for the entire data set
my_y_pred = model.predict(my_X_test)

# print the initial scoring of the model
print('\n----------------------------------------------')
print("R^2 Score: {:.3f}".format(r2_score(my_y_test,my_y_pred)))
print("RMSE Score: {:.3f}".format(sqrt(mean_squared_error(my_y_test,my_y_pred))))

# convert the model back to median 1970's thousands of dollars
real_my_y_pred = scaler_y.inverse_transform(my_y_pred)

# define the actual median 1970's thousands of dollars
real_y_test = scaler_y.inverse_transform(my_y_test)



# Plot outputs of predicted values vs actual values to visualize model
prediction_data_fig, ax = plt.subplots()
plt.scatter(real_y_test, real_my_y_pred, color='blue',alpha=0.5)
plt.plot([0,35],[0,35], color = 'green',label="Fit Line")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Actual Median Values (Thousands, USD)")
plt.ylabel("Predicted Median Values (Thousands, USD)")
plt.title("Predicted Values vs. Actual Values")
prediction_data_fig.savefig('prediction_data_fig' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  

