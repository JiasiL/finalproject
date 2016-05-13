# -*- coding: utf-8 -*-

"""

Created on Tue Apr 5 23:41:48 2016

@author: Jiasi Li

"""



import numpy as np
from scipy import stats
import pandas as pd
from sklearn import ensemble



###############################################################################
# Read training set and generate the target-act-value list and matrix
# Read the descriptor csv file
# Read test set and generate the prediction matrix
# Read test result file and generate the true-act-value list
###############################################################################


train_df = pd.read_csv('ACT4_competition_training.csv')
del train_df['MOLECULE']
train_y = np.array(train_df['Act'].tolist())
del train_df['Act']
train_x = train_df.as_matrix()
   
test_df = pd.read_csv('ACT4_competition_test.csv')
test_labels = test_df['MOLECULE'].tolist()
del test_df['MOLECULE']
test_x = test_df.as_matrix()

result_df = pd.read_csv('ACT4_competition_test_result.csv')
result_y = np.array(result_df['Act'].tolist())


###############################################################################
# Build the model using Extremely Randomized Trees
# Fit model to training set
###############################################################################


etr = ensemble.ExtraTreesRegressor(n_estimators=100, 
                                   bootstrap=False, 
                                   oob_score=False, 
                                   verbose=1,
                                   random_state=0,
                                   n_jobs=4)

print("Fitting model")
etr.fit(train_x, train_y)


###############################################################################
# Apply model to test set
# and generate the prediction result: "submission.csv"
###############################################################################


print("Predicting on the test data")
prediction = etr.predict(test_x)
pred = np.array(prediction)

print("Writing out the prediction")
submission = pd.DataFrame({"Prediction Act": prediction, 
                           "MOLECULE": test_labels})    
submission.to_csv("prediction.csv", index=False)


###############################################################################
# Calculate the Pearson Correlation Coefficient
# to evaluate the accuracy of the prediction results
###############################################################################


Pearson_correlation_coefficient, tailed_pvalue = stats.pearsonr(pred, result_y)
#slope, intercept, rvalue, pvalue, stderr= stats.linregress(pred, result_y)
# r2 = r_square(result_y, pred)
print("Predictions for activity will be evaluated using Pearson correlation coefficient:")
print("It is:", Pearson_correlation_coefficient**2)


print(">>>>>> End >>>>>")
    

    
       
