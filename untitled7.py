# -*- coding: utf-8 -*-


import numpy as np
from scipy import stats
import pandas as pd
from sklearn import ensemble


# load data
train_df = pd.read_csv('ACT4_competition_training.csv')
del train_df['MOLECULE']
train_y = np.array(train_df['Act'].tolist())
del train_df['Act']
train_x = train_df.as_matrix()
   
train_des = pd.read_csv('ACT4_competition_training_descriptor.csv')
train_descriptor = train_des['Descriptor'].tolist()

test_df = pd.read_csv('ACT4_competition_test.csv')
test_labels = test_df['MOLECULE'].tolist()
del test_df['MOLECULE']
test_x = test_df.as_matrix()

result_df = pd.read_csv('ACT4_competition_test_result.csv')
result_y = np.array(result_df['Act'].tolist())

# build the model
etr = ensemble.ExtraTreesRegressor(bootstrap=False, verbose=1, oob_score=False, n_jobs=4, n_estimators=40)

# run the model
print("Fitting model")
etr.fit(train_x, train_y)
importance = pd.DataFrame({"Descriptor": train_descriptor, "Feature importance": etr.feature_importances_})
importance.to_csv("importance.csv", index=False)
print("Predicting on the test data")
prediction = etr.predict(test_x)
pred = np.array(prediction)
print("Writing out the prediction")
submission = pd.DataFrame({"Prediction Act": prediction, "MOLECULE": test_labels})    
submission.to_csv("submission.csv", index=False)
slope, intercept, rvalue, pvalue, stderr= stats.linregress(pred, result_y)
print("R^2 is:", rvalue**2)
print(">>>>>> End >>>>>")
    

    
       
