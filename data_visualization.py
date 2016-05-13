# -*- coding: utf-8 -*-

"""

Created on Tue Mar 1 20:40:31 2016

@author: Jiasi Li

"""



import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pylab as Plot
import pandas as pd



###############################################################################
# Read training set and generate the target-act-value list
###############################################################################


train_df = pd.read_csv('ACT4_competition_training.csv')
del train_df['MOLECULE']
train_y = train_df['Act'].tolist()
del train_df['Act']


###############################################################################
# Call PCA to processing the dataset
# Call the t-SNE algorithm 
# and fit the algorithm to the training set
###############################################################################


pca = PCA()
train_pca = pca.fit_transform(train_df)

tsne = TSNE(n_components=2, init='pca', random_state=0)
train_tsne = tsne.fit_transform(train_pca)


###############################################################################
# Generate the scatter plot
###############################################################################


Plot.scatter(train_tsne[:,0], train_tsne[:,1], 20, c=train_y);
Plot.show();


###############################################################################
# Generate the feature importance file for the descriptors
###############################################################################


des = pd.read_csv('ACT4_competition_descriptor.csv')
descriptor = des['Descriptor'].tolist()

importance = pd.DataFrame({"Descriptor": descriptor, 
                           "Feature importance": pca.mean_})
importance.to_csv("descriptor_importance.csv", index=False)






