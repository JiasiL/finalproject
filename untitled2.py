# -*- coding: utf-8 -*-


import numpy as np
from sklearn.manifold import TSNE
import pylab as Plot
import pandas as pd


train_df = pd.read_csv('ACT4_competition_training.csv')
del train_df['MOLECULE']
train_y = train_df['Act'].tolist()
del train_df['Act']


tsne = TSNE(n_components=2, init='pca', random_state=0)


train_tsne = tsne.fit_transform(train_df)


Plot.scatter(train_tsne[:,0], train_tsne[:,1], 20, train_y);
Plot.show();

