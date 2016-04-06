# -*- coding: utf-8 -*-


import numpy as np
from sklearn.manifold import TSNE
import pylab as Plot
import pandas as pd

data = pd.read_csv('2.csv') 
X=data.values[0:,0:]
label = pd.read_csv('4.csv')
label2=label.values[0:,0:]

Y=np.array(X)

tsne = TSNE(n_components=2, init='pca', random_state=0)

train_tsne = tsne.fit_transform(Y)


Plot.scatter(train_tsne[:,0], train_tsne[:,1], 20, label2);
Plot.show();

