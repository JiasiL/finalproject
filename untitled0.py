# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 13:23:38 2016

@author: Reeshark
"""

# coding: utf-8
import numpy as Math
import pylab as Plot
import pandas as pd

label = pd.read_csv('4.csv')
Y2 = pd.read_csv('3.csv')
Y=Y2.values[0:,0:]
label2=label.values[0:,0:]
Plot.scatter(Y[:,0], Y[:,1], 20, label2);
Plot.show();