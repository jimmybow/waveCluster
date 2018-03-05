# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 19:11:29 2018

@author: jimmybow
"""

from sklearn import datasets
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from waveCluster import *
import pandas as pd

iris = datasets.load_iris()
data = iris.data
tags = waveCluster(data, scale=50, threshold=-0.1, plot=True)
true_tags = iris.target
draw2Darray(data[:,0], data[:,2], tags)
draw2Darray(data[:,0], data[:,2], true_tags)

print(pd.Series.value_counts(tags))
# 標準化的互信息評分: normalized_mutual_info_score
print(nmi(true_tags, tags))
