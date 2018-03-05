# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 19:11:29 2018

@author: jimmybow
"""

from sklearn import cluster, preprocessing
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import pandas as pd
import sys
import os
sys.exit()
os.chdir(os.path.abspath(os.path.dirname(__file__)))

for i in range(4):
    df = pd.read_csv('waveData_{}.csv'.format(i), header = None)
    data = df.iloc[:,:2].as_matrix()
    data = preprocessing.MinMaxScaler().fit_transform(data)
    km = cluster.KMeans(n_clusters = 5, random_state = 42).fit(data)
    tags = km.labels_
    true_tags = df.iloc[:,2]
    draw2Darray(data[:,0], data[:,1], tags)
    draw2Darray(data[:,0], data[:,1], true_tags)
    
    # 標準化的互信息評分: normalized_mutual_info_score
    print('score =', nmi(true_tags, tags))
