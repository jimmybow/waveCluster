# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 19:11:29 2018

@author: jimmybow
"""

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from waveCluster import *
import pandas as pd
import sys
import os
sys.exit()
os.chdir(os.path.abspath(os.path.dirname(__file__)))
# threshold 同時代表抗噪音的能力，設定最低代表完全不過濾噪音。
# scale 會影響最終分出來的群的個數
for i in range(4):
    df = pd.read_csv('waveData_{}.csv'.format(i), header = None)
    data = df.iloc[:,:2].as_matrix()
    tags = waveCluster(data, scale=128, threshold=0.5, plot=True)
    true_tags = df.iloc[:,2]
    draw2Darray(data[:,0], data[:,1], tags)
    draw2Darray(data[:,0], data[:,1], true_tags)
    
    # pd.Series.value_counts(tags)
    print('total group =', len(pd.unique(tags)))
    # 標準化的互信息評分: normalized_mutual_info_score
    print('score =', nmi(true_tags, tags))
