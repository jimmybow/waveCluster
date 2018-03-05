# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 19:00:21 2018

@author: jimmybow
"""

from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from waveCluster import *

data = np.array(read_sample(FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS))
tags = waveCluster(data, scale=144, threshold=-0.5, plot=True)
true_tags = np.arange(len(data))>=400
draw2Darray(data[:,0], data[:,1], tags)
draw2Darray(data[:,0], data[:,1], true_tags)

print(pd.Series.value_counts(tags))
# 標準化的互信息評分: normalized_mutual_info_score
print(nmi(true_tags, tags))
