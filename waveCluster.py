import matplotlib.pyplot as plt
import pywt
import numpy as np
import random
from math import *


def uniGenerate(a, b, n):
    # generate n uniform distributted real numbers in the range of (a,b), duplication is permitted
    uniList = []
    for i in range(0,n):
        uniList.append(random.uniform(a,b))
    return np.array(uniList)

def gaussGenerate(mu, sigma, n):
    # generate n Gauss distributted real numbers in the range of (a,b), duplication is permitted 
    gaussList = []
    for i in range(0,n):
        gaussList.append(random.gauss(mu,sigma))
    return np.array(gaussList)

def experimentalDatasetGenerate(noiseFraction = 0.5):
    # generate 2D experimental synthetic dataset
    # let's set the approximate range of the experimental picture is about 100*100
    clusterScale = 800
    c1x = uniGenerate(10,45,clusterScale)
    c1y = c1x + gaussGenerate(0,1,clusterScale)
    c2x = uniGenerate(15,45,clusterScale)
    c2y = c2x + gaussGenerate(0,1,clusterScale) + [15]
    c3dist = gaussGenerate(20,1.5,clusterScale*3)
    c3angle = uniGenerate(0, 2*pi, clusterScale*3)
    c3x = c3dist * np.cos(c3angle) + [75]
    c3y = c3dist * np.sin(c3angle) + [75]
    c4dist = gaussGenerate(8,1,clusterScale)
    c4angle = uniGenerate(0, 2*pi, clusterScale)
    c4x = c4dist * np.cos(c4angle) + [75]
    c4y = c4dist * np.sin(c4angle) + [75]
    c5x = gaussGenerate(60,3,clusterScale)
    c5y = gaussGenerate(15,2,clusterScale)
    cx = np.concatenate((c1x, c2x, c3x, c4x, c5x))
    cy = np.concatenate((c1y, c2y, c3y, c4y, c5y))
    tags = np.array([1]*clusterScale + [2]*clusterScale + [3]*clusterScale*3 + [4]*clusterScale + [5]*clusterScale)
    dataset = np.concatenate((cx, cy, tags)).reshape(3,clusterScale*7).T
    noiseNum = floor(7*clusterScale*noiseFraction/(1-noiseFraction))
    noisex = uniGenerate(0,100,noiseNum)
    noisey = uniGenerate(0,100,noiseNum)
    noiseSet = np.concatenate((noisex, noisey, np.array([0]*noiseNum))).reshape(3,noiseNum).T
    c1judge = np.logical_and(np.logical_and(10 < noiseSet[:,0], noiseSet[:,0] < 45), \
                             np.power(noiseSet[:,1] - noiseSet[:,0], 2) < 3)
    noiseSet[c1judge,2] = 1
    c2judge = np.logical_and(np.logical_and(15 < noiseSet[:,0], noiseSet[:,0] < 45), \
                             np.power(noiseSet[:,1] - noiseSet[:,0] - [15], 2) < 3)
    noiseSet[c2judge,2] = 2
    c3judge = np.logical_and(17 < np.sqrt(np.power(noiseSet[:,0] - [75], 2) + np.power(noiseSet[:,1] - [75], 2)), \
                             np.sqrt(np.power(noiseSet[:,0] - [75], 2) + np.power(noiseSet[:,1] - [75], 2))< 23)
    noiseSet[c3judge,2] = 3
    c4judge = np.logical_and(6 < np.sqrt(np.power(noiseSet[:,0] - [75], 2) + np.power(noiseSet[:,1] - [75], 2)), \
                             np.sqrt(np.power(noiseSet[:,0] - [75], 2) + np.power(noiseSet[:,1] - [75], 2))< 10)
    noiseSet[c4judge,2] = 4
    c5judge = np.sqrt(np.power(noiseSet[:,0] - 60,2) + np.power(noiseSet[:,1]-15,2)) < 5
    noiseSet[c5judge,2] = 5
    dataset = np.concatenate((dataset, noiseSet))
    return dataset

def scale_01_data(rawData):
    # normalize the raw dataset
    dim = rawData.shape[1]  
    # the rawData has at least 2 raw, 1 for signal 1 for lable
    minList = [np.amin(rawData[:,x]) for x in range(0, dim)]
    maxList = [np.amax(rawData[:,x])+0.001 for x in range(0, dim)] 
    # add the [0] and [1] because there is a 'raw of lable', and 0.001 to avoid 1
    toZero = rawData - np.array(minList)
    normData = toZero / (np.array(maxList) - np.array(minList))
    return(normData)

def map2ScaleDomain(dataset, scale=128):
    # map the dataset into scale domain for wavelet transform
    if scale <= 0 or not(isinstance(scale, int)):
        raise ValueError('scale must be a positive interger')
    dim = dataset.shape[1]
    length = dataset.shape[0]
    sd_data = {}
    for i in range(0,length):
        num = 0
        for j in reversed(range(0, dim)):     # start from the most weighted dimension
            num += (dataset[i,j]//(1/scale))*pow(scale, j)  # let the numbering start from '0'!
        num = int(num)
        if sd_data.get(num, 'N/A')=='N/A':
            sd_data[num] = 1
        else: sd_data[num] += 1
    # print(("%d points have been mapped to scale domain \n") % (len(sd_data)))
    return(sd_data)

def ndWT(data, dim, scale, wave):
    # calculate 1 order n dimensional wavelet transform with numbered grids
    wavelets = {'db1':[0.707, 0.707], 'bior1.3':[-0.09, 0.09, 0.707, 0.707, 0.09, -0.09], \
                'db2':[-0.13, 0.224, 0.836, 0.483]}
    lowFreq = {}
    convolutionLen = len(wavelets.get(wave))-1
    lineLen = ceil(scale/2) + ceil((convolutionLen-2)/2)
    for inDim in range(0, dim):
        for key in data.keys():
            coordinate = [] # coordinate start from 0
            tempkey = key
            for i in range(0,dim):
                # get the coordinate for a numbered grid
                if i <= dim-inDim-1:
                    coordinate.append(tempkey//pow(scale, (dim-1-i)))
                    tempkey = tempkey%pow(scale, (dim-1-i))
                else:
                    coordinate.append(tempkey//pow(lineLen, (dim-1-i)))
                    tempkey = tempkey%pow(lineLen, (dim-1-i))
            coordinate.reverse()
            # if inDim == 0:  print(coordinate,'\n')
            startCoord = ceil((coordinate[inDim]+1)/2)-1    # to calculate ndwt, signal should start from 1, temperally convert
            startNum = 0    # numbered lable for next level of data
            for i in range(0, dim):
                if i <= inDim:
                    if i == inDim:
                        startNum += startCoord*pow(lineLen, i)
                    else:
                        startNum += coordinate[i]*pow(lineLen, i)
                else:
                    startNum += coordinate[i]*pow(scale, i)
            wavelet = wavelets.get(wave)   # for convolution
            # wavelet.reverse()
            for i in range(0, convolutionLen//2+1):  
                if startCoord+i >= lineLen: # coordinate start from 0 
                    break
                if lowFreq.get(int(startNum+pow(lineLen, inDim)*i), 'N/A') == 'N/A':
                    lowFreq[int(startNum+pow(lineLen, inDim)*i)] = \
                            data[key]*wavelet[int((startCoord+1+i)*2-(coordinate[inDim]+1))]
                else:
                    lowFreq[int(startNum+pow(lineLen, inDim)*i)] += \
                            data[key]*wavelet[int((startCoord+1+i)*2-(coordinate[inDim]+1))]
        data = lowFreq
        lowFreq = {}
    # print(("after dwt, there are %d datapoints") % (len(data)))
    return(data)

# start node checking
class node():
    def __init__(self,key=0,value=0):
        self.key = key
        self.value = value
        self.process = False
        self.cluster = None
    def around(self,scale=1,dim=1):
        aroundNodeKey = []
        coordinate = []
        for inDim in range(0,dim):
            # we can't afford diagnal searching
            dimCoord = self.key//pow(scale,inDim)
            if dimCoord == 0:
                aroundNodeKey.append(self.key+pow(scale,inDim))
            elif dimCoord == scale-1:
                aroundNodeKey.append(self.key-pow(scale,inDim))
            else:
                aroundNodeKey.append(self.key+pow(scale,inDim))
                aroundNodeKey.append(self.key-pow(scale,inDim))
        return(aroundNodeKey)

# def checkNode(data,point,scale,dim):
#     point.process = True
#     aroundKey = [point]
#     for around in point.around(scale,dim):
#         if not(data.get(around,'N/A')=='N/A'):
#             if data.get(around).process == False:
#                 aroundPoints = checkNode(data,data.get(around),scale,dim)
#                 aroundKey = aroundKey + aroundPoints
#     return(aroundKey)
# 
# def clustering(data,startNode,scale,dim,cutMiniCluster):
#     result = {1:checkNode(data,startNode,scale,dim)}
#     for point in data.values():
#         if not(point.process):
#             pointLink = checkNode(data,point,scale,dim)
#             if len(pointLink)==1:
#                 if pointLink[0].value <= cutMiniCluster:
#                     continue
#             result[len(result)+1] = pointLink
#     return(result)

def bfs(equal_pair,maxQueue):
    if equal_pair == []:
        return(equal_pair)
    group = {x:[] for x in range(1,maxQueue)}
    result = []
    for x,y in equal_pair:
        group[x].append(y)
        group[y].append(x)
    for i in range(1,maxQueue):
        if i in group:
            if group[i] == []:
                del group[i]
            else:
                queue = [i]
                for j in queue:
                    if j in group:
                        queue += group[j]
                        del group[j]
                record = list(set(queue))
                record.sort()
                result.append(record)
    return(result)

def build_key_cluster(nodes,equal_list,cutMiniCluster):
    cluster_key = {}
    for point in nodes.values():
        flag = 0
        for cluster in equal_list:
            if point.cluster in cluster:
                point.cluster = cluster[0]
                if cluster_key.get(cluster[0],'N/A') == 'N/A':
                    cluster_key[cluster[0]] = [point]
                    flag = 1
                else:
                    cluster_key[cluster[0]].append(point)
                    flag = 1
                break
        if flag == 0:
            if cluster_key.get(point.cluster,'N/A') == 'N/A':
                cluster_key[point.cluster] = [point]
            else:
                cluster_key[point.cluster].append(point)
    count = 1
    result = {}
    # ipdb.set_trace()
    for cluster in cluster_key.keys():
        # ipdb.set_trace()
        if len(cluster_key[cluster]) == 1:
            if cluster_key[cluster][0].value < cutMiniCluster:
                continue
        for p in cluster_key[cluster]:
            result[p.key] = count
        count += 1
    return(result)

def clustering(data, scale, dim, cutMiniCluster):
    equal_pair = []
    cluster_flag = 1
    for point in data.values():
        point.process = True
        for around in point.around(scale, dim):
            if not(data.get(around, 'N/A')=='N/A'):
                around = data.get(around)
                if around.cluster is not None:
                    if point.cluster is None:
                        point.cluster = around.cluster
                    elif point.cluster != around.cluster:
                        mincluster = min(point.cluster, around.cluster)
                        maxcluster = max(point.cluster, around.cluster)
                        equal_pair += [(mincluster,maxcluster)]
        if point.cluster is None:
            point.cluster = cluster_flag
            cluster_flag += 1
    #-------------------
    # def maptest():
    #     board = np.zeros((int(scale),int(scale)),dtype=np.int) 
    #     for point in data.values():
    #         x = point.key // scale
    #         y = point.key % scale
    #         board[int(x),int(y)] = point.cluster
    #     print(board)
    #-------------------
    equal_pair = set(equal_pair)
    equal_list = bfs(equal_pair,cluster_flag)
    # ipdb.set_trace()
    # maptest()
    result = build_key_cluster(data,equal_list,cutMiniCluster)
    return(result)

def thresholding(data,threshold,scale,dim):
    nodes = {}
    result = {}
    startNode = node(0)
    avg = 0
    for key,value in data.items():
        if value >= threshold:
            nodes[key]=node(key,value)
            avg += value
            if value > startNode.value:
                startNode = node(key,value)
    cutMiniCluster = avg/len(nodes)
    # ipdb.set_trace()
    # clusters = clustering(nodes,startNode,scale,dim,cutMiniCluster)
    clusters = clustering(nodes,scale,dim,cutMiniCluster)
    # for tag in clusters.keys():
    #     for point in clusters.get(tag):
    #         result[point.key] = tag
    return(clusters)

def findThreshold(data,threshold):
    value = list(data.values())
    value.sort(reverse=True)
    # 'cutMiniCluster' is used to throw away the single grid
    x = [i for i in range(1,len(value)+1)]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x,value)
    ax.axhline(y=threshold,xmin=0,xmax=1,color='r')
    plt.show()

def markData(normData,cluster,scale):
    dim = normData.shape[1]
    # there is a column for tags
    tags = []
    for point in range(0,normData.shape[0]):
        number = 0
        for inDim in range(0,dim):
            number += (normData[point,inDim]//(1/scale))*pow(scale,inDim)
        if cluster.get(int(number),'N/A')=='N/A':
            tags.append(0)
        else:
            tags.append(cluster.get(int(number)))
    return(tags)

def waveCluster(data,scale=50,wavelet='db2',threshold=0.5,plot=False):
    # the input data can only have one more row represent tags at the end
    waveletlen = {'db1':0,'db2':1,'bior1.3':2}
    normData = scale_01_data(data)
    dim = normData.shape[1]
    dataDic = map2ScaleDomain(normData,scale)
    dwtResult = ndWT(dataDic,dim,scale,wavelet)
    if plot: findThreshold(dwtResult,threshold)
    lineLen = scale//2+waveletlen.get(wavelet)
    result = thresholding(dwtResult,threshold,lineLen,dim)
    tags = markData(normData,result,lineLen)
    return(tags)

def draw2Darray(x,y,tag):
    # draw a picture for 2d dataset
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    color = tag / np.amax(tag)
    rgb = plt.get_cmap('jet')(color)
    ax.scatter(x,y,color = rgb)
    plt.show()

def heatmap2D(data,lineLen):
    # draw the 2d map
    intensity = np.zeros((lineLen,lineLen))
    x = list(range(0,lineLen))
    y = x
    for key in data.keys():
        xIn = key % (lineLen) 
        yIn = key // (lineLen) 
        intensity[int(xIn),int(yIn)] = data.get(key)
    x, y = np.meshgrid(x,y)
    plt.pcolormesh(x,y,intensity.T)
    plt.colorbar()
    plt.show()

