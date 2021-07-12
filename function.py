from os import remove
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import xlrd
from matplotlib.colors import LogNorm
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from DBCV import DBCV
from scipy.spatial.distance import euclidean

def plot_input(fig,axarr,i,j,value,color,label,vmin=np.nan,vmax=np.nan):
    if np.isnan(vmin):
        vmin=np.nanmin(value)
    if np.isnan(vmax):
        vmax=np.nanmax(value)
    a=axarr[i][j].imshow(value, extent=(np.amin(data['x']), np.amax(data['x']), 
         np.amax(data['z']),np.amin(data['z'])), vmin=vmin, vmax=vmax,
        cmap=color)
    cb = fig.colorbar(a,ax=axarr[i][j])
    cb.set_label(label)


def show_data(dataset,nr,nc):
    fig, axarr = plt.subplots(nr, nc, figsize=(15, 15)) # nr is the number of rows
                                                        # nc is the number of columns
    plot_input(fig,axarr,0,0,dataset[0],cm.viridis_r,'Vp(m/s)')
    plot_input(fig,axarr,0,1,dataset[1],cm.viridis_r,'Vs(m/s)')
    plot_input(fig,axarr,1,0,dataset[2],cm.magma_r,'Density(kg/m^3)')
    plot_input(fig,axarr,1,1,dataset[3],cm.magma_r,'Vp/Vs')
    plot_input(fig,axarr,2,0,dataset[4],cm.magma_r,'Qp')
    plot_input(fig,axarr,2,1,dataset[5],cm.magma_r,'Qs')
    plt.show()

def loadData(dataset):
    data = []
    for i in range(len(dataset.files)-2):
        data.append(dataset[dataset.files[i]])
    return data


def removenans(dataset):
# Function that removes points with nan values from array
# Store the indices of the removed points so that they can be added back later (after clustering)
# Option 1:
# - add a dimension to the array where you store the initial indices
# - remove elements with nans and place removed elements in new array
# - perorm clustering
# - write separate function that restores removed elements
# inputs = full dataset
# outputs = trimmed dataset and array of removed elements
    new_data = []
    for i in range(len(dataset['z'])):
        for j in range(len(dataset['x'])):
            for k in range(len(dataset.files)-2):
                new_data.append(dataset[dataset.files[k]][i][j])
            new_data.append(dataset['x'][j])
            new_data.append(dataset['z'][i])
    new_data = np.array(new_data).reshape(-1,8)
    nan_list = np.isnan(new_data).any(axis=1)
    data_noNan = new_data[~nan_list,:]
    return new_data,data_noNan,nan_list

#def restorenans(dataset,nanarray):
# - paste removed elements to back of array
# - sort elements by original index 
# inputs = trimmed dataset and array of removed elements
# outputs = restored dataset


def preprocessing(dataset):
    data_preprocessing = []
    for i in range(dataset.shape[1]-2):
        mean=np.nanmean(dataset[:,i])
        var=np.nanstd(dataset[:,i])
        y=(dataset[:,i]-mean)/var
        data_preprocessing.append(y)
    data_preprocessing.append(dataset[:,-2])
    data_preprocessing.append(dataset[:,-1])
    # data_preprocessing[1]=np.clip(data_preprocessing[1],None,2)
    # data_preprocessing[3]=np.clip(data_preprocessing[3],None,2)
    data_preprocessing = np.array(data_preprocessing).T
    print(data_preprocessing.shape)
    return data_preprocessing
    # for i in range(len(dataset.files)-2):
    #     # print(np.isnan(dataset[dataset.files[i]]))
    #      data = dataset[dataset.files[i]]
    #      data_inf = np.isinf(data)
    #      data[data_inf] = 0
    #      data_nan = np.isnan(data)
    #      data[data_nan] = 0
    #      scaler = StandardScaler()
    #      scaler.fit(data)
    #      data_normalized = scaler.transform(data)
    #      data_preprocessing.append(data_normalized)
    # return data_preprocessing

def showDistribution(dataset):
    for i in range(dataset.shape[1]):
        ax = plt.subplot(3,2,i+1)
        data_distribution = dataset[:,i]
        value = np.isfinite(data_distribution)
        ax.set_title(data.files[i])
        plt.hist(data_distribution[value])
    plt.tight_layout()
    plt.show()

def crossplot(dataset):
    for i in range(5):
        ax = plt.subplot(3,2,i+1)
        plt.scatter(dataset[:,0], dataset[:,i+1])
        ax.set_xlabel(data.files[0])
        ax.set_ylabel(data.files[i+1])
    plt.tight_layout()
    plt.show()
# def dataForm(dataset):
#      n = len(dataset)
#      row_num = dataset[0].shape[0]
#      col_num = dataset[0].shape[1]
#      data = np.zeros((row_num*col_num,n))
#      for i in range(row_num):
#          for j in range(col_num):
#              for dim in range(n):
#                  data[i*col_num+j,dim] = dataset[dim][i,j]
#      return data

def distEclud(vecA, vecB):
     return np.sqrt(np.sum(np.power(vecA - vecB, 2))) 

def randCent(dataset, k):
     n = dataset.shape[1]
     centroids = np.mat(np.zeros((k,n)))   
     for i in range(n):
         data_min = np.min(dataset[:,i])
         data_max = np.max(dataset[:,i])
         data_range = np.float(data_max-data_min)
         index = np.random.randint(dataset[0].size)
         centroids[:,i] = data_min + data_range * np.random.rand(k, 1)
     return centroids

def kMeans(n, dataset,iter=300):
    cluster_k = KMeans(n_clusters=n, max_iter=iter,random_state=0).fit(dataset)
    result_k = cluster_k.labels_
    return result_k

def Hdbscan(n, dataset):
    cluster_h = hdbscan.HDBSCAN(min_cluster_size=n, gen_min_span_tree=True)
    cluster_h.fit(dataset)
    result_h = cluster_h.labels_
    return result_h

def output_2D(result, nan_list):
    output = np.zeros(len(nan_list))
    output[~nan_list] = result
    output[nan_list] = -1
    print(len(output))
    output = output.reshape(len(data['z']),len(data['x']))
    print(output.shape)
    return output

def plotResult(dataset, result):
    x,y = np.meshgrid(dataset['x'],dataset['z'])
    plt.scatter(x,y,c=result,cmap=plt.cm.RdYlBu)
    
    plt.gca().invert_yaxis()

# #plt.scatter(centroids[i][0],centroids[i][1],linewidth=3,s=300,marker='+',color='black')
    plt.show()

def kmeans_param(dataset):
    tuned_parameters = [{'n_clusters':np.arange(2,8,1),'max_iter':np.arange(300,800,100), 'random_state':[0]}]
    gsearch = GridSearchCV(estimator = KMeans(),param_grid = tuned_parameters,cv=5)
    gsearch.fit(dataset)
    return gsearch.best_params_

def hdbscan_param(dataset):
    score = 10
    n = 0
    for i in range(2,8):
        result_h = Hdbscan(i, dataset)
        dbscore = DBCV(dataset, result_h, dist_function=euclidean)
        print(dbscore)
        if dbscore < score:
            score = dbscore
            n = i
    return n

def calculateAri(result1, result2):
    ari = metrics.adjusted_rand_score(result1,result2)
    return ari

def calculateAmi(result1, result2):
    ami = metrics.adjusted_mutual_info_score(result1, result2)
    return ami

def calculateVm(result1, result2):
    vm = metrics.v_measure_score(result1, result2) 
    return vm

def calculateSC(dataset, result):
    sc = metrics.silhouette_score(dataset, result, metric='euclidean')
    return sc

def calculateCH(dataset, result):
    ch = metrics.calinski_harabasz_score(dataset, result)
    return ch

def calculateDB(dataset, result):
    dbscore = metrics.davies_bouldin_score(dataset, result)
    return dbscore

def plotInputData(input_data):
    x,y = np.meshgrid(input_data['x'],input_data['z'])
    plt.scatter(x,y,c=input_data['classes'],cmap=plt.cm.RdYlBu)
    plt.gca().invert_yaxis()
    plt.show()

# def kMeans(dataset, k, distMeans =distEclud, createCent = randCent):
#      samples_num = dataset.shape[0]
#      clusterAssment = np.mat(np.zeros((samples_num,2)))    
#      centroids = createCent(dataset, k)
#      clusterChanged = True   
#      while clusterChanged:
#          clusterChanged = False
#          for i in range(samples_num):
#                 minDist = np.inf
#                 minIndex = -1
#                 for j in range(k):
#                     distJI = distMeans(centroids[j,:], dataset[i:])
#                     if distJI < minDist:
#                         minDist = distJI
#                         minIndex = j
#                 if clusterAssment[i,0] != minIndex: 
#                     clusterChanged = True
#                 clusterAssment[i,:] = minIndex,minDist**2   
#          clusterAssment = np.array(clusterAssment)
#          for cent in range(k):  
#              ptsInClust = dataset[np.nonzero(clusterAssment[:,0] == cent)[0]]  
#              centroids[cent,:] = np.mean(ptsInClust, axis = 0)  
#      return centroids, clusterAssment


# def HDBSCAN(dataset):
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
#     clusterer.fit(dataset)
#     print(clusterer.labels_)


# def showCluster(dataSet, k, centroids, clusterAssment):
# 	numSamples = dataSet.shape[0]

 
# 	mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
# 	if k > len(mark):
# 		print ("Sorry! Your k is too large!")
# 		return 1
 
# 	# draw all samples
# 	for i in xrange(numSamples):
# 		markIndex = int(clusterAssment[i, 0])
# 		plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
 
# 	mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
# 	# draw the centroids
# 	for i in range(k):
# 		plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)
 
# 	plt.show()

data = np.load('C:/Users/12928/Desktop/SyntheticDatasets/Model5b/output_fields_smooth.npz')
# print(len(data['x']))
# new_data,data_noNan,nan_list = removenans(data)
# data_preprocessing = preprocessing(data_noNan) 
# result_k = kMeans(4, data_preprocessing[:,:-2])
# output_k = output_2D(result_k,nan_list)
# plotResult(data, output_k)
# result_h = Hdbscan(4, data_preprocessing)
# dbscore = DBCV(data_preprocessing, result_h, dist_function=euclidean)
# print(dbscore)

#print(data[data.files[0]].shape)
#data_list = loadData(data)
# data_inf = np.isinf(data_list[3])
# data_list[3][data_inf] = np.nan

# data_list[0]= 1./data_list[0]
# data_list[1]= 1./data_list[1]
# data_inf = np.isinf(data_list[1])
# data_list[1][data_inf] = np.nan


# show_data(data_list,3,2)

#data_list[3]=np.clip(data_list[3],None,2)
#HDBSCAN(data_noNan)
#showDistribution(new_data)
# crossplot(new_data[:,0],new_data[:,1])


#data_preprocessing = preprocessing(data_noNan)







# showDistribution(data_preprocessing)


#data1 = loadData(data)
# crossplot(data1)
#data2 = preprocessing(data1)
#crossplot(data2)
# dataDistribution(data)
# show_batch(data,3,2)

#show_newData(data_new,3,2)
# data_new = dataForm(data_new)

# for i in range(len(data.files)-2):
#         ax = plt.subplot(3,2,i+1)
#         data_distribution = data[data.files[i]].reshape(1,-1)
#         value = np.isfinite(data_distribution)
#         ax.set_title(data.files[i])
#         # print(value)
#         # print(data_distribution[value])
#         plt.hist(data_distribution[value])
#         #plt.hist(data_distribution)
# plt.tight_layout()
# plt.show()

# plt.hist(a,bins=5)
# print(a)
# plt.show()  



# col = ['HotPink','Aqua','Chartreuse','yellow','red','blue','green','grey','orange'] 
# for i in range(data_noNan.shape[0]):
#     plt.scatter(data_noNan[i][0],data_noNan[i][1],color=col[cluster_h.labels_[i]])

# # #plt.scatter(centroids[i][0],centroids[i][1],linewidth=3,s=300,marker='+',color='black')
# plt.show()

