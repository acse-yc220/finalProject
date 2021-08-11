from os import remove
from hdbscan import validity
from matplotlib import colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import xlrd
from matplotlib.colors import LogNorm
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering,DBSCAN,OPTICS
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from DBCV import DBCV
from scipy.spatial.distance import euclidean
from sklearn.metrics import f1_score   
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE

def plot_input(fig,axarr,i,j,value,color,label,vmin=np.nan,vmax=np.nan):
    if np.isnan(vmin):
        vmin=np.nanmin(value)
    if np.isnan(vmax):
        vmax=np.nanmax(value)
    a=axarr[i][j].imshow(value, extent=(np.amin(data['x']), np.amax(data['x']), 
         np.amax(data['z']),np.amin(data['z'])), vmin=vmin, vmax=vmax,
        cmap=color)
    axarr[i][j].set_xlabel('x')
    axarr[i][j].set_ylabel('z')
    cb = fig.colorbar(a,ax=axarr[i][j])
    cb.set_label(label)


def show_data(dataset,nr,nc):
    for i in range(len(dataset)):
        data_inf = np.isinf(dataset[i])
        dataset[i][data_inf] = np.nan
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

def subsample(data,n):
    x = data[:,-2]
    z = data[:,-1]
    nx = np.unique(x).size
    nz = np.unique(z).size
    dm = np.reshape(data,(nz,nx,data.shape[1]))
    dmo = dm[::n,::n,:]
    datasub = np.reshape(dmo,(dmo.shape[0]*dmo.shape[1],dmo.shape[2]))
    return datasub

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
    dataset[:,3]=np.clip(dataset[:,3],None,2)
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

def kMeans(n, dataset):
    cluster_k = KMeans(n_clusters=n,random_state=0).fit(dataset)
    result_k = cluster_k.labels_
    return result_k

def Hdbscan(dataset, min_size = 5,min_sample=5,method='eom',epsilon=0.0,alpha=1.0):
    cluster_h = hdbscan.HDBSCAN(min_cluster_size=min_size, min_samples=min_sample,
                               gen_min_span_tree=True,cluster_selection_method=method,
                               cluster_selection_epsilon=epsilon,
                               alpha=alpha)
    cluster_h.fit(dataset)
    result_h = cluster_h.labels_
    score = cluster_h.relative_validity_
    return result_h, score

def spectral(n, dataset):
    cluster_sc = SpectralClustering(n_clusters=n,random_state=8).fit(dataset)
    result_sc = cluster_sc.labels_
    return result_sc

def ag(n, dataset,linkage='ward'):
    cluster_ag = AgglomerativeClustering(n_clusters=n,linkage=linkage).fit(dataset)
    result_ag = cluster_ag.labels_
    return result_ag

def dbscan(dataset, eps=0.5,min_sample=10,leaf_size=30):
    cluster = DBSCAN(eps=eps, min_samples=min_sample,leaf_size=leaf_size)
    cluster.fit(dataset)
    result = cluster.labels_
    return result

def optics(dataset, max_eps=np.inf,min_sample=10,xi=0.05,min_cluster_size=None):
    cluster = OPTICS(max_eps=max_eps, min_samples=min_sample,xi=xi,min_cluster_size=min_cluster_size)
    cluster.fit(dataset)
    result = cluster.labels_
    return result

def output_2D(result, nan_list):
    output = np.zeros(len(nan_list))
    output[~nan_list] = result
    output[nan_list] = -2
    output = output.reshape(len(data['z']),len(data['x']))
    return output

def output_addNan(result, nan_list):
    output = np.zeros(len(nan_list))
    output[~nan_list] = result
    output[nan_list] = -2
    return output

# def plotResult(dataset, result):
#     x,y = np.meshgrid(dataset['x'],dataset['z'])
#     plt.scatter(x,y,c=result,cmap=plt.cm.RdYlBu)
    
#     plt.gca().invert_yaxis()

# # #plt.scatter(centroids[i][0],centroids[i][1],linewidth=3,s=300,marker='+',color='black')
#     plt.show()

def plotResult(dataset, result):
    plt.scatter(dataset[:,-2],dataset[:,-1],c=result,cmap=plt.cm.RdYlBu)
    plt.gca().invert_yaxis()
    plt.show()

def Kmeans_cluster_scores(dataset):
    ch = []
    sc = []
    db = []
    x = range(2,15,1)  
    for i in x:
        result = kMeans(i,dataset)
        score = metrics.calinski_harabasz_score(dataset, result)
        ch.append(score)
        score = metrics.silhouette_score(dataset, result, metric='euclidean')
        sc.append(score)
        score=metrics.davies_bouldin_score(dataset, result)
        db.append(score)                  
    fig, axarr = plt.subplots(3, 1, figsize=(9, 10)) # nr is the number of rows # nc is the number of columns                 
    a0=axarr[0].plot(x,ch)
    axarr[0].set_ylabel('CH')
    a1=axarr[1].plot(x,sc)
    axarr[1].set_ylabel('SL')
    a2=axarr[2].plot(x,db)
    axarr[2].set_ylabel('DB')
    axarr[2].set_xlabel('Number of clusters')
    plt.show()

def calculatef1(pred, true):
    true_label = true['classes'].reshape(-1,1)  
    predict_label = pred.reshape(-1,1)    
    f1 = f1_score(true_label,predict_label, average='micro')
    print("F1-score is: ",f1)

def purity_score(true, pred):
    true_label = true['classes'].reshape(-1,1)  
    predict_label = pred.reshape(-1,1) 
    y_voted_labels = np.zeros(true_label.shape)
    print("shape: ",y_voted_labels.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(true_label)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        true_label[true_label==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(true_label)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(predict_label):
        hist, _ = np.histogram(true_label[predict_label==cluster], bins=bins)
        print(hist)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[predict_label==cluster] = winner
    return accuracy_score(true_label, y_voted_labels)
    
    

def plot_dimensionality_reduction(dataset, result):
    projection = TSNE().fit_transform(dataset)
    sns.set_style('white')
    sns.set_color_codes()
    color_palette = sns.color_palette('Paired', 15)
    cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in result]

    plt.scatter(*projection.T,c=result,cmap=plt.cm.RdYlBu, s=1)
    plt.show()

def plot_cluster_distribution(output):
    print(int(max(output)))
    bins=np.arange(int(max(output)+3))-1.5
    print(bins)
    plt.hist(output,bins)
    plt.tight_layout()
    plt.xlim(bins[0],bins[-1])
    plt.show()



def hdbscan_param(dataset,method='eom',epsilon=0):
    score = 0
    ch = 0
    min_size = 0
    min_sample = 0
    #scores=[]
    print("Running grid search over hyperparameters")
    for i in range(10,800,50):
        for j in range(1,100,5):
            result_h, validity_score = Hdbscan(dataset,min_size=i,min_sample=j,method=method)
            ch_score = metrics.calinski_harabasz_score(dataset, result_h)
#            print(validity_score,ch_score)

            if validity_score > score:
                score = validity_score
                min_size = i
                min_sample = j
                print('Validity: '+str(validity_score)+', min_cluster_size: '+str(min_size)+
                      ', min_sample_size: '+str(min_sample))
            if ch_score > ch:
                ch = ch_score
                ch_min_size = i
                ch_min_sample = j
                print('CH score: '+str(ch_score)+', min_cluster_size: '+str(ch_min_size)+
                      ', min_sample_size: '+str(ch_min_sample))
            #scores.append([i,j,validity_score,ch_score])
    print("Optimal values of hyperparameters:")
    print("Validity: "+str(score)+", min_cluster_size: "+str(min_size)+", min_sample_size: "+str(min_sample))
    print("CH score: "+str(ch)+", min_cluster_size: "+str(ch_min_size)+", min_sample_size: "+str(ch_min_sample))


def hdbscan_cluster(dataset, min_sample):
    score = []
    ch = []
    for i in range(10,1500,10):
        result_h, validity_score = Hdbscan(dataset, min_size = i, min_sample=min_sample)
        score.append(validity_score)
        ch_score = metrics.calinski_harabasz_score(dataset, result_h)
        ch.append(ch_score)
    x = range(10,1500,10)
    fig, axarr = plt.subplots(2,1,figsize=(9,10))
    a0 = axarr[0].plot(x,score)
    axarr[0].set_ylabel('Relative Validity score')
    a1 = axarr[1].plot(x,ch)
    axarr[1].set_ylabel('CH')
    axarr[1].set_xlabel('Min Cluster Size')
    plt.show()

def hdbscan_sample(dataset, min_size):
    score = []
    ch = []
    for i in range(1,100,5):
        result_h, validity_score = Hdbscan(dataset, min_size = min_size, min_sample=i)
        score.append(validity_score)
        ch_score = metrics.calinski_harabasz_score(dataset, result_h)
        ch.append(ch_score)
    x = range(1,100,5)
    fig, axarr = plt.subplots(2,1,figsize=(9,10))
    a0 = axarr[0].plot(x,score)
    axarr[0].set_ylabel('Relative Validity score')
    a1 = axarr[1].plot(x,ch)
    axarr[1].set_ylabel('CH')
    axarr[1].set_xlabel('Min Sample Size')
    plt.show()

def dbscan_param(dataset):
    score = 0
    ch = 0
    eps = 0
    min_sample = 0
    print("Running grid search over hyperparameters")
    for i in np.arange(0.05,0.2,0.01):
        for j in range(2,20,1):
            result_db = dbscan(dataset, eps=i,min_sample=j)
            ch_score = metrics.calinski_harabasz_score(dataset, result_db)
            if ch_score > ch:
                ch = ch_score
                eps = i
                min_sample = j
                print('CH score: '+str(ch_score)+', eps: '+str(eps)+
                      ', min_sample_size: '+str(min_sample))
            #scores.append([i,j,validity_score,ch_score])
    print("Optimal values of hyperparameters:")
    print("CH score: "+str(ch)+", eps: "+str(eps)+", min_sample_size: "+str(min_sample))

def ag_param(dataset):
    ch = []
    sc = []
    db = []
    x = range(2,15,1)  
    for i in x:
        result = ag(i,dataset)
        score = metrics.calinski_harabasz_score(dataset, result)
        ch.append(score)
        score = metrics.silhouette_score(dataset, result, metric='euclidean')
        sc.append(score)
        score=metrics.davies_bouldin_score(dataset, result)
        db.append(score)                  
    plot_validation(x,ch,sc,db) 

def sc_param(dataset):
    ch = []
    sc = []
    db = []
    x = range(2,15,1)  
    for i in x:
        result = spectral(i,dataset)
        score = metrics.calinski_harabasz_score(dataset, result)
        ch.append(score)
        score = metrics.silhouette_score(dataset, result, metric='euclidean')
        sc.append(score)
        score=metrics.davies_bouldin_score(dataset, result)
        db.append(score)     
    plot_validation(x,ch,sc,db)             


def plot_validation(x,ch,sc,db):
    fig, axarr = plt.subplots(3, 1, figsize=(9, 10))  
    a0=axarr[0].plot(x,ch)
    axarr[0].set_ylabel('CH')
    a1=axarr[1].plot(x,sc)
    axarr[1].set_ylabel('SL')
    a2=axarr[2].plot(x,db)
    axarr[2].set_ylabel('DB')
    axarr[2].set_xlabel('Number of clusters')
    plt.show()

def calculateAri(result1, result2):
    ari = metrics.adjusted_rand_score(result1,result2)
    return ari

def calculateAmi(result1, result2):
    ami = metrics.adjusted_mutual_info_score(result1, result2)
    return ami

def calculateVm(result1, result2):
    vm = metrics.v_measure_score(result1, result2) 
    return vm

def plotInputData(input_data):
    x,y = np.meshgrid(input_data['x'],input_data['z'])
    plt.scatter(x,y,c=input_data['classes'],cmap=plt.cm.RdYlBu)
    plt.gca().invert_yaxis()
    plt.show()

def kmeans_param_ch(dataset):
    score = 0
    cluster = 0
    for i in range(2,15,1):
        result_k = kMeans(i,dataset)
        ch = metrics.calinski_harabasz_score(dataset,result_k)
        if ch >score:
            score = ch
            cluster = i
    print("{'the best number of cluster': "+str(cluster)+" }")

def kmeans_param_sc(dataset):
    score = 0
    cluster = 0
    for i in range(2,15,1):
        result_k = kMeans(i,dataset)
        sc = metrics.silhouette_score(dataset,result_k)
        if sc >score:
            score = sc
            cluster = i
    print("{'the best number of cluster': "+str(cluster)+" }")

def kmeans_param_db(dataset):
    score = 2
    cluster = 0
    for i in range(2,15,1):
        result_k = kMeans(i,dataset)
        db = metrics.davies_bouldin_score(dataset,result_k)
        if db < score:
            score = db
            cluster = i
    print("{'the best number of cluster': "+str(cluster)+" }")


def ag_param_ch(dataset):
    score = 0
    cluster = 0
    for i in range(2,15,1):
        result_ag = ag(i,dataset)
        ch = metrics.calinski_harabasz_score(dataset,result_ag)
        if ch >score:
            score = ch
            cluster = i
    print("{'the best number of cluster': "+str(cluster)+" }")

def ag_param_sc(dataset):
    score = 0
    cluster = 0
    for i in range(2,15,1):
        result_ag = ag(i,dataset)
        sc = metrics.silhouette_score(dataset,result_ag)
        if sc >score:
            score = sc
            cluster = i
    print("{'the best number of cluster': "+str(cluster)+" }")

def ag_param_db(dataset):
    score = 2
    cluster = 0
    for i in range(2,15,1):
        result_ag = ag(i,dataset)
        db = metrics.davies_bouldin_score(dataset,result_ag)
        if db < score:
            score = db
            cluster = i
    print("{'the best number of cluster': "+str(cluster)+" }")

def sc_param_ch(dataset):
    score = 0
    cluster = 0
    for i in range(2,15,1):
        result_sc = spectral(i,dataset)
        ch = metrics.calinski_harabasz_score(dataset,result_sc)
        if ch >score:
            score = ch
            cluster = i
    print("{'the best number of cluster': "+str(cluster)+" }")

def sc_param_sc(dataset):
    score = 0
    cluster = 0
    for i in range(2,15,1):
        result_sc = spectral(i,dataset)
        sc = metrics.silhouette_score(dataset,result_sc)
        if sc >score:
            score = sc
            cluster = i
    print("{'the best number of cluster': "+str(cluster)+" }")

def sc_param_db(dataset):
    score = 2
    cluster = 0
    for i in range(2,15,1):
        result_sc = spectral(i,dataset)
        db = metrics.davies_bouldin_score(dataset,result_sc)
        if db < score:
            score = db
            cluster = i
    print("{'the best number of cluster': "+str(cluster)+" }")


def op_param(dataset):
    score = 0
    ch = 0
    min_sample = 0
    xi= 0
    cluster_size = 0
    #scores=[]
    print("Running grid search over hyperparameters")
    for i in range(10,70,10):
        for j in np.arange(0.005,0.01,0.001):
            for k in range(10,200,10):
                result_op = optics(dataset,min_sample = i,xi=j,min_cluster_size = k)
                ch_score = metrics.calinski_harabasz_score(dataset, result_op)
    #            print(validity_score,ch_score)
                if ch_score > ch:
                    ch = ch_score
                    min_sample = i
                    xi = j
                    cluster_size = k
                    print('CH score: '+str(ch)+', min sample: '+str(min_sample)+
                        ', xi: '+str(xi)+', min cluster size: '+str(cluster_size))
            #scores.append([i,j,validity_score,ch_score])
    print("Optimal values of hyperparameters:")
    print('CH score: '+str(ch)+', min sample: '+str(min_sample)+
                        ', xi: '+str(xi)+', min cluster size: '+str(cluster_size))


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