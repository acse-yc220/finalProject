from os import remove
from hdbscan import validity
from matplotlib import colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering,DBSCAN
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from scipy.spatial.distance import euclidean 
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

def preprocessing(dataset):
    data_preprocessing = []
    for i in range(dataset.shape[1]-2):
        mean=np.nanmean(dataset[:,i])
        var=np.nanstd(dataset[:,i])
        y=(dataset[:,i]-mean)/var
        data_preprocessing.append(y)
    data_preprocessing.append(dataset[:,-2])
    data_preprocessing.append(dataset[:,-1])
    data_preprocessing = np.array(data_preprocessing).T
    return data_preprocessing


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

def output_2D(result, nan_list):
    output = np.zeros(len(nan_list))
    output[~nan_list] = result
    output[nan_list] = -2
    output = output.reshape(len(data['z']),len(data['x']))
    return output


def input_deleteNan(input, nan_list):
    input_delete = input[~nan_list]
    return input_delete


def plotResult(dataset, result):
    plt.scatter(dataset[:,-2],dataset[:,-1],c=result,cmap=plt.cm.RdYlBu_r)
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

def crossplot_result(dataset, result):
    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
    a=axarr[0].scatter(dataset[:,1],dataset[:,0],c=result,cmap=plt.cm.RdYlBu_r,s=1)
    b=axarr[1].scatter(dataset[:,0],dataset[:,3],c=result,cmap=plt.cm.RdYlBu_r,s=1)
    cb = fig.colorbar(a,ax=axarr[1],ticks=np.arange(np.min(result),np.max(result)+1))
    plt.show()


def plot_dimensionality_reduction(dataset, result):
    projection = TSNE().fit_transform(dataset)
    sns.set_style('white')
    sns.set_color_codes()
    color_palette = sns.color_palette('Paired', 15)
    cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in result]

    plt.scatter(*projection.T,c=result,cmap=plt.cm.RdYlBu_r, s=1)
    plt.show()

def plot_cluster_distribution(output):
    bins=np.arange(int(max(output)+3))-1.5
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
    plt.scatter(x,y,c=input_data['classes'],cmap=plt.cm.RdYlBu_r)
    plt.gca().invert_yaxis()
    plt.show()

data = np.load('../syntheticData/Model5b/output_fields_smooth.npz')