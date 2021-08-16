import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
from sklearn import metrics

def loadData(data_vp,data_vs,vsmask):
    xy = vsmask[:,:2]
    mask = vsmask[:,2].reshape(-1,1)
    vp = data_vp[:,2].reshape(-1,1)*mask*1000
    vs = data_vs[:,2].reshape(-1,1)*mask*1000
    vpvs=vp*vs
    vpvs_d = vp/vs
    # Set water layer to nan
    remove=[]
    for i,x in enumerate(vpvs_d):
        if vpvs_d[i]<=1.75 and vp[i] < 5000:
            remove.append(i)

    vp[remove] = np.nan
    vs[remove] = np.nan
    vpvs[remove] = np.nan
    vpvs_d[remove] = np.nan
    data = np.concatenate((vp,vs,vpvs,vpvs_d,xy),axis=1)
    return data

def showData(dataset):
    for i in range(len(dataset)):
        data_inf = np.isinf(dataset[i])
        dataset[i][data_inf] = np.nan
    data_column = ['Vp','Vs','Vp*Vs','Vp/Vs']
    color_column = [cm.viridis_r,  cm.viridis_r, cm.magma_r, cm.magma_r, cm.magma_r]
    fig=plt.figure(figsize=(12,6))
    for i in range(dataset.shape[1]-2):
        ax = fig.add_subplot(2,2,i+1)
        a = ax.scatter(dataset[:,-2],dataset[:,-1],c=dataset[:,i],cmap=color_column[i],s=3)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        cb = fig.colorbar(a,ax=ax)
        cb.set_label(data_column[i])
        ax.invert_yaxis()
    plt.show()

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

    nan_list = np.isnan(dataset).any(axis=1)
    data_noNan = dataset[~nan_list,:]
    return data_noNan,nan_list


def showDistribution(dataset):
    data_column = ['Vp','Vs','Vp*Vs','Vp/Vs']
   # dataset[:,3]=np.clip(dataset[:,3],None,2)
    for i in range(dataset.shape[1]-2):
        ax = plt.subplot(2,2,i+1)
        data_distribution = dataset[:,i]
        value = np.isfinite(data_distribution)
        ax.set_title(data_column[i])
        plt.hist(data_distribution[value])
    plt.tight_layout()
    plt.show()

def crossplot(dataset):
    data_column = ['Vp','Vs','Vp*Vs','Vp/Vs']
    for i in range(3):
        ax = plt.subplot(2,2,i+1)
        plt.scatter(dataset[:,0], dataset[:,i+1])
        ax.set_xlabel(data_column[0])
        ax.set_ylabel(data_column[i+1])
    plt.tight_layout()
    plt.show()

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

def output(result, nan_list):
    output = np.zeros(len(nan_list))
    output[~nan_list] = result
    output[nan_list] = -2
    return output

def plotResult(dataset, result):
    plt.scatter(dataset[:,-2],dataset[:,-1],c=result,cmap=plt.cm.RdYlBu)
    plt.gca().invert_yaxis()
    plt.show()

def dbscan(dataset, eps=0.5,min_sample=10,leaf_size=30):
    cluster = DBSCAN(eps=eps, min_samples=min_sample,leaf_size=leaf_size)
    cluster.fit(dataset)
    result = cluster.labels_
    return result

def dbscan_param(dataset):
    score = 0
    ch = 0
    eps = 0
    min_sample = 0
    print("Running grid search over hyperparameters")
    for i in np.arange(0.2,0.3,0.01):
        for j in range(50,800,10):
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