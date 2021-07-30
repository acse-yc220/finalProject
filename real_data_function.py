import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans

def loadData(data_vp,data_vs,xy):
    xy = xy[:,:2]
    vp = data_vp[:,2].reshape(-1,1)
    vs = data_vs[:,2].reshape(-1,1)
    vpvs=vp*vs
    vpvs_d = vp/vs
    poisson = (vpvs_d**2-2)/(vpvs_d**2-1)/2
    data = np.concatenate((vp,vs,vpvs,vpvs_d,poisson,xy),axis=1)
    return data

def plot_input(dataset,fig,axarr,i,value,color,label,vmin=np.nan,vmax=np.nan):
    if np.isnan(vmin):
        vmin=np.nanmin(value)
    if np.isnan(vmax):
        vmax=np.nanmax(value)
    a=axarr[i].scatter(dataset[:,-2],dataset[:,-1],c=value,cmap=color)
    axarr[i].set_xlabel('x')
    axarr[i].set_ylabel('y')
    cb = fig.colorbar(a,ax=axarr[i])
    cb.set_label(label)


def showData(dataset,nr,nc):
    for i in range(len(dataset)):
        data_inf = np.isinf(dataset[i])
        dataset[i][data_inf] = np.nan
    fig, axarr = plt.subplots(nr, nc, figsize=(15, 15)) # nr is the number of rows
                                                        # nc is the number of columns
    plot_input(dataset,fig,axarr,0,dataset[:,0],cm.viridis_r,'Vp(m/s)')
    plot_input(dataset,fig,axarr,1,dataset[:,1],cm.viridis_r,'Vs(m/s)')
    plot_input(dataset,fig,axarr,2,dataset[:,2],cm.magma_r,'Vp*Vs')
    plot_input(dataset,fig,axarr,3,dataset[:,3],cm.magma_r,'Vp/Vs')
    plot_input(dataset,fig,axarr,4,dataset[:,4],cm.magma_r,'Poisson')
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
    data_column = ['Vp','Vs','Vp*Vs','Vp/Vs','Poisson']
   # dataset[:,3]=np.clip(dataset[:,3],None,2)
    for i in range(dataset.shape[1]-2):
        ax = plt.subplot(3,2,i+1)
        data_distribution = dataset[:,i]
        value = np.isfinite(data_distribution)
        ax.set_title(data_column[i])
        plt.hist(data_distribution[value])
    plt.tight_layout()
    plt.show()

def crossplot(dataset):
    data_column = ['Vp','Vs','Vp*Vs','Vp/Vs','Poisson']
    for i in range(4):
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
