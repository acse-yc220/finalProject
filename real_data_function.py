import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans

def loadData(dataset):
    coordinates = []
    for line in dataset:
        x,y,z = line.split()
        coordinates.append([float(z),float(x), float(y)])
    dataset.close()
    coordinates = np.array(coordinates).reshape(-1,3)
    return coordinates

def show_data(dataset):
    plt.scatter(dataset[:,1],dataset[:,2],c=dataset[:,0],cmap=cm.viridis_r)
    plt.gca().invert_yaxis()
    plt.show()

def showDistribution(dataset,fig_title):
    data_distribution = dataset[:,0]
    plt.title(fig_title)
    plt.hist(data_distribution)
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

def plotResult(dataset, result):
    plt.scatter(dataset[:,1],dataset[:,2],c=result,cmap=plt.cm.RdYlBu)
    plt.gca().invert_yaxis()
    plt.show()
