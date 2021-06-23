import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import xlrd
from matplotlib.colors import LogNorm
from sklearn.preprocessing import StandardScaler

def plot_input(fig,axarr,i,j,value,color,label):
    a=axarr[i][j].imshow(value, extent=(np.amin(data['x']), np.amax(data['x']), 
         np.amax(data['z']),np.amin(data['z'])),
        cmap=color)
    cb = fig.colorbar(a,ax=axarr[i][j])
    cb.set_label(label)
    #plt.xlim(0，20)
   # h = plt.contourf(value,cmap = cm.viridis_r)
  #  c = plt.colorbar(h) 

# def show_batch(dataset, nr, nc):
#     fig, axarr = plt.subplots(nr, nc, figsize=(15, 15)) # nr is the number of rows
#                                                         # nc is the number of columns
#     plot_input(fig,axarr,0,0,dataset['vp'],cm.viridis_r,'Vp(m/s)')
#     plot_input(fig,axarr,0,1,dataset['vs'],cm.viridis_r,'Vs(m/s)')
#     plot_input(fig,axarr,1,0,dataset['dn'],cm.magma_r,'Density(kg/m^3)')
#     plot_input(fig,axarr,1,1,dataset['vpvs'],cm.magma_r,'Vp/Vs')
#     plot_input(fig,axarr,2,0,dataset['qp'],cm.magma_r,'Qp^(-1)')
#     plot_input(fig,axarr,2,1,dataset['qs'],cm.magma_r,'Qs^(-1)')
#     plt.show()

def show_data(dataset,nr,nc):
    fig, axarr = plt.subplots(nr, nc, figsize=(15, 15)) # nr is the number of rows
                                                        # nc is the number of columns
    plot_input(fig,axarr,0,0,dataset[0],cm.viridis_r,'Vp(m/s)')
    plot_input(fig,axarr,0,1,dataset[1],cm.viridis_r,'Vs(m/s)')
    plot_input(fig,axarr,1,0,dataset[2],cm.magma_r,'Density(kg/m^3)')
    plot_input(fig,axarr,1,1,dataset[3],cm.magma_r,'Vp/Vs')
    plot_input(fig,axarr,2,0,dataset[4],cm.magma_r,'Qp^(-1)')
    plot_input(fig,axarr,2,1,dataset[5],cm.magma_r,'Qs^(-1)')
    plt.show()

def loadData(dataset):
    data = []
    for i in range(len(dataset.files)-2):
        data.append(dataset[dataset.files[i]])
    return data

def preprocessing(dataset):
    data_preprocessing = []
    for i in range(len(dataset)):
        data_inf = np.isinf(dataset[i])
        dataset[i][data_inf] = 0
        data_nan = np.isnan(dataset[i])
        dataset[i][data_nan] = 0
        scaler = StandardScaler()
        scaler.fit(dataset[i])
        data_normalized = scaler.transform(dataset[i])
        data_preprocessing.append(data_normalized)
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
    for i in range(len(dataset)):
        ax = plt.subplot(3,2,i+1)
        data_distribution = dataset[i].reshape(1,-1)
        value = np.isfinite(data_distribution)
        ax.set_title(data.files[i])
        plt.hist(data_distribution[value])
    plt.tight_layout()
    plt.show()

def crossplot(dataset):
    fig, ax = plt.subplots(3,2,figsize=(15, 20),subplot_kw={"projection": "3d"})
    for i in range(len(dataset)):
        xx, yy = np.meshgrid(data['x'], data['z'] )
        data_inf = np.isinf(dataset[i])
        dataset[i][data_inf] = 0
        data_nan = np.isnan(dataset[i])
        dataset[i][data_nan] = 0
        zz = dataset[i]
        row_index = i//2
        col_index = i%2
        surf = ax[row_index][col_index].plot_surface(xx, yy, zz, cmap='viridis', edgecolor='none',
                        linewidth=0, antialiased=True,  rstride=1, cstride=1,)
        ax[row_index][col_index].set_xlabel(r'x')
        ax[row_index][col_index].set_ylabel(r'z')
        ax[row_index][col_index].set_zlabel(data.files[i])
        cb = fig.colorbar(surf, shrink=0.5, aspect=5,ax=ax[row_index][col_index])
        cb.set_label(data.files[i])
    plt.tight_layout()
    plt.show()

def dataForm(dataset):
     n = len(dataset)
     row_num = dataset[0].shape[0]
     col_num = dataset[0].shape[1]
     data = np.zeros((row_num*col_num,n))
     for i in range(row_num):
         for j in range(col_num):
             for dim in range(n):
                 data[i*col_num+j,dim] = dataset[dim][i,j]
     return data

def distEclud(vecA, vecB):
     return np.sqrt(np.sum(np.power(vecA - vecB, 2))) 

def randCent(dataset, k):
     n = dataset.shape[1]
     centroids = np.zeros((k,n))   
     for i in range(k):
         index = np.random.randint(dataset[0].size)
         centroids[i,:] = dataset[index,:]
     return centroids

def kMeans(dataset, k, distMeans =distEclud, createCent = randCent):
     samples_num = dataset.shape[0]
     clusterAssment = np.mat(np.zeros((samples_num,2)))    
     centroids = createCent(dataset, k)
     clusterChanged = True   
     while clusterChanged:
         clusterChanged = False
         for i in range(samples_num):
                minDist = np.inf
                minIndex = -1
                for j in range(k):
                    distJI = distMeans(centroids[j,:], dataset[i:])
                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j
                if clusterAssment[i,0] != minIndex: 
                    clusterChanged = True
                clusterAssment[i,:] = minIndex,minDist**2   
         print(clusterAssment)
         clusterAssment = np.array(clusterAssment)
         for cent in range(k):  
             ptsInClust = dataset[np.nonzero(clusterAssment[:,0] == cent)[0]]  
             centroids[cent,:] = np.mean(ptsInClust, axis = 0)  
     return centroids, clusterAssment



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

data = np.load('C:/Users/12928/Desktop/acse9/SyntheticDatasets/Model5b/output_fields.npz')
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

#centroids,clusterAssment = kMeans(data_new,4)

# print(myCentroids)
# print(clusterAssment)
# showCluster(data_new, 4, centroids, clusterAssment)

