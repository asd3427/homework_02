import numpy as np  
import operator  
import matplotlib.pyplot as plt  
from sklearn import datasets,decomposition,manifold  
from itertools import cycle  
from mpl_toolkits.mplot3d import Axes3D  
def load_data():  
    swiss_roll =datasets.make_swiss_roll(n_samples=1000)  
    return swiss_roll[0],np.floor(swiss_roll[1])  
  
def LLE_components(*data):  
    X,Y=data  
    for n in [3,2,1]:  
        lle=manifold.LocallyLinearEmbedding(n_components=n)  
        lle.fit(X)  
        print("n = %d 重建误差："%n,lle.reconstruction_error_)  
  
def LLE_neighbors(*data):  
    X,Y=data  
    Neighbors=[1,2,3,4,5,15,30,100,Y.size-1]  
  
    fig=plt.figure("LLE",figsize=(9, 9))  
    
    for i,k in enumerate(Neighbors):  
        lle=manifold.LocallyLinearEmbedding(n_components=2,n_neighbors=k,eigen_solver='dense')  
        X_r=lle.fit_transform(X)  
        ax=fig.add_subplot(3,3,i+1)  
        ax.scatter(X_r[:,0],X_r[:,1],marker='o',c=Y,alpha=0.5)   
        ax.set_title("k = %d"%k)     
        plt.xticks(fontsize=10, color="darkorange")    
        plt.yticks(fontsize=10, color="darkorange")   
    plt.suptitle("LLE")  
    plt.show()  
  
X,Y=load_data()  
fig = plt.figure('data')  
ax = Axes3D(fig)  
ax.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o',c=Y)  
LLE_components(X,Y)  
LLE_neighbors(X,Y)  