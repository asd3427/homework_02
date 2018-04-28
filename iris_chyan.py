import numpy as np
import scipy
from sklearn import neighbors, datasets
from sklearn.decomposition import PCA   
n_neighbors = 3 #K=3
Predictmatrix = np.zeros((150,1),int)
# import some data to play with
iris = datasets.load_iris()
#print(iris)
pca=PCA(n_components=2)#PCA降維度
newData=pca.fit_transform(iris.data)    

X = newData
y = iris.target
ACC = 0

for i in range(150):#leave one out 迴圈
    test = X[i]  
    test = test.reshape(1,-1)
    train = scipy.delete(X,i,0)
    y = scipy.delete(iris.target,i,0)
    clf = neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(train, y)
    Predict = clf.predict(test)    
    Predictmatrix[i] = Predict
    if Predictmatrix[i] == iris.target[i]:
        ACC = ACC + 1
   
print("K=3正確個數:",ACC)
ACC = ACC /150    
print ("K=3正確率:",ACC)   

n_neighbors = 15  #K=15
Predictmatrix = np.zeros((150,1),int)
# import some data to play with
iris = datasets.load_iris()
#print(iris)
pca=PCA(n_components=2)
newData=pca.fit_transform(iris.data)    

X = newData
y = iris.target
ACC = 0                          

for i in range(150):#leave one out 迴圈
    test = X[i]  
    test = test.reshape(1,-1)
    train = scipy.delete(X,i,0)
    y = scipy.delete(iris.target,i,0)
    clf = neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(train, y)
    Predict = clf.predict(test)    
    Predictmatrix[i] = Predict
    if Predictmatrix[i] == iris.target[i]:
        ACC = ACC + 1
    
    
    
    #print (Predictmatrix)
print("K=15正確個數:",ACC)
ACC = ACC /150    
print ("K=15正確率:",ACC)   

    
                            
