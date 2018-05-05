import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn import manifold,datasets



iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

lle =manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='standard')
X_r3 = lle.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
print('explained variance ratio (first two components): %s'
      % str(lda.explained_variance_ratio_))
   

plt.figure()
point = ['y*','r+', 'c>']
lw=2


for point, i, target_name in zip(point, [0, 1, 2], target_names):
        plt.plot(X_r[y == i, 0], X_r[y == i, 1], point, alpha=1,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.figure()
point2= ['y*','r+', 'c>']
for point2, i, target_name in zip(point2, [0, 1, 2], target_names):
        plt.plot(X_r2[y == i, 0], X_r2[y == i, 1], point2,alpha=.8,lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')
plt.figure()
point3= ['y*','r+', 'c>']
for point3, i, target_name in zip(point3, [0, 1, 2], target_names):
        plt.plot(X_r3[y == i, 0], X_r3[y == i, 1],point3, alpha=.8,lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LLE of IRIS dataset')

plt.show()