import matplotlib.pyplot as plt

<<<<<<< HEAD
=======
from sklearn import datasets
>>>>>>> fc48cd6e6455e18bc5799597d4314f2c82ef9ca4
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import LocallyLinearEmbedding
import pandas as pd
<<<<<<< HEAD
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

digits = datasets.load_digits(n_class=9)
lle =LocallyLinearEmbedding()
X = digits.data
y= digits.target
target_names = digits.target_names
n_samples, n_features = X.shape
n_neighbors = 2
X_r3 = lle.fit(X, y).transform(X)
colors = ['red', 'turquoise', 'darkorange']
lw = 2
=======

wine =datasets.load_wine()
X = wine.data
y = wine.target
target_names = wine.target_names


pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

lle =LocallyLinearEmbedding(n_neighbors=50)
X_r3 = lle.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['red', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')
plt.figure()

>>>>>>> fc48cd6e6455e18bc5799597d4314f2c82ef9ca4
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r3[y == i, 0], X_r3[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LLE of IRIS dataset')

<<<<<<< HEAD
plt.show()
=======
plt.show()


>>>>>>> fc48cd6e6455e18bc5799597d4314f2c82ef9ca4
