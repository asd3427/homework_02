import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import LocallyLinearEmbedding
import pandas as pd
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
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r3[y == i, 0], X_r3[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LLE of IRIS dataset')

plt.show()