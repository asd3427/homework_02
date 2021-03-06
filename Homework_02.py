import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn import manifold,datasets

def iris():
    
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    lle =manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=2,
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
def digits():
    
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    target_names = digits.target_names

    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    lle =manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=2,
                                      method='standard') # method 有四種 每種的圖都不一樣 有的會取隨機數 所以圖繪不一樣 如下所顯示 這裡使用標準
                                                     # standard,modified,hessian,ltsa
    X_lle = lle.fit(X,y).transform(X)

# Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

    plt.figure()
#----------------------------------pca------------------------------------------------

    point = ['y*','r+', 'c>','<','^','o','x','.','d','s']
    lw=2
    for point, i, target_name in zip(point,  range(0,11), target_names):
        plt.plot(X_r[y == i, 0], X_r[y == i, 1], point, alpha=1,
                label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of Digits dataset')

    plt.figure()
#----------------------------------lda------------------------------------------------

    point2= ['y*','r+', 'c>','<','^','o','x','.','d','s']
    for point2, i, target_name in zip(point2, range(0,11), target_names):
        plt.plot(X_r2[y == i, 0], X_r2[y == i, 1], point2,alpha=.8,lw=lw,
                label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of Digits dataset')
    plt.figure()
#----------------------------------lle------------------------------------------------

    point3=  ['y*','r+', 'c>','<','^','o','x','.','d','s']
    for point3, i, target_name in zip(point3,range(0,11), target_names):
        plt.plot(X_lle[y == i, 0], X_lle[y == i, 1],point3, alpha=.8,lw=lw,
                label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LLE of Digits dataset')
    plt.show()


def wine():
    
    wine = datasets.load_wine()
    X = wine.data
    y = wine.target
    target_names = wine.target_names

    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    lle =manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=2,
                                      method='standard')
    X_r3 = lle.fit(X, y).transform(X)

# Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

    plt.figure()
    point = ['y*','r+', 'c>']
    lw=2


    for point, i, target_name in zip(point, [0, 1, 2], target_names):
        plt.plot(X_r[y == i, 0], X_r[y == i, 1], point, alpha=1,
                label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of Wine dataset')

    plt.figure()
    point2= ['y*','r+', 'c>']
    for point2, i, target_name in zip(point2, [0, 1, 2], target_names):
        plt.plot(X_r2[y == i, 0], X_r2[y == i, 1], point2,alpha=.8,lw=lw,
                label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of Wine dataset')
    plt.figure()
    point3= ['y*','r+', 'c>']
    for point3, i, target_name in zip(point3, [0, 1, 2], target_names):
        plt.plot(X_r3[y == i, 0], X_r3[y == i, 1],point3, alpha=.8,lw=lw,
                label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LLE of Wine dataset')

    plt.show()
iris()
digits()
wine()

