import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt

dataset= pd.read_json('iris.json')
dataset= dataset.drop(columns= 'species')
X= dataset.iloc[:, :].values

from sklearn.decomposition import PCA
pca= PCA(n_components= 2)
X= pca.fit_transform(X)
#explained_variance= pca.explained_variance_ratio_

from sklearn.cluster import KMeans
wcss= list()
for i in range(1, 11):
    kmeans= KMeans(n_clusters= i, init= 'k-means++', n_init= 10, max_iter= 300)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans= KMeans(n_clusters= 3, init= 'k-means++', n_init= 10, max_iter= 300)
y_kmeans= kmeans.fit_predict(X)

plt.scatter(X[y_kmeans== 1, 0], X[y_kmeans== 1, 1], c= 'red', s= 100, label= 'setosa')
plt.scatter(X[y_kmeans== 0, 0], X[y_kmeans== 0, 1], c= 'blue', s= 100, label= 'versicolor')
plt.scatter(X[y_kmeans== 2, 0], X[y_kmeans== 2, 1], c= 'yellow', s= 100, label= 'virginica')
plt.title("Iris")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()