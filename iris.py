import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset= pd.read_json('iris.json')
dataset= dataset.drop(columns= 'species')
X= dataset.iloc[:, :-1].values

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

plt.scatter(X[y_kmeans== 1, 0], X[y_kmeans== 1, 2], c= 'red', s= 100, label= 'setosa')
plt.scatter(X[y_kmeans== 0, 0], X[y_kmeans== 0, 2], c= 'blue', s= 100, label= 'versicolor')
plt.scatter(X[y_kmeans== 2, 0], X[y_kmeans== 2, 2], c= 'yellow', s= 100, label= 'virginica')
plt.title("Iris (Petal Length vs. Sepal Length)")
plt.xlabel("Petal Length")
plt.ylabel("Sepal Length")
plt.legend()
plt.show()