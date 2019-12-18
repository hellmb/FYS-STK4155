import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# generate random data in 2D space
X  = -2 * np.random.rand(100,2)
X1 = 1 + 2 * np.random.rand(50,2)

# set half of X to X1 so that we have two groups of 50 points each
X[50:100,:] = X1


X = -2 * np.random.rand(100, 2)
X1 = 2 + 2 * np.random.rand(20, 2)
X2 = 2 * np.random.rand(25, 2)
X3 = 3 * np.random.rand(25, 2)

# set half of X to X1 so that we have two groups of 50 points each
X[30:50, :] = X1
X[50:75, :] = X2
X[75:100, :] = X3

# process the randomly generated data
Kmean = KMeans(n_clusters=4)            # set number of clusters to 2
Kmean.fit(X)

# find centroids
centroid_values = Kmean.cluster_centers_

print(centroid_values.shape)

# display cluster centroids
plt.scatter(X[:,0], X[:,1], s=50, c='b')
dark_col  = ['#880E4F', '#01579B', '#33691E']
for i in range(3):
    plt.scatter(centroid_values[i][0], centroid_values[i][1], s=200, color=dark_col[i])
# plt.scatter(centroid_values[0,0], centroid_values[0,1], s=200, c='g', marker='s')
# plt.scatter(centroid_values[1,0], centroid_values[1,1], s=200, c='r', marker='s')
plt.show()

# test algorithm
labels = Kmean.labels_

sample_test = np.array([-3,-3])
second_test = sample_test.reshape(1,-1)

predict = Kmean.predict(second_test)

print(predict)

