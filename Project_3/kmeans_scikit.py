import data
import time
import pickle
import numpy as np
import plot_results
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

cube = data.sst_cube()
cube = cube[0:140000,:]

# cube = data.simple_data()


k_start = 50
k_end = 50
k = 10
store_time = np.zeros(1)
for i in range(len(store_time)):
# for k in range(k_start, k_end + 1):
    # start = time.time()
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, max_iter=300, n_jobs=4, verbose=0, n_init=1)
    # kmeans = KMeans(n_clusters=50, init='k-means++', max_iter=300, n_jobs=36, verbose=0)
    start = time.time()
    kmeans.fit(cube)
    sklearn_centroids = kmeans.cluster_centers_
    sklearn_labels = kmeans.labels_
    n_iter = kmeans.n_iter_
    inertia = kmeans.inertia_
    end = time.time()

    print(f'k-means clustering stopped at {n_iter} iterations')
    print(f'Time elapsed is {end - start} seconds')

    store_time[i] = end - start

print(store_time)
print(np.mean(store_time))

# file = open('files/sk_end_centroid.pickle', 'wb')
# pickle.dump(sklearn_centroids, file, protocol=4)
# file.close()
