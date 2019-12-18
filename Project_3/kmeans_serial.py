import time
import numpy as np

class SerialKMeansClustering():
    """
    k-means clustering class
    """

    def __init__(self, data, k=2, init_centroid='random', tol=1e-4, max_iter=300):

        self.data = data
        self.k    = k
        self.init_centroid = init_centroid
        self.tol  = tol
        self.max_iter = max_iter

        self.N = self.data.shape[0]

        # define centroids array on all processors
        self.centroids = np.zeros(((self.k, self.data.shape[-1])))

    def define_centroids(self):
        """
        define initial centroids
        """

        if self.init_centroid == 'random':

            # np.random.seed(10)
            random_indices = np.random.choice(self.data.shape[0], size=self.k)
            self.centroids = self.data[random_indices]

        elif self.init_centroid == 'kmeans++':

            self.centroids = np.zeros((self.k, self.data.shape[-1]))
            # np.random.seed(30)
            random_index = np.random.choice(self.data.shape[0])

            self.centroids[0] = self.data[random_index]

            distance = np.linalg.norm(self.data - self.centroids[0], axis=1)

            i = 1
            while i < self.k:
                prob = distance**2
                random_index = np.random.choice(self.data.shape[0], size=1, p=prob/np.sum(prob))
                self.centroids[i] = self.data[random_index]

                new_distance = np.linalg.norm(self.data - self.centroids[i], axis=1)
                distance = np.min(np.vstack((distance, new_distance)), axis=0)

                i += 1

            print(self.centroids)

    def cluster_data(self):
        """
        clustering algorithm
        """

        self.distances = np.zeros((self.data.shape[0], self.k))
        self.clusters  = np.zeros(self.data.shape[0])
        self.sum_dist  = np.zeros(self.k)

        self.iter = 0
        error = 1000
        self.prev_wcss = 1
        while error > self.tol:

            # measure distance to centres
            for i in range(self.k):
                self.distances[:, i] = np.linalg.norm(self.data - self.centroids[i], axis=1)

            self.clusters[:] = np.argmin(self.distances, axis=1)

            # update centroids
            self.prev_centroids = self.centroids.copy()

            for i in range(self.k):
                if self.data[self.clusters == i].shape[0] == 0:
                    self.centroids[i] = self.prev_centroids[i]
                else:
                    self.centroids[i] = np.mean(self.data[self.clusters == i], axis=0)

            self.animate_centroids.append(self.centroids.copy())
            self.animate_clusters.append(self.clusters.copy())

            # calculate within cluster sum of squares with new centroids
            for i in range(self.k):
                self.sum_dist[i] = np.sum(np.linalg.norm(
                    self.data[self.clusters == i] - self.centroids[i], axis=1)**2)

            # calculate relative error
            self.iter += 1

            self.wcss = np.sum(self.sum_dist)
            error = np.abs(1 - self.wcss/self.prev_wcss)
            print(f'Iteration: {self.iter} Relative error: {error} WCSS: {self.wcss}')

            self.prev_wcss = self.wcss

            if self.iter == self.max_iter-1 and error > self.tol:
                print('Maximum number of iterations reached. No convergence.')
                error = 0

    def fit(self):
        """
        fit data with k-means clustering
        """

        self.define_centroids()
        self.animate_centroids = []
        self.animate_clusters  = []
        self.animate_centroids.append(self.centroids.copy())
        self.animate_clusters.append(np.zeros(self.data.shape[0]))

        self.start = time.time()

        self.cluster_data()

        self.end = time.time()

        print(f'k-means clustering stopped at {self.iter+1} iterations')
        print(f'Time elapsed is {self.end-self.start} seconds')