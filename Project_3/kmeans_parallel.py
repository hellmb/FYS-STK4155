import numpy as np
from mpi4py import MPI

class ParallelKMeansClustering():
    """
    k-means clustering class
    """

    def __init__(self, data, mpi_comm, n_procs, rank, k=2, init_centroid='random', tol=1e-4, max_iter=300):

        self.data = data
        self.k    = k
        self.init_centroid = init_centroid
        self.tol  = tol
        self.max_iter = max_iter

        self.N = self.data.shape[0]

        # define communicator, number of processors and rank
        self.comm = mpi_comm
        self.n_procs = n_procs
        self.rank = rank

        # define empty arrays for storing points per process
        self.pN = np.zeros(self.n_procs, dtype='i')

        # define centroids array on all processors
        self.centroids = np.zeros(((self.k, self.data.shape[-1])))

        # define points per process
        if self.rank == 0:
            self.pN[:] = int(self.N/self.n_procs)

            if self.N%self.n_procs != 0:
                self.pN[-1] += int(self.N%self.n_procs)

            # define global clusters on root process
            self.global_clusters = np.empty(self.data.shape[0], dtype='i')
            self.sum_global_dist = np.empty(self.k, dtype='f')
        else:
            # set global clusters to 'None' on the remaining processes
            self.global_clusters = None
            self.sum_global_dist = None

        # broadcast points-per-process to all processes
        self.comm.Bcast(self.pN, root=0)


    def split_matrix(self):
        """
        split data (matrix-like) into sub-matrices for each process
        """
        if self.rank == 0:
            start = 0
            end   = self.pN[self.rank]
        else:
            # calculate sum of elements in pN up to process index
            sum_pN = 0
            for i in range(self.rank+1):
                sum_pN += self.pN[i]
            start = self.pN[self.rank - 1] * self.rank
            end   = sum_pN

        self.mpi_data = self.data[start:end,:]

    def define_centroids(self):
        """
        define initial centroids
        """

        if self.init_centroid == 'random':

            random_indices = np.random.choice(self.data.shape[0],size=self.k)
            self.centroids = self.data[random_indices]

        elif self.init_centroid == 'kmeans++':

            self.centroids = np.zeros((self.k, self.data.shape[-1]))

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

    def cluster_data(self):
        """
        clustering algorithm
        """

        self.distances = np.zeros((self.mpi_data.shape[0], self.k))
        self.local_clusters = np.zeros(self.pN[self.rank], dtype='i')
        self.sum_local_dist = np.zeros(self.k, dtype='f')

        self.iter = 0
        error = 1000
        self.prev_wcss = 1
        while error > self.tol:

            # measure distance to centres
            for i in range(self.k):
                self.distances[:, i] = np.linalg.norm(self.mpi_data - self.centroids[i], axis=1)

            self.local_clusters[:] = np.argmin(self.distances, axis=1)

            self.comm.Gatherv(self.local_clusters, [self.global_clusters, self.pN], root=0)

            # update centroids
            if self.rank == 0:

                self.prev_centroids = self.centroids.copy()

                for i in range(self.k):
                    if self.data[self.global_clusters == i].shape[0] == 0:
                        self.centroids[i] = self.prev_centroids[i]
                    else:
                        self.centroids[i] = np.mean(self.data[self.global_clusters == i], axis=0)

            # broadcast new centroids to all processes
            self.comm.Bcast(self.centroids, root=0)

            # calculate within cluster sum of squares with new centroids
            for i in range(self.k):
                self.sum_local_dist[i] = np.sum(np.linalg.norm(
                    self.mpi_data[self.local_clusters == i] - self.centroids[i], axis=1)**2)

            # reduce with sum to root process
            self.comm.Reduce(self.sum_local_dist, self.sum_global_dist, op=MPI.SUM, root=0)

            # calculate relative error
            if self.rank == 0:

                self.iter += 1

                self.wcss = np.sum(self.sum_global_dist)
                error = np.abs(1 - self.wcss/self.prev_wcss)

                self.prev_wcss = self.wcss

                if self.iter == self.max_iter-1 and error > self.tol:
                    print('Maximum number of iterations reached. No convergence.')
                    error = 0

            # broadcast value of error to all processes
            error = self.comm.bcast(error, root=0)

    def fit(self):
        """
        fit data with k-means clustering
        """

        if self.rank == 0:
            self.define_centroids()

        # broadcast centroids to all processes
        self.comm.Bcast(self.centroids, root=0)

        self.split_matrix()

        self.comm.Barrier()
        self.start = MPI.Wtime()

        self.cluster_data()

        self.end = MPI.Wtime()

        if self.rank == 0:
            print(f'k-means clustering stopped at {self.iter+1} iterations')
            print(f'Time elapsed is {self.end-self.start} seconds')

            self.mean_std = self.mean_standard_deviation()

    def mean_standard_deviation(self):
        """
        calculate the mean standrad deviation
        :return: mean standard deviation, float
        """

        self.std = []

        for i in range(self.k):
            points_in_cluster = self.data[self.global_clusters == i]
            if points_in_cluster.shape[0] > 0:

                # calculate euclidean distance
                cluster_distance = np.linalg.norm(points_in_cluster - self.centroids[i], axis=1)

                # calculate standard deviation
                cluster_std = np.std(cluster_distance)

                # append to list of standard deviations
                self.std.append(cluster_std)
            else:
                print(f'Points in cluster {i} is {points_in_cluster.shape[0]}. No standard deviation calculated.')

        return np.mean(self.std)