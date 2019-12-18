import sys
import data
import pickle
import plot_results
import numpy as np
from mpi4py import MPI
from kmeans_parallel import ParallelKMeansClustering

if __name__ == '__main__':

    # initialise MPI
    mpi_comm = MPI.COMM_WORLD
    n_procs  = mpi_comm.Get_size()
    rank     = mpi_comm.Get_rank()

    if rank == 0:
        if len(sys.argv) == 1:
            print('No argument passed. Valid input arguments are "simple_data", "training_data" and "full_data".')
            sys.exit()
        if sys.argv[1] != 'simple_data' and sys.argv[1] != 'training_data' and sys.argv[1] != 'full_data':
            print('Invalid input argument. Valid input arguments are "simple_data", "training_data" and "full_data".')
            sys.exit()

        # import dataset
        if sys.argv[1] == 'simple_data':
            data_set = data.simple_data()
        elif sys.argv[1] == 'training_data':
            data_set = data.sst_cube(purpose='training')
        elif sys.argv[1] == 'full_data':
            data_set = data.sst_cube(purpose='full_run')

        # define list to store mean standard deviation
        store_mean_std = []
    else:
        if len(sys.argv) == 1:
            print('No argument passed. Valid input arguments are "simple_data", "training_data" and "full_data".')
            sys.exit()
        if sys.argv[1] != 'simple_data' and sys.argv[1] != 'training_data' and sys.argv[1] != 'full_data':
            print('Invalid input argument. Valid input arguments are "simple_data", "training_data" and "full_data".')
            sys.exit()
        data_set = None

    # broadcast cube from rank 0 to all other ranks
    data_set = mpi_comm.bcast(data_set, root=0)

    if sys.argv[1] == 'simple_data':
        k = 10
        store_time = np.zeros(10)
        for i in range(len(store_time)):
            run = ParallelKMeansClustering(data_set,
                                           mpi_comm=mpi_comm,
                                           n_procs=n_procs,
                                           rank=rank,
                                           k=k,
                                           init_centroid='kmeans++',
                                           max_iter=300)
            run.fit()
            if rank == 0:
                print(f'Cluster: {k}')
                store_time[i] = run.end-run.start

        if rank == 0:
            print(store_time)
            print(f'Mean time: {np.mean(store_time)}')

    elif sys.argv[1] == 'training_data':
        # run k-means algorithm for different number of clusters
        k = 50
        i = 0
        while i < 10:
            run = ParallelKMeansClustering(data_set,
                                           mpi_comm=mpi_comm,
                                           n_procs=n_procs,
                                           rank=rank,
                                           k=k,
                                           init_centroid='kmeans++',
                                           tol=1E-5,
                                           max_iter=300)
            run.fit()

            if rank == 0:
                print(f'Cluster: {k}')
                store_mean_std.append(run.mean_std)
                end_centroid = run.centroids

                # write to file
                # file = open('files/end_centroid'+str(i)+'.pickle', 'wb')
                # pickle.dump(end_centroid, file, protocol=4)
                # file.close()

            i += 1

        if rank == 0:
            store_mean_std = np.array(store_mean_std)
            k_range = np.linspace(10, k, len(store_mean_std))
            # plot_results.decide_n_clusters(store_mean_std, k_range, savefig=True)

    elif sys.argv[1] == 'full_data':
        # do stuff
        k = 50
        run = ParallelKMeansClustering(data_set,
                                       mpi_comm=mpi_comm,
                                       n_procs=n_procs,
                                       rank=rank,
                                       k=k,
                                       init_centroid='kmeans++',
                                       max_iter=300)
        run.fit()

    if rank == 0:
        # finalise mpi
        MPI.Finalize()


### PUT IN README.md ###
# mpiexec --mca shmem posix --oversubscribe -np 6 python main.py #

