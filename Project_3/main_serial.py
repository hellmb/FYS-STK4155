import data
import plot_results
from kmeans_serial import SerialKMeansClustering

if __name__ == '__main__':

    # import data
    simple_data = data.simple_data()

    k = 10

    run = SerialKMeansClustering(simple_data,
                                 k=k,
                                 init_centroid='random',
                                 tol=1e-4,
                                 max_iter=300)

    run.fit()

    plot_results.animate_kmeans(simple_data, run.animate_centroids, run.animate_clusters, k)
