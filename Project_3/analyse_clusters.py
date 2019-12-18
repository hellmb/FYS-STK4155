import data as dt
import pickle
import plot_results
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def wcss(data, clusters, centroids, n_clusters):
    """
    calculate the within-cluster sum of squares
    :param data: training data set, matrix
    :param clusters: clusters, matrix
    :param centroids: centroids, matrix
    :param n_clusters: number fo clusters, int
    :return: within-cluster sum of squares
    """

    sum_dist = np.zeros(n_clusters)

    # calculate within cluster sum of squares with new centroids
    for i in range(n_clusters):
        sum_dist[i] = np.sum(np.linalg.norm(data[clusters == i] - centroids[i], axis=1) ** 2)

    wcss = np.sum(sum_dist)

    return wcss

def animate_time_series(data, centroids, n_clusters, wav, savefig=False):
    """
    animate time series with specific clusters highlighted in colour
    :param data: data set, matrix
    :param centroids: centroids, matrix
    :param n_clusters: number of clusters, int
    :param wav: wavelength point, int
    :param savefig: True/False
    """

    sz = data.shape
    reshaped_data = np.reshape(data, (sz[0]*sz[1], sz[2], sz[-1]))

    fig = plt.figure()
    frames = []

    for i in range(data.shape[2]):

        print(f'Scan number: {i}')
        image = data[:, :, i, wav]

        distances = np.zeros((reshaped_data.shape[0], n_clusters))
        clusters  = np.zeros(reshaped_data.shape[0])

        for k in range(n_clusters):
            distances[:, k] = np.linalg.norm(reshaped_data[:,i,:] - centroids[k], axis=1)

        clusters[:] = np.argmin(distances, axis=1)

        cluster_to_image = np.reshape(clusters, (sz[0], sz[1]))

        y1, x1   = np.where(cluster_to_image == 1)
        y6, x6   = np.where(cluster_to_image == 6)
        y14, x14 = np.where(cluster_to_image == 14)
        y19, x19 = np.where(cluster_to_image == 19)
        y45, x45 = np.where(cluster_to_image == 45)

        cmap = plt.cm.binary
        norm = plt.Normalize(image.min(), image.max())
        rgba = cmap(norm(image))

        conv = 255
        rgba[y1, x1, :3]   = 236 / conv, 112 / conv, 99 / conv
        rgba[y6, x6, :3]   = 175 / conv, 122 / conv, 197 / conv
        rgba[y14, x14, :3] = 84 / conv, 153 / conv, 199 / conv
        rgba[y19, x19, :3] = 244 / conv, 208 / conv, 63 / conv
        rgba[y45, x45, :3] = 244 / conv, 143 / conv, 177 / conv

        frames.append([plt.imshow(rgba, origin='lower', animated=True)])

    anim = animation.ArtistAnimation(fig, frames, interval=100, blit=True, repeat_delay=0)

    if savefig:
        writer = animation.FFMpegWriter(bitrate=5000, fps=10)
        anim.save('movies/full_run_wav' + str(wav) + '.mp4', writer=writer, dpi=300)

def action_shots(data, centroids, n_clusters, wav):
    """
    save 'action shots' of the data set overplotted with specific clusters
    :param data: data set, matrix
    :param centroids: centroids, matrix
    :param n_clusters: number of clusters, int
    :param wav: wavelength point, int
    """

    sz = data.shape
    reshaped_data = np.reshape(data, (sz[0] * sz[1], sz[2], sz[-1]))

    for i in range(data.shape[2]):

        print(f'Scan number: {i}')
        image = data[:, :, i, wav]

        distances = np.zeros((reshaped_data.shape[0], n_clusters))
        clusters  = np.zeros(reshaped_data.shape[0])

        for k in range(n_clusters):
            distances[:, k] = np.linalg.norm(reshaped_data[:,i,:] - centroids[k], axis=1)

        clusters[:] = np.argmin(distances, axis=1)

        cluster_to_image = np.reshape(clusters, (sz[0], sz[1]))

        y1, x1 = np.where(cluster_to_image == 1)
        y6, x6 = np.where(cluster_to_image == 6)
        y14, x14 = np.where(cluster_to_image == 14)
        y19, x19 = np.where(cluster_to_image == 19)
        y45, x45 = np.where(cluster_to_image == 45)

        cmap = plt.cm.binary
        norm = plt.Normalize(image.min(), image.max())
        rgba = cmap(norm(image))

        conv = 255
        rgba[y1, x1, :3] = 236 / conv, 112 / conv, 99 / conv
        rgba[y6, x6, :3] = 175 / conv, 122 / conv, 197 / conv
        rgba[y14, x14, :3] = 84 / conv, 153 / conv, 199 / conv
        rgba[y19, x19, :3] = 244 / conv, 208 / conv, 63 / conv
        rgba[y45, x45, :3] = 244 / conv, 143 / conv, 177 / conv

        plot_results.plot_action_shots(rgba, i, wav, savefig=True)

if __name__ == '__main__':

    # optimal number of clusters
    n_clusters = 50

    # find optimal centroid
    optimal_centroid = False
    if optimal_centroid:

        # import training data
        file = open('files/cube_training_scans.pickle', 'rb')
        data = pickle.load(file)
        file.close()

        # reshape data to fit centroid
        sz = data.shape
        cube = np.reshape(data, (sz[0]*sz[1]*sz[2], sz[-1]))

        wcss_score = np.zeros(10)
        for j in range(10):
            file1 = open('files/end_centroid'+str(j)+'.pickle', 'rb')
            centroids = pickle.load(file1)
            file1.close()

            distances = np.zeros((cube.shape[0], n_clusters))
            clusters  = np.zeros(cube.shape[0])

            wcss_score[j] = wcss(cube, clusters, centroids, n_clusters)

        index_min_wcss = np.argmin(wcss_score)

        print(f'End centroid from file end_centroid'+str(index_min_wcss)+'.pickle gives lowest WCSS score.')

    # plot different spectral line features
    plot_features = False
    if plot_features:

        centroids = dt.centroid_data(purpose='dev')

        plot_results.various_clusters(centroids, method='dev', savefig=False)

    # make animation highlighting the pixels of clusters with high intensity in the centroids
    make_animation = True
    if make_animation:

        # import files
        # file1 = open('files/cube_stokes_i.pickle', 'rb')
        # data = pickle.load(file1)
        # file1.close()

        data = dt.sst_cube(purpose='full_run')
        centroids = dt.centroid_data(purpose='dev')

        # set wavelength point
        wav = 3

        # animate_time_series(data, centroids, n_clusters, wav, savefig=True)
        # action_shots(data, centroids, n_clusters, wav)

    # analyse Scikit-Learn centroids
    sklearn= False
    if sklearn:

        centroids = dt.centroid_data(purpose='sklearn')

        plot_results.various_clusters(centroids, method='sklearn', savefig=False)
