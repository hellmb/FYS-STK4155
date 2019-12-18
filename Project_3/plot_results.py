import pickle
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.metrics import silhouette_samples, silhouette_score

# set general plotting font consistent with LaTeX
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def parallel_time():
    """
    plot performance of developed and scikit-learn models
    """

    n_procs = np.arange(1,7)

    time_sklearn = np.array([2.8634217023849486,
                             3.120748448371887,
                             3.1267561674118043,
                             2.8316067934036253,
                             2.9997848510742187,
                             2.861952519416809])

    time_devcode =np.array([1.5680895,
                            0.9582897999999999,
                            0.8100597999999998,
                            0.7608845999999998,
                            0.7472497999999999,
                            0.7535217999999999])

    convergence_sklearn = np.array([7, 11, 11, 11, 11, 11])
    convergence_devcode = np.array([12, 12, 12, 12, 12, 12])

    color = ['#00897B', '#1E88E5', '#D81B60']

    plt.figure(figsize=(10,8))
    plt.plot(n_procs, time_sklearn/convergence_sklearn, color=color[0], label='Scikit-Learn')
    plt.plot(n_procs, time_devcode/convergence_devcode, color=color[2], label='Developed code')
    plt.title('Model performance', fontsize=25)
    plt.xlabel('Number of processors', fontsize=17)
    plt.ylabel('Mean time/iteration [s]', fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.show()

def parallel_time2():
    """
    plot performance of developed code when changing the number of clusters
    """

    n_procs = np.arange(1,7)

    time_k10 = np.array([1.5169908999999993, 1.0497001, 0.8711897000000001, 0.8274102999999998, 0.8235206999999999, 0.9675295])
    time_k12 = np.array([1.7918140000000005, 1.2326469000000004, 0.9704652000000002, 0.9836439000000002, 1.0031432999999996, 1.0009900000000003])
    time_k14 = np.array([1.9769065000000006, 1.4160117000000003, 1.233975499999999, 1.1636884000000003, 1.2301367000000003, 1.0866142000000003])
    time_k16 = np.array([4.6023388, 2.7905425000000017, 2.4531515000000006, 2.2719541000000008, 2.4122511999999987, 2.1806848000000016])
    time_k18 = np.array([6.314952899999996, 3.8996376999999995, 3.6727096999999973, 3.0269727000000017, 3.3880529000000004, 2.910036500000004])
    time_k20 = np.array([6.4989469, 3.5275626999999945, 3.3051444000000005, 3.005359600000001, 2.9318711999999976, 2.9975556000000028])

    conv_k10 = np.array([12, 12, 12, 12, 12, 12])
    conv_k12 = np.array([12, 12, 12, 12, 12, 12])
    conv_k14 = np.array([12, 12, 12, 12, 12, 12])
    conv_k16 = np.array([20, 20, 20, 20, 20, 20])
    conv_k18 = np.array([27, 27, 27, 27, 27, 27])
    conv_k20 = np.array([23, 23, 23, 23, 23, 23])

    color = ['#00897B', '#1E88E5', '#D81B60', '#80CBC4', '#90CAF9', '#F8BBD0']

    plt.figure(figsize=(10,8))
    plt.plot(n_procs, time_k10 / conv_k10, color=color[0], label=r'$k$ = 10')
    plt.plot(n_procs, time_k12 / conv_k12, color=color[1], label=r'$k$ = 12')
    plt.plot(n_procs, time_k14 / conv_k14, color=color[2], label=r'$k$ = 14')
    plt.plot(n_procs, time_k16 / conv_k16, color=color[3], label=r'$k$ = 16')
    plt.plot(n_procs, time_k18 / conv_k18, color=color[4], label=r'$k$ = 18')
    plt.plot(n_procs, time_k20 / conv_k20, color=color[5], label=r'$k$ = 20')
    plt.title('Model performance', fontsize=25)
    plt.xlabel('Number of processors', fontsize=17)
    plt.ylabel('Mean time/iteration [s]', fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.show()

def decide_n_clusters(savefig=True):
    """
    plot mean standard deviation against number of clusters for training data
    :param savefig: boolean, True/False
    """

    file1 = open('files/mean_std.pickle', 'rb')
    mean_std = pickle.load(file1)
    file1.close()

    file2 = open('files/n_clusters.pickle', 'rb')
    n_clusters = pickle.load(file2)
    file2.close()

    n_clusters = n_clusters[0:-1]
    fig = plt.figure(figsize=(10,8))
    plt.plot(n_clusters, mean_std[0:-1], color='#D81B60', label=r'$\hat{\sigma}_k$')
    plt.axvline(x=50, color='#F8BBD0', linestyle='--')
    plt.title('Finding the optimal number of clusters', fontsize=25)
    plt.xlabel(r'Number of clusters', fontsize=17)
    plt.ylabel(r'Mean standard deviation', fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.show()

    if savefig:
        fig.savefig('figures/decide_n_clusters.png', dpi=200)

def silhouette(data, global_clusters, k):
    """
    silhouette analysis
    :param data: data, matrix
    :param global_clusters: global clusters, matrix
    :param k: number of clusters, int
    """

    plt.figure(figsize=(10, 8))
    plt.xlim([-0.25, 1])
    plt.ylim([0, data.shape[0] + (k + 1) * 10])

    sil_avg = silhouette_score(data, global_clusters)
    print(f'For k = {k} the average silhouette score is {sil_avg}\n')

    sample_sil_val = silhouette_samples(data, global_clusters)

    y_lower = 10
    for i in range(k):
        cluster_sil_val = sample_sil_val[global_clusters == i]
        cluster_sil_val.sort()

        size_cluster = cluster_sil_val.shape[0]

        y_upper = y_lower + size_cluster

        color = cm.nipy_spectral(float(i) / k)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_sil_val,
                          facecolor=color,
                          edgecolor=color,
                          alpha=0.7)

        plt.text(-0.2, y_lower + 0.5 * size_cluster, str(i))
        y_lower = y_upper + 10

        plt.title('Silhouette plot for score of all the values', fontsize=20)
        plt.xlabel('Silhouette coefficient values',fontsize=17)
        plt.ylabel('Cluster label',fontsize=17)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.axvline(x=sil_avg, color='red', linestyle='--')

        plt.yticks([])

        plt.suptitle((f'Silhouette analysis for K-means clustering with n_clusters = {k}'),
                     fontsize=22, fontweight='bold')

def animate_kmeans(data, animate_centroids, animate_clusters, k):
    """
    animate k-means clustering
    :param data: data, matrix
    :param animate_centroids: centroids, 3D list
    :param animate_clusters: clusters, 3D list
    :param k: number of clusters
    """

    dark_col  = ['#880E4F', '#01579B', '#33691E', '#B71C1C', '#0D47A1',
                 '#F57F17', '#004D40', '#1A237E', '#BF360C', '#4A148C']
    light_col = ['#F48FB1', '#81D4FA', '#C5E1A5', '#EF9A9A', '#90CAF9',
                 '#FFF176', '#26A69A', '#5C6BC0', '#FF8A65', '#BA68C8']

    fig_anim = plt.figure()
    frames = []

    for i in range(len(np.array(animate_centroids))):
        gather_frames = []
        for j in range(k):
            gather_frames.append(plt.scatter(data[animate_clusters[i] == j, 0], data[animate_clusters[i] == j, 1],
                                            color=light_col[j]))
            gather_frames.append(plt.scatter(animate_centroids[i][j, 0], animate_centroids[i][j, 1],
                                            color=dark_col[j], marker='*', s=150))
        frames.append(gather_frames)
    anim = animation.ArtistAnimation(fig_anim, frames, interval=300, blit=True)
    # anim.save('movies/simple_kmeans_rand.mp4', dpi=500)
    plt.show()

    # plot single frames
    plot_single_frames = False
    if plot_single_frames:
        for i in range(len(np.array(animate_centroids))):
            fig = plt.figure()
            for j in range(k):
                plt.scatter(data[animate_clusters[i] == j, 0], data[animate_clusters[i] == j, 1],
                                            color=light_col[j])
                plt.scatter(animate_centroids[i][j, 0], animate_centroids[i][j, 1],
                            color=dark_col[j], marker='*', s=150)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
            fig.savefig('figures/simple_random/rand_'+str(i)+'.png', dpi=200)
            plt.show()

def various_clusters(centroids, method='dev', savefig=False):
    """
    plot varius clusters
    :param centroids: centroids, matrix
    :param method: method decsription, string ('dev' or 'sklearn')
    :param savefig: True/False
    """

    colors = ['#EC7063', '#AF7AC5', '#5499C7', '#45B39D', '#FF5722',
              '#F4D03F', '#F39C12', '#EC407A', '#F48FB1', '#BA68C8']

    fig = plt.figure(figsize=(10, 8))

    if method == 'sklearn':
        illustrate_lines = np.array([1, 6, 11, 12, 14, 27, 40, 44, 46])
        plt.title(r'Ca II 8542 $\AA$ cluster centroids ' + '(Scikit-Learn)', fontsize=25)
        figsave = 'figures/various_clusters_sk.png'
    else:
        illustrate_lines = np.array([1, 6, 14, 16, 17, 19, 29, 42, 45])
        plt.title(r'Ca II 8542 $\AA$ cluster centroids', fontsize=25)
        figsave = 'figures/various_clusters.png'

    i = 0
    for k in illustrate_lines:
        plt.plot(np.linspace(0, 10, 11), centroids[k][:-1], color=colors[i], label=r'$c$ = %d' % k)
        i += 1

    plt.xlabel('Wavelength point', fontsize=17)
    plt.ylabel('Intensity', fontsize=17)
    plt.ylim([4000, 28000])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

    if savefig:
        fig.savefig(figsave, dpi=500)

def plot_action_shots(rgba, i, wav, savefig=False):
    """
    plot 'action shots' from entire time series with highlighted clusters
    :param rgba: image in rgba colors, matrix
    :param i: scan number, int
    :param wav: wavelength point, int
    :param savefig: True/False
    """

    # colormap with 5 discrete colors
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",['#F48FB1', '#F4D03F', '#5499C7', '#AF7AC5', '#EC7063'],5)

    fig = plt.figure(figsize=(10,8))
    plt.imshow(rgba, origin='lower')
    plt.title(r'Ca II 8542 $\AA$', fontsize=25)
    plt.xlabel(r'$x$ [pixels]', fontsize=17)
    plt.ylabel(r'$y$ [pixels]', fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    cbar_ax = fig.add_axes([0.82, 0.2, 0.02, 0.6])
    cbar_ax.axes.get_xaxis().set_visible(False)
    cbar_ax.axes.get_yaxis().set_visible(False)
    cbar = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap, orientation='vertical')

    # set each discrete color to its cluster value
    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate([r'$k$ = 45', r'$k$ = 19', r'$k$ = 14', r'$k$ = 6', r'$k$ = 1']):
        cbar.ax.text(1.1, (2*j+1)/10., lab, ha='left', va='center', fontsize=15)

    # plt.show()

    if savefig:
        fig.savefig('figures/action_shots/action_shot_scan' + str(i) + '_wac' + str(wav) + '.png', dpi=500)