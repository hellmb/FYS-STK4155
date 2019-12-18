import sys
import pickle
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

def simple_data():
    """
    simple data set for testing the k-means algorithm
    :return: randomised data set, X
    """

    X, y = make_blobs(n_samples=1000, centers=10, random_state=30, cluster_std=2.5)

    return X

def sst_cube(purpose='training'):
    """
    import SST data cube
    :param purpose: 'training' or 'full_run'
    :return: data
    """

    if purpose == 'training':
        # import file
        file = open('files_all/cube_training_scans.pickle', 'rb')
        data = pickle.load(file)
        file.close()
        data = data[:,:,3:8,:]

        # convert each scan to a 3d array
        sz = data.shape
        data = np.reshape(data, (sz[0] * sz[1] * sz[2], sz[-1]))
    elif purpose == 'full_run':
        # import file
        file = open('files_all/cube_stokes_i.pickle', 'rb')
        data = pickle.load(file)
        file.close()
    else:
        print('Insert purpose.')
        sys.exit()

    return data

def centroid_data(purpose='dev'):
    """
    import centroid data
    :param purpose: 'dev' or 'sklearn
    :return: centroids
    """

    if purpose == 'dev':
        file2 = open('files/end_centroid5.pickle', 'rb')
        centroids = pickle.load(file2)
        file2.close()
    elif purpose == 'sklearn':
        file = open('files/sk_end_centroid.pickle', 'rb')
        centroids = pickle.load(file)
        file.close()
    else:
        print('Insert purpose.')
        sys.exit()

    return centroids