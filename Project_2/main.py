import sys
import data
import numpy as np
from logistic_regression import LogisticRegression
from neural_network import NeuralNetwork
from sklearn.datasets import load_breast_cancer

if __name__ == '__main__':

    # check for input arguments
    if len(sys.argv) == 1:
        print('No arguments passed. Please specify classification method ("log" or "nn").')
        sys.exit()

    arg = sys.argv[1]

    # initialise features and targets using credit card data
    X, y = data.design_matrix()

    X = X[:,1:10]
    # print(X)
    # sys.exit()

    if arg == 'log':

        lr = LogisticRegression(X, y, n_boots=1, benchmark=False)
        lr.logistic_regression()

    elif arg == 'nn':

        # one-hot encode targets
        y = data.onehotencode(y)

        num_targets = np.sum(y,axis=0)
        print('Ratio of targets [0,1]: ',num_targets[0]/np.sum(num_targets))

        nn = NeuralNetwork(X, y, eta=0.01, lamb=0.1, minibatch_size=100, epochs=2000, n_boots=1, nodes=[1])
        nn.mlp()

    elif arg == 'bc_log':

        data_bc = load_breast_cancer()

        # split data set into features and targets
        X = data_bc.data
        y = data_bc.target
        y = y[:,np.newaxis]

        # normalise features
        X = data.normalise_cancer_data(X,y)

        lr = LogisticRegression(X, y, n_boots=2, benchmark=True)
        lr.logistic_regression()

    elif arg == 'bc_nn':

        data_bc = load_breast_cancer()

        # split data set into features and targets
        X = data_bc.data
        y = data_bc.target

        # normalise features
        X = data.normalise_cancer_data(X,y)

        # one-hot encode targets
        y = data.onehotencode(y[:,np.newaxis])

        num_targets = np.sum(y,axis=0)
        print('Ratio of targets [0,1]: ',num_targets[0]/np.sum(num_targets))

        nn = NeuralNetwork(X, y, eta=0.01, lamb=0.1, minibatch_size=50, epochs=2000, n_boots=1, nodes=[8,10])
        nn.mlp()

    else:
        print('Invalid input argument. Please specify "log" or "nn".')
