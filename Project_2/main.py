import sys
import data
import numpy as np
from logistic_regression import LogisticRegression
from neural_network import NeuralNetwork
from neural_network_linreg import NeuralNetworkLinearRegression
from sklearn.datasets import load_breast_cancer


if __name__ == '__main__':

    # check for input arguments
    if len(sys.argv) == 1:
        print('No arguments passed. Please specify classification method ("log" or "nn").')
        sys.exit()

    arg = sys.argv[1]

    if arg == 'log':

        # initialise features and targets using credit card data
        X, y = data.preprocessing(remove_data=True)

        lr = LogisticRegression(X, y, eta=0.01, lamb=0, minibatch_size=100, epochs=100, n_boots=1, benchmark=False)
        lr.logistic_regression()

    elif arg == 'nn':

        # initialise features and targets using credit card data
        X, y = data.preprocessing(remove_data=True)

        # one-hot encode targets
        y = data.onehotencode(y)

        num_targets = np.sum(y,axis=0)
        print('Ratio of targets [0,1]: ',num_targets[0]/np.sum(num_targets))

        nn = NeuralNetwork(X, y, eta=0.01, lamb=0, minibatch_size=100, epochs=1000, folds=10, nodes=[50], benchmark=False)
        nn.mlp()

    elif arg == 'linreg':

        x1 = np.linspace(0, 1, 100)
        x2 = np.linspace(0, 1, 100)

        # create meshgrid
        mx, my = np.meshgrid(x1, x2)

        # target data
        y = data.franke_function(mx, my, noise=True)

        # get features by ravelling meshgrid
        feature1 = np.ravel(mx)
        feature2 = np.ravel(my)

        # create matrix of features
        X = np.column_stack((feature1, feature2))

        # create targets by ravelling y
        y = np.ravel(y)

        nn = NeuralNetworkLinearRegression(X, y, mx, my, eta=0.01, lamb=0, minibatch_size=10, epochs=500, folds=10, nodes=[10,8,3], benchmark=False)
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

        nn = NeuralNetwork(X, y, eta=0.001, lamb=0, minibatch_size=100, epochs=500, n_boots=1, folds=10, nodes=[10,8], benchmark=False)
        nn.mlp()

    else:
        print('Invalid input argument. Please specify "log" or "nn".')
