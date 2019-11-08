import sys
import data
import numpy as np
import plotting_function
from logistic_regression import LogisticRegression
from neural_network import NeuralNetwork
from neural_network_linreg import NeuralNetworkLinearRegression


if __name__ == '__main__':

    # check for input arguments
    if len(sys.argv) == 1:
        print('No arguments passed. Please specify classification method ("log" or "nn").')
        sys.exit()

    arg = sys.argv[1]

    if arg == 'log':

        # initialise features and targets using credit card data
        X, y = data.preprocessing(remove_data=True)

        lr = LogisticRegression(X, y, eta=0.001, minibatch_size=100, epochs=100, folds=10, benchmark=False)
        lr.logistic_regression()

    elif arg == 'nn':

        # initialise features and targets using credit card data
        X, y = data.preprocessing(remove_data=True)

        # one-hot encode targets
        y = data.onehotencode(y)

        num_targets = np.sum(y,axis=0)
        print('Ratio of targets [0,1]: ',num_targets[0]/np.sum(num_targets))

        # nn = NeuralNetwork(X, y, eta=0.01, lamb=1, minibatch_size=100, epochs=150, folds=10, nodes=[50], benchmark=False)
        # nn.mlp()

        explore_regularisation = True
        if explore_regularisation:
            store_acc_train = []
            store_acc_test  = []
            list_of_lambdas = [1E-4, 1E-3, 1E-2, 1E-1, 1]
            for lamb in list_of_lambdas:
                nn = NeuralNetwork(X, y, eta=0.01, lamb=lamb, minibatch_size=100, epochs=50, folds=10, nodes=[50], benchmark=False)
                nn.mlp()
                store_acc_train.append(nn.acc_epoch_train)
                store_acc_test.append(nn.acc_epoch_test)
            plotting_function.test_regularisation(nn.epochs, store_acc_train, store_acc_test, list_of_lambdas, savefig=True)


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

        # nn = NeuralNetworkLinearRegression(X, y, mx, my, eta=0.01, lamb=0, minibatch_size=100, epochs=300, folds=10, nodes=[8,4,3], benchmark=True)
        # nn.mlp()

        explore_regularisation = False
        if explore_regularisation:
            store_acc_train = []
            store_acc_test  = []
            list_of_lambdas = [1E-4, 1E-3, 1E-2, 1E-1, 1]
            for lamb in list_of_lambdas:
                nn = NeuralNetworkLinearRegression(X, y, mx, my, eta=0.01, lamb=lamb, minibatch_size=100, epochs=300, folds=10, nodes=[8,4,3], benchmark=False)
                nn.mlp()
                store_acc_train.append(nn.acc_epoch_train)
                store_acc_test.append(nn.acc_epoch_test)
            plotting_function.test_regularisation(nn.epochs, store_acc_train, store_acc_test, list_of_lambdas, savefig=True)

    else:
        print('Invalid input argument. Please specify "log" or "nn".')
