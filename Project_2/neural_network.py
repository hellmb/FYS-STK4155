import sys
import plotting_function
import numpy as np
from machine_learning import MachineLearning
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

import matplotlib.pyplot as plt

class NeuralNetwork(MachineLearning):
    """
    neural network class
    inherits from class MachineLearning
    """

    def __init__(self, X, y, eta, lamb, minibatch_size, epochs, folds, nodes, benchmark=False):
        """
        initialise the instance of the class
        param X: features
        param y: targets
        param nodes: list of the number of nodes in each hidden layer
        """

        # define features and targets
        self.X = X
        self.y = y

        # keep X and y unshuffled
        self.X_unshuffled = X
        self.y_unshuffled = y

        self.n_input    = X.shape[0]
        self.n_features = X.shape[1]

        # define hidden layers and nodes (list)
        self.hidden_layers = len(nodes)
        self.nodes         = nodes

        # learning rate
        self.eta = eta

        # regularisation hyper-parameter
        self.lamb = lamb

        # size of mini-batches
        self.minibatch_sz = minibatch_size

        # define epochs
        self.max_epoch = epochs
        self.epochs    = np.linspace(0,self.max_epoch,self.max_epoch+1)

        # define number of k-folds
        self.folds = folds

        self.benchmark = benchmark

    def weights_biases(self):
        """
        function that calculates the weights and biases
        all weights and biases for every layer are defined in lists
        """

        self.weights = []
        self.biases  = []

        for n in range(self.hidden_layers):
            if n == 0:
                input_to_node = self.n_features
            else:
                input_to_node = w.shape[1]

            # w = np.random.randn(input_to_node,self.nodes[n]) * np.sqrt(1./input_to_node)
            w = (2/np.sqrt(input_to_node)) * np.random.random_sample((input_to_node,self.nodes[n])) - (1/np.sqrt(input_to_node))
            # w = np.random.randn(input_to_node,self.nodes[n]) * np.sqrt(2/(input_to_node+self.nodes[n]))
            self.weights.append(np.array(w))

            b = np.zeros(self.nodes[n]) #+ 0.01
            self.biases.append(b[:,np.newaxis])

        # define output weights and biases
        w_out = np.random.rand(w.shape[1],self.y.shape[1])
        self.weights.append(np.array(w_out))

        b_out = np.zeros(w_out.shape[1]) #+ 0.01
        self.biases.append(b_out[:,np.newaxis])

    def feed_forward(self, X):
        """
        function performing the feed-forward pass
        param X: features
        """

        self.a = []
        self.a.append(np.array(X))

        i = 1
        for a, b, w in zip(self.a, self.biases, self.weights):
            z = np.dot(a,w) + b.T

            if self.y.shape[1] > 2 and i == len(self.weights):
                # softmax activation (multiclass) for output layer
                self.a.append(np.array(self.softmax(z)))
            else:
                self.a.append(np.array(self.sigmoid(z)))

            i += 1


    def backpropagation(self, y):
        """
        function performing back-propagation
        param y: targets
        """

        delta = []

        # compute output error
        delta.append(np.array(self.a[-1] - y).T)

        # back propagation for remaining layers
        for l in range(1,self.hidden_layers+1):
            delta_l = (self.a[-l-1] * (1 - self.a[-l-1])).T * (self.weights[-l] @ delta[-1])

            # append to list of deltas
            delta.append(np.array(delta_l))

        # update weights and biases
        for l in range(1,self.hidden_layers+2):
            regularisation = self.lamb * self.weights[-l]/self.weights[-l].shape[1]
            self.weights[-l] = self.weights[-l] - self.eta*((self.a[-l-1].T @ delta[l-1].T)/self.minibatch_sz - regularisation)
            self.biases[-l]  = self.biases[-l] - (self.eta*np.sum(delta[l-1],axis=1,keepdims=True))/self.minibatch_sz

    def kfold(self):
        """
        k-fold cross-validation
        """

        self.array_setup_kfold()

        kfolds = KFold(n_splits=self.folds)

        train_index = []
        test_index  = []
        for i_train, i_test in kfolds.split(self.X):
            train_index.append(np.array(i_train))
            test_index.append(np.array(i_test))

        # check max and min accuracy for each fold
        max_accuracy = 0
        min_accuracy = 100000

        for k in range(self.folds):

            print('Fold number %d' % k)

            # split into training and test data
            self.X_train = self.X[train_index[k]]
            self.y_train = self.y[train_index[k]]

            self.X_test = self.X[test_index[k]]
            self.y_test = self.y[test_index[k]]

            # define weights and biases
            self.weights_biases()

            # empty arrays to store accuracy and cost for epochs
            self.temporary_arrays()

            for j in range(len(self.epochs)):
                for i in range(0,self.X_train.shape[0],self.minibatch_sz):
                    self.feed_forward(self.X_train[i:i+self.minibatch_sz,:])
                    self.backpropagation(self.y_train[i:i+self.minibatch_sz,:])

                # prediction from training data
                self.feed_forward(self.X_train)
                self.acc_epoch_train[j]  = self.accuracy_nn(self.y_train, self.a[-1])
                self.cost_epoch_train[j] = self.binary_cross_entropy(self.y_train, self.a[-1], self.weights[-1], self.lamb)

                # prediction from test data
                self.feed_forward(self.X_test)
                self.predictions.append(self.a[-1])
                self.acc_epoch_test[j]  = self.accuracy_nn(self.y_test, self.a[-1])
                self.cost_epoch_test[j] = self.binary_cross_entropy(self.y_test, self.a[-1], self.weights[-1], self.lamb)

                if j%50 == 0:
                    print('Acc train: ',self.acc_epoch_train[j])
                    print('Acc test:  ',self.acc_epoch_test[j])
                    print('Cost train: ',self.cost_epoch_train[j])
                    print('Cost test:  ',self.cost_epoch_test[j])
                    print('Max weight: ',np.max(self.weights[0]))
                    print('Max weight: ',np.max(self.weights[1]))

            # store max accuracy score
            if self.acc_epoch_test[-1] > max_accuracy:
                self.best_weights = self.weights
                self.best_biases  = self.biases

                self.acc_train[:,0] = self.acc_epoch_train
                self.acc_test[:,0]  = self.acc_epoch_test
                self.cost_train[:,0] = self.cost_epoch_train
                self.cost_test[:,0]  = self.cost_epoch_test

                # update max_accuracy
                max_accuracy = np.max(self.acc_epoch_test)

            # store min accuracy score
            if self.acc_epoch_test[-1] < min_accuracy:
                self.acc_train[:,1] = self.acc_epoch_train
                self.acc_test[:,1]  = self.acc_epoch_test
                self.cost_train[:,1] = self.cost_epoch_train
                self.cost_test[:,1]  = self.cost_epoch_test

                # update min_accuracy
                min_accuracy = self.acc_epoch_test[-1]

    def mlp(self):
        """
        multilayer perceptron
        """

        # shuffle X and y before splitting into training and test data
        random_index = np.arange(self.X.shape[0])
        np.random.shuffle(random_index)
        self.X = self.X[random_index,:]
        self.y = self.y[random_index,:]

        # self.bootstrap()
        self.kfold()

        self.statistical_analysis()

        # validation of neural network
        if self.benchmark:
            # also test for small sample
            # set X = X[0:2,:] and and run code
            # testing purpose --> include in benchmarks for validating the neural network
            # set X = X[0:500,:]
            # random_index = np.arange(self.y_train.shape[0])
            # np.random.shuffle(random_index)
            # self.y_train = self.y_train[random_index,:]

            self.keras_nn()

    def statistical_analysis(self):
        """
        statistical analysis on best model
        """

        # plot accuracy and cost for training and test data
        plotting_function.accuracy_kfold(self.epochs, self.acc_train, self.acc_test, self.folds, savefig=False)
        plotting_function.cost_kfold(self.epochs, self.cost_train, self.cost_test, self.folds, savefig=False)

        # define best weights and biases
        # self.weights = self.best_weights
        # self.biases  = self.best_biases

        # feed-forward with best weights and biases
        # self.feed_forward(self.X_unshuffled)




    def keras_nn(self):
        """
        keras neural network
        """

        # split into training and test data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

        # neural network with keras
        model = Sequential()
        model.add(Dense(self.nodes[0], input_dim=self.n_features, activation='sigmoid'))

        # add more hidden layers
        if self.hidden_layers > 1:
            for l in range(1,self.hidden_layers):
                model.add(Dense(self.nodes[l], activation='sigmoid'))

        # add output layer
        model.add(Dense(self.y.shape[1], activation='softmax'))

        sgd = SGD(learning_rate=self.eta)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # training step
        history = model.fit(self.X_train, self.y_train, epochs=self.max_epoch, batch_size=self.minibatch_sz)
        # print(history.history.keys())

        # plot -> put in plotting_function
        plt.plot(history.history['accuracy'])
        plt.show()
        plt.plot(history.history['loss'])
        plt.show()

    def array_setup_bootstrap(self):
        """
        function for defining empty arrays for bootstrap
        """

        # empty arrays to store accuracy for every epoch
        self.acc_train = np.zeros((len(self.epochs), self.n_boots))
        self.acc_test  = np.zeros((len(self.epochs), self.n_boots))

        # empty arrays to store cost/loss for every epoch
        self.cost_train = np.zeros((len(self.epochs), self.n_boots))
        self.cost_test  = np.zeros((len(self.epochs), self.n_boots))

    def temporary_arrays(self):
        """
        temporary (empty) arrays
        """

        self.acc_epoch_train  = np.zeros(len(self.epochs))
        self.acc_epoch_test   = np.zeros(len(self.epochs))
        self.cost_epoch_train = np.zeros(len(self.epochs))
        self.cost_epoch_test  = np.zeros(len(self.epochs))

    def array_setup_kfold(self):
        """
        function for defining empty arrays for k-fold cross-validation
        """

        # store predictions for each k-fold
        self.predictions = []

        # empty arrays to store accuracy for every epoch
        self.acc_train = np.zeros((len(self.epochs), self.folds))
        self.acc_test  = np.zeros((len(self.epochs), self.folds))

        # empty arrays to store cost/loss for every epoch
        self.cost_train = np.zeros((len(self.epochs), self.folds))
        self.cost_test  = np.zeros((len(self.epochs), self.folds))





# end of code
