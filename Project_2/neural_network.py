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

    def __init__(self, X, y, eta, lamb, minibatch_size, epochs, n_boots, folds, nodes, benchmark=False):
        """
        initialise the instance of the class
        param X: features
        param y: targets
        param nodes: list of the number of nodes in each hidden layer
        """

        # define features and targets
        self.X = X
        self.y = y

        self.n_input    = X.shape[0]
        self.n_features = X.shape[1]

        # define hidden layers and nodes (list)
        self.hidden_layers = len(nodes)
        self.nodes         = nodes

        # learning rate
        self.eta = eta

        # lambda (regularisation hyper-parameter)
        self.lamb = lamb

        # mini-batches
        self.minibatch_sz = minibatch_size                          # size of mini-batches

        # define epochs
        self.max_epoch = epochs
        self.epochs    = np.linspace(0,self.max_epoch,self.max_epoch+1)

        # define number of bootstraps
        self.n_boots = n_boots

        # define numder of k-folds
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

        self.p = []

        for a, b, w in zip(self.a, self.biases, self.weights):
            z = np.dot(a,w) + b.T
            self.a.append(np.array(self.sigmoid(z)))

            # self.p.append(np.array(self.softmax(z)))

    def backpropagation(self, y):
        """
        function performing back-propagation
        param y: targets
        """

        delta = []

        # compute output error
        delta.append(np.array(self.a[-1] - y).T)
        # delta.append(np.array(self.p[-1] - y).T)

        # back propagation for remaining layers
        for l in range(1,self.hidden_layers+1):
            delta_l = (self.a[-l-1] * (1 - self.a[-l-1])).T * (self.weights[-l] @ delta[-1])
            # elf    = self.weights[-l] @ delta[-1]
            # rudolf = self.a[-l-1]*(1-self.a[-l-1])
            # santa  = rudolf.T*elf

            # append to list of deltas
            delta.append(np.array(delta_l))

        # update weights and biases
        for l in range(1,self.hidden_layers+2):
            regularisation = self.lamb * self.weights[-l]/self.weights[-l].shape[1]
            self.weights[-l] = self.weights[-l] - self.eta*((self.a[-l-1].T @ delta[l-1].T)/self.minibatch_sz - regularisation)
            self.biases[-l]  = self.biases[-l] - (self.eta*np.sum(delta[l-1],axis=1,keepdims=True))/self.minibatch_sz

    def bootstrap(self):
        """
        bootstrap algorithm
        """

        for k in range(self.n_boots):

            print('Bootstrap number ', k)

            # define weights and biases
            self.weights_biases()

            # empty arrays to store accuracy and cost for epochs
            acc_epoch_train  = np.zeros(len(self.epochs))
            acc_epoch_test   = np.zeros(len(self.epochs))
            cost_epoch_train = np.zeros(len(self.epochs))
            cost_epoch_test  = np.zeros(len(self.epochs))

            for j in range(len(self.epochs)):
                for i in range(0,self.X_train.shape[0],self.minibatch_sz):
                    self.feed_forward(self.X_train[i:i+self.minibatch_sz,:])
                    self.backpropagation(self.y_train[i:i+self.minibatch_sz,:])

                # calculate accuracy for training data
                self.feed_forward(self.X_train)
                acc_epoch_train[j]  = self.accuracy_nn(self.y_train, self.a[-1])
                cost_epoch_train[j] = self.cost_function_nn(self.y_train, self.a[-1], self.weights[-1], self.lamb)

                # calculate accuracy for test data
                self.feed_forward(self.X_test)
                acc_epoch_test[j]  = self.accuracy_nn(self.y_test, self.a[-1])
                cost_epoch_test[j] = self.cost_function_nn(self.y_test, self.a[-1], self.weights[-1], self.lamb)

                if j%50 == 0:
                    print('Acc train: ',acc_epoch_train[j])
                    print('Acc test:  ',acc_epoch_test[j])
                    print('Cost train: ',cost_epoch_train[j])
                    print('Cost test:  ',cost_epoch_test[j])
                    print('Max weight: ',np.max(self.weights[0]))
                    print('Max weight: ',np.max(self.weights[1]))

                # create random indices for every bootstrap
                random_index = np.arange(self.X_train.shape[0])
                np.random.shuffle(random_index)

                # resample X_train and y_train
                self.X_train = self.X_train[random_index,:]
                self.y_train = self.y_train[random_index,:]

            # store accuracy for every bootstrap
            self.acc_train[:,k]  = acc_epoch_train
            self.acc_test[:,k]   = acc_epoch_test
            self.cost_train[:,k] = cost_epoch_train
            self.cost_test[:,k]  = cost_epoch_test

            # plot accuracy for every epoch
            plotting_function.accuracy_epoch(self.epochs, self.acc_train, self.acc_test, savefig=False)
            plotting_function.cost_epoch(self.epochs, self.cost_train, self.cost_test, savefig=False)

    def kfold(self):
        """
        k-fold cross-validation
        """

        kfolds = KFold(n_splits=self.folds)

        train_index = []
        test_index  = []
        for i_train, i_test in kfolds.split(self.X):
            train_index.append(np.array(i_train))
            test_index.append(np.array(i_test))

        for k in range(self.folds):
            # split into training and test data
            self.X_train = self.X[train_index[k]]
            self.y_train = self.y[train_index[k]]

            self.X_test = self.X[test_index[k]]
            self.y_test = self.y[test_index[k]]

            # define weights and biases
            self.weights_biases()

            # empty arrays to store accuracy and cost for epochs
            acc_epoch_train  = np.zeros(len(self.epochs))
            acc_epoch_test   = np.zeros(len(self.epochs))
            cost_epoch_train = np.zeros(len(self.epochs))
            cost_epoch_test  = np.zeros(len(self.epochs))

            for j in range(len(self.epochs)):
                for i in range(0,self.X_train.shape[0],self.minibatch_sz):
                    self.feed_forward(self.X_train[i:i+self.minibatch_sz,:])
                    self.backpropagation(self.y_train[i:i+self.minibatch_sz,:])

                # calculate accuracy for training data
                self.feed_forward(self.X_train)
                acc_epoch_train[j]  = self.accuracy_nn(self.y_train, self.a[-1])
                cost_epoch_train[j] = self.cost_function_nn(self.y_train, self.a[-1], self.weights[-1], self.lamb)

                # calculate accuracy for test data
                self.feed_forward(self.X_test)
                acc_epoch_test[j]  = self.accuracy_nn(self.y_test, self.a[-1])
                cost_epoch_test[j] = self.cost_function_nn(self.y_test, self.a[-1], self.weights[-1], self.lamb)

                if j%50 == 0:
                    print('Acc train: ',acc_epoch_train[j])
                    print('Acc test:  ',acc_epoch_test[j])
                    print('Cost train: ',cost_epoch_train[j])
                    print('Cost test:  ',cost_epoch_test[j])
                    print('Max weight: ',np.max(self.weights[0]))
                    print('Max weight: ',np.max(self.weights[1]))

            # store accuracy for every bootstrap
            self.acc_train[:,k]  = acc_epoch_train
            self.acc_test[:,k]   = acc_epoch_test
            self.cost_train[:,k] = cost_epoch_train
            self.cost_test[:,k]  = cost_epoch_test

            # plot all accuracy scores
            plt.plot(self.epochs, acc_epoch_train, label='Train, fold %s' % k)
            plt.plot(self.epochs, acc_epoch_test, label='Test, fold %s' % k)
            plt.title(r'Accuracy as a function of epoch for $k$-fold cross-validation', fontsize=20)
            plt.xlabel('Epoch', fontsize=15)
            plt.ylabel(r'$\epsilon$', fontsize=15)
            plt.legend(loc='lower right', fontsize=15)

            # plot accuracy for every epoch
            # plotting_function.accuracy_epoch(self.epochs, self.acc_train, self.acc_test, savefig=False)
            # plotting_function.cost_epoch(self.epochs, self.cost_train, self.cost_test, savefig=False)

        plt.show()

    def keras_nn(self):
        """
        keras neural network
        """

        # neural network with keras
        model = Sequential()
        model.add(Dense(self.nodes[0], input_dim=self.n_features, activation='sigmoid'))

        # add more hidden layers
        if self.hidden_layers > 1:
            for l in range(1,self.hidden_layers):
                model.add(Dense(self.nodes[l], activation='sigmoid'))

        # add output layer
        model.add(Dense(self.y.shape[1], activation='sigmoid'))

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

    def mlp(self):
        """
        multilayer perceptron
        """

        # shuffle X and y before splitting into training and test data
        random_index = np.arange(self.X.shape[0])
        np.random.shuffle(random_index)
        self.X = self.X[random_index,:]
        self.y = self.y[random_index,:]

        # split into training and test data
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

        # set up empty arrays to store accuracy and cost for every bootstrap
        self.array_setup()

        # bootstrap
        # self.bootstrap()
        self.kfold()

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


    def array_setup(self):
        """
        function for defining empty arrays for bootstrap
        """

        # empty arrays to store accuracy for every epoch
        # self.acc_train = np.zeros((len(self.epochs), self.n_boots))
        # self.acc_test  = np.zeros((len(self.epochs), self.n_boots))
        #
        # # empty arrays to store cost/loss for every epoch
        # self.cost_train = np.zeros((len(self.epochs), self.n_boots))
        # self.cost_test  = np.zeros((len(self.epochs), self.n_boots))

        self.acc_train = np.zeros((len(self.epochs), self.folds))
        self.acc_test  = np.zeros((len(self.epochs), self.folds))

        # empty arrays to store cost/loss for every epoch
        self.cost_train = np.zeros((len(self.epochs), self.folds))
        self.cost_test  = np.zeros((len(self.epochs), self.folds))





# end of code
