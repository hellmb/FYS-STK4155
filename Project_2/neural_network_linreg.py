import sys
import plotting_function
import numpy as np
from machine_learning import MachineLearning
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

import matplotlib.pyplot as plt

class NeuralNetworkLinearRegression(MachineLearning):
    """
    linear regression neural network class
    inherits from class MachineLearning
    """

    def __init__(self, X, y, mx, my, eta, lamb, minibatch_size, epochs, folds, nodes, benchmark=False):
        """
        initialise the instance of the class
        param X: features
        param y: targets
        param nodes: list of the number of nodes in each hidden layer
        """

        # define features and targets
        self.X = X
        self.y = y[:, np.newaxis]

        # keep X and y unshuffled
        self.X_unshuffled = X
        self.y_unshuffled = y

        # meshgrid
        self.mx = mx
        self.my = my

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

            w = (2/np.sqrt(input_to_node)) * np.random.random_sample((input_to_node,self.nodes[n])) - (1/np.sqrt(input_to_node))
            self.weights.append(np.array(w))

            b = np.zeros(self.nodes[n])
            self.biases.append(b[:,np.newaxis])

        # define output weights and biases
        w_out = np.random.rand(w.shape[1],self.y.shape[1])
        self.weights.append(np.array(w_out))

        b_out = np.zeros(w_out.shape[1])
        self.biases.append(b_out[:,np.newaxis])

    def feed_forward(self, X):
        """
        function performing the feed-forward pass
        param X: features
        """

        self.a = []
        self.a.append(np.array(X))

        self.z = []

        i = 1
        for a, b, w in zip(self.a, self.biases, self.weights):
            self.z.append(np.array(np.dot(a,w) + b.T))

            if i == len(self.weights):
                # linear activation
                self.a.append(np.array(self.z[-1]))
            else:
                self.a.append(np.array(self.relu(self.z[-1])))

            i += 1

    def backpropagation(self, y):
        """
        function performing back-propagation
        param y: targets
        """

        delta = []

        # compute output error
        relu_output_derivative = self.relu_derivative(self.z[-1])
        delta.append(np.array(relu_output_derivative * (self.a[-1] - y)).T)

        # back propagation for remaining layers
        for l in range(1,self.hidden_layers+1):
            relu_derivative = self.relu_derivative(self.z[-l-1])
            delta_l = relu_derivative.T * (self.weights[-l] @ delta[-1])

            # append to list of deltas
            delta.append(np.array(delta_l))

        # update weights and biases
        for l in range(1,self.hidden_layers+2):
            regularisation = self.lamb * self.weights[-l]
            self.weights[-l] = self.weights[-l] - self.eta*((self.a[-l-1].T @ delta[l-1].T) + regularisation)/self.minibatch_sz
            self.biases[-l]  = self.biases[-l] - (self.eta*np.sum(delta[l-1],axis=1,keepdims=True))/self.minibatch_sz

    def kfold(self):
        """
        k-fold cross-validation
        """

        self.array_setup()

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
                self.acc_epoch_train[j]  = self.r2_score(self.y_train, self.a[-1])
                self.cost_epoch_train[j] = self.mean_squared_error(self.y_train, self.a[-1], self.weights[-1], self.lamb)

                # prediction from test data
                self.feed_forward(self.X_test)
                self.acc_epoch_test[j]  = self.r2_score(self.y_test, self.a[-1])
                self.cost_epoch_test[j] = self.mean_squared_error(self.y_test, self.a[-1], self.weights[-1], self.lamb)

                if j%50 == 0:
                    print(j)
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
                max_accuracy = self.acc_epoch_test[-1]

            # store min accuracy score
            if self.acc_epoch_test[-1] < min_accuracy:
                self.acc_train[:,1] = self.acc_epoch_train
                self.acc_test[:,1]  = self.acc_epoch_test
                self.cost_train[:,1] = self.cost_epoch_train
                self.cost_test[:,1]  = self.cost_epoch_test

                # update min_accuracy
                min_accuracy = self.acc_epoch_test[-1]

            if self.benchmark:
                # run keras neural network
                self.keras_nn()

                # store last accuracy score for each fold
                self.last_acc_train[k] = self.acc_epoch_train[-1]
                self.last_acc_test[k]  = self.acc_epoch_test[-1]
                self.keras_acc_train[k] = self.r2_score(self.y_train, self.keras_predict_train)
                self.keras_acc_test[k]  = self.r2_score(self.y_test, self.keras_predict_test)

    def mlp(self):
        """
        multilayer perceptron
        """

        # shuffle X and y before splitting into training and test data
        random_index = np.arange(self.X.shape[0])
        np.random.shuffle(random_index)
        self.X = self.X[random_index,:]
        self.y = self.y[random_index,:]

        # run k-fold cross-validation
        self.kfold()

        # statistical analysis
        self.statistical_analysis()

    def statistical_analysis(self):
        """
        statistical analysis on best model
        """

        # define best weights and biases
        self.weights = self.best_weights
        self.biases  = self.best_biases

        # feed-forward with best weights and biases
        self.feed_forward(self.X_unshuffled)

        # plot franke function target and prediction
        y_target  = np.reshape(self.y_unshuffled, (self.mx.shape[0], self.my.shape[0]))
        y_predict = np.reshape(self.a[-1], (self.mx.shape[0], self.my.shape[0]))
        plotting_function.plot_surface(self.mx, self.my, y_target, y_predict, savefig=False)

        # plot accuracy and cost
        plotting_function.accuracy_kfold(self.epochs, self.acc_train, self.acc_test, savefig=False)
        plotting_function.cost_kfold(self.epochs, self.cost_train, self.cost_test, savefig=False)

        if self.benchmark:
            plotting_function.accuracy_keras(self.last_acc_train, self.last_acc_test, self.keras_acc_train, self.keras_acc_test, self.folds, savefig=False)

    def keras_nn(self):
        """
        keras neural network
        """

        # split into training and test data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

        # neural network with keras
        model = Sequential()
        model.add(Dense(self.nodes[0], input_dim=self.n_features, activation='relu'))

        # add more hidden layers
        if self.hidden_layers > 1:
            for l in range(1,self.hidden_layers):
                model.add(Dense(self.nodes[l], activation='relu'))

        # add output layer
        model.add(Dense(self.y.shape[1]))

        sgd = SGD(learning_rate=self.eta)
        model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse'])

        # training step
        model.fit(self.X_train, self.y_train, epochs=self.max_epoch, batch_size=self.minibatch_sz)

        # keras predictions
        self.keras_predict_train = model.predict(self.X_train)
        self.keras_predict_test  = model.predict(self.X_test)

    def temporary_arrays(self):
        """
        temporary (empty) arrays
        """

        self.acc_epoch_train  = np.zeros(len(self.epochs))
        self.acc_epoch_test   = np.zeros(len(self.epochs))
        self.cost_epoch_train = np.zeros(len(self.epochs))
        self.cost_epoch_test  = np.zeros(len(self.epochs))

    def array_setup(self):
        """
        function for defining empty arrays for k-fold cross-validation
        """

        self.acc_train = np.zeros((len(self.epochs), 2))
        self.acc_test  = np.zeros((len(self.epochs), 2))

        # empty arrays to store cost/loss for every epoch
        self.cost_train = np.zeros((len(self.epochs), 2))
        self.cost_test  = np.zeros((len(self.epochs), 2))

        # store last accuracy score for each k-fold
        self.last_acc_train = np.zeros(self.folds)
        self.last_acc_test  = np.zeros(self.folds)

        # empty arrays to store accuracy for every epoch
        self.keras_acc_train = np.zeros(self.folds)
        self.keras_acc_test  = np.zeros(self.folds)

# end of code
