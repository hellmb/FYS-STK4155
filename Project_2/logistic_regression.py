import plotting_function
import numpy as np
import matplotlib.pyplot as plt
from machine_learning import MachineLearning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from warnings import filterwarnings

class LogisticRegression(MachineLearning):
    """
    logistic regression class
    inherits from class MachineLearning
    """

    def __init__(self, X, y, n_boots, benchmark=False):
        """
        initialise the instance of the class
        """

        # initialise features and targets
        self.X = X
        self.y = y

        # set up quantities
        self.n           = self.y.shape[0]       # number of data points
        self.M           = 5                     # size of mini-batches
        self.minibatches = int(self.n/self.M)    # number of mini-batches

        # define epochs
        self.max_epoch = 100
        self.epochs    = np.linspace(0,self.max_epoch,self.max_epoch+1)

        self.n_boots = n_boots
        self.benchmark = benchmark

    def array_setup(self):
        """
        function for defining empty arrays for bootstrap
        """

        # empty arrays to store accuracy for every epoch and bootstrap
        self.acc_train = np.zeros((len(self.epochs), self.n_boots))
        self.acc_test  = np.zeros((len(self.epochs), self.n_boots))

        # empty arrays to store accuracy for every epoch and bootstrap
        self.cost_train = np.zeros((len(self.epochs), self.n_boots))
        self.cost_test  = np.zeros((len(self.epochs), self.n_boots))

        # set up arrays for storing maximum accuracy for benchmarking
        if self.benchmark:
            self.sgd_train = np.zeros(self.n_boots)
            self.sgd_test  = np.zeros(self.n_boots)
            self.dc_train  = np.zeros(self.n_boots)
            self.dc_test   = np.zeros(self.n_boots)

    def bootstrap(self):
        """
        bootstrap algorithm
        """

        for k in range(self.n_boots):

            print('Bootstrap number ', k)

            # define random beta
            beta = np.random.rand(self.X.shape[1], 1)

            # empty array to store accuracy and cost for epochs
            acc_epoch_train  = np.zeros(len(self.epochs))
            acc_epoch_test   = np.zeros(len(self.epochs))
            cost_epoch_train = np.zeros(len(self.epochs))
            cost_epoch_test  = np.zeros(len(self.epochs))

            for j in range(len(self.epochs)):
                for i in range(0,self.n,self.M):
                    beta = self.gradient_descent(self.X_train[i:i+self.M,:], self.y_train[i:i+self.M,:], beta, self.M)
                    # cost = self.cost_function(self.X_train, self.y_train, beta)

                ypred_train = np.dot(self.X_train,beta)
                ypred_test  = np.dot(self.X_test,beta)

                # calculate accuracy and cost for each epoch
                acc_epoch_train[j] = self.accuracy_log(self.y_train, ypred_train)
                acc_epoch_test[j]  = self.accuracy_log(self.y_test, ypred_test)
                cost_epoch_train[j] = self.cost_function(self.X_train, self.y_train, beta)
                cost_epoch_test[j]  = self.cost_function(self.X_test, self.y_test, beta)

                # create random indices for every bootstrap
                random_index = np.random.randint(self.X_train.shape[0], size=self.X_train.shape[0])

                # resample X_train and y_train
                self.X_train = self.X_train[random_index,:]
                self.y_train = self.y_train[random_index,:]

            # store accuracy for every bootstrap
            self.acc_train[:,k]  = acc_epoch_train
            self.acc_test[:,k]   = acc_epoch_test
            self.cost_train[:,k] = cost_epoch_train
            self.cost_test[:,k]  = cost_epoch_test

            # run sklearn SGD classifier and logistic regression for benchmarking
            if self.benchmark:

                # ignore convergence warning from sklearn
                filterwarnings('ignore')

                # use sklearn stochastic gradient descent
                clf_sgd = SGDClassifier(loss='log',penalty='none',learning_rate='constant',eta0=0.001,
                                    fit_intercept=False,max_iter=self.max_epoch,shuffle=True)
                                    #,verbose=1)

                clf_sgd.fit(self.X_train,self.y_train.ravel())

                self.sgd_train[k] = clf_sgd.score(self.X_train, self.y_train)
                self.sgd_test[k]  = clf_sgd.score(self.X_test, self.y_test)

                self.dc_train[k]  = np.max(self.acc_train)
                self.dc_test[k]   = np.max(self.acc_test)

    def logistic_regression(self):
        """
        perform logistic regression
        """

        # shuffle X and y before splitting into training and test data
        index = np.arange(self.X.shape[0])
        np.random.shuffle(index)
        self.X = self.X[index,:]
        self.y = self.y[index,:]

        # split into training and test data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

        # define empty arrays
        self.array_setup()

        # bootstrap
        self.bootstrap()

        # plot cost for training and test data
        plotting_function.cost_epoch(self.epochs,self.cost_train,self.cost_test,savefig=False)

        # plot accuracy for benchmarking
        if self.benchmark:
            plotting_function.plot_benchmark(self.sgd_train,self.sgd_test,self.dc_train,self.dc_test,savefig=False)


# end of code
