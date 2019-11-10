import plotting_function
import numpy as np
import matplotlib.pyplot as plt
from machine_learning import MachineLearning
from sklearn.model_selection import KFold
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from warnings import filterwarnings

class LogisticRegression(MachineLearning):
    """
    logistic regression class
    inherits from class MachineLearning
    """

    def __init__(self, X, y, eta, minibatch_size, epochs, folds, benchmark=False):
        """
        initialise the instance of the class
        """

        # initialise features and targets
        self.X = X
        self.y = y

        # set up quantities
        self.n            = self.y.shape[0]                  # number of data points
        self.minibatch_sz = minibatch_size                   # size of mini-batches
        self.minibatches  = int(self.n/self.minibatch_sz)    # number of mini-batches

        # define epochs
        self.max_epoch = epochs
        self.epochs    = np.linspace(0,self.max_epoch,self.max_epoch+1)

        # define hyper-parameters
        self.eta  = eta

        # number of k-folds
        self.folds = folds

        # set to True to run benchmarks
        self.benchmark = benchmark

    def array_setup(self):
        """
        function for defining empty arrays for bootstrap
        """

        # empty arrays to store accuracy for every epoch and bootstrap
        self.acc_train = np.zeros((len(self.epochs), self.folds))
        self.acc_test  = np.zeros((len(self.epochs), self.folds))

        # empty arrays to store accuracy for every epoch and bootstrap
        self.cost_train = np.zeros((len(self.epochs), self.folds))
        self.cost_test  = np.zeros((len(self.epochs), self.folds))

        # set up arrays for storing maximum accuracy for benchmarking
        if self.benchmark:
            self.sgd_train = np.zeros(self.folds)
            self.sgd_test  = np.zeros(self.folds)
            self.dc_train  = np.zeros(self.folds)
            self.dc_test   = np.zeros(self.folds)

    def temporary_arrays(self):
        """
        temporary (empty) arrays
        """

        self.acc_epoch_train  = np.zeros(len(self.epochs))
        self.acc_epoch_test   = np.zeros(len(self.epochs))
        self.cost_epoch_train = np.zeros(len(self.epochs))
        self.cost_epoch_test  = np.zeros(len(self.epochs))

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

            # define random beta
            beta = np.random.rand(self.X.shape[1], 1)

            # split into training and test data
            self.X_train = self.X[train_index[k]]
            self.y_train = self.y[train_index[k]]

            self.X_test = self.X[test_index[k]]
            self.y_test = self.y[test_index[k]]

            # empty arrays to store accuracy and cost for epochs
            self.temporary_arrays()

            for j in range(len(self.epochs)):
                for i in range(0,self.X_train.shape[0],self.minibatch_sz):
                    beta = self.gradient_descent(self.X_train[i:i+self.minibatch_sz,:], self.y_train[i:i+self.minibatch_sz,:], beta, self.eta, self.minibatch_sz)

                # prediction from training data
                ypred_train = np.dot(self.X_train,beta)
                ypred_test  = np.dot(self.X_test,beta)

                self.acc_epoch_train[j] = self.accuracy_log(self.y_train, ypred_train)
                self.acc_epoch_test[j]  = self.accuracy_log(self.y_test, ypred_test)
                self.cost_epoch_train[j] = self.cost_function(self.X_train, self.y_train, beta)
                self.cost_epoch_test[j]  = self.cost_function(self.X_test, self.y_test, beta)

                if j%50 == 0:
                    print('Acc train: ',self.acc_epoch_train[j])
                    print('Acc test:  ',self.acc_epoch_test[j])
                    print('Cost train: ',self.cost_epoch_train[j])
                    print('Cost test:  ',self.cost_epoch_test[j])

            # store max accuracy score
            if self.acc_epoch_test[-1] > max_accuracy:
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

            # plot accuracy for benchmarking
            if self.benchmark:
                # ignore convergence warning from sklearn
                filterwarnings('ignore')

                # use sklearn stochastic gradient descent
                clf_sgd = SGDClassifier(loss='log',penalty='none',learning_rate='constant',eta0=0.001,
                                    fit_intercept=False,max_iter=self.max_epoch,shuffle=True)

                clf_sgd.fit(self.X_train,self.y_train.ravel())

                self.sgd_train[k] = clf_sgd.score(self.X_train, self.y_train)
                self.sgd_test[k]  = clf_sgd.score(self.X_test, self.y_test)

                self.dc_train[k]  = self.acc_epoch_train[-1]
                self.dc_test[k]   = self.acc_epoch_test[-1]

    def logistic_regression(self):
        """
        perform logistic regression
        """

        # shuffle X and y before splitting into training and test data
        index = np.arange(self.X.shape[0])
        np.random.shuffle(index)
        self.X = self.X[index,:]
        self.y = self.y[index,:]

        self.kfold()

        self.statistical_analysis()

    def statistical_analysis(self):
        """
        statistical analysis of models
        """

        if not self.benchmark:
            plotting_function.accuracy_kfold(self.epochs,self.acc_train,self.acc_test,savefig=False)
            plotting_function.cost_kfold(self.epochs,self.cost_train,self.cost_test,savefig=False)
        else:
            plotting_function.benchmark_sgd(self.sgd_train, self.sgd_test, self.dc_train, self.dc_test, self.folds, savefig=False)


# end of code
