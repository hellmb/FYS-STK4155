import data
import numpy as np
import matplotlib.pyplot as plt
from machine_learning import MachineLearning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

class LogisticRegression(MachineLearning):
    """
    logistic regression class
    inherits from class MachineLearning
    """

    def __init__(self):
        """
        initialise the instance of the class
        """

        # initialise features and target
        self.X, self.y = data.design_matrix()

        # set up quantities
        self.n           = self.y.shape[0]       # number of data points
        self.M           = 5                     # size of mini-batches
        self.minibatches = int(self.n/self.M)    # number of mini-batches

        # define epochs
        self.max_epoch = 10
        self.epochs    = np.linspace(0,self.max_epoch,self.max_epoch+1)

    def bootstrap(self, n_boots, benchmark=False):
        """
        bootstrap algorithm
        param n_boots: number of bootstraps
        param benchmark: set to True to run sklearn SGD
        """

        # reshape y if they are multidimensional
        if len(self.y.shape) > 1:
            self.y = np.ravel(self.y)

        # split into training and test data
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)

        # add axis to y_train and y_test
        y_train = y_train[:,np.newaxis]
        y_test  = y_test[:,np.newaxis]

        # create empty arrays to store test predicitions for each bootstrap
        zpred_test_store = np.empty((y_test.shape[0], n_boots))

        # empty arrays to store accuracy for every epoch and bootstrap
        self.acc_train = np.zeros((len(self.epochs), n_boots))
        self.acc_test  = np.zeros((len(self.epochs), n_boots))

        # perform MSE on the different training and test data
        for k in range(n_boots):

            print('Bootstrap number ', k)

            # define random beta
            beta = np.random.rand(self.X.shape[1], 1)

            # empty array to store accuracy for epochs
            acc_epoch_train = np.zeros(len(self.epochs))
            acc_epoch_test  = np.zeros(len(self.epochs))

            for j in range(len(self.epochs)):
                for i in range(0,self.n,self.M):
                    beta = self.gradient_descent(X_train[i:i+self.M,:], y_train[i:i+self.M,:], beta, self.M)

                    cost = self.cost_function(X_train, y_train, beta)

                ypred_train = np.dot(X_train,beta)
                ypred_test  = np.dot(X_test,beta)

                # calculate cost and accuracy for each
                acc_epoch_train[j] = self.accuracy(y_train, ypred_train)
                acc_epoch_test[j]  = self.accuracy(y_test, ypred_test)

                # create random indices for every bootstrap
                random_index = np.random.randint(X_train.shape[0], size=X_train.shape[0])

                # resample X_train and y_train
                X_tmp = X_train[random_index,:]
                y_tmp = y_train[random_index]

            # store accuracy for every bootstrap
            self.acc_train[:,k] = acc_epoch_train
            self.acc_test[:,k]  = acc_epoch_test

            if benchmark:
                # use sklearn stochastic gradient descent
                clf = SGDClassifier(loss='log',penalty='none',learning_rate='constant',eta0=0.001,
                                    fit_intercept=False,max_iter=self.max_epoch,shuffle=True)

                clf.fit(X_train,y_train.ravel())

                clf_acc_train = clf.score(X_train, y_train)
                clf_acc_test  = clf.score(X_test, y_test)

                print('sklearn train:        ',clf_acc_train)
                print('sklearn test:         ',clf_acc_test)

                print('Developed code train: ',np.max(self.acc_train))
                print('Developed code test:  ',np.max(self.acc_test))

                ### tomorrow: create file that writes the accuracy scores
                ### for each bootstrap for both sklearn and developed code


    def regression_analysis(self):
        """
        perform logistic regression analysis
        """

        self.bootstrap(1, benchmark=True)

        # for i in range(5):
        #     plt.subplot(1, 2, 1)
        #     plt.plot(self.acc_train[:,i])
        #     plt.xlabel('Epoch')
        #     plt.ylabel('Accuracy')
        #
        #     plt.subplot(1, 2, 2)
        #     plt.plot(self.acc_test[:,i])
        #     plt.xlabel('Epoch')
        #     plt.ylabel('Accuracy')
        #
        # plt.show()



# end of code
