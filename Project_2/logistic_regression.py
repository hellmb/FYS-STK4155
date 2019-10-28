import data
import numpy as np
import matplotlib.pyplot as plt
from machine_learning import MachineLearning

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
        self.n = self.y.shape[0]                 # number of data points
        self.M = 5                               # size of mini-batches
        self.minibatches = int(self.n/self.M)    # number of mini-batches

        # define random beta
        self.beta_rand = np.random.rand(self.X.shape[1], 1)


    def regression_analysis(self):
        """
        perform logistic regression analysis
        """

        # ml = MachineLearning()

        cost_old = self.cost_function(self.X, self.y, self.beta_rand)

        # test gradient descent method
        max_epoch = 100
        epochs = np.linspace(0,max_epoch,max_epoch+1)

        beta = self.beta_rand

        avg_cost = np.zeros(len(epochs))
        for e in range(len(epochs)):
            i = 0
            cost = np.zeros(self.minibatches+1)
            batches = np.zeros(self.minibatches+1)
            for batch in range(0,self.n,self.M):
                beta = self.gradient_descent(self.X[batch:batch+self.M,:], self.y[batch:batch+self.M,:], beta, self.M)

                cost[i] = self.cost_function(self.X[batch:batch+self.M,:], self.y[batch:batch+self.M,:], beta)
                i += 1

            avg_cost[e] = np.sum(cost)/(self.minibatches+1)

        plt.plot(epochs, avg_cost)
        plt.show()
