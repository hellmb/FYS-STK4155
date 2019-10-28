import data
from logistic_regression import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    lr = LogisticRegression()
    lr.regression_analysis()

    # # initialise features and target
    # X, y = data.design_matrix()
    #
    # # set up quantities
    # n = y.shape[0]            # number of data points
    # M = 5                     # size of mini-batches
    # minibatches = int(n/M)    # number of mini-batches
    #
    # # define random beta
    # # beta_ols  = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))
    # beta_rand = np.random.rand(X.shape[1], 1)
    #
    # cost_old = cost_function(X, y, beta_rand)
    #
    # # test gradient descent method
    # max_epoch = 100
    # epochs = np.linspace(0,max_epoch,max_epoch+1)
    #
    # beta = beta_rand
    #
    # avg_cost = np.zeros(len(epochs))
    # for e in range(len(epochs)):
    #     i = 0
    #     cost = np.zeros(minibatches+1)
    #     batches = np.zeros(minibatches+1)
    #     for batch in range(0,n,M):
    #         beta = gradient_descent(X[batch:batch+M,:], y[batch:batch+M,:], beta, M)
    #
    #         cost[i] = cost_function(X[batch:batch+M,:], y[batch:batch+M,:], beta)
    #         i += 1
    #
    #     avg_cost[e] = np.sum(cost)/(minibatches+1)
    #
    # plt.plot(epochs, avg_cost)
    # plt.show()
