import numpy as np

class MachineLearning:
    """
    class containing machine learning functions
    """

    def __init__(self):
        """
        initialise the instance of the class
        """


    def sigmoid(self, X, beta):
        """
        calculate the sigmoid function (probability of y = 1)
        """

        y_predict = np.dot(X, beta)
        p = np.exp(y_predict)/(1. + np.exp(y_predict))
        # p0 = 1 - p1

        return p #p1, p0


    def cost_function(self, X, y, beta):
        """
        cost/loss function
        param X: design matrix (features), matrix
        param y: target, array
        param beta: beta, array
        """

        N = len(y)

        p = self.sigmoid(X, beta)

        C = -np.sum(y*np.log10(p) + (1 - y)*np.log10(1 - p))/N

        return C


    def gradient_descent(self, X, y, beta, N):
        """
        stochastic gradient descent with mini-batches
        param X: design matrix (features), matrix
        param y: target, array
        param beta: previous beta, array
        param N: batch size, int
        """

        # learning rate - put in input arg?
        gamma = 0.001

        p = self.sigmoid(X, beta)

        # calculate the gradient of the cost function
        dC_dbeta = -(np.dot(X.T,(y - p)))/N

        # stochastic gradient descent
        new_beta = beta - gamma*dC_dbeta

        return new_beta
