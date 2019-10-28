import numpy as np

class MachineLearning:
    """
    class containing machine learning functions
    """

    def __init__(self):
        """
        initialise the instance of the class
        """

    def mean_squared_error(self, y, y_predict):
        """
        function for calculating the mean squared error (MSE)
        param y: function array
        param y_predict: predicted function array
        """

        len_y = len(np.ravel(y))

        mse = np.sum((y - y_predict)**2)/len_y

        return mse

    def r2_score(self, y, y_predict):
        """
        function for calculating the R2 score
        param y: function array
        param y_predict: predicted function array
        """

        len_y = len(np.ravel(y))

        # calculate mean value of y_predict
        mean_y_predict = np.sum(y_predict)/len_y

        r2score = 1. - np.sum((y - y_predict)**2)/np.sum((y - mean_y_predict)**2)

        return r2score

    def accuracy(self, y, y_predict):
        """
        function for calculating the accuracy score
        """

        if len(y.shape) > 1:
            y = np.ravel(y)

        if len(y_predict.shape) > 1:
            y_predict = np.ravel(y_predict)

        # the values of y_predict are not binary
        y_predict[y_predict < 0.5] = 0
        y_predict[y_predict >= 0.5] = 1

        # calculate indicator function
        I = np.zeros(len(y))

        # more elegant way of doing this?
        for i in range(0,len(y)-2):
            # print(i)
            if y[i] == y_predict[i]:
                I[i] = 1
            else:
                I[i] = 0

        accuracy = np.sum(I)/len(I)

        return accuracy

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
        eta0 = 0.001

        p = self.sigmoid(X, beta)

        # calculate the gradient of the cost function
        dC_dbeta = -(np.dot(X.T,(y - p)))/N

        # stochastic gradient descent
        new_beta = beta - eta0*dC_dbeta

        return new_beta
