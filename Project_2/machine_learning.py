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

    def accuracy_log(self, y, y_predict):
        """
        function for calculating the accuracy score (logistic regression)
        """

        len_y = len(np.ravel(y))

        # y = np.ravel(y)

        # print(y.shape)
        # print(y.dtype)

        # print(y_predict.shape)
        # print(y_predict.dtype)

        # y_predict = np.random.rand(len_y,1)

        # print(y_predict.shape)
        # print(y_predict.dtype)

        # print('yp before')
        # print(y_predict)

        # the values of y_predict are not binary
        y_predict[y_predict < 0.5] = 0
        y_predict[y_predict >= 0.5] = 1

        # print('yp after')
        # print(y_predict)
        #
        # print(y == y_predict.astype(int))

        accuracy = np.sum(y == y_predict.astype(int))/len_y

        return accuracy

    def accuracy_nn(self, yt, yp):
        """
        function for calculating the accuracy score (neural network)
        """

        # print('y target:  ',yt[:5,:])
        # print('y predict: ',yp[:5,:])

        # inputs are one-hot encoded -> need to fix!!
        find_max_yt = np.argmax(yt,axis=1)
        find_max_yp = np.argmax(yp,axis=1)

        accuracy = np.sum(find_max_yt == find_max_yp)/len(find_max_yt)

        return accuracy


    def sigmoid(self, theta):
        """
        calculate the sigmoid function
        """

        sigma = np.exp(theta)/(1. + np.exp(theta))

        return sigma

    def softmax(self,z):
        """
        calculate probabilities using the sofmax function
        """

        softmax = np.exp(z)/np.sum(np.exp(z),axis=1,keepdims=True)

        return softmax

    def cost_function(self, X, y, beta, lamb):
        """
        cost/loss function
        param X: design matrix (features), matrix
        param y: targets, array
        param beta: beta, array
        """

        N = len(y)

        y_predict = np.dot(X, beta)

        p = self.sigmoid(y_predict)

        C = -np.sum(y*np.log(p) + (1 - y)*np.log(1 - p))/N - lamb/(2*N) * np.sum(beta[-1]**2)

        return C

    def cost_function_nn(self, yt, a, w, lamb):
        """
        cost/loss function for neural network
        param yt: targets
        param a: prediction
        param w: weights
        param lamb: regularisation hyper-parameter
        """
        N = len(yt)

        C = -np.sum(yt*np.log(a[-1]) + (1 - yt)*np.log(1 - a[-1]))/N - lamb/(2*N) * np.sum(w[-1]**2)

        return C


    def gradient_descent(self, X, y, beta, N):
        """
        stochastic gradient descent with mini-batches
        param X: design matrix (features), matrix
        param y: targets, array
        param beta: previous beta, array
        param N: batch size, int
        """

        # learning rate - put in input arg?
        eta0 = 0.001

        y_predict = np.dot(X, beta)

        p = self.sigmoid(y_predict)

        # calculate the gradient of the cost function
        dC_dbeta = -(np.dot(X.T,(y - p)))/N

        # stochastic gradient descent
        new_beta = beta - eta0*dC_dbeta

        return new_beta
