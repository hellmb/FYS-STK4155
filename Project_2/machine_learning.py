import data
import numpy as np

class MachineLearning:
    """
    class containing machine learning functions
    """

    def __init__(self):
        """
        initialise the instance of the class
        """

    def mean_squared_error(self, yt, yp, w, lamb):
        """
        sum of squares
        param yt: targets
        param yp: prediction
        param w: weights (beta)
        param lamb: regularisation hyper-parameter
        """

        # add regularisation to the cost
        mse = np.sum((yt - yp)**2)/len(yt) + lamb/len(yt) * np.sum(np.square(w[-1]))

        return mse

    def r2_score(self, yt, yp):
        """
        function for calculating the R2 score
        param yt: function array
        param yp: predicted function array
        """

        mean_y_predict = np.sum(yp)/len(yt)
        r2score = 1. - np.sum((yt - yp)**2)/np.sum((yt - mean_y_predict)**2)

        return r2score

    def accuracy_log(self, yt, yp):
        """
        function for calculating the accuracy score (logistic regression)
        param yt: targets
        param yp: predicted targets
        """

        len_y = len(np.ravel(yt))

        # set values of y_predict to binary integers
        yp[yp < 0.5] = 0
        yp[yp >= 0.5] = 1

        accuracy = np.sum(yt == yp.astype(int))/len_y

        return accuracy

    def accuracy_nn(self, yt, yp):
        """
        function for calculating the accuracy score (neural network)
        param yt: targets
        param yp: predicted targets
        """

        # find maximum target and prediction arrays
        find_max_yt = np.argmax(yt,axis=1)
        find_max_yp = np.argmax(yp,axis=1)

        accuracy = np.sum(find_max_yt == find_max_yp)/len(find_max_yt)

        return accuracy

    def relu(self, theta):
        """
        rectified linear unit (ReLU) activation function
        param theta: input to function
        """

        relu = np.maximum(0, theta)

        return relu

    def relu_derivative(self, theta):
        """
        derivative of the ReLU activation function
        param theta: input to function
        """

        theta[theta <= 0] = 0
        theta[theta > 0]  = 1

        return theta

    def sigmoid(self, theta):
        """
        calculate the sigmoid function
        param theta: input to function
        """

        sigma = np.exp(theta)/(1. + np.exp(theta))

        return sigma

    def softmax(self, theta):
        """
        calculate probabilities using the sofmax function
        param theta: input to function
        """

        softmax = np.exp(theta)/np.sum(np.exp(theta),axis=1,keepdims=True)

        return softmax

    def cost_function(self, X, y, beta):
        """
        cost/loss function
        param X: design matrix (features), matrix
        param y: targets, array
        param beta: beta, array
        """

        N = len(y)

        y_predict = np.dot(X, beta)

        p = self.sigmoid(y_predict)

        C = -np.sum(y*np.log(p) + (1 - y)*np.log(1+1E-15 - p))/N

        return C

    def binary_cross_entropy(self, yt, a, w, lamb):
        """
        cost/loss function for neural network
        param yt: targets
        param a: prediction
        param w: weights
        param lamb: regularisation hyper-parameter
        """

        score = 0
        for j in range(yt.shape[1]):
            score += np.sum(yt[:,j] * np.log(1e-15 + a[:,j]) + (1 - yt[:,j])*np.log(1 - a[:,j]))
        C = score/len(yt)

        # add regularisation to the cost
        frobenius_norm = np.linalg.norm(w[-1])
        cost = - C - (lamb * frobenius_norm)/len(yt)

        return cost


    def gradient_descent(self, X, y, beta, eta, N):
        """
        stochastic gradient descent with mini-batches
        param X: design matrix (features), matrix
        param y: targets, array
        param beta: previous beta, array
        param eta: learning rate
        param N: batch size, int
        """

        y_predict = np.dot(X, beta)

        p = self.sigmoid(y_predict)

        # calculate the gradient of the cost function
        dC_dbeta = -(np.dot(X.T,(y - p)))/N

        # stochastic gradient descent
        new_beta = beta - eta*dC_dbeta

        return new_beta