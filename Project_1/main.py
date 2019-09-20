import file_handling
import plotting_function
import matplotlib
import numpy as np
from random import random, seed
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold

# set general plotting font consistent with LaTeX
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

class RegressionAnalysis():

    def __init__(self, m=2, noise=False):
        """
        initialise the instance of the class
        """

        # array dimension
        self.dim = 100

        # random values (sorted)
        self.x = np.sort(np.random.uniform(0, 1, self.dim))
        self.y = np.sort(np.random.uniform(0, 1, self.dim))

        # create meshgrid
        self.x, self.y = np.meshgrid(self.x, self.y)

        # polynomial degree
        self.m = m

        # noise
        self.noise = noise

    def FrankeFunction(self, x, y):
        """
        function to calculate the Franke function
        input x, y: data points in x- and y-direction
        input noise: set to True to add noise to the function
        """

        term1 = 0.75*np.exp(-((9*x-2)**2)/4. - ((9*y-2)**2)/4.)
        term2 = 0.75*np.exp(-((9*x+1)**2)/49. - ((9*y+1)**2)/10.)
        term3 = 0.5*np.exp(-((9*x-7)**2)/4. - ((9*y-3)**2)/4.)
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

        z = term1 + term2 + term3 + term4

        if self.noise:
            z += 0.1*np.random.randn(self.dim,1)

        return z

    def DesignMatrix(self, x, y):
        """
        function for creating the design matrix
        the matrix has dimension (nxl) where n is the number of data points
        and l is the number of terms in the polynomial with degree m (maybe write in report instead)
        input x, y: data points in x- and y-direction
        input m: polynomial degree
        """

        print('You are fitting a polynomial of degree %d' % self.m)

        # reshape x and y if they are multidimensional
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        n = len(x)
        l = int((self.m+1)*(self.m+2)/2)        # number of beta values
        # print('Dimension of design matrix: (%d, %d)' % (n, l))

        # set up the design matrix
        X = np.ones((n, l))
        for i in range(1, self.m+1):
            index = int(i*(i+1)/2)
            for j in range(i+1):
                X[:,index+j] = x**(i-j) * y**j

        return X

    def BetaValues(self, X, z):
        """
        compute beta values using matrix inversion
        input X: design matrix
        input z: Franke function
        """

        beta = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,z))

        return beta

    def OrdinaryLeastSquares(self):
        """
        function performing ordinary least squares (OLS) on the Franke function
        """

        # compute the Franke function
        self.z = self.FrankeFunction(self.x, self.y)

        # compute the design matrix based on the polynomial degree m
        self.X = self.DesignMatrix(self.x, self.y)

        # reshape matrix into vector in order to compute the beta values
        z_vec = np.ravel(self.z)

        # calculate beta values
        self.beta = self.BetaValues(self.X, z_vec)

        # compute predicted Franke function
        self.z_predict = np.dot(self.X,self.beta)

    def ConfidenceInterval(self, zr, zr_predict):
        """
        function for calculating the confidence interval with a 95 % confidence level
        input z: flattened Franke function array
        input z_predict: flattened predicted Franke function array
        """

        # array dimensions
        n = len(zr)
        l = len(self.beta)

        # calculate sigma squared (unbiased) and standard deviation
        sigma_squared = sum((zr - zr_predict)**2)/(n - l - 1)
        sigma         = np.sqrt(sigma_squared)

        # variance of beta
        XTX_inv       = np.linalg.inv(np.dot(self.X.T,self.X))
        self.var_beta = sigma_squared*XTX_inv

        # Z-score for a 95% confidence interval is 1.96
        Z_score = 1.96

        # create array for storing the confidence interval for beta
        self.con_int = np.zeros((l,2))

        print('       Beta          Lower percentile   Upper percentile')
        for i in range(l):
            self.con_int[i,0] = self.beta[i] - Z_score*np.sqrt(XTX_inv[i,i])*sigma
            self.con_int[i,1] = self.beta[i] + Z_score*np.sqrt(XTX_inv[i,i])*sigma

            print(self.beta[i], self.con_int[i, 0], self.con_int[i,1])


    def MeanSquaredError(self, zr, zr_predict):
        """
        function for calculating the mean squared error (MSE)
        input z: flattened Franke function array
        input z_predict: flattened predicted Franke function array
        """

        mse = sum((zr - zr_predict)**2)/len(zr)

        # print('MSE: %.5f' % self.mse)

        return mse

    def R2Score(self, zr, zr_predict):
        """
        function for calculating the R2 score
        input z: flattened Franke function array
        input z_predict: flattened predicted Franke function array
        """

        # calculate mean value of z_predict
        mean_z_predict = sum(zr_predict)/len(zr)

        r2score = 1. - sum((zr - zr_predict)**2)/sum((zr - mean_z_predict)**2)

        # print('r2score: %.5f' % self.r2score)

        return r2score

    def TestTrainSplit(self, X, z, split=0.8):
        """
        function for splitting data into training data and test data
        input X: design matrix
        input z: Franke function
        """





    def KFoldCrossValidation(self, X, z, folds):
        """
        k-fold cross validation
        """

        # shuffle X and z equally (index-wise) -- must be a better way of doing this
        z = np.ravel(z)
        randomise = np.arange(len(z))
        np.random.shuffle(randomise)
        X = X[randomise,:]
        z = z[randomise]
        #z = np.reshape(z, (self.dim,self.dim))

        # split X and z into k folds
        X_k = np.array_split(X, folds)
        z_k = np.array_split(z, folds)


        # empty arrays for training and test MSE
        mse_train = np.zeros(folds)
        mse_test  = np.zeros(folds)

        # perform MSE on the different training and test data
        for k in range(folds):

            # set up training data for X and z
            X_train = X_k
            X_train = np.delete(X_train, k, 0)      # delete fold with index k
            X_train = np.concatenate(X_train)       # join sequence of arrays

            z_train = z_k
            z_train = np.delete(z_train, k, 0)
            z_train = np.concatenate(z_train)

            print(X_train.shape, z_train.shape)
            # z_train = np.ravel(z_train)             # use np.ravel for correct dimensions

            # set test data equal to the deleted fold in training data
            X_test = X_k[k]
            z_test = z_k[k]
            #z_test = np.ravel(z_test)

            # perform OLS
            beta_train = self.BetaValues(X_train, z_train)

            # calculate z_predict for training and test data
            zpred_train = np.dot(X_train,beta_train)
            zpred_test = np.dot(X_test, beta_train)

            # append MSE to lists
            mse_train[k] = self.MeanSquaredError(z_train, zpred_train)
            mse_test[k]  = self.MeanSquaredError(z_test, zpred_test)


        print(mse_train)
        print(mse_test)
        # print(mse_test-mse_train)

        zpred_train_mesh = np.reshape(zpred_train, ())
        zpred_test_mesh  = np.reshape(zpred_test, (self.dim,self.dim))

        # plotting_function.PlotMultiple3D(self.x, self.y, zpred_train_mesh, zpred_test_mesh, zpred_train, zpred_test, self.m, self.dim)

        # perform statistical analysis like mean, average, standard deviation etc.


    def RavelArrays(self):
        """
        general function for flattening the x-, y-, z- and z_predict-arrays
        """

        # ravel x, y, z and z_predict
        self.xr         = np.ravel(self.x)
        self.yr         = np.ravel(self.y)
        self.zr         = np.ravel(self.z)
        self.zr_predict = np.ravel(self.z_predict)


    def Benchmark(self, x, y, z, z_predict, m):
        """
        function for creating benchmarks
        inputs x, y, z and z_predict: flattened x-, y-, z- and z_predict-arrays
        input m: polynomial degree
        """

        ### benchmarking beta values ###

        # expand x and y to have an additional dimension
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)

        # concatenate in order to create an input array acceptable in fit_transform
        input_arr = np.concatenate((x.T, y.T), axis=1)

        poly   = PolynomialFeatures(degree=m)
        Xp     = poly.fit_transform(input_arr)
        linreg = LinearRegression(fit_intercept=False)  # set fit_intercept to False to get all beta values
        linreg.fit(Xp, z)

        self.beta_bench = linreg.coef_

        ### benchmark mean squared error (MSE) ###
        self.mse       = self.MeanSquaredError(z, z_predict)
        self.mse_bench = mean_squared_error(z, z_predict)

        ### benchmark R2 score ###
        self.r2score       = self.R2Score(z, z_predict)
        self.r2score_bench = r2_score(z, z_predict)

        ### benchmark k_fold cross validation ###
        # validate k-fold cv --> put in benchmarks
        # k_fold = KFold(n_splits=folds, shuffle=True)
        # for train_ind, test_ind in k_fold.split(X):
        #     print(train_ind, test_ind)
        #     print('sklearn:')
        #     print(len(X[train_ind]))

        ### write to file ###
        file_handling.BenchmarksToFile(self.beta,
                                       self.beta_bench,
                                       self.mse,
                                       self.mse_bench,
                                       self.r2score,
                                       self.r2score_bench)



    def TaskA(self):
        """
        run project task a
        """

        self.OrdinaryLeastSquares()
        self.RavelArrays()
        self.ConfidenceInterval(self.zr, self.zr_predict)

        mse     = self.MeanSquaredError(self.zr, self.zr_predict)
        r2score = self.R2Score(self.zr, self.zr_predict)

        # plotting_function.Plot3D(self.x, self.y, self.z_predict, self.m, self.dim)
        # plotting_function.ErrorBars(self.beta, self.con_int, self.m)
        # plotting_function.PlotMultiple3D(self.x, self.y, self.z, self.zr_predict, self.zr, self.zr_predict, self.m, self.dim)

    def TaskB(self):
        """
        run project task b
        """
        self.OrdinaryLeastSquares()
        self.KFoldCrossValidation(self.X, self.z, 5)



if __name__ == '__main__':
    # for m in range(1,6):
    #     run = RegressionAnalysis(m=m, noise=False)
    #     run.TaskA()
    # plotting_function.Plot3D(run.x, run.y, run.z_predict, run.m, run.dim)
    run = RegressionAnalysis(m=5, noise=False)
    run.TaskB()
