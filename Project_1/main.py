import file_handling
import plotting_function
import matplotlib
import numpy as np
from random import random, seed
from imageio import imread
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# set general plotting font consistent with LaTeX
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

class RegressionAnalysis():

    def __init__(self, dim = 100, m=2, lambda_val=0, noise=False, method='OLS'):
        """
        initialise the instance of the class
        """

        # array dimension
        self.dim = dim

        # random values (sorted)
        x = np.sort(np.random.uniform(0, 1, self.dim))
        y = np.sort(np.random.uniform(0, 1, self.dim))

        # create meshgrid
        self.x, self.y = np.meshgrid(x, y)

        # polynomial degree
        self.m = m

        # value of lambda for Ridge and Lasso regression
        # default is zero for OLS
        self.lambda_ = lambda_val

        # noise, boolean
        self.noise = noise

        # regression method, string
        # default method is OLS
        self.method = method

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

        print('Polynomial degree: ', self.m)

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

        self.l = l

        return X

    def BetaValues(self, X, z):
        """
        compute beta values using matrix inversion
        input X: design matrix
        input z: input function
        input lambda_val: complexity parameter
        """

        if len(z.shape) > 1:
            z = np.ravel(z)

        if self.method == 'ridge':
            beta = np.dot(np.linalg.inv(np.dot(X.T,X) + np.dot(self.lambda_,np.identity(self.l))),np.dot(X.T,z))
        elif self.method == 'lasso':
            lasso = Lasso(alpha=self.lambda_, fit_intercept=False)
            lasso.fit(X, z)
            beta = lasso.coef_
        else:
            beta = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,z))

        return beta

    def ConfidenceInterval(self, z, z_predict):
        """
        function for calculating the confidence interval with a 95 % confidence level
        input z: function array
        input z_predict: predicted function array
        """

        # reshape z and z_predict if they are multidimensional
        if len(z.shape) > 1:
            z = np.ravel(z)
            z_predict = np.ravel(z_predict)

        # array dimensions
        n = len(z)
        l = len(self.beta)

        # calculate sigma squared (unbiased) and standard deviation
        sigma_squared = sum((z - z_predict)**2)/(n - l - 1)
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

    def MeanSquaredError(self, z, z_predict):
        """
        function for calculating the mean squared error (MSE)
        input z: function array
        input z_predict: predicted function array
        """

        len_z = len(np.ravel(z))

        mse = np.sum((z - z_predict)**2)/len_z

        return mse

    def R2Score(self, z, z_predict):
        """
        function for calculating the R2 score
        input z: function array
        input z_predict: predicted function array
        """

        len_z = len(np.ravel(z))

        # calculate mean value of z_predict
        mean_z_predict = np.sum(z_predict)/len_z

        r2score = 1. - np.sum((z - z_predict)**2)/np.sum((z - mean_z_predict)**2)

        return r2score

    def KFoldCrossValidation(self, X, z, folds, shuffle=True):
        """
        k-fold cross validation
        """

        # reshape z if they are multidimensional
        if len(z.shape) > 1:
            z = np.ravel(z)

        if shuffle:
            # shuffle X and z equally
            randomise = np.arange(len(z))
            np.random.shuffle(randomise)

            # randomise X and z
            X = X[randomise,:]
            z = z[randomise]

        # split X and z into k folds
        X_k = np.array_split(X, folds)
        z_k = np.array_split(z, folds)

        # arrays for training and test MSE
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

            # set test data equal to the deleted fold in training data
            X_test = X_k[k]
            z_test = z_k[k]

            # perform OLS
            beta_train = self.BetaValues(X_train, z_train)

            # calculate z_predict for training and test data
            zpred_train = np.dot(X_train, beta_train)
            zpred_test = np.dot(X_test, beta_train)

            # append MSE to lists
            mse_train[k] = self.MeanSquaredError(z_train, zpred_train)
            mse_test[k]  = self.MeanSquaredError(z_test, zpred_test)

        # perform statistical analysis like mean, average, standard deviation etc.
        mean_train = np.mean(mse_train)
        mean_test  = np.mean(mse_test)

        std_train = np.std(mse_train)
        std_test  = np.std(mse_test)

        print('k-fold cross validation:')
        print('Mean train:               ', mean_train)
        print('Mean test:                ', mean_test)
        print('Standard deviation train: ', std_train)
        print('Standard deviation test:  ', std_test)

    def Bootstrap(self, X, z, n_boots):
        """
        bootstrap algorithm
        """

        # reshape z if they are multidimensional
        if len(z.shape) > 1:
            z = np.ravel(z)

        # split into training and test data
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

        # create empty arrays to store test predicitions for each bootstrap
        zpred_test_store = np.empty((z_test.shape[0], n_boots))

        # arrays for storing mse for training and test data
        mse_train = np.zeros(n_boots)
        mse_test  = np.zeros(n_boots)

        # arrays for storing mse for training and test data
        r2s_train = np.zeros(n_boots)
        r2s_test  = np.zeros(n_boots)

        # perform MSE on the different training and test data
        for k in range(n_boots):

            # create random indices for every bootstrap
            random_index = np.random.randint(X_train.shape[0], size=X_train.shape[0])

            # resample X_train and z_train
            X_tmp = X_train[random_index,:]
            z_tmp = z_train[random_index]

            # obtain beta values for training data
            beta_train = self.BetaValues(X_tmp, z_tmp)

            # calculate z_predict for training and test data
            zpred_test_store[:,k] = np.dot(X_test, beta_train)
            zpred_train = np.dot(X_tmp, beta_train)

            # calculate MSE for training and test data
            mse_train[k] = self.MeanSquaredError(z_tmp, zpred_train)
            mse_test[k]  = self.MeanSquaredError(z_test, zpred_test_store[:,k])

            # calculate r2 score for training and test data
            r2s_train[k] = self.R2Score(z_tmp, zpred_train)
            r2s_test[k]  = self.R2Score(z_test, zpred_test_store[:,k])

            # store the best beta array
            if (k > 0) and (mse_train[k] < mse_train[k-1]):
                self.best_beta = beta_train

        # calculate bias an variance for the test data
        bias_test, var_test = self.BiasVarianceTradeoff(z_test, zpred_test_store)

        self.avg_mse_train = np.mean(mse_train)
        self.avg_mse_test  = np.mean(mse_test)

        self.avg_r2s_train = np.mean(r2s_train)
        self.avg_r2s_test  = np.mean(r2s_test)

        print('R2 score train: ', self.avg_mse_train)
        print('R2 score test:  ', self.avg_r2s_test)
        print('MSE train:      ', self.avg_mse_train)
        print('MSE test:       ', self.avg_mse_test)
        print('Bias^2 test:    ', bias_test)
        print('Variance test:  ', var_test)
        print(f'{self.avg_mse_test} >= {bias_test+var_test}')


    def BiasVarianceTradeoff(self, z_test, zpred_test):
        """
        function for calculating the bias-variance tradeoff (using k-fold cross validation to train/test data)
        input z_test: test data for the Franke function
        input zpred_test: array of dimension (z_test.shape[0], number_of_bootstraps)
        """

        bias_squared = np.mean((z_test[:,None] - np.mean(zpred_test, axis=1, keepdims=True))**2)
        variance     = np.mean(np.var(zpred_test, axis=1, keepdims=True))

        return bias_squared, variance

    def Benchmark(self):
        """
        function for creating benchmarks
        inputs x, y, z and z_predict: x-, y-, z- and z_predict-arrays
        input m: polynomial degree
        """

        ### benchmarking beta values ###

        # reshape x, y and z if they are multidimensional
        if len(self.x.shape) > 1:
            x = np.ravel(self.x)
            y = np.ravel(self.y)
            z = np.ravel(self.z)

        # expand x and y to have an additional dimension
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)

        # concatenate in order to create an input array acceptable in fit_transform
        input_arr = np.concatenate((x.T, y.T), axis=1)

        poly   = PolynomialFeatures(degree=self.m)
        Xp     = poly.fit_transform(input_arr)
        linreg = LinearRegression(fit_intercept=False)  # set fit_intercept to False to get all beta values
        linreg.fit(Xp, z)

        beta_bench = linreg.coef_

        ### benchmark mean squared error (MSE) ###
        mse       = self.MeanSquaredError(self.z, self.z_predict)
        mse_bench = mean_squared_error(self.z, self.z_predict)

        ### benchmark R2 score ###
        r2score       = self.R2Score(self.z, self.z_predict)
        r2score_bench = r2_score(self.z.ravel(), self.z_predict.ravel())    # r2score from sklearn has different ways of calculating when array is multidimensional

        # print(mse, mse_bench)
        print(r2score, r2score_bench)

        ### write to file ###
        file_handling.BenchmarksToFile(self.beta,
                                       beta_bench,
                                       mse,
                                       mse_bench,
                                       r2score,
                                       r2score_bench)

    def TaskA(self, plot=True):
        """
        run project task a
        """

        # calculate the Franke function and design matrix
        self.z = self.FrankeFunction(self.x, self.y)
        self.X = self.DesignMatrix(self.x, self.y)

        # beta values for ordinary least squares method
        self.beta = self.BetaValues(self.X, self.z)

        # calculate z_predict
        self.z_predict = np.dot(self.X,self.beta)
        self.z_predict = np.reshape(self.z_predict, (self.dim, self.dim))

        self.ConfidenceInterval(self.z, self.z_predict)

        mse     = self.MeanSquaredError(self.z, self.z_predict)
        r2score = self.R2Score(self.z, self.z_predict)

        print('MSE:     ', mse)
        print('r2score: ', r2score)

        if plot:
            # plotting_function.Plot3D(self.x, self.y, self.z_predict, self.m, self.dim, function='franke', savefig=False)
            # plotting_function.ErrorBars(self.beta, self.con_int, self.m, savefig=False)
            plotting_function.PlotDuo3D(self.x, self.y, self.z, self.z_predict, self.m, self.dim, lambda_val=0, function='franke', method='OLS', savefig=False)

    def TaskB(self):
        """
        run project task b
        """

        # calculate the Franke function and design matrix
        self.z = self.FrankeFunction(self.x, self.y)
        self.X = self.DesignMatrix(self.x, self.y)

        # k-fold cross validation (OLS)
        self.KFoldCrossValidation(self.X, self.z, folds=10, shuffle=True)

    def TaskC(self):
        """
        run project task c
        """

        # calculate the Franke function and design matrix
        self.z = self.FrankeFunction(self.x, self.y)
        self.X = self.DesignMatrix(self.x, self.y)

        # bootstrap method (OLS)
        self.Bootstrap(self.X, self.z, n_boots=100)

        # self.KFoldCrossValidation(self.X, self.z, folds=10, shuffle=True)

    def TaskD(self):
        """
        run project task d
        """

        # calculate the Franke function and design matrix
        self.z = self.FrankeFunction(self.x, self.y)
        self.X = self.DesignMatrix(self.x, self.y)

        # self.RidgeRegression(lambda_val)
        # self.ConfidenceInterval(self.z, self.z_predict)
        # self.mse = self.MeanSquaredError(self.z, self.z_predict)
        # self.r2score = self.R2Score(self.z, self.z_predict)

        self.Bootstrap(self.X, self.z, n_boots=100)

    def TaskE(self):
        """
        run project task e
        """

        # calculate the Franke function and design matrix
        self.z = self.FrankeFunction(self.x, self.y)
        self.X = self.DesignMatrix(self.x, self.y)

        # self.LassoRegression(lambda_val)
        # self.ConfidenceInterval(self.z, self.z_predict)
        # self.mse = self.MeanSquaredError(self.z, self.z_predict)
        # self.r2score = self.R2Score(self.z, self.z_predict

        self.Bootstrap(self.X, self.z, n_boots=100)

    def TaskG(self):
        """
        run project task g
        """

        # load terrain data
        terrain = imread('srtm_data_oslo.tif')

        # plt.figure()
        # plt.imshow(terrain, cmap='gray')
        # plt.show()

        # reduce the size of the terrain to 300x300
        # reduced_terrain = terrain[1200:1200+self.dim, 1200:1200+self.dim]
        reduced_terrain = terrain[1000:1000+self.dim, 200:200+self.dim]

        # set z equal to the normalised reduced_terrain array
        self.z = (reduced_terrain-reduced_terrain.min())/(reduced_terrain.max() - reduced_terrain.min())

        # plotting_function.Plot3D(self.x, self.y, self.z, self.m, self.dim, function='terrain', savefig=False)

        # calculate design matrix
        self.X = self.DesignMatrix(self.x, self.y)

        self.Bootstrap(self.X, self.z, n_boots=100)

        # calculate z_predict with the best beta
        self.z_predict = np.dot(self.X,self.best_beta)
        self.z_predict = np.reshape(self.z_predict, (self.dim, self.dim))

if __name__ == '__main__':

    # set tasks to run
    run_task_A = False
    run_task_B = False
    run_task_C = False
    run_task_D = False
    run_task_E = False
    run_task_G = True

    if run_task_A:
        max_degree = 1
        for m in range(1,max_degree+1):
            run = RegressionAnalysis(dim=100, m=m, noise=False)
            run.TaskA(plot=True)

    if run_task_B:
        max_degree = 5
        for m in range(1,max_degree+1):
            run = RegressionAnalysis(dim=100, m=m, noise=False)
            run.TaskB()

    if run_task_C:
        max_degree = 5
        mse_train  = np.zeros(max_degree)
        mse_test   = np.zeros(max_degree)
        m_array    = np.linspace(1,max_degree,max_degree)

        for m in range(1,max_degree+1):
            print(m)
            run = RegressionAnalysis(dim=100, m=m, noise=False)
            run.TaskC()
            mse_train[m-1] = run.avg_mse_train
            mse_test[m-1]  = run.avg_mse_test

        # plot model complexity vs. predicted error
        plotting_function.PlotMSETestTrain(m_array, mse_train, mse_test, max_degree, savefig=False)

    if run_task_D:

        max_degree = 5
        m_array    = np.linspace(1,max_degree,max_degree)

        # create list of different lambda values
        list_of_lambdas = [0.001, 0.1, 1, 10, 10000]

        mse_train  = np.zeros((max_degree, len(list_of_lambdas)))
        mse_test   = np.zeros((max_degree, len(list_of_lambdas)))
        r2s_train = np.zeros((max_degree, len(list_of_lambdas)))
        r2s_test  = np.zeros((max_degree, len(list_of_lambdas)))

        for l in range(len(list_of_lambdas)):
            for m in range(1,max_degree+1):
                run = RegressionAnalysis(dim=100, m=m, lambda_val=l, noise=False, method='ridge')
                run.TaskD()

                mse_train[m-1,l] = run.avg_mse_train
                mse_test[m-1,l]  = run.avg_mse_test
                r2s_train[m-1,l] = run.avg_r2s_train
                r2s_test[m-1,l]  = run.avg_r2s_test

        # plot MSE and R2 score
        plotting_function.PlotMultipleMSETestTrain(m_array, mse_train, mse_test, list_of_lambdas, max_degree, 'Ridge', savefig=False)
        plotting_function.PlotMultipleR2STestTrain(m_array, r2s_train, r2s_test, list_of_lambdas, max_degree, 'Ridge', savefig=False)

    if run_task_E:

        max_degree = 5
        m_array    = np.linspace(1,max_degree,max_degree)

        # create list of different lambda values
        list_of_lambdas = [0.001, 0.1, 1, 100, 1000, 10000]

        mse_train = np.zeros((max_degree, len(list_of_lambdas)))
        mse_test  = np.zeros((max_degree, len(list_of_lambdas)))
        r2s_train = np.zeros((max_degree, len(list_of_lambdas)))
        r2s_test  = np.zeros((max_degree, len(list_of_lambdas)))

        for l in range(len(list_of_lambdas)):
            for m in range(1,max_degree+1):
                run = RegressionAnalysis(dim=100, m=m, lambda_val=l, noise=False, method='lasso')
                run.TaskE()

                mse_train[m-1,l] = run.avg_mse_train
                mse_test[m-1,l]  = run.avg_mse_test
                r2s_train[m-1,l] = run.avg_r2s_train
                r2s_test[m-1,l]  = run.avg_r2s_test

        # plot MSE and R2 score
        plotting_function.PlotMultipleMSETestTrain(m_array, mse_train, mse_test, list_of_lambdas, max_degree, 'Lasso', savefig=False)
        plotting_function.PlotMultipleR2STestTrain(m_array, r2s_train, r2s_test, list_of_lambdas, max_degree, 'Lasso', savefig=False)

    if run_task_G:
        max_degree = 9
        # for m in range(9,max_degree+1):
        #     run = RegressionAnalysis(dim=100, m=m, lambda_val=0.01, noise=False, method='OLS')
        #     run.TaskG()
        #     run.z_ols = run.z_predict
        #     plotting_function.PlotDuo3D(run.x, run.y, run.z, run.z_ols, run.m, run.dim, lambda_val=0, function='terrain', method='OLS', savefig=False)

        lambda_val = 0.001

        run1 = RegressionAnalysis(dim=100, m=9, lambda_val=lambda_val, noise=False, method='OLS')
        run1.TaskG()

        run2 = RegressionAnalysis(dim=100, m=9, lambda_val=lambda_val, noise=False, method='ridge')
        run2.TaskG()
        # plotting_function.PlotDuo3D(run2.x, run2.y, run2.z, run2.z_predict, run2.m, run2.dim, lambda_val=lambda_val, function='terrain', method='ridge', savefig=False)

        run3 = RegressionAnalysis(dim=100, m=9, lambda_val=lambda_val, noise=False, method='lasso')
        run3.TaskG()

        plotting_function.PlotCuatro3D(run1.x, run1.y, run1.z, run1.z_predict,
                                       run2.x, run2.y, run2.z_predict,
                                       run3.x, run3.y, run3.z_predict,
                                       run1.m, run1.dim, lambda_val=lambda_val, function='terrain', savefig=False)





    # run.Benchmark()
