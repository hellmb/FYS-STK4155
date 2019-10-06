import sys
import matplotlib
import numpy as np
from random import random
from imageio import imread
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# import separate scripts
import file_handling
import plotting_function

# set general plotting font consistent with LaTeX
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

class RegressionAnalysis():

    def __init__(self, dim=100, m=2, lambda_val=0, noise=False, method='OLS'):
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
        self.lambda_ = lambda_val

        # noise, boolean
        self.noise = noise

        # regression method, string
        self.method = method

    def FrankeFunction(self, x, y):
        """
        function to calculate the Franke function
        param x, y: data points in x- and y-direction
        """

        term1 = 0.75*np.exp(-((9*x-2)**2)/4. - ((9*y-2)**2)/4.)
        term2 = 0.75*np.exp(-((9*x+1)**2)/49. - ((9*y+1)**2)/10.)
        term3 = 0.5*np.exp(-((9*x-7)**2)/4. - ((9*y-3)**2)/4.)
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

        z = term1 + term2 + term3 + term4

        if self.noise:
            z += 0.1*np.random.randn(self.dim,1)

        return z

    def Terrain(self):
        """
        function that reads and processes the real terrain data
        """

        # load terrain data
        terrain = imread('srtm_data_oslo.tif')

        # reduce the size of the terrain to 200x200
        reduced_terrain = terrain[1000:1000+self.dim, 200:200+self.dim]

        # set z equal to the normalised reduced_terrain array
        z = (reduced_terrain-reduced_terrain.min())/(reduced_terrain.max() - reduced_terrain.min())

        return z

    def DesignMatrix(self, x, y):
        """
        function for creating the design matrix
        param x, y: data points in x- and y-direction
        """

        # reshape x and y if they are multidimensional
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        n = len(x)
        l = int((self.m+1)*(self.m+2)/2)        # number of beta values

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
        param X: design matrix
        param z: input function
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

    def ConfidenceInterval(self, z, z_predict, X, beta):
        """
        function for calculating the confidence interval with a 95 % confidence level
        param z: function array
        param z_predict: predicted function array
        param X: design matrix
        param beta: beta array
        """

        # reshape z and z_predict if they are multidimensional
        if len(z.shape) > 1:
            z = np.ravel(z)
            z_predict = np.ravel(z_predict)

        # array dimensions
        n = len(z)
        l = len(beta)

        # calculate sigma squared (unbiased) and standard deviation
        sigma_squared = sum((z - z_predict)**2)/(n - l - 1)
        sigma         = np.sqrt(sigma_squared)

        # variance of beta
        XTX_inv       = np.linalg.inv(np.dot(X.T,X))
        var_beta = sigma_squared*XTX_inv

        # Z-score for a 95% confidence interval is 1.96
        Z_score = 1.96

        # create array for storing the confidence interval for beta
        self.con_int = np.zeros((l,2))

        for i in range(l):
            self.con_int[i,0] = beta[i] - Z_score*np.sqrt(XTX_inv[i,i])*sigma
            self.con_int[i,1] = beta[i] + Z_score*np.sqrt(XTX_inv[i,i])*sigma


    def MeanSquaredError(self, z, z_predict):
        """
        function for calculating the mean squared error (MSE)
        param z: function array
        param z_predict: predicted function array
        """

        len_z = len(np.ravel(z))

        mse = np.sum((z - z_predict)**2)/len_z

        return mse

    def R2Score(self, z, z_predict):
        """
        function for calculating the R2 score
        param z: function array
        param z_predict: predicted function array
        """

        len_z = len(np.ravel(z))

        # calculate mean value of z_predict
        mean_z_predict = np.sum(z_predict)/len_z

        r2score = 1. - np.sum((z - z_predict)**2)/np.sum((z - mean_z_predict)**2)

        return r2score

    def Bootstrap(self, X, z, n_boots):
        """
        bootstrap algorithm
        param X: design matrix
        param z: function array
        param n_boots: number of bootstraps
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
            if (k > 0) and (mse_test[k] < mse_test[k-1]):
                self.best_beta = beta_train
            else:
                self.best_beta = beta_train

        # calculate bias an variance for the test data
        self.bias_test, self.var_test = self.BiasVarianceTradeoff(z_test, zpred_test_store)

        # take the average of the training and test mse and r2 score
        self.avg_mse_train = np.mean(mse_train)
        self.avg_mse_test  = np.mean(mse_test)

        self.avg_r2s_train = np.mean(r2s_train)
        self.avg_r2s_test  = np.mean(r2s_test)


    def BiasVarianceTradeoff(self, z_test, zpred_test):
        """
        function for calculating the bias-variance tradeoff
        param z_test: test data array
        param zpred_test: test data array of dimension (z_test.shape[0], number_of_bootstraps)
        """

        bias_squared = np.mean((z_test[:,None] - np.mean(zpred_test, axis=1, keepdims=True))**2)
        variance     = np.mean(np.var(zpred_test, axis=1, keepdims=True))

        return bias_squared, variance

    def Benchmark(self):
        """
        function for creating benchmarks
        """

        ### benchmarking beta values ###

        z = self.FrankeFunction(self.x, self.y)
        X = self.DesignMatrix(self.x, self.y)
        beta = self.BetaValues(X, z)
        z_predict = np.dot(X,beta)

        # reshape x, y and z if they are multidimensional
        if len(self.x.shape) > 1:
            x = np.ravel(self.x)
            y = np.ravel(self.y)
            z = np.ravel(z)
            z_predict = np.ravel(z_predict)

        # expand x and y to have an additional dimension
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)

        # concatenate in order to create an input array acceptable in fit_transform
        input_arr = np.concatenate((x.T, y.T), axis=1)

        poly   = PolynomialFeatures(degree=self.m)
        Xp     = poly.fit_transform(input_arr)
        linreg = LinearRegression(fit_intercept=False)
        linreg.fit(Xp, z)

        beta_bench = linreg.coef_

        ### benchmark mean squared error (MSE) ###
        mse       = self.MeanSquaredError(z, z_predict)
        mse_bench = mean_squared_error(z, z_predict)

        ### benchmark R2 score ###
        r2score       = self.R2Score(z, z_predict)
        r2score_bench = r2_score(z.ravel(), z_predict.ravel())

        ### write to file ###
        file_handling.BenchmarksToFile(beta,
                                       beta_bench,
                                       mse,
                                       mse_bench,
                                       r2score,
                                       r2score_bench)

if __name__ == '__main__':

    write_benchmarks = False
    if write_benchmarks:
        # write benchmark values to file
        bench = RegressionAnalysis(dim=100, m=5, noise=False)
        bench.Benchmark()

    # check for input arguments
    if len(sys.argv) == 1:
        print('No arguments passed. Run "python main.py franke" or "python main.py terrain" in terminal.')
        sys.exit()

    arg = sys.argv[1]

    # list of methods
    methods = ['OLS', 'ridge', 'lasso']

    # list of different lambda values to test
    list_of_lambdas = [0.0001, 0.001, 0.1, 1, 10, 100, 10000]

    # maximum polynomial degree
    max_degree = 5

    # number of bootstraps
    n_boots = 1000

    if arg == 'franke':

        for method in methods:
            if method == 'OLS':

                # define empty arrays for different values
                mse_train    = np.zeros(max_degree)
                mse_ols_test = np.zeros(max_degree)
                r2s_train    = np.zeros(max_degree)
                r2s_ols_test = np.zeros(max_degree)
                bias         = np.zeros(max_degree)
                variance     = np.zeros(max_degree)

                m_array    = np.linspace(1,max_degree,max_degree)

                for m in range(1,max_degree+1):
                    run = RegressionAnalysis(dim=100, m=m, noise=True)

                    # calculate the Franke function
                    z = run.FrankeFunction(run.x, run.y)

                    # compute desing matrix
                    X = run.DesignMatrix(run.x, run.y)

                    # resample data
                    run.Bootstrap(X, z, n_boots=n_boots)

                    # calculate z_predict with the best beta
                    z_predict = np.dot(X,run.best_beta)
                    z_predict = np.reshape(z_predict, (run.dim, run.dim))

                    # calculate the confidence interval
                    run.ConfidenceInterval(z, z_predict, X, run.best_beta)

                    # fill empty arrays with MSE adn R2 score for training and test data
                    mse_ols_test[m-1] = run.avg_mse_test
                    mse_train[m-1]    = run.avg_mse_train
                    r2s_train[m-1]    = run.avg_r2s_train
                    r2s_ols_test[m-1] = run.avg_r2s_test
                    bias[m-1]         = run.bias_test**(1./2)
                    variance[m-1]     = run.var_test

                    # calculate confidence interval for maximum polynomial degree
                    if m == max_degree:
                        ci_ols      = run.con_int
                        ci_beta_ols = run.best_beta

                # plot results
                plotting_function.PlotMSETestTrain(m_array, mse_train, mse_ols_test, max_degree, function=arg, savefig=False)
                plotting_function.PlotR2STestTrain(m_array, r2s_train, r2s_ols_test, max_degree, function=arg, savefig=False)
                plotting_function.PlotBiasVariance(m_array, bias, variance, max_degree, function=arg, savefig=False)

            else:

                # define empty arrays for different values
                mse_train = np.zeros((max_degree, len(list_of_lambdas)))
                mse_test  = np.zeros((max_degree, len(list_of_lambdas)))
                r2s_train = np.zeros((max_degree, len(list_of_lambdas)))
                r2s_test  = np.zeros((max_degree, len(list_of_lambdas)))
                bias      = np.zeros((max_degree, len(list_of_lambdas)))
                variance  = np.zeros((max_degree, len(list_of_lambdas)))
                m_array   = np.linspace(1,max_degree,max_degree)

                for l in range(len(list_of_lambdas)):
                    for m in range(1,max_degree+1):
                        run = RegressionAnalysis(dim=100, m=m, lambda_val=l, noise=True, method='ridge')

                        # calculate the Franke function
                        z = run.FrankeFunction(run.x, run.y)

                        # compute desing matrix
                        X = run.DesignMatrix(run.x, run.y)

                        # resample data
                        run.Bootstrap(X, z, n_boots=n_boots)

                        # calculate z_predict with the best beta
                        z_predict = np.dot(X,run.best_beta)
                        z_predict = np.reshape(z_predict, (run.dim, run.dim))

                        # calculate the confidence interval
                        run.ConfidenceInterval(z, z_predict, X, run.best_beta)

                        # fill arrays with MSE and R2 score for training and test data
                        mse_train[m-1,l] = run.avg_mse_train
                        mse_test[m-1,l]  = run.avg_mse_test
                        r2s_train[m-1,l] = run.avg_r2s_train
                        r2s_test[m-1,l]  = run.avg_r2s_test
                        bias[m-1,l]      = run.bias_test**(1./2)
                        variance[m-1,l]  = run.var_test

                        # calculate confidence interval for maximum polynomial degree
                        if list_of_lambdas[l] == min(list_of_lambdas) and method == 'ridge':
                            ci_ridge      = run.con_int
                            ci_beta_ridge = run.best_beta

                        if list_of_lambdas[l] == min(list_of_lambdas) and method == 'lasso':
                            ci_lasso      = run.con_int
                            ci_beta_lasso = run.best_beta

                # change method name to have capital letter in plot titles
                if method == 'ridge':
                    string_name = 'Ridge'
                    mse_ridge_test = np.zeros(max_degree)
                    r2s_ridge_test = np.zeros(max_degree)
                    mse_ridge_test = mse_test[:,0]
                    r2s_ridge_test = r2s_test[:,0]
                else:
                    string_name = 'Lasso'
                    mse_lasso_test = np.zeros(max_degree)
                    r2s_lasso_test = np.zeros(max_degree)
                    mse_lasso_test = mse_test[:,0]
                    r2s_lasso_test = r2s_test[:,0]

                # plot MSE, R2 score and bias-variance tradeoff
                plotting_function.PlotMultipleMSETestTrain(m_array, mse_train, mse_test, list_of_lambdas, max_degree, string_name, function=arg, savefig=False)
                plotting_function.PlotMultipleR2STestTrain(m_array, r2s_train, r2s_test, list_of_lambdas, max_degree, string_name, function=arg, savefig=False)
                plotting_function.PlotMultipleBiasVariance(m_array, bias, variance, list_of_lambdas, max_degree, string_name, function=arg, savefig=False)

                if method == 'lasso':
                    # plot surface for best lambda value for all regression methods
                    # at the end of the loop (when the lasso method has been run)

                    # print table values in latex format
                    print('\nMSE for OLS, Ridge and Lasso')
                    for m in range(max_degree):
                        print('& %.3f & %.3f & %.3f' % (mse_ols_test[m], mse_ridge_test[m], mse_lasso_test[m]))

                    print('\nR2 score for OLS, Ridge and Lasso')
                    for m in range(max_degree):
                        print('& %.3f & %.3f & %.3f' % (r2s_ols_test[m],r2s_ridge_test[m], r2s_lasso_test[m]))

                    print('\nBeta values and confidence intervals for OLS, Ridge and Lasso for 5th degree polynomial')
                    for k in range(len(ci_beta_ols)):
                        print('%.3f & (%.3f, %.3f) & %.3f & (%.3f, %.3f) & %.3f & (%.3f, %.3f)' % (ci_beta_ols[k],
                        ci_ols[k,0], ci_ols[k,1],ci_beta_ridge[k], ci_ridge[k,0], ci_ridge[k,1],ci_beta_lasso[k],
                        ci_lasso[k,0], ci_lasso[k,1],))

                    # create list of lambda values to plot as surfaces
                    lambda_  = [min(list_of_lambdas), 1]

                    for lam in lambda_:

                        run1 = RegressionAnalysis(dim=100, m=max_degree, lambda_val=lam, noise=True, method='OLS')
                        z1   = run1.FrankeFunction(run1.x, run1.y)
                        X1   = run1.DesignMatrix(run1.x, run1.y)
                        run1.Bootstrap(X1, z1, n_boots=n_boots)
                        z_predict1 = np.dot(X1,run1.best_beta)
                        z_predict1 = np.reshape(z_predict1, (run1.dim, run1.dim))

                        run2 = RegressionAnalysis(dim=100, m=max_degree, lambda_val=lam, noise=True, method='ridge')
                        z2   = run2.FrankeFunction(run2.x, run2.y)
                        X2   = run2.DesignMatrix(run2.x, run2.y)
                        run2.Bootstrap(X2, z2, n_boots=n_boots)
                        z_predict2 = np.dot(X2,run2.best_beta)
                        z_predict2 = np.reshape(z_predict2, (run2.dim, run2.dim))

                        run3 = RegressionAnalysis(dim=100, m=max_degree, lambda_val=lam, noise=True, method='lasso')
                        z3   = run3.FrankeFunction(run3.x, run3.y)
                        X3   = run3.DesignMatrix(run3.x, run3.y)
                        run3.Bootstrap(X3, z3, n_boots=n_boots)
                        print(run3.best_beta)
                        z_predict3 = np.dot(X3,run3.best_beta)
                        z_predict3 = np.reshape(z_predict3, (run3.dim, run3.dim))

                        plotting_function.PlotCuatro3D(run1.x, run1.y, z1, z_predict1,
                                                       run2.x, run2.y, z_predict2,
                                                       run3.x, run3.y, z_predict3,
                                                       run1.m, run1.dim, lambda_val=lam, function=arg, savefig=False)

    elif arg == 'terrain':

        for method in methods:
            if method == 'OLS':

                # define empty arrays for different values
                mse_train    = np.zeros(max_degree)
                mse_ols_test = np.zeros(max_degree)
                r2s_train    = np.zeros(max_degree)
                r2s_ols_test = np.zeros(max_degree)
                bias         = np.zeros(max_degree)
                variance     = np.zeros(max_degree)

                m_array    = np.linspace(1,max_degree,max_degree)

                for m in range(1,max_degree+1):
                    run = RegressionAnalysis(dim=200, m=m, noise=True)

                    # calculate the Franke function
                    z = run.Terrain()

                    # compute desing matrix
                    X = run.DesignMatrix(run.x, run.y)

                    # resample data
                    run.Bootstrap(X, z, n_boots=n_boots)

                    # calculate z_predict with the best beta
                    z_predict = np.dot(X,run.best_beta)
                    z_predict = np.reshape(z_predict, (run.dim, run.dim))

                    # calculate the confidence interval
                    run.ConfidenceInterval(z, z_predict, X, run.best_beta)

                    # fill empty arrays with MSE adn R2 score for training and test data
                    mse_ols_test[m-1] = run.avg_mse_test
                    mse_train[m-1]    = run.avg_mse_train
                    r2s_train[m-1]    = run.avg_r2s_train
                    r2s_ols_test[m-1] = run.avg_r2s_test
                    bias[m-1]         = run.bias_test**(1./2)
                    variance[m-1]     = run.var_test

                    # calculate confidence interval for maximum polynomial degree
                    if m == max_degree:
                        ci_ols      = run.con_int
                        ci_beta_ols = run.best_beta

                # plot results
                plotting_function.PlotMSETestTrain(m_array, mse_train, mse_ols_test, max_degree, function=arg, savefig=False)
                plotting_function.PlotR2STestTrain(m_array, r2s_train, r2s_ols_test, max_degree, function=arg, savefig=False)
                plotting_function.PlotBiasVariance(m_array, bias, variance, max_degree, function=arg, savefig=False)

            else:

                # define empty arrays for different values
                mse_train = np.zeros((max_degree, len(list_of_lambdas)))
                mse_test  = np.zeros((max_degree, len(list_of_lambdas)))
                r2s_train = np.zeros((max_degree, len(list_of_lambdas)))
                r2s_test  = np.zeros((max_degree, len(list_of_lambdas)))
                bias      = np.zeros((max_degree, len(list_of_lambdas)))
                variance  = np.zeros((max_degree, len(list_of_lambdas)))
                m_array   = np.linspace(1,max_degree,max_degree)

                for l in range(len(list_of_lambdas)):
                    for m in range(1,max_degree+1):
                        run = RegressionAnalysis(dim=200, m=m, lambda_val=l, noise=True, method='ridge')

                        # calculate the Franke function
                        z = run.Terrain()

                        # compute desing matrix
                        X = run.DesignMatrix(run.x, run.y)

                        # resample data
                        run.Bootstrap(X, z, n_boots=n_boots)

                        # calculate z_predict with the best beta
                        z_predict = np.dot(X,run.best_beta)
                        z_predict = np.reshape(z_predict, (run.dim, run.dim))

                        # calculate the confidence interval
                        run.ConfidenceInterval(z, z_predict, X, run.best_beta)

                        # fill arrays with MSE and R2 score for training and test data
                        mse_train[m-1,l] = run.avg_mse_train
                        mse_test[m-1,l]  = run.avg_mse_test
                        r2s_train[m-1,l] = run.avg_r2s_train
                        r2s_test[m-1,l]  = run.avg_r2s_test
                        bias[m-1,l]      = run.bias_test**(1./2)
                        variance[m-1,l]  = run.var_test

                        # calculate confidence interval for maximum polynomial degree
                        if list_of_lambdas[l] == min(list_of_lambdas) and method == 'ridge':
                            ci_ridge      = run.con_int
                            ci_beta_ridge = run.best_beta

                        if list_of_lambdas[l] == min(list_of_lambdas) and method == 'lasso':
                            ci_lasso      = run.con_int
                            ci_beta_lasso = run.best_beta

                # change method name to have capital letter for plotting purposes
                if method == 'ridge':
                    string_name = 'Ridge'
                    mse_ridge_test = np.zeros(max_degree)
                    r2s_ridge_test = np.zeros(max_degree)
                    mse_ridge_test = mse_test[:,0]
                    r2s_ridge_test = r2s_test[:,0]
                else:
                    string_name = 'Lasso'
                    mse_lasso_test = np.zeros(max_degree)
                    r2s_lasso_test = np.zeros(max_degree)
                    mse_lasso_test = mse_test[:,0]
                    r2s_lasso_test = r2s_test[:,0]

                # plot MSE, R2 score and bias-variance tradeoff
                plotting_function.PlotMultipleMSETestTrain(m_array, mse_train, mse_test, list_of_lambdas, max_degree, string_name, function=arg, savefig=False)
                plotting_function.PlotMultipleR2STestTrain(m_array, r2s_train, r2s_test, list_of_lambdas, max_degree, string_name, function=arg, savefig=False)
                plotting_function.PlotMultipleBiasVariance(m_array, bias, variance, list_of_lambdas, max_degree, string_name, function=arg, savefig=False)


                if method == 'lasso':
                    # plot surface for best lambda value for all regression methods
                    # at the end of the loop (when the lasso method has been run)

                    # print table values in latex format
                    print('\nMSE for OLS, Ridge and Lasso')
                    for m in range(max_degree):
                        print('& %.3f & %.3f & %.3f' % (mse_ols_test[m], mse_ridge_test[m], mse_lasso_test[m]))

                    print('\nR2 score for OLS, Ridge and Lasso')
                    for m in range(max_degree):
                        print('& %.3f & %.3f & %.3f' % (r2s_ols_test[m],r2s_ridge_test[m], r2s_lasso_test[m]))

                    print('\nBeta values and confidence intervals for OLS, Ridge and Lasso for 5th degree polynomial')
                    for k in range(len(ci_beta_ols)):
                        print('%.3f & (%.3f, %.3f) & %.3f & (%.3f, %.3f) & %.3f & (%.3f, %.3f)' % (ci_beta_ols[k],
                        ci_ols[k,0], ci_ols[k,1],ci_beta_ridge[k], ci_ridge[k,0], ci_ridge[k,1],ci_beta_lasso[k],
                        ci_lasso[k,0], ci_lasso[k,1],))

                    # create list of lambda values to plot as surfaces
                    lambda_  = [min(list_of_lambdas), 1]

                    for lam in lambda_:

                        run1 = RegressionAnalysis(dim=200, m=max_degree, lambda_val=lam, noise=True, method='OLS')
                        z1   = run1.Terrain()
                        X1   = run1.DesignMatrix(run1.x, run1.y)
                        run1.Bootstrap(X1, z1, n_boots=n_boots)
                        z_predict1 = np.dot(X1,run1.best_beta)
                        z_predict1 = np.reshape(z_predict1, (run1.dim, run1.dim))

                        run2 = RegressionAnalysis(dim=200, m=max_degree, lambda_val=lam, noise=True, method='ridge')
                        z2   = run2.Terrain()
                        X2   = run2.DesignMatrix(run2.x, run2.y)
                        run2.Bootstrap(X2, z2, n_boots=n_boots)
                        z_predict2 = np.dot(X2,run2.best_beta)
                        z_predict2 = np.reshape(z_predict2, (run2.dim, run2.dim))

                        run3 = RegressionAnalysis(dim=200, m=max_degree, lambda_val=lam, noise=True, method='lasso')
                        z3   = run3.Terrain()
                        X3   = run3.DesignMatrix(run3.x, run3.y)
                        run3.Bootstrap(X3, z3, n_boots=n_boots)
                        z_predict3 = np.dot(X3,run3.best_beta)
                        z_predict3 = np.reshape(z_predict3, (run3.dim, run3.dim))

                        plotting_function.PlotCuatro3D(run1.x, run1.y, z1, z_predict1,
                                                       run2.x, run2.y, z_predict2,
                                                       run3.x, run3.y, z_predict3,
                                                       run1.m, run1.dim, lambda_val=lam, function=arg, savefig=False)

    else:
        print('Invalid input argument. Please specify "franke" or "terrain".')
