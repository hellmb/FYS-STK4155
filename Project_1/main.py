import define_colormap
import file_handling
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

class RegressionAnalysis():

    def __init__(self):
        """
        initialise the instance of the class
        """

        # random values (sorted)
        self.x = np.sort(np.random.uniform(0, 1, 100))
        self.y = np.sort(np.random.uniform(0, 1, 100))

        # create meshgrid
        self.x, self.y = np.meshgrid(self.x, self.y)

    def FrankeFunction(self, x, y, noise=False):
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

        if noise:
            z += 0.1*np.random.randn(100,1)

        return z

    def DesignMatrix(self, x, y, m):
        """
        function for creating the design  matrix
        the matrix has dimension (nxl) where n is the number of data points
        and l is the number of terms in the polynomial with degree m (maybe write in report instead)
        input x, y: data points in x- and y-direction
        input m: polynomial degree
        """

        print('You are fitting a polynomial of degree %d' % m)

        # reshape x and y if they are multidimensional
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        n = len(x)
        l = int((m+1)*(m+2)/2)        # number of beta values
        print('Dimension of design matrix: (%d, %d)' % (n, l))

        # set up the design matrix
        X = np.ones((n, l))
        for i in range(1, m+1):
            index = int(i*(i+1)/2)
            for j in range(i+1):
                X[:,index+j] = x**(i-j) * y**j

        return X

    def ComputeBetaValues(self, X, z):
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
        self.X = self.DesignMatrix(self.x, self.y, m=2)

        # reshape matrix into vector in order to compute the beta values
        z_vec = np.ravel(self.z)

        # calculate beta values
        self.beta = self.ComputeBetaValues(self.X, z_vec)

        # compute predicted Franke function
        self.z_predict = np.dot(self.X,self.beta)

    def ConfidenceInterval(self, z, z_predict):
        """
        function for calculating the confidence interval with a 95 % confidence level
        input z: flattened Franke function array
        input z_predict: flattened predicted Franke function array
        """

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
        self.con_int = np.zeros((len(self.beta),2))

        print('       Beta          Lower percentile   Upper percentile')
        for i in range(len(self.beta)):
            self.con_int[i,0] = self.beta[i] - Z_score*np.sqrt(XTX_inv[i,i])*sigma
            self.con_int[i,1] = self.beta[i] + Z_score*np.sqrt(XTX_inv[i,i])*sigma

            print(self.beta[i], self.con_int[i, 0], self.con_int[i,1])


    def MeanSquaredError(self, z, z_predict):
        """
        function for calculating the mean squared error (MSE)
        input z: flattened Franke function array
        input z_predict: flattened predicted Franke function array
        """

        self.mse = sum((z - z_predict)**2)/len(z)

    def R2Score(self, z, z_predict):
        """
        function for calculating the R2 score
        input z: flattened Franke function array
        input z_predict: flattened predicted Franke function array
        """

        # calculate mean value of z_predict
        mean_z_predict = sum(z_predict)/len(z)

        self.r2score = 1. - sum((z - z_predict)**2)/sum((z - mean_z_predict)**2)

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

        # self.OrdinaryLeastSquares()

        ### benchmarking beta values ###

        # expand to an additional dimension
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
        self.MeanSquaredError(z, z_predict)
        self.mse_bench = mean_squared_error(z, z_predict)

        ### benchmark R2 score ###
        self.R2Score(z, z_predict)
        self.r2score_bench = r2_score(z, z_predict)

        ### write to file ###
        file_handling.BenchmarksToFile(self.beta,
                                       self.beta_bench,
                                       self.mse,
                                       self.mse_bench,
                                       self.r2score,
                                       self.r2score_bench)

    def Plot3D(self):
        """
        create 3D plot of the Franke function
        """

        # calculate the Franke function
        z = self.FrankeFunction(self.x, self.y)

        # plot figure
        fig = plt.figure()
        ax  = fig.gca(projection='3d')

        # define costum colormap
        cmap = define_colormap.DefineColormap('arctic')

        # plot surface
        surf = ax.plot_surface(self.x, self.y, z, cmap=cmap, linewidth=0, antialiased=False)

        # Customize the z axis.
        # ax.set_zlim(-0.10, 1.40)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # add colorbar
        fig.colorbar(surf)

        plt.show()

    def PlotMultiple3D(self):
        """
        create multiple 3D subplot of the Franke function
        """

        # calculate predicted Franke function
        self.OrdinaryLeastSquares()
        self.z_predict = np.reshape(self.z_predict, (100,100))

        # plot figure
        fig = plt.figure(figsize=(15,6))
        ax1  = fig.add_subplot(1, 2, 1, projection='3d')

        # define costum colormap
        cmap = define_colormap.DefineColormap('arctic')

        # plot surface
        surf1 = ax1.plot_surface(self.x, self.y, self.z, cmap=cmap, linewidth=0, antialiased=False)

        ax2  = fig.add_subplot(1, 2, 2, projection='3d')
        surf2 = ax2.plot_surface(self.x, self.y, self.z_predict, cmap=cmap, linewidth=0, antialiased=False)

        # add colorbar
        # fig.colorbar(surf2)

        plt.show()

if __name__ == '__main__':
    run = RegressionAnalysis()
    run.OrdinaryLeastSquares()
    run.RavelArrays()
    run.ConfidenceInterval(run.zr, run.zr_predict)
    run.MeanSquaredError(run.zr, run.zr_predict)
    run.Benchmark(run.xr, run.yr, run.zr, run.zr_predict, m=2)
    # run.PlotMultiple3D()
