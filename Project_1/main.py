import define_colormap
import file_handling
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


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

        # self.x = np.arange(0, 1, 0.05)
        # self.y = np.arange(0, 1, 0.05)

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

        # reshape x and y if they are multidimensional
        if len(x.shape) > 1:
            print('x and y are in a meshgrid, and are being reshaped')
            x = np.ravel(x)
            y = np.ravel(y)

        n = len(x)
        l = int((m+1)*(m+2)/2)        # number of beta values
        print('Dimension of design matrix: (%d x %d)' % (n, l))

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

        # return z_predict

    def Benchmark(self, m):
        """
        function for creating benchmarks
        input m: polynomial degree
        """

        self.OrdinaryLeastSquares()

        # set up x- and y-arrays to a dimension that fit_transform accepts
        x = np.ravel(self.x)
        y = np.ravel(self.y)

        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)

        input_arr = np.concatenate((x.T, y.T), axis=1)

        poly = PolynomialFeatures(degree=m)
        Xp   = poly.fit_transform(input_arr)

        self.z = np.ravel(self.z)
        linreg = LinearRegression(fit_intercept=False)
        linreg.fit(Xp, self.z)

        self.beta2 = linreg.coef_

        # write to file
        file_handling.BenchmarksToFile(self.beta, self.beta2)


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
    # run.OrdinaryLeastSquares()
    # run.PlotMultiple3D()
    run.Benchmark(m=2)
