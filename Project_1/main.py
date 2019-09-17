import define_colormap
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


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
        and l is the number of terms in the polynomial with degree m
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
        """

        beta = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,z))

        return beta

    def OrdinaryLeastSquares(self):
        """
        function performing ordinary least squares (OLS) on the Franke function
        """

        # compute the Franke function
        z = self.FrankeFunction(self.x, self.y)

        # compute the design matrix based on the polynomial degree m
        X = self.DesignMatrix(self.x, self.y, m=5)

        # reshape matrix into vector in order to compute the beta values
        z_vec = np.ravel(z)

        # calculate beta values
        beta = self.ComputeBetaValues(X, z_vec)

        print(X.shape, beta.shape)

        # compute predicted Franke function
        z_predict = np.dot(X,beta)

        return z_predict

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

        # calculate the Franke function
        z = self.FrankeFunction(self.x, self.y)

        print(z.shape)

        # calculate predicted Franke function
        z_predict = self.OrdinaryLeastSquares()
        np.reshape(z_predict, (100,100))

        print(z_predict)
        print(z_predict.shape)


        # plot figure
        fig = plt.figure()
        ax1  = fig.add_subplot(1, 2, 1, projection='3d')

        # define costum colormap
        cmap = define_colormap.DefineColormap('arctic')

        # plot surface
        surf1 = ax1.plot_surface(self.x, self.y, z, cmap=cmap, linewidth=0, antialiased=False)
        # ax1.view_init(elev=15)

        ax2  = fig.add_subplot(1, 2, 2, projection='3d')
        surf2 = ax2.plot_surface(self.x, self.y, z_predict, cmap=cmap, linewidth=0, antialiased=False)
        # ax2.view_init(elev=15)

        # Customize the z axis.
        # ax.set_zlim(-0.10, 1.40)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # add colorbar
        # fig.colorbar(surf2)

        # plt.show()

if __name__ == '__main__':
    run = RegressionAnalysis()
    # run.OrdinaryLeastSquares()
    run.PlotMultiple3D()
