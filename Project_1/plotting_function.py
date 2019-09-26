import define_colormap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

def Plot3D(x, y, z, m, dim, savefig=False):
    """
    create 3D plot of the Franke function
    input z: function to be plotted
    """

    # plot figure
    fig = plt.figure(figsize=(10,8))
    ax  = fig.gca(projection='3d')

    # resize array if z is the predicted function
    if len(z) > dim:
        z = np.reshape(z, (dim,dim))
        ax.set_title('Predicted Franke function with polynomial of degree %d' % m, fontsize=20)
        name_string = 'surface_p' + str(m)
    else:
        ax.set_title('Franke function', fontsize=20)
        name_string = 'surface_franke_function'

    # define costum colormap
    cmap = define_colormap.DefineColormap('arctic')

    # plot surface
    surf = ax.plot_surface(x, y, z, cmap=cmap, linewidth=0, antialiased=False)
    ax.view_init(azim=45, elev=20)
    ax.set_xlabel(r'$x$', fontsize=15)
    ax.set_ylabel(r'$y$', fontsize=15)
    ax.set_zlabel(r'$z$', fontsize=15)

    # add colorbar
    fig.colorbar(surf, shrink=0.6)

    plt.show()

    if savefig:
        fig.savefig('Figures/'+name_string+'.png', format='png', dpi=500, transparent=False)


def PlotMultiple3D(x, y, z, z_predict, m, dim, savefig=False):
    """
    create multiple 3D subplot of the Franke function
    """

    # plot figure
    fig = plt.figure(figsize=(15,6))

    # define costum colormap
    cmap = define_colormap.DefineColormap('arctic')

    # plot surface
    ax1  = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(x, y, z, cmap=cmap, linewidth=0, antialiased=False)
    ax1.view_init(azim=45, elev=20)
    ax1.set_title('Franke function', fontsize=20)
    ax1.set_xlabel(r'$x$', fontsize=15)
    ax1.set_ylabel(r'$y$', fontsize=15)
    ax1.set_zlabel(r'$z$', fontsize=15)

    ax2  = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(x, y, z_predict, cmap=cmap, linewidth=0, antialiased=False)
    ax2.view_init(azim=45, elev=20)
    ax2.set_title('Predicted Franke function with polynomial of degree %d' % m, fontsize=20)
    ax2.set_xlabel(r'$x$', fontsize=15)
    ax2.set_ylabel(r'$y$', fontsize=15)
    ax2.set_zlabel(r'$z$', fontsize=15)

    ### add colorbar ###
    # ravel z and z_predict
    zr         = np.ravel(z)
    zr_predict = np.ravel(z_predict)
    # determine maximum and minimum values of the colorbar
    determine_colormap = np.array([max(zr), max(zr_predict), min(zr), min(zr_predict)])
    min_val = determine_colormap.min()
    max_val = determine_colormap.max()

    # normalise values
    norm = cm.colors.Normalize(vmax=max_val, vmin=min_val)
    map = cm.ScalarMappable(cmap=cmap, norm=norm)

    # make the colorbar axis
    p0 = ax1.get_position().get_points().flatten()
    p1 = ax2.get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p0[0], 0, p1[2]-p0[0], 0.02])

    # plot colorbar
    map.set_array([])
    cbar = plt.colorbar(map, orientation='horizontal', cax=ax_cbar, shrink=0.7, fraction=.05, ticklocation='top')

    plt.show()

    if savefig:
        name_string = 'dual_surface_p' + str(m)
        fig.savefig('Figures/'+name_string+'.png', format='png', dpi=500, transparent=False)

def ErrorBars(beta, con_int, m, savefig=False):
    """
    function for creating error bar plot for confidence intervals
    """

    errorbar_size = np.zeros(len(beta))

    for i in range (len(beta)):
        errorbar_size[i] = con_int[i,1] - con_int[i,0]

    # add wiggle room to the y-axis
    y_range = np.linspace(min(errorbar_size)-min(errorbar_size), max(errorbar_size)+max(errorbar_size), len(beta))

    fig = plt.figure(figsize=(10,6))
    plt.errorbar(beta, y_range, yerr=errorbar_size, capsize=3, linestyle="None",
                    marker="s", markersize=7, color="black", mfc="black", mec="black")
    plt.title(r'95% confidence intervals of $\beta$', fontsize=20)
    plt.xlabel(r'$\beta$', fontsize=15)
    plt.ylabel(r'Confidence interval', fontsize=15)
    plt.show()

    if savefig:
        name_string = 'confidence_interval_p' + str(m)
        fig.savefig('Figures/'+name_string+'.png', format='png', dpi=500, transparent=False)

def PlotMSE(m_array, mse, list_of_lambdas, max_degree, reg_name, savefig=False):
    """
    plot the mean squared error against model complexity
    """

    fig = plt.figure(figsize=(10,6))
    for l in range(len(list_of_lambdas)):
        plt.plot(m_array, mse[:,l], label=r'$\lambda$ = %.3f' % list_of_lambdas[l])

    plt.title('Polynomial degree vs. mean squared error for '+reg_name+' regression', fontsize=20)
    plt.xlabel('Polynomial degree', fontsize=15)
    plt.ylabel('Mean squared error', fontsize=15)
    plt.xticks(np.arange(1,max_degree+1, step=1))
    plt.legend(fontsize=12)
    plt.show()

    if savefig:
        fig.savefig('Figures/mse_degree_'+reg_name+'.png', format='png', dpi=500)

def PlotR2Score(m_array, r2score, list_of_lambdas, max_degree, reg_name, savefig=False):
    """
    plot the mean squared error against model complexity
    """

    fig = plt.figure(figsize=(10,6))
    for l in range(len(list_of_lambdas)):
        plt.plot(m_array, r2score[:,l], label=r'$\lambda$ = %.3f' % list_of_lambdas[l])

    plt.title(r'Polynomial degree vs. $R^2$ score for '+reg_name+' regression', fontsize=20)
    plt.xlabel('Polynomial degree', fontsize=15)
    plt.ylabel(r'$R^2$ score', fontsize=15)
    plt.xticks(np.arange(1,max_degree+1, step=1))
    plt.legend(fontsize=12)
    plt.show()

    if savefig:
        fig.savefig('Figures/r2score_degree_'+reg_name+'.png', format='png', dpi=500)

def PlotMSETestTrain(m_array, mse_train, mse_test, max_degree, savefig=False):
    """
    plot the mean squared error against model complexity
    """

    fig = plt.figure(figsize=(10,6))
    plt.plot(m_array, mse_train, label='Train')
    plt.plot(m_array, mse_test, label='Test')
    plt.title('Model complexity vs. predicted error', fontsize=20)
    plt.xlabel('Polynomial degree', fontsize=15)
    plt.ylabel('Mean squared error', fontsize=15)
    plt.xticks(np.arange(1,max_degree+1, step=1))
    plt.legend(fontsize=12)
    plt.show()

    if savefig:
        fig.savefig('Figures/error_model_complexity.png', format='png', dpi=500)

def PlotMultipleMSETestTrain(m_array, mse_train, mse_test, list_of_lambdas, max_degree, reg_name, savefig=False):
    """
    plot the mean squared error against model complexity for different values of lambda
    """

    fig = plt.figure(figsize=(10,6))
    color_train = ['#880E4F','#311B92','#0D47A1','#006064','#1B5E20','#FF6F00','#BF360C']
    color_test  = ['#EC407A','#7986CB','#42A5F5','#80CBC4','#9CCC65','#FFD54F','#FFAB91']

    for l in range(len(list_of_lambdas)):
        # for m in range(max_degree):
        # plot model complexity vs. predicted error
        plt.plot(m_array, mse_train[:,l], color=color_train[l], label=r'Train, $\lambda$ = %.3f' % list_of_lambdas[l])
        plt.plot(m_array, mse_test[:,l], color=color_test[l], label=r'Test, $\lambda$ = %.3f' % list_of_lambdas[l])
        plt.legend(fontsize=12)

    plt.title('Model complexity vs. predicted error for '+reg_name+' regression', fontsize=20)
    plt.xlabel('Polynomial degree', fontsize=15)
    plt.ylabel('Mean squared error', fontsize=15)
    plt.xticks(np.arange(1,max_degree+1, step=1))
    plt.show()

    if savefig:
        fig.savefig('Figures/error_model_complexity_'+reg_name+'.png', format='png', dpi=500)

def PlotMultipleR2STestTrain(m_array, r2s_train, r2s_test, list_of_lambdas, max_degree, reg_name, savefig=False):
    """
    plot the mean squared error against model complexity for different values of lambda
    """

    fig = plt.figure(figsize=(10,6))
    color_train = ['#880E4F','#311B92','#0D47A1','#006064','#1B5E20','#FF6F00','#BF360C']
    color_test  = ['#EC407A','#7986CB','#42A5F5','#80CBC4','#9CCC65','#FFD54F','#FFAB91']

    for l in range(len(list_of_lambdas)):
        # for m in range(max_degree):
        # plot model complexity vs. predicted error
        plt.plot(m_array, r2s_train[:,l], color=color_train[l], label=r'Train, $\lambda$ = %.3f' % list_of_lambdas[l])
        plt.plot(m_array, r2s_test[:,l], color=color_test[l], label=r'Test, $\lambda$ = %.3f' % list_of_lambdas[l])
        plt.legend(fontsize=12)

    plt.title(r'Model complexity vs. $R^2$ score for '+reg_name+' regression', fontsize=20)
    plt.xlabel('Polynomial degree', fontsize=15)
    plt.ylabel(r'$R^2$ score', fontsize=15)
    plt.xticks(np.arange(1,max_degree+1, step=1))
    plt.show()

    if savefig:
        fig.savefig('Figures/r2score_model_complexity_'+reg_name+'.png', format='png', dpi=500)
