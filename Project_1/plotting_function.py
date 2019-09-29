import define_colormap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

def Plot3D(x, y, z, m, dim, function='franke', savefig=False):
    """
    create 3D plot of the Franke function
    input z: function to be plotted
    """

    if function == 'franke':
        title_string1 = r'Predicted Franke function with polynomial of %d$^{th}$ degree'
        title_string2 = 'Franke function'
        name_string1  = 'surface_franke_p' + str(m)
        name_string2  = 'surface_franke_function'
    elif function == 'terrain':
        title_string1 = r'Predicted terrain with polynomial of %d$^{th}$ degree'
        title_string2 = 'Original terrain'
        name_string1  = 'surface_terrain_p' + str(m)
        name_string2  = 'surface_terrain_function'
    else:
        print('Insert function ("franke" or "terrain").')
        raise NameError(function)

    # plot figure
    fig = plt.figure(figsize=(10,8))
    ax  = fig.gca(projection='3d')

    # resize array if z is the predicted function
    if len(z) > dim:
        z = np.reshape(z, (dim,dim))
        ax.set_title(title_string1 % m, fontsize=20)
        name_string = name_string1
    else:
        ax.set_title(title_string2, fontsize=20)
        name_string = name_string2

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


def PlotDuo3D(x, y, z, z_predict, m, dim, lambda_val=0, function='franke', method='OLS', savefig=False):
    """
    create multiple 3D subplot
    """

    if function == 'franke':
        title_string1 = 'Franke function'
        name_string   = 'dual_surface_franke_p' + str(m)
    elif function == 'terrain':
        title_string1 = 'Original terrain'
        name_string   = 'dual_surface_terrain_p' + str(m)
    else:
        print('Insert function ("franke" or "terrain").')
        raise NameError(function)

    if method == 'ridge':
        title_string2 = r'Ridge regression with $\lambda$ = %.3f,'
    elif method == 'lasso':
        title_string2 = r'Lasso regression with $\lambda$ = %.3f'
    else:
        title_string2 = 'Ordinary least squares'

    # plot figure
    fig = plt.figure(figsize=(15,6))

    fig.suptitle(r'Fitting a %d$^{th}$ degree polynomial' % m, fontsize=20)

    # define costum colormap
    cmap = define_colormap.DefineColormap('arctic')

    # plot surface
    ax1   = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(x, y, z, cmap=cmap, linewidth=0, antialiased=False)
    ax1.view_init(azim=45, elev=20)
    ax1.set_title(title_string1, fontsize=20)
    ax1.set_xlabel(r'$x$', fontsize=15)
    ax1.set_ylabel(r'$y$', fontsize=15)
    ax1.set_zlabel(r'$z$', fontsize=15)

    ax2   = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(x, y, z_predict, cmap=cmap, linewidth=0, antialiased=False)
    ax2.view_init(azim=45, elev=20)
    # ax2.set_title(title_string2, fontsize=20)
    ax2.set_xlabel(r'$x$', fontsize=15)
    ax2.set_ylabel(r'$y$', fontsize=15)
    ax2.set_zlabel(r'$z$', fontsize=15)

    if method != 'OLS':
        ax2.set_title(title_string2 % lambda_val, fontsize=20)
        # surf2._facecolors2d=surf2._facecolors3d
        # surf2._edgecolors2d=surf2._edgecolors3d
        # ax2.legend()
    else:
        ax2.set_title(title_string2, fontsize=20)

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

def PlotCuatro3D(x1, y1, z, z_ols, x2, y2, z_ridge, x3, y3, z_lasso, m, dim, lambda_val=0, function='franke', savefig=False):
    """
    plot four 3D subplots in one figure
    """

    if function == 'franke':
        title_string1 = 'Franke function'
        name_string   = 'cuatro_surface_franke_p' + str(m)
    elif function == 'terrain':
        title_string1 = 'Original terrain'
        name_string   = 'cuatro_surface_terrain_p' + str(m)
    else:
        print('Insert function ("franke" or "terrain").')
        raise NameError(function)

    title_string2 = 'Ordinary least squares'
    title_string3 = r'Ridge regression with $\lambda$ = %.3f'
    title_string4 = r'Lasso regression with $\lambda$ = %.3f'

    # plot figure
    fig = plt.figure(figsize=(12,9))

    fig.suptitle(r'Fitting a %d$^{th}$ degree polynomial' % m, fontsize=20)

    # define costum colormap
    cmap = define_colormap.DefineColormap('arctic')

    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0,0], projection='3d')
    ax2 = fig.add_subplot(gs[0,1], projection='3d')
    ax3 = fig.add_subplot(gs[1,0], projection='3d')
    ax4 = fig.add_subplot(gs[1,1], projection='3d')

    # plot surface
    surf1 = ax1.plot_surface(x1, y1, z, cmap=cmap, linewidth=0, antialiased=False)
    ax1.view_init(azim=45, elev=20)
    ax1.set_title(title_string1, fontsize=18)
    ax1.set_xlabel(r'$x$', fontsize=15)
    ax1.set_ylabel(r'$y$', fontsize=15)
    ax1.set_zlabel(r'$z$', fontsize=15)

    surf2 = ax2.plot_surface(x1, y1, z_ols, cmap=cmap, linewidth=0, antialiased=False)
    ax2.view_init(azim=45, elev=20)
    ax2.set_title(title_string2, fontsize=18)
    ax2.set_xlabel(r'$x$', fontsize=15)
    ax2.set_ylabel(r'$y$', fontsize=15)
    ax2.set_zlabel(r'$z$', fontsize=15)

    surf3 = ax3.plot_surface(x2, y2, z_ridge, cmap=cmap, linewidth=0, antialiased=False)
    ax3.view_init(azim=45, elev=20)
    ax3.set_title(title_string3 % lambda_val, fontsize=18)
    ax3.set_xlabel(r'$x$', fontsize=15)
    ax3.set_ylabel(r'$y$', fontsize=15)
    ax3.set_zlabel(r'$z$', fontsize=15)

    surf4 = ax4.plot_surface(x3, y3, z_lasso, cmap=cmap, linewidth=0, antialiased=False)
    ax4.view_init(azim=45, elev=20)
    ax4.set_title(title_string4 % lambda_val, fontsize=18)
    ax4.set_xlabel(r'$x$', fontsize=15)
    ax4.set_ylabel(r'$y$', fontsize=15)
    ax4.set_zlabel(r'$z$', fontsize=15)

    ### add colorbar ###

    # ravel z and z_predict
    zr     = np.ravel(z)
    zr_ols = np.ravel(z_ols)

    # determine maximum and minimum values of the colorbar
    determine_colormap = np.array([max(zr), max(zr_ols), min(zr), min(zr_ols)])
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
