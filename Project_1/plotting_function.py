import define_colormap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

def Plot3D(x, y, z, m, function='franke', savefig=False):
    """
    create 3D plot of the Franke function
    input z: function to be plotted
    """

    if function == 'franke':
        title_string = 'Franke function'
        name_string  = 'surface_franke'
    elif function == 'terrain':
        title_string = 'Original terrain'
        name_string  = 'surface_terrain'
    else:
        print('Insert function ("franke" or "terrain").')
        raise NameError(function)

    # plot figure
    fig = plt.figure(figsize=(10,8))
    ax  = fig.gca(projection='3d')

    ax.set_title(title_string, fontsize=20)
    name_string = name_string

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


def PlotDuo3D(x, y, z_f, z_t, m, dim, savefig=False):
    """
    create multiple 3D subplot
    """

    # plot figure
    fig = plt.figure(figsize=(15,6))

    # define costum colormap
    cmap = define_colormap.DefineColormap('arctic')

    # plot surface
    ax1   = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(x, y, z_f, cmap=cmap, linewidth=0, antialiased=False)
    ax1.view_init(azim=45, elev=20)
    ax1.set_title('Franke function', fontsize=20)
    ax1.set_xlabel(r'$x$', fontsize=15)
    ax1.set_ylabel(r'$y$', fontsize=15)
    ax1.set_zlabel(r'$z$', fontsize=15)

    ax2   = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(x, y, z_t, cmap=cmap, linewidth=0, antialiased=False)
    ax2.view_init(azim=45, elev=20)
    ax2.set_title('Real terrain data', fontsize=20)
    ax2.set_xlabel(r'$x$', fontsize=15)
    ax2.set_ylabel(r'$y$', fontsize=15)
    ax2.set_zlabel(r'$z$', fontsize=15)

    ### add colorbar ###

    # ravel z and z_predict
    zfr         = np.ravel(z_f)
    ztr = np.ravel(z_t)

    # determine maximum and minimum values of the colorbar
    determine_colormap = np.array([max(zfr), max(ztr), min(zfr), min(ztr)])
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
        fig.savefig('Figures/duo_franke_terrain.png', format='png', dpi=500, transparent=False)

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

def PlotMSE(m_array, mse, list_of_lambdas, max_degree, method='OLS', savefig=False):
    """
    plot the mean squared error against model complexity
    """

    fig = plt.figure(figsize=(10,6))
    for l in range(len(list_of_lambdas)):
        plt.plot(m_array, mse[:,l], label=r'$\lambda$ = %.3f' % list_of_lambdas[l])

    plt.title('Polynomial degree vs. mean squared error for '+method+' regression', fontsize=20)
    plt.xlabel('Polynomial degree', fontsize=15)
    plt.ylabel('Mean squared error', fontsize=15)
    plt.xticks(np.arange(1,max_degree+1, step=1))
    plt.legend(fontsize=12)
    plt.show()

    if savefig:
        fig.savefig('Figures/mse_degree_'+method+'.png', format='png', dpi=500)

def PlotR2Score(m_array, r2score, list_of_lambdas, max_degree, method='OLS', savefig=False):
    """
    plot the mean squared error against model complexity
    """

    fig = plt.figure(figsize=(10,6))
    for l in range(len(list_of_lambdas)):
        plt.plot(m_array, r2score[:,l], label=r'$\lambda$ = %.3f' % list_of_lambdas[l])

    plt.title(r'Polynomial degree vs. $R^2$ score for '+method+' regression', fontsize=20)
    plt.xlabel('Polynomial degree', fontsize=15)
    plt.ylabel(r'$R^2$ score', fontsize=15)
    plt.xticks(np.arange(1,max_degree+1, step=1))
    plt.legend(fontsize=12)
    plt.show()

    if savefig:
        fig.savefig('Figures/r2score_degree_'+method+'.png', format='png', dpi=500)

def PlotMSETestTrain(m_array, mse_train, mse_test, max_degree, function='franke', savefig=False):
    """
    plot the mean squared error against model complexity
    """

    fig = plt.figure(figsize=(10,6))
    plt.plot(m_array, mse_train, color='#880E4F', label='Train')
    plt.plot(m_array, mse_test, color='#EC407A',  label='Test')
    plt.title('Model complexity vs. predicted error for the OLS method', fontsize=20)
    plt.xlabel('Polynomial degree', fontsize=15)
    plt.ylabel('Mean squared error', fontsize=15)
    plt.xticks(np.arange(1,max_degree+1, step=1))
    plt.legend(fontsize=12)
    plt.show()

    if savefig:
        fig.savefig('Figures/error_complexity_OLS_'+function+'.png', format='png', dpi=500)

def PlotR2STestTrain(m_array, r2s_train, r2s_test, max_degree, function='franke', savefig=False):
    """
    plot the mean squared error against model complexity
    """

    fig = plt.figure(figsize=(10,6))
    plt.plot(m_array, r2s_train, color='#880E4F', label='Train')
    plt.plot(m_array, r2s_test, color='#EC407A',  label='Test')
    plt.title(r'Model complexity vs. $R^2$ score for the OLS method', fontsize=20)
    plt.xlabel('Polynomial degree', fontsize=15)
    plt.ylabel(r'$R^2$ score', fontsize=15)
    plt.xticks(np.arange(1,max_degree+1, step=1))
    plt.legend(fontsize=12)
    plt.show()

    if savefig:
        fig.savefig('Figures/r2score_complexity_OLS_'+function+'.png', format='png', dpi=500)

def PlotBiasVariance(m_array, bias, variance, max_degree, function='franke', savefig=False):
    """
    plot the bias-variance tradeoff against model complexity
    """

    fig = plt.figure(figsize=(10,6))
    plt.plot(m_array, bias, color='#880E4F', label='Bias')
    plt.plot(m_array, variance, color='#EC407A',  label='Variance')
    plt.title(r'Bias-variance tradeoff for the OLS method', fontsize=20)
    plt.xlabel('Polynomial degree', fontsize=15)
    plt.ylabel(r'Bias-variance', fontsize=15)
    plt.xticks(np.arange(1,max_degree+1, step=1))
    plt.legend(fontsize=12)
    plt.show()

    if savefig:
        fig.savefig('Figures/bias-variance_complexity_OLS_'+function+'.png', format='png', dpi=500)

def PlotMultipleMSETestTrain(m_array, mse_train, mse_test, list_of_lambdas, max_degree, method='ridge', function='franke', savefig=False):
    """
    plot the mean squared error against model complexity for different values of lambda
    """

    fig = plt.figure(figsize=(10,6))
    color_train = ['#880E4F','#311B92','#0D47A1','#006064','#1B5E20','#FF6F00','#BF360C']
    color_test  = ['#EC407A','#7986CB','#42A5F5','#80CBC4','#9CCC65','#FFD54F','#FFAB91']

    for l in range(len(list_of_lambdas)):
        plt.plot(m_array, mse_train[:,l], color=color_train[l], label=r'Train, $\lambda$ = %s' % str(list_of_lambdas[l]))
        plt.plot(m_array, mse_test[:,l], color=color_test[l], label=r'Test, $\lambda$ = %s' % str(list_of_lambdas[l]))
        plt.legend(fontsize=12)

    plt.title('Model complexity vs. predicted error for '+method+' regression', fontsize=20)
    plt.xlabel('Polynomial degree', fontsize=15)
    plt.ylabel('Mean squared error', fontsize=15)
    plt.xticks(np.arange(1,max_degree+1, step=1))
    plt.show()

    if savefig:
        fig.savefig('Figures/error_complexity_'+method+'_'+function+'.png', format='png', dpi=500)

def PlotMultipleR2STestTrain(m_array, r2s_train, r2s_test, list_of_lambdas, max_degree, method='ridge', function='franke', savefig=False):
    """
    plot the mean squared error against model complexity for different values of lambda
    """

    fig = plt.figure(figsize=(10,6))
    color_train = ['#880E4F','#311B92','#0D47A1','#006064','#1B5E20','#FF6F00','#BF360C']
    color_test  = ['#EC407A','#7986CB','#42A5F5','#80CBC4','#9CCC65','#FFD54F','#FFAB91']

    for l in range(len(list_of_lambdas)):
        plt.plot(m_array, r2s_train[:,l], color=color_train[l], label=r'Train, $\lambda$ = %s' % str(list_of_lambdas[l]))
        plt.plot(m_array, r2s_test[:,l], color=color_test[l], label=r'Test, $\lambda$ = %s' % str(list_of_lambdas[l]))
        plt.legend(fontsize=12)

    plt.title(r'Model complexity vs. $R^2$ score for '+method+' regression', fontsize=20)
    plt.xlabel('Polynomial degree', fontsize=15)
    plt.ylabel(r'$R^2$ score', fontsize=15)
    plt.xticks(np.arange(1,max_degree+1, step=1))
    plt.show()

    if savefig:
        fig.savefig('Figures/r2score_complexity_'+method+'_'+function+'.png', format='png', dpi=500)

def PlotMultipleBiasVariance(m_array, bias, variance, list_of_lambdas, max_degree, method='ridge', function='franke', savefig=False):
    """
    function for plotting the bias-variance tradeoff
    """

    fig = plt.figure(figsize=(10,6))
    color_train = ['#880E4F','#311B92','#0D47A1','#006064','#1B5E20','#FF6F00','#BF360C']
    color_test  = ['#EC407A','#7986CB','#42A5F5','#80CBC4','#9CCC65','#FFD54F','#FFAB91']

    for l in range(len(list_of_lambdas)):
        plt.plot(m_array, bias[:,l], color=color_train[l], label=r'Bias, $\lambda$ = %s' % str(list_of_lambdas[l]))
        plt.plot(m_array, variance[:,l], color=color_test[l], label=r'Variance, $\lambda$ = %s' % str(list_of_lambdas[l]))
        plt.legend(fontsize=12, loc='upper left')

    plt.title(r'Bias-variance tradeoff for '+method+' regression', fontsize=20)
    plt.xlabel('Polynomial degree', fontsize=15)
    plt.ylabel(r'Bias-variance', fontsize=15)
    plt.xticks(np.arange(1,max_degree+1, step=1))
    plt.show()

    if savefig:
        fig.savefig('Figures/bias-variance_complexity_'+method+'_'+function+'.png', format='png', dpi=500)

def PlotCuatro3D(x1, y1, z, z_ols, x2, y2, z_ridge, x3, y3, z_lasso, m, dim, lambda_val, function='franke', savefig=False):
    """
    plot four 3D subplots in one figure
    """

    if function == 'franke':
        title_string1 = 'Franke function'
        name_string   = 'cuatro_surface_franke_p' + str(m) + '_' + str(lambda_val)
    elif function == 'terrain':
        title_string1 = 'Original terrain'
        name_string   = 'cuatro_surface_terrain_p' + str(m) + '_' + str(lambda_val)
    else:
        print('Insert function ("franke" or "terrain").')
        raise NameError(function)

    title_string2 = 'Ordinary least squares'
    title_string3 = r'Ridge regression with $\lambda$ = %s'
    title_string4 = r'Lasso regression with $\lambda$ = %s'

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
    ax3.set_title(title_string3 % str(lambda_val), fontsize=18)
    ax3.set_xlabel(r'$x$', fontsize=15)
    ax3.set_ylabel(r'$y$', fontsize=15)
    ax3.set_zlabel(r'$z$', fontsize=15)

    surf4 = ax4.plot_surface(x3, y3, z_lasso, cmap=cmap, linewidth=0, antialiased=False)
    ax4.view_init(azim=45, elev=20)
    ax4.set_title(title_string4 % str(lambda_val), fontsize=18)
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

    if savefig:
        fig.savefig('Figures/'+name_string+'.png', format='png', dpi=500)
