import define_colormap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

def Plot3D(x, y, z, m, dim):
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

    # plt.show()

    fig.savefig('Figures/'+name_string+'.png', format='png', dpi=500, transparent=False)


def PlotMultiple3D(x, y, z, z_predict, zr, zr_predict, m, dim):
    """
    create multiple 3D subplot of the Franke function
    input zr: flattened true values of z
    input zr_predict: flattened predicted values of z
    """

    # calculate predicted Franke function
    # self.OrdinaryLeastSquares()
    z_predict = np.reshape(z_predict, (dim,dim))

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

    # plt.show()

    name_string = 'dual_surface_p' + str(m)
    fig.savefig('Figures/'+name_string+'.png', format='png', dpi=500, transparent=False)

def ErrorBars(beta, con_int, m):
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
    # plt.show()

    name_string = 'confidence_interval_p' + str(m)
    fig.savefig('Figures/'+name_string+'.png', format='png', dpi=500, transparent=False)
