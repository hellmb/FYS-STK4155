import matplotlib
import numpy as np
import define_colormap
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# set general plotting font consistent with LaTeX
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def accuracy_epoch(epoch, acc_train, acc_test, savefig=False):
    """
    plot the accuracy versus epoch
    param epoch: 1D array of epochs
    param cost_train: 2D array of accuracy for training data
    param cost_test: 2D array of accuracy for test data
    """

    fig = plt.figure(figsize=(10,6))
    plt.plot(epoch, acc_test[:,0], color='#EC407A', label='Test')
    plt.plot(epoch, acc_train[:,0], color='#AD1457', label='Train')
    plt.title('Accuracy as a function of epoch', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel(r'$\epsilon$', fontsize=15)
    plt.legend(loc='lower right', fontsize=15)
    plt.show()

    if savefig:
        fig.savefig('Figures/accuracy_epochs.png', dpi=200)

def cost_epoch(epoch, cost_train, cost_test, savefig=False):
    """
    plot the cost/loss versus epoch
    param epoch: 1D array of epochs
    param cost_train: 2D array of cost for training data
    param cost_test: 2D array of cost for test data
    """

    fig = plt.figure(figsize=(10,6))
    plt.plot(epoch, cost_test[:,0], color='#EC407A', label='Test')
    plt.plot(epoch, cost_train[:,0], color='#AD1457', label='Train')
    plt.title('Cost/loss as a function of epoch', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel(r'$C(\beta)$', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

    if savefig:
        fig.savefig('Figures/loss_epochs.png', dpi=200)

def accuracy_kfold(epoch, acc_train, acc_test, folds, savefig=False):
    """
    plot accuracy for all k-folds
    """

    color_train = ['#880E4F','#006064']
    color_test  = ['#EC407A','#80CBC4']

    fig = plt.figure(figsize=(10,6))
    plt.plot(epoch, acc_train[:,0], color=color_train[0], label=r'Train$_{\epsilon^{max}}$')
    plt.plot(epoch, acc_test[:,0], color=color_test[0], label=r'Test$_{\epsilon^{max}}$')
    plt.plot(epoch, acc_train[:,1], color=color_train[1], label=r'Train$_{\epsilon^{min}}$')
    plt.plot(epoch, acc_test[:,1], color=color_test[1], label=r'Test$_{\epsilon^{min}}$')
    plt.title(r'Accuracy as a function of epoch for $k$-fold cross-validation', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel(r'$\epsilon$', fontsize=15)
    plt.legend(loc='lower right', fontsize=15)
    plt.show()

    plt.show()

    if savefig:
        fig.savefig('Figures/kfold_accuracy_epochs.png', dpi=200)

def cost_kfold(epoch, cost_train, cost_test, folds, savefig=False):
    """
    plot cost/loss for all k-folds
    """

    color_train = ['#880E4F','#006064']
    color_test  = ['#EC407A','#80CBC4']

    fig = plt.figure(figsize=(10,6))
    plt.plot(epoch, cost_train[:,0], color=color_train[0], label=r'Train$_{\epsilon^{max}}$')
    plt.plot(epoch, cost_test[:,0], color=color_test[0], label=r'Test$_{\epsilon^{max}}$')
    plt.plot(epoch, cost_train[:,1], color=color_train[1], label=r'Train$_{\epsilon^{min}}$')
    plt.plot(epoch, cost_test[:,1], color=color_test[1], label=r'Test$_{\epsilon^{min}}$')
    plt.title(r'Cost/loss as a function of epoch for $k$-fold cross-validation', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel(r'$C(\beta)$', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

    if savefig:
        fig.savefig('Figures/kfold_cost_epochs.png', dpi=200)

def terrain(x, y, zt, zp, savefig=False):
    """
    plot terrain data
    """

    # plot figure
    fig = plt.figure(figsize=(15,6))

    # define costum colormap
    cmap = define_colormap.DefineColormap('candy')

    # plot surface
    ax1   = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(x, y, zt, cmap=cmap, linewidth=0, antialiased=False)
    ax1.view_init(azim=45, elev=20)
    ax1.set_title('Franke function', fontsize=20)
    ax1.set_xlabel(r'$x$', fontsize=15)
    ax1.set_ylabel(r'$y$', fontsize=15)
    ax1.set_zlabel(r'$z$', fontsize=15)

    ax2   = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(x, y, zp, cmap=cmap, linewidth=0, antialiased=False)
    ax2.view_init(azim=45, elev=20)
    ax2.set_title('Predicted Franke function', fontsize=20)
    ax2.set_xlabel(r'$x$', fontsize=15)
    ax2.set_ylabel(r'$y$', fontsize=15)
    ax2.set_zlabel(r'$z$', fontsize=15)

    ### add colorbar ###

    # ravel z and z_predict
    ztr = np.ravel(zt)
    zpr = np.ravel(zp)

    # determine maximum and minimum values of the colorbar
    determine_colormap = np.array([max(ztr), max(zpr), min(ztr), min(zpr)])
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
        fig.savefig('Figures/predicted_terrain.png', format='png', dpi=200, transparent=False)


def plot_benchmark(sgd_train, sgd_test, dc_train, dc_test, savefig=False):
    """
    function for plotting accuracy (benchmark)
    """

    fig = plt.figure(figsize=(10,6))
    plt.plot(sgd_test, color='#64B5F6', label=r'SGDClassifier test')
    plt.plot(sgd_train, color='#1565C0', label=r'SGDClassifier  train')
    # plt.plot(lr_test, color='#FFB74D', label=r'LogisticRegression test')
    # plt.plot(lr_train, color='#F4511E', label=r'LogisticRegression  train')
    plt.plot(dc_test, color='#EC407A', label=r'Developed code test')
    plt.plot(dc_train, color='#AD1457', label=r'Developed code train')
    plt.title('Accuracy score for the developed method and scikit-learn', fontsize=20)
    plt.xlabel('Number of bootstraps', fontsize=15)
    plt.ylabel('Maximum accuracy', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

    if savefig:
        fig.savefig('Figures/accuracy_sklearn_devcode.png', dpi=200)

def all_accuracy_kfold(epoch, acc_train, acc_test, folds, savefig=False):
    """
    plot accuracy for all k-folds
    """

    color_train = ['#880E4F','#311B92','#0D47A1','#006064','#1B5E20','#FF6F00','#BF360C']
    color_test  = ['#EC407A','#7986CB','#42A5F5','#80CBC4','#9CCC65','#FFD54F','#FFAB91']

    fig = plt.figure(figsize=(10,6))

    for k in range(folds):
        # plot all accuracy scores for every k-fold
        plt.plot(epoch, acc_train[:,k], label='Train, fold %s' % k)
        plt.plot(epoch, acc_test[:,k], label='Test, fold %s' % k)
        plt.title(r'Accuracy as a function of epoch for $k$-fold cross-validation', fontsize=20)
        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel(r'$\epsilon$', fontsize=15)
        plt.legend(loc='lower right', fontsize=15)

    plt.show()

    if savefig:
        fig.savefig('Figures/kfold_accuracy_epochs.png', dpi=200)

def all_cost_kfold(epoch, cost_train, cost_test, folds, savefig=False):
    """
    plot accuracy for all k-folds
    """

    color_train = ['#880E4F','#311B92','#0D47A1','#006064','#1B5E20','#FF6F00','#BF360C']
    color_test  = ['#EC407A','#7986CB','#42A5F5','#80CBC4','#9CCC65','#FFD54F','#FFAB91']

    fig = plt.figure(figsize=(10,6))

    for k in range(folds):
        # plot all accuracy scores for every k-fold
        plt.plot(epoch, cost_train[:,k], label='Train, fold %s' % k)
        plt.plot(epoch, cost_test[:,k], label='Test, fold %s' % k)
        plt.title(r'Cost/loss as a function of epoch for $k$-fold cross-validation', fontsize=20)
        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel(r'$C(\beta)$', fontsize=15)
        plt.legend(loc='lower right', fontsize=15)

    plt.show()

    if savefig:
        fig.savefig('Figures/kfold_cost_epochs.png', dpi=200)
