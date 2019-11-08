import matplotlib
import numpy as np
import pandas as pd
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

def accuracy_kfold(epoch, acc_train, acc_test, savefig=False):
    """
    plot accuracy for all k-folds
    """

    color_train = ['#880E4F','#006064']
    color_test  = ['#EC407A','#80CBC4']

    fig = plt.figure(figsize=(10,6))
    plt.plot(epoch, acc_train[:,0]*100, color=color_train[0], label=r'Train$_{\epsilon^{max}}$')
    plt.plot(epoch, acc_test[:,0]*100, color=color_test[0], label=r'Test$_{\epsilon^{max}}$')
    plt.plot(epoch, acc_train[:,1]*100, color=color_train[1], label=r'Train$_{\epsilon^{min}}$')
    plt.plot(epoch, acc_test[:,1]*100, color=color_test[1], label=r'Test$_{\epsilon^{min}}$')
    plt.title(r'Accuracy as a function of epoch', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel(r'$\epsilon$ [%]', fontsize=15)
    plt.legend(loc='lower right', fontsize=15)
    plt.show()

    plt.show()

    if savefig:
        fig.savefig('Figures/kfold_accuracy_epochs.png', dpi=200)

def cost_kfold(epoch, cost_train, cost_test, savefig=False):
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
    plt.title(r'Cost as a function of epoch', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel(r'$C(\beta)$', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

    if savefig:
        fig.savefig('Figures/kfold_cost_epochs.png', dpi=200)

def accuracy_keras(acc_train, acc_test, keras_acc_train, keras_acc_test, folds, savefig=False):
    """
    plot maximum accuracy scores from keras and developed neural network
    """

    x = np.linspace(1, folds, folds)

    color_train = ['#880E4F','#006064']
    color_test  = ['#EC407A','#80CBC4']

    fig = plt.figure(figsize=(10,6))
    plt.plot(x, acc_train*100, color=color_train[0], linestyle='--', label=r'Train')
    plt.plot(x, acc_test*100, color=color_test[0], label=r'Test')
    plt.plot(x, keras_acc_train*100, color=color_train[1], linestyle='--', label=r'Keras train')
    plt.plot(x, keras_acc_test*100, color=color_test[1], label=r'Keras test')
    plt.title(r'Final accuracy score for developed and keras neural networks', fontsize=20)
    plt.xlabel(r'Number of $k$-folds', fontsize=15)
    plt.ylabel(r'$\epsilon$ [%]', fontsize=15)
    plt.legend(loc='lower right', fontsize=15)
    plt.show()

    plt.show()

    if savefig:
        fig.savefig('Figures/final_accuracy_dev_keras.png', dpi=200)

def accuracy_scikit(acc_train, acc_test, sk_acc_train, sk_acc_test, folds, savefig=False):
    """
    plot maximum accuracy scores from scikit-learn and developed neural network
    """

    x = np.linspace(1, folds, folds)

    color_train = ['#880E4F','#006064']
    color_test  = ['#EC407A','#80CBC4']

    fig = plt.figure(figsize=(10,6))
    plt.plot(x, acc_train*100, color=color_train[0], linestyle='--', label=r'Train')
    plt.plot(x, acc_test*100, color=color_test[0], label=r'Test')
    plt.plot(x, sk_acc_train*100, color=color_train[1], linestyle='--', label=r'Scikit-learn train')
    plt.plot(x, sk_acc_test*100, color=color_test[1], label=r'Scikit-learn test')
    plt.title(r'Final accuracy score for developed and scikit-learn neural networks', fontsize=20)
    plt.xlabel(r'Number of $k$-folds', fontsize=15)
    plt.ylabel(r'$\epsilon$ [%]', fontsize=15)
    plt.legend(loc='lower right', fontsize=15)
    plt.show()

    plt.show()

    if savefig:
        fig.savefig('Figures/final_accuracy_dev_scikit.png', dpi=200)

def plot_surface(x, y, zt, zp, savefig=False):
    """
    plot terrain data
    """

    # plot figure
    fig = plt.figure(figsize=(18,8))

    # define costum colormap
    cmap = define_colormap.DefineColormap('arctic')

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
        fig.savefig('Figures/predicted_surface.png', format='png', dpi=200, transparent=False)

def golden_test1(epoch, acc_train, acc_test, savefig=False):
    """
    plot first golden test
    """

    color_train = ['#880E4F']
    color_test  = ['#EC407A']

    fig = plt.figure(figsize=(10,6))
    plt.plot(epoch, acc_train, color=color_train[0], label=r'Train')
    plt.plot(epoch, acc_test, color=color_test[0], label=r'Test')
    plt.title(r'Neural network validation: few input samples', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel(r'Accuracy', fontsize=15)
    plt.legend(loc='lower right', fontsize=15)
    plt.show()

    plt.show()

    if savefig:
        fig.savefig('Benchmarks/golden_test1.png', dpi=200)

def golden_test2(epoch, cost_train, cost_test, savefig=False):
    """
    plot second golden test
    """

    color_train = ['#880E4F']
    color_test  = ['#EC407A']

    fig = plt.figure(figsize=(10,6))
    plt.plot(epoch, cost_train, color=color_train[0], label=r'Train')
    plt.plot(epoch, cost_test, color=color_test[0], label=r'Test')
    plt.title(r'Neural network validation: shuffling labels', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel(r'Accuracy', fontsize=15)
    plt.legend(loc='lower right', fontsize=15)
    plt.show()

    plt.show()

    if savefig:
        fig.savefig('Benchmarks/golden_test2.png', dpi=200)

def benchmark_sgd(sgd_train, sgd_test, dc_train, dc_test, folds, savefig=False):
    """
    function for plotting accuracy (benchmark)
    """
    print('Hello')
    x = np.linspace(1, folds, folds)

    color_train = ['#880E4F','#006064']
    color_test  = ['#EC407A','#80CBC4']

    fig = plt.figure(figsize=(10,6))
    plt.plot(x, sgd_test*100, color=color_train[0], label=r'Scikit-learn test')
    plt.plot(x, sgd_train*100, color=color_train[0], linestyle='--', label=r'Scikit-learn train')
    plt.plot(x, dc_test*100, color=color_train[1], label=r'Test')
    plt.plot(x, dc_train*100, color=color_train[1], linestyle='--', label=r'Train')
    plt.title('Accuracy score for the developed method and scikit-learn', fontsize=20)
    plt.xlabel(r'Number of $k$-folds', fontsize=15)
    plt.ylabel('$\epsilon$ [%]', fontsize=15)
    plt.legend(loc='lower right', fontsize=15)
    plt.show()

    if savefig:
        fig.savefig('Figures/accuracy_sklearn_devcode.png', dpi=200)

def plot_3d_hist(df):
    """
    plot 3d histogram of credit card data features
    """

    pay0 = df.loc[:, df.columns == 'PAY_0'].values
    pay2 = df.loc[:, df.columns == 'PAY_2'].values
    pay3 = df.loc[:, df.columns == 'PAY_3'].values
    pay4 = df.loc[:, df.columns == 'PAY_4'].values
    pay5 = df.loc[:, df.columns == 'PAY_5'].values
    pay6 = df.loc[:, df.columns == 'PAY_6'].values

    df_pay    = [pay0, pay2, pay3, pay4, pay5, pay6]
    df_colors = ['#880E4F', '#311B92', '#0D47A1', '#006064', '#1B5E20', '#FF6F00']
    df_space  = [0, 10, 20, 30, 40, 50]
    # df_labels = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']

    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20)

    i = 0
    xs = np.linspace(-2, 9, 12)
    for c, z in zip(df_colors, df_space):

        ys = df_pay[i]

        hist, xedges = np.histogram(ys, bins=12, range=[-2,9])
        ax.bar(xs, hist, zs=z, zdir='y', color=c, alpha=0.8, width=1)

        i += 1

    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[1] = 'PAY_0'
    labels[2] = 'PAY_2'
    labels[3] = 'PAY_3'
    labels[4] = 'PAY_4'
    labels[5] = 'PAY_5'
    labels[6] = 'PAY_6'

    ax.set_title('History of past payment', fontsize=20)
    ax.set_xlabel('Repayment status', fontsize=15)
    ax.set_yticklabels(labels)
    ax.set_zlabel('Observations count', fontsize=15)
    plt.show()

    fig.savefig('Figures/pay_hist.png', dpi=200)

def test_regularisation(epoch, acc_train, acc_test, lambdas, savefig=False):
    """
    plot accuracy for train and test data for different regularisation hyper parameters
    """

    color_train = ['#880E4F','#311B92','#0D47A1','#006064','#1B5E20','#FF6F00','#BF360C']
    color_test  = ['#EC407A','#7986CB','#42A5F5','#80CBC4','#9CCC65','#FFD54F','#FFAB91']

    fig = plt.figure(figsize=(10,6))
    for i in range(len(lambdas)):
        plt.plot(epoch, acc_train[i][:], color=color_train[i], linestyle='--', label=r'Train, $\lambda= %s$' % str(lambdas[i]))
        plt.plot(epoch, acc_test[i][:], color=color_test[i], label=r'Test, $\lambda= %s$' % str(lambdas[i]))
    plt.title(r'Accuracy for different values of $\lambda$', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel(r'$\epsilon$ [%]', fontsize=15)
    plt.legend(loc='lower right', fontsize=15)
    plt.show()

    if savefig:
        fig.savefig('Figures/lambdas_accuracy.png', dpi=200)

def test_eta(epoch, acc_train, acc_test, etas, savefig=True):
    """
    plot accuracy for train and test data for different learning rates
    """

    color_train = ['#880E4F','#311B92','#0D47A1','#006064','#1B5E20','#FF6F00','#BF360C']
    color_test  = ['#EC407A','#7986CB','#42A5F5','#80CBC4','#9CCC65','#FFD54F','#FFAB91']

    fig = plt.figure(figsize=(10,6))
    for i in range(len(etas)):
        plt.plot(epoch, acc_train[i][:]*100, color=color_train[i], linestyle='--', label=r'Train, $\eta = %s$' % str(etas[i]))
        plt.plot(epoch, acc_test[i][:]*100, color=color_test[i], label=r'Test, $\eta = %s$' % str(etas[i]))
    plt.title(r'Accuracy for different values of the learning rate $\eta$', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel(r'$\epsilon$ [%]', fontsize=15)
    plt.legend(loc='lower right', fontsize=15)
    plt.show()

    if savefig:
        fig.savefig('Figures/etas_accuracy.png', dpi=200)

def test_minibatches(epoch, acc_train, acc_test, minibatches, savefig=True):
    """
    plot accuracy for train and test data for different learning rates
    """

    color_train = ['#880E4F','#311B92','#0D47A1','#006064','#1B5E20','#FF6F00','#BF360C']
    color_test  = ['#EC407A','#7986CB','#42A5F5','#80CBC4','#9CCC65','#FFD54F','#FFAB91']

    fig = plt.figure(figsize=(10,6))
    for i in range(len(minibatches)):
        plt.plot(epoch, acc_train[i][:]*100, color=color_train[i], linestyle='--', label=r'Train, mb$_{size} = %s$' % str(minibatches[i]))
        plt.plot(epoch, acc_test[i][:]*100, color=color_test[i], label=r'Test, mb$_{size} = %s$' % str(minibatches[i]))
    plt.title(r'Accuracy for different mini-batch sizes', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel(r'$\epsilon$ [%]', fontsize=15)
    plt.legend(loc='lower right', fontsize=15)
    plt.show()

    if savefig:
        fig.savefig('Figures/minibatch_accuracy.png', dpi=200)

# end of code
