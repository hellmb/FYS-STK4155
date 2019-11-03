import matplotlib
import numpy as np
import matplotlib.pyplot as plt

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
