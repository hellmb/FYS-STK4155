import os
import numpy as np
import pandas as pd
import plotting_function
from imageio import imread
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from warnings import filterwarnings

def preprocessing(remove_data=False):
    """
    read and preprocess credit card data
    param remove_data: removes uncategorised observations in features
    """

    # ignore warnings
    filterwarnings('ignore')

    # get path to data file
    cwd = os.getcwd()
    file = cwd + '/default of credit card clients.xls'

    # read excel file and store NA values in dictionary
    na_values = {}
    df = pd.read_excel(file, header=1, index_col=0, na_values=na_values)

    if remove_data:
        # remove uncategorised values from EDUCATION, MARRIAGE and PAY_*
        df = df.drop(df[df.EDUCATION < 1].index)
        df = df.drop(df[df.EDUCATION > 4].index)

        df = df.drop(df[df.MARRIAGE < 1].index)
        df = df.drop(df[df.MARRIAGE > 3].index)

        # df_pay = [df.PAY_0, df.PAY_2, df.PAY_3, df.PAY_4, df.PAY_5, df.PAY_6]
        # vals   = [-2, 0]
        # for s in df_pay:
        #     for i in vals:
        #         df = df.drop(df[s == i].index)


    # remove lines of zeros from BILL_AMT* and PAY_AMT*
    df = df.drop(df[(df.BILL_AMT1 == 0) &
                    (df.BILL_AMT2 == 0) &
                    (df.BILL_AMT3 == 0) &
                    (df.BILL_AMT4 == 0) &
                    (df.BILL_AMT5 == 0) &
                    (df.BILL_AMT6 == 0)].index)

    df = df.drop(df[(df.PAY_AMT1 == 0) &
                    (df.PAY_AMT2 == 0) &
                    (df.PAY_AMT3 == 0) &
                    (df.PAY_AMT4 == 0) &
                    (df.PAY_AMT5 == 0) &
                    (df.PAY_AMT6 == 0)].index)

    # divide data into features and targets
    features = df.loc[:, df.columns != 'default payment next month'].values
    targets  = df.loc[:, df.columns == 'default payment next month'].values

    # use column transformer to one-hot encode categorical features and scale the other features
    preprocessor = ColumnTransformer([('ohe', OneHotEncoder(categories='auto'), [1,2,3]),
                                      ('ss', StandardScaler(), [0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22])])


    # transform X
    features = preprocessor.fit_transform(features, targets)

    # scale data
    # ss  = StandardScaler()
    # features2 = ss.fit_transform(features2)

    return features, targets

def onehotencode(targets):
    """
    one-hot encode targets
    """

    onehotencoder = OneHotEncoder()
    target_new = onehotencoder.fit_transform(targets).toarray()

    return target_new

def franke_function(x, y, noise=False):
    """
    function to calculate the Franke function
    param x, y: data points in x- and y-direction
    """

    term1 = 0.75*np.exp(-((9*x-2)**2)/4. - ((9*y-2)**2)/4.)
    term2 = 0.75*np.exp(-((9*x+1)**2)/49. - ((9*y+1)**2)/10.)
    term3 = 0.5*np.exp(-((9*x-7)**2)/4. - ((9*y-3)**2)/4.)
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    z = term1 + term2 + term3 + term4

    if noise:
        z += 0.1*np.random.randn(x.shape[0],1)

    return z

def visualise_data():
    """
    visualise payment status
    """

    # ignore warnings
    filterwarnings('ignore')

    # get path to data file
    cwd = os.getcwd()
    file = cwd + '/default of credit card clients.xls'

    # read excel file and store NA values in dictionary
    na_values = {}
    df = pd.read_excel(file, header=1, index_col=0, na_values=na_values)

    # plot payment status histograms
    plotting_function.plot_3d_hist(df)
