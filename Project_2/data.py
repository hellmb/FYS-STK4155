import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, Normalizer
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

        df_pay = [df.PAY_0, df.PAY_2, df.PAY_3, df.PAY_4, df.PAY_5, df.PAY_6]
        vals   = [-2, 0]
        for s in df_pay:
            for i in vals:
                df = df.drop(df[s == i].index)


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

    # find the unique values per feature
    onehotencoder = OneHotEncoder(categories='auto')

    # use column transformer to one-hot encode the gender feature and normalise all other features with the L2 norm
    preprocessor = ColumnTransformer([('onehotencoder', onehotencoder, [1,2,3]),
                                      ('norm1', Normalizer(norm='l1'), [0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22])])

    # decide on normaliser!!

    # transform X
    features = preprocessor.fit_transform(features, targets)

    return features, targets

def onehotencode(targets):
    """
    one-hot encode targets
    """

    onehotencoder = OneHotEncoder(categories='auto')
    preprocess   = ColumnTransformer([('onehot',onehotencoder,[0])])

    target_new = preprocess.fit_transform(targets)

    return target_new

def normalise_cancer_data(features,targets):
    """
    normalise cancer data feature matrix
    """

    preprocess = ColumnTransformer([('norm2',Normalizer(norm='l2'),[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])])

    norm_features = preprocess.fit_transform(features,targets)

    return norm_features

def design_matrix():
    """
    function that returns the design matrix X
    """

    # get feature matrix and targets
    X, y = preprocessing(remove_data=True)

    # set up the design matrix X
    X = np.c_[np.ones((X.shape[0], 1)), X]

    return X, y
