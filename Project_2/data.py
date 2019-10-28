import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.compose import ColumnTransformer

# make class?
def preprocessing(remove_data=False):
    """
    read and preprocess credit card data
    param remove_data: removes uncategorised observations in features
    """

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

    # divide data into features and target
    features = df.loc[:, df.columns != 'default payment next month'].values
    target   = df.loc[:, df.columns == 'default payment next month'].values

    # find the unique values per feature
    onehotencoder = OneHotEncoder(categories='auto')

    # use column transformer to one-hot encode the gender feature and normalise all other features with the L1 norm
    ### includ one-hot for SEX, EDUCATION and MARRIAGE - should PAY* be included as well? ###
    preprocessor = ColumnTransformer([('onehotencoder', onehotencoder, [1,2,3]),
                                      ('norm2', Normalizer(norm='l2'), [0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22])])

    # decide on normaliser!!

    # transform X
    features = preprocessor.fit_transform(features, target)

    return features, target

def design_matrix():
    """
    function that returns the design matrix X
    """

    # get feature matrix and target
    X, y = preprocessing(remove_data=True)

    # set up the design matrix X
    X = np.c_[np.ones((X.shape[0], 1)), X]

    return X, y
