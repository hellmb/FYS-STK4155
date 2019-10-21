import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

cwd = os.getcwd()
file = cwd + '/default of credit card clients.xls'

na_values = {}
df = pd.read_excel(file, header=1, index_col=0, na_values=na_values)

# remove uncategorised values from EDUCATION and MARRIAGE
df = df.drop(df[df.EDUCATION < 1].index)
df = df.drop(df[df.EDUCATION > 4].index)

df = df.drop(df[df.MARRIAGE < 1].index)
df = df.drop(df[df.MARRIAGE > 3].index)

# remove uncategorised values from PAY
df_pay = [df.PAY_0, df.PAY_2, df.PAY_3, df.PAY_4, df.PAY_5, df.PAY_6]
vals   = [-2, 0]
for s in df_pay:
    for i in vals:
        df = df.drop(df[s == i].index)


# remove zeros from BILL_AMT and PAY_AMT
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
X = df.loc[:, df.columns != 'default payment next month'].values
y = df.loc[:, df.columns == 'default payment next month'].values


# use OneHotEncoder to scale each row
onehotencoder = OneHotEncoder(categories='auto')
preprocessor  = ColumnTransformer([("", onehotencoder, [3])], remainder='passthrough')

# X = preprocessor.fit_transform(X)

print(onehotencoder.fit_transform(X))

print(X)
y.shape
