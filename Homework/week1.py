import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

x = np.random.rand(100,1)                   # random values given a shape (100 data points)
y = 5*x*x+0.1*np.random.randn(100,1)        # returns a sample from the standard normal distribution

### compute parametrisation of the data set fitting a second-order polynomial ###

# set up design matrix X
X    = np.zeros((100,1))
X[:] = 1
X    = np.append(X, x, axis=1)
X    = np.append(X, x**2, axis=1)

# compute beta values
beta = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))

print(beta)

### use scikit-learn to compute the parametrisation ###
poly = PolynomialFeatures(degree=2)
Xp   = poly.fit_transform(x)

# set fit_intercept=False when using linreg.coef_ to obtain all beta values
# set fit_intercept=True when using linreg.intercept_ to find the intercepting value beta0
linreg = LinearRegression(fit_intercept=False)
linreg.fit(Xp,y)

beta2 = linreg.coef_

print(beta2)

### use scikit-learn to compute the mean square error and the R2 score ###
ypredict = linreg.predict(Xp)

# averaged square difference between the estimated values and the actual values
mse = mean_squared_error(y, ypredict)

# statistical measure of how close the data are to the fitted regression line, and is always between 0 and 1
# 0% indicates that the model explains none of the variability of the response data around its mean
# 100% indicates that the model explains all the variability of the response data around its mean (perfect prediction)
r2s = r2_score(y, ypredict)

print('Mean squared error: %.3f' % mse)
print('R2 score: %.3f' % r2s)
