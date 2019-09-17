import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.random.rand(100,1)                   # random values given a shape (100 data points)
y = 5*x*x+0.1*np.random.randn(100,1)        # returns a sample from the standard normal distribution

### write the Ridge regression method and compute the parametrisation for different values of lambda ###
