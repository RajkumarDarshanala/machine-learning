# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:52:26 2020

@author: draj7
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[7.5]]))

#finding accuracy in prediction by taking a point 
# note it is not optimal prediction because i have taken only on point to calculate accuracy so it is not final accuracy
#it is just to show that model is a good predictor
(1-(y[5]-lin_reg_2.predict(poly_reg.fit_transform([[6]])))/y[5])*100
