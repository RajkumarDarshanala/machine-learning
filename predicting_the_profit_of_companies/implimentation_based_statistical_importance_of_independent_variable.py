import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#calculating accuracy and error
y_error=y_test-y_pred
SUM = sum(y_error)
error  = SUM / sum(y_test)
accuracy= 1-error

#measuring statistical dependencies and folowing backward elimination
import statsmodels.api as sm
X = np.append(arr=np.ones((50,1)).astype(int),values =X,axis=1 )
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X_opt[:,1]
from sklearn.model_selection import train_test_split
X_train_opt, X_test_opt, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)

regressor.fit(X_train_opt,y_train)
y_pred_opt=regressor.predict(X_test_opt)

y_error_opt=y_test-y_pred_opt
SUM_opt = sum(y_error_opt)
error_opt  = SUM_opt / sum(y_test)
accuracy_opt= 1-error_opt
