# -*- coding: utf-8 -*-
"""
Created on Thu May 19 22:15:14 2022

@author: rInasDS
"""

import numpy as np


class CustomLinearRegression:
    
    d = dict()
    
    def __init__(self, fit_intercept=True):

        self.fit_intercept = fit_intercept
        self.coefficient = ...
        self.intercept = ...
      

    def fit(self, X, y):
        self.X = X
        if self.fit_intercept is True:
            ones = np.array([1 for i in range(len(X.T))])
            X2 = np.vstack((ones, self.X))
            self.X = X2
            self.B = np.linalg.solve(X2 @ X2.T, X2 @ y)
            self.B = np.insert(self.B, 0, 0)
            self.intercept = self.B[1]
            self.coefficient = self.B[1:]
            self.d = {'Intercept': self.intercept,
                      'Coefficient': self.coefficient[1:]}
        else:    # if self.fit_intercept is False:
            self.B = np.linalg.solve(X @ X.T, X @ y)
            self.intercept = self.B[0]
            self.coefficient = self.B[0:]
            self.d = {'Intercept': self.intercept,
                      'Coefficient': self.coefficient}

    def predict(self, X):
        self.yhat = self.coefficient @ self.X
        return self.yhat
        
    def r2_score(self, y, yhat):  # r2 = SSxy / root(SSxx × SSyy)
        r2 = ((sum(y * self.yhat) - 1 / len(y) * (sum(y) * sum(self.yhat))) /\
              ((sum(y ** 2) - 1 / len(y) * sum(y) ** 2) * (sum(self.yhat ** 2) - 1 /\
                                                           len(y) * sum(self.yhat) ** 2)) ** 0.5) ** 2
        self.d['R2'] = r2  

    def rmse(self, y, yhat):
        rmse = ((y - self.yhat) ** 2).mean() ** 0.5
        self.d['RMSE'] = rmse  


x = np.array([2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7.87, 7.87])
w = np.array([65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 66.6, 96.1, 100., 85.9,94.3])
s = np.array([15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 15.2,15.2])
y = np.array([24., 21.6, 34.7, 33.4, 36.2, 28.7, 22.9, 27.1, 16.5, 18.9, 15.])

clr = CustomLinearRegression(fit_intercept=True)   
clr.fit((np.vstack((x, w, s))), y)
y_pred = clr.predict(np.vstack((x, w, s)))
clr.r2_score(y, y_pred)
clr.rmse(y, y_pred)


# -------------------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
#import pandas as pd
#import numpy as np


#df = pd.read_csv('data.csv')
#X = np.array([df['f1'],df['f2'],df['f3']]).T
#y = np.array(df['y'])
X = np.array([x,w,s]).T
y = np.array(y)
model = LinearRegression(fit_intercept=True).fit(X, y)  
y_pred = model.predict(X)
inter = model.intercept_
coef = model.coef_
r2 = r2_score(y, y_pred)
RMSE = np.sqrt(mean_squared_error(y, y_pred))

# Comparing two dictionaries
clr.d['Intercept'] -= inter
clr.d['Coefficient'] -= coef
clr.d['R2'] -= r2
clr.d['RMSE'] -= RMSE
print(clr.d)
