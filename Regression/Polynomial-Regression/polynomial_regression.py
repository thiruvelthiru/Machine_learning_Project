# Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
plt.scatter(X,y)
plt.show()

# Fitting Polynomial Regression to the dataset 
poly = PolynomialFeatures(degree = 6) 
X_poly = poly.fit_transform(X) 
poly.fit(X_poly, y) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y) 


# Visualising the Polynomial Regression results 
plt.scatter(X, y, color = 'blue') 
plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red') 
plt.title('Salary Prediction With Experience') 
plt.xlabel('Years of Experience') 
plt.ylabel('Salary') 
plt.show() 

#Prediction test
lin2.predict(poly.fit_transform([[6]])) 
