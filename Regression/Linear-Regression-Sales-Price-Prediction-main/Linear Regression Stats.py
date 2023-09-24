import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(11.0,8.0)
data=pd.read_csv('check.csv')
X=data.iloc[:,0]
Y=data.iloc[:,1]

plt.scatter(X,Y)
plt.show()


#Manual Implementation
# Building the model
X_mean = np.mean(X)
Y_mean = np.mean(Y)
n = len(X)
num = 0
den = 0
print(X_mean)
print(Y_mean)
for i in range(len(X)):
    num += (X[i] - X_mean)*(Y[i] - Y_mean)
    print(num)
    den += (X[i] - X_mean)**2
    
m = num / den
c = Y_mean - m*X_mean
print (m, c)
# Making predictions
Y_pred = m*X + c
plt.scatter(X, Y) # actual
plt.scatter(X, Y_pred, color='green')
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red') # predicted
plt.show()


#Calculating the error RMSE & R2- Score
#RMSE - Root-mean-square deviation
#R2 -  R2 score is used to evaluate the performance of a linear regression model. It is the amount 
#of the variation in the output dependent attribute which is predictable from the input independent variable(s).
rmse = 0
for i in range(n):
    y_pred= c + m* X[i]
    rmse += (Y[i] - y_pred) ** 2 
rmse = np.sqrt(rmse/n)
sum_pred = 0
sum_act = 0
for i in range(n):
 y_pred = (m*X[i]+c)
 sum_pred += (Y[i]-y_pred)**2
 sum_act +=(Y[i]-Y_mean)**2
r2 = 1-(sum_pred/sum_act)
print('rmse :',rmse)
print('R2-Score :',r2)





#Predict the Y value for 3000
def predict(x):
    y = m*x + c
    print(y)
predict(6)


#Implementation Using Scikit-Learn

#scikit-learn -Scikit-learn is a free software machine learning library for the Python programming language.
# It features various classification, regression and clustering algorithms.


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(11.0,8.0)
data=pd.read_csv('check.csv')
X=data.iloc[:,0]
Y=data.iloc[:,1]
x = np.array(X).reshape(-1,1)
y = np.array(Y).reshape(-1,1) 
lr = LinearRegression()
lr.fit(x,y)
y_pred = lr.predict(x)

mse = mean_squared_error(y,y_pred)
rmse = np.sqrt(mse)
r2_score = lr.score(x,y)
print('rmse :',rmse)
print('R2 Score :',r2_score)



print('Prediction Value :',lr.predict([[20]]))
