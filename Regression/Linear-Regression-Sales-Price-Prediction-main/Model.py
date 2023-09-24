#Linear Regression

# Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error #error calculation
from sklearn.metrics import r2_score #error calculation

# Importing the dataset
data=pd.read_csv('Salesdata.csv')
data.columns
# Check outliers in the data.
sns.boxplot(data['Newspaper'])
sns.boxplot(data['TV'])
sns.boxplot(data['Radio'])

# Let's see the correlation between different variables.
sns.heatmap(data.corr(),cmap="magma_r",annot=True)
plt.show()
#In the Heatmap, the variable TV seems to be most correlated with Sales. 
#So let's take and perform Linear Regression using with TV as our feature variable.

X=data['TV']
Y=data['Sales']
xval=X.values
yval=Y.values
x=xval.reshape(-1,1)
y=yval.reshape(-1,1)

# Splitting the dataset into the Training set and Test set
#21
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.25,random_state=21)

lin_reg=LinearRegression(normalize=True)
lin_reg.fit(X_train,Y_train)
#print(lin_reg.intercept_) 

# Predicting the Test set results
test_pred = lin_reg.predict(X_test)


# Visualising the Train set results
plt.scatter(X_train, Y_train, color = '#88c939')
plt.plot(X_train, lin_reg.predict(X_train), color = 'red')
plt.title('Sales Price Prediction (training set)')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()


# Visualising the Test set results
plt.scatter(X_test, Y_test, color = '#88c939')
plt.plot(X_train, lin_reg.predict(X_train), color = 'red')
plt.title('Sales Price Prediction (Test set)')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()

#Evaluating the Model
score=r2_score(Y_test,test_pred)
print("R2 Score is =",score) #printing the accuracy
print("MSE is =",mean_squared_error(Y_test,test_pred))
print("RMSE of is =",np.sqrt(mean_squared_error(Y_test,test_pred)))

#To save model
import pickle
pickle.dump(lin_reg, open('model.pkl','wb'))


#load and predict
import math
model = pickle.load( open('model.pkl','rb'))
res=model.predict([[180]])
res1=res[0][0]
print(round(res1))




