import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import metrics

data = pd.read_csv('purhchase_data.csv')
data.head()

X = data.iloc[:,[2,3]].values
y = data.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.25, random_state= 0)
#feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#Fitting Decision Tree Classification to the Training set
classifier = DecisionTreeClassifier()
classifier = classifier.fit(X_train,y_train)
#prediction
y_pred = classifier.predict(X_test)#Accuracy
print('Accuracy Score:', metrics.accuracy_score(y_test,y_pred))
#Confusion Matrix
confusion_matrix(y_test, y_pred) 


#Test prediction
user_age_salary=[[30,100000]]
scaled_result = sc_X.transform(user_age_salary)
res=classifier.predict(scaled_result)
if res==1:
    print("He will purchase")
else:
    print("He do not purchase")


